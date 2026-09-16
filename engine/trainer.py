from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from engine.early_stopping import EarlyStopping
from engine.metrics import MetricTracker
from utils.checkpoint import save_checkpoint
from utils.logger import TrainingLogger
from utils.timer import Timer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[Any] = None,
        cfg: Optional[Dict[str, Any]] = None,
        experiment_name: str = "run",
        num_classes: int = 102,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.cfg = cfg or {}
        self.experiment_name = experiment_name
        self.num_classes = num_classes

        train_cfg = self.cfg.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.use_amp = train_cfg.get("amp", False) and device.type == "cuda"
        self.grad_clip = train_cfg.get("gradient_clip")
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.scheduler_per_batch = (
            self.scheduler is not None
            and type(self.scheduler).__name__ == "OneCycleLR"
        )

        es_cfg = train_cfg.get("early_stopping", {})
        
        self.monitor = es_cfg.get("monitor", "val_macro_f1")
        self.monitor_mode = es_cfg.get("mode", "max")
        
        self.early_stopping = None
        
        if es_cfg.get("enabled", True):
            self.early_stopping = EarlyStopping(
                patience=es_cfg.get("patience", 10),
                min_delta=es_cfg.get("min_delta", 0.001),
                mode=self.monitor_mode,
            )

        log_dir = self.cfg.get("logging", {}).get("log_dir", "outputs/logs")
        self.logger = TrainingLogger(
            log_dir,
            experiment_name,
            console=False,
        )

        ckpt_dir = self.cfg.get("logging", {}).get(
            "checkpoint_dir", "outputs/checkpoints"
        )
        self.ckpt_dir = Path(ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.ckpt_dir / f"{experiment_name}_best.pth"

        self.best_metric = ( float("-inf") if self.monitor_mode == "max" else float("inf") )
        self.history = []

    def _train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        tracker = MetricTracker(
            num_classes=self.num_classes,
            average=self.cfg.get("evaluation", {}).get("average", "macro"),
        )
        pbar = tqdm(loader, desc="Train", leave=False)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=self.use_amp,):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None and self.scheduler_per_batch:
                self.scheduler.step()

            tracker.update(logits, labels, loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return tracker.compute()

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker(
            num_classes=self.num_classes,
            average=self.cfg.get("evaluation", {}).get("average", "macro"),
        )
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with torch.amp.autocast(
                    "cuda",
                    enabled=self.use_amp,):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            tracker.update(logits, labels, loss.item())
        return tracker.compute()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        total_timer = Timer()
        total_timer.start()

        for epoch in range(1, self.epochs + 1):
            epoch_timer = Timer()
            epoch_timer.start()

            train_metrics = self._train_one_epoch(train_loader)
            val_metrics = self._validate(val_loader)

            if self.scheduler is not None and not self.scheduler_per_batch:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = epoch_timer.stop()

            row = {
                "train_loss": train_metrics.get("loss", 0.0),
                "train_accuracy": train_metrics.get("accuracy", 0.0),
                "val_loss": val_metrics.get("loss", 0.0),
                "val_accuracy": val_metrics.get("accuracy", 0.0),
                "val_precision": val_metrics.get("precision", 0.0),
                "val_recall": val_metrics.get("recall", 0.0),
                "val_f1": val_metrics.get("f1", 0.0),
                "val_macro_f1": val_metrics.get("macro_f1", 0.0),
                "val_weighted_f1": val_metrics.get("weighted_f1", 0.0),
                "lr": lr,
                "epoch_time": epoch_time,
            }
            self.logger.log(epoch, row)
            self.history.append(row)

            print(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"train_loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} | "
                f"val_loss={row['val_loss']:.4f} acc={row['val_accuracy']:.4f} "
                f"macro_f1={row['val_macro_f1']:.4f} | lr={lr:.2e} | time={epoch_time:.1f}s"
            )

            monitor_val = row.get(
                self.monitor,
                row.get("val_macro_f1", row["val_accuracy"]),
            )

            is_better = (
                monitor_val > self.best_metric
                if self.monitor_mode == "max"
                else monitor_val < self.best_metric
            )
            if is_better:
                self.best_metric = monitor_val
                save_checkpoint(
                    self.best_path,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch=epoch,
                    metrics=row,
                    config=self.cfg,
                )
                print(f"  → saved best checkpoint ({self.monitor}={monitor_val:.4f})")

            if self.early_stopping is not None:
                if self.early_stopping(monitor_val, epoch=epoch):
                    print(f"Early stopping at epoch {epoch}")
                    break

        total_time = total_timer.stop()
        self.logger.close()

        return {
            "best_metric": self.best_metric,
            "best_checkpoint": str(self.best_path),
            "total_training_time": total_time,
            "history": self.history,
        }