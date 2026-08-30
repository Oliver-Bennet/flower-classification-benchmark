import torch
import torch.nn as nn
from tqdm import tqdm
from engine.metrics import Metrics
from utils.logger import Logger
from utils.checkpoint import save_checkpoint
from utils.plot import plot_curves

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, scheduler, 
                 criterion, device, num_classes, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.config = config
        
        self.logger = Logger(log_dir=config.get('log_dir', 'outputs/logs'))
        self.best_acc = 0.0

    def train_one_epoch(self, epoch):
        self.model.train()
        metrics = Metrics(self.num_classes)
        
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            metrics.update(outputs, labels, loss, images.size(0))
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return metrics.compute()

    def validate(self, epoch):
        self.model.eval()
        metrics = Metrics(self.num_classes)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                metrics.update(outputs, labels, loss, images.size(0))
        
        return metrics.compute()

    def fit(self, epochs, start_epoch=0):
        for epoch in range(start_epoch, epochs):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log(epoch, train_metrics, val_metrics, current_lr)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_acc
            if is_best:
                self.best_acc = val_metrics['accuracy']
            
            save_path = f"outputs/checkpoints/epoch_{epoch}.pth"
            save_checkpoint(
                self.model, self.optimizer, epoch, val_metrics, 
                save_path, is_best=is_best
            )
        
        # Lưu history + vẽ đồ thị
        self.logger.save()
        plot_curves(self.logger.history)
        
        print(f"\nTraining finished. Best Val Acc: {self.best_acc:.4f}")