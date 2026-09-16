from __future__ import annotations

from typing import Optional


class EarlyStopping:

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False
        self.best_epoch: Optional[int] = None

    def __call__(self, score: float, epoch: int = 0) -> bool:
    
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "counter": self.counter,
            "best_score": self.best_score,
            "should_stop": self.should_stop,
            "best_epoch": self.best_epoch,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "mode": self.mode,
        }

    def reset(self) -> None:
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = None

    def load_state_dict(self, state: dict) -> None:
        self.counter = state["counter"]
        self.best_score = state["best_score"]
        self.should_stop = state["should_stop"]
        self.best_epoch = state["best_epoch"]
        self.patience = state["patience"]
        self.min_delta = state["min_delta"]
        self.mode = state["mode"]