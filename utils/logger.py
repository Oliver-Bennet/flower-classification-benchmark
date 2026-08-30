import os
import json
from datetime import datetime

class Logger:
    def __init__(self, log_dir="outputs/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, epoch, train_metrics, val_metrics, lr):
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_acc'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_acc'].append(val_metrics['accuracy'])
        self.history['lr'].append(lr)

        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"LR: {lr:.6f}")

    def save(self, filename=None):
        if filename is None:
            filename = f"history_{self.start_time}.json"
        path = os.path.join(self.log_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {path}")