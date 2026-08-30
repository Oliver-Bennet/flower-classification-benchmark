import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

class Metrics:
    def __init__(self, num_classes, average='macro'):
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.total_loss = 0.0
        self.total_samples = 0

    def update(self, preds, labels, loss, batch_size):
        # preds: logits hoặc predicted class
        if preds.dim() > 1:
            preds = torch.argmax(preds, dim=1)
        
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        self.total_loss += loss.item() * batch_size
        self.total_samples += batch_size

    def compute(self):
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average=self.average, zero_division=0)
        recall = recall_score(labels, preds, average=self.average, zero_division=0)
        f1 = f1_score(labels, preds, average=self.average, zero_division=0)
        cm = confusion_matrix(labels, preds)
        avg_loss = self.total_loss / self.total_samples if self.total_samples > 0 else 0

        return {
            'loss': avg_loss,
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }