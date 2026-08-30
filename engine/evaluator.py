import torch
from tqdm import tqdm
from engine.metrics import Metrics
from utils.plot import plot_confusion_matrix

class Evaluator:
    def __init__(self, model, test_loader, criterion, device, num_classes, class_names):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names

    def evaluate(self):
        self.model.eval()
        metrics = Metrics(self.num_classes)
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                metrics.update(outputs, labels, loss, images.size(0))
        
        results = metrics.compute()
        
        print("\n===== Test Results =====")
        print(f"Loss      : {results['loss']:.4f}")
        print(f"Accuracy  : {results['accuracy']:.4f}")
        print(f"Precision : {results['precision']:.4f}")
        print(f"Recall    : {results['recall']:.4f}")
        print(f"F1-score  : {results['f1']:.4f}")
        
        # Vẽ confusion matrix
        plot_confusion_matrix(results['confusion_matrix'], self.class_names)
        
        return results