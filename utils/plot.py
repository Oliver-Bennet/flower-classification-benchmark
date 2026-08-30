import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def plot_curves(history, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_acc_curves.png'), dpi=150)
    plt.close()
    print(f"Curves saved to {save_dir}/loss_acc_curves.png")

def plot_confusion_matrix(cm, class_names, save_dir="outputs/plots"):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_dir}/confusion_matrix.png")