import torch
import os

def save_checkpoint(model, optimizer, epoch, metrics, path, is_best=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(state, path)
    
    if is_best:
        best_path = path.replace('.pth', '_best.pth')
        torch.save(state, best_path)
        print(f"Best model saved to {best_path}")

def load_checkpoint(model, optimizer=None, path=None, device='cuda'):
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    print(f"Loaded checkpoint from epoch {epoch}")
    return epoch, metrics