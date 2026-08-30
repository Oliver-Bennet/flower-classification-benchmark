import torch
import torch.nn as nn
import yaml
import argparse
from torch.utils.data import DataLoader

from datasets.flower_dataset import FlowerDataset, get_transforms
from models import MLP, SimpleCNN, CNNTransformer, create_vit, create_swin, create_convnext, create_maxvit
from engine.trainer import Trainer
from engine.evaluator import Evaluator
from utils.checkpoint import load_checkpoint

def get_model(name, num_classes):
    if name == 'mlp':
        return MLP(num_classes=num_classes)
    elif name == 'cnn':
        return SimpleCNN(num_classes=num_classes)
    elif name == 'cnn_transformer':
        return CNNTransformer(num_classes=num_classes)
    elif name == 'vit':
        return create_vit(num_classes=num_classes, model_name='vit_tiny_patch16_224', pretrained=False)
    elif name == 'swin':
        return create_swin(num_classes=num_classes, model_name='swin_tiny_patch4_window7_224', pretrained=False)
    elif name == 'convnext':
        return create_convnext(num_classes=num_classes, model_name='convnext_tiny', pretrained=False)
    elif name == 'maxvit':
        return create_maxvit(num_classes=num_classes, model_name='maxvit_tiny_tf_224', pretrained=False)
    else:
        raise ValueError(f"Unknown model: {name}")

def main(args):
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_tf, val_tf = get_transforms()
    train_dataset = FlowerDataset(config['dataset']['root'], transform=train_tf, split='train')
    val_dataset   = FlowerDataset(config['dataset']['root'], transform=val_tf, split='valid')  # hoặc 'val'
    test_dataset  = FlowerDataset(config['dataset']['root'], transform=val_tf, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config['dataloader']['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=config['dataloader']['batch_size'], 
                              shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset, batch_size=config['dataloader']['batch_size'], 
                              shuffle=False, num_workers=4)
    
    num_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}")
    
    # Model
    model = get_model(args.model, num_classes)
    print(f"Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Loss
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training'].get('weight_decay', 0.05))
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training'].get('weight_decay', 0.05))
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    start_epoch = 0

    if args.resume is not None:
        start_epoch, metrics = load_checkpoint(
            model=model,
            optimizer=optimizer,
            path=args.resume,
            device=device
        )
        start_epoch += 1
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        config={'log_dir': f'outputs/logs/{args.model}'}
    )
    
    # Train
    trainer.fit(
        epochs=config['training']['epochs'],
        start_epoch=start_epoch
    )
    
    # Evaluate trên test set
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        class_names=train_dataset.classes
    )
    results = evaluator.evaluate()
    
    # Lưu kết quả
    import json
    with open(f'outputs/results/{args.model}_results.json', 'w') as f:
        json.dump({
            'model': args.model,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'params': sum(p.numel() for p in model.parameters())
        }, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                    choices=['mlp', 'cnn', 'cnn_transformer', 'vit', 'swin', 'convnext', 'maxvit'])
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    main(args)