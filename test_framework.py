import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from engine.trainer import Trainer
from engine.evaluator import Evaluator

# Dummy data
x = torch.randn(100, 3, 224, 224)
y = torch.randint(0, 5, (100,))
dataset = TensorDataset(x, y)
loader = DataLoader(dataset, batch_size=8)

# Dummy model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3*224*224, 128),
    nn.ReLU(),
    nn.Linear(128, 5)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

config = {'log_dir': 'outputs/logs'}

trainer = Trainer(model, loader, loader, optimizer, None, criterion, device, 5, config)
trainer.fit(epochs=2)

evaluator = Evaluator(model, loader, criterion, device, 5, [f'class_{i}' for i in range(5)])
evaluator.evaluate()