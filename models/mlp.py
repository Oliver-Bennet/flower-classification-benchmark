import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes=5, img_size=224, hidden_dims=[512, 256, 128]):
        super().__init__()
        input_dim = 3 * img_size * img_size  # flatten RGB image
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten: (B, 3, H, W) → (B, 3*H*W)
        return self.network(x)