import torch
import torch.nn as nn

class CNNTransformer(nn.Module):
    def __init__(self, num_classes=5, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # CNN Backbone (giống SimpleCNN nhưng bỏ classifier)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 112
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 56
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 28
            
            nn.Conv2d(128, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 14 → feature map 14x14
        )
        
        # Flatten spatial dimensions → sequence length = 14*14 = 196
        self.seq_len = 14 * 14
        
        # Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # CNN feature extraction
        x = self.backbone(x)                    # (B, embed_dim, 14, 14)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)        # (B, 196, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)                       # (B, embed_dim)
        
        return self.head(x)