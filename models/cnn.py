import torch
import torch.nn as nn

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=5):
#         super().__init__()
        
#         self.features = nn.Sequential(
#             # Block 1
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 112x112
            
#             # Block 2
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 56x56
            
#             # Block 3
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 28x28
            
#             # Block 4
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),  # 14x14
#         )
        
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(128, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),          # 224 -> 112
            nn.Dropout(0.25),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),          # 112 -> 56
            nn.Dropout(0.25),

            # Block 3
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),          # 56 -> 28
            nn.Dropout(0.3),

            # Block 4
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),          # 28 -> 14
        )

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),

            nn.Linear(256,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256,num_classes)

        )

    def forward(self,x):

        x = self.features(x)

        x = self.classifier(x)

        return x