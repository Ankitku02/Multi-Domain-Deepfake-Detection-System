import torch
import torch.nn as nn

class FrequencyBranch(nn.Module):
    """
    Frequency Branch: Extracts features from 2D FFT & DCT magnitude maps.
    Processes input of shape (B, 2, 224, 224).
    """
    def __init__(self, in_channels=2, feature_dim=128):
        super().__init__()
        print("Initializing Frequency Branch CNN...")
        
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 28x28 -> 14x14
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # (B, 128, 1, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # Input shape: (B, 2, 224, 224)
        x = self.features(x)  # Shape: (B, 128, 1, 1)
        x = self.fc(x)        # Shape: (B, feature_dim)
        return x
