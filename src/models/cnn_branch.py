import torch
import torch.nn as nn
import timm

class CNNBranch(nn.Module):
    """
    CNN Branch: Extracts local spatial visual features from face crops.
    Typically uses pretrained EfficientNet-B0.
    """
    def __init__(self, model_name="efficientnet_b0", pretrained=True, feature_dim=1280):
        super().__init__()
        print(f"Initializing CNN Branch: {model_name}...")
        try:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            print(f"Successfully loaded pretrained {model_name}.")
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights for {model_name}. Fallback to random initialization. Error: {e}")
            self.model = timm.create_model(model_name, pretrained=False, num_classes=0)
            
        self.num_features = self.model.num_features
        
        # Project output features to match configured CNN feature dim
        if self.num_features != feature_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.num_features, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        # Input shape: (B, 3, 224, 224)
        features = self.model(x)  # Shape: (B, num_features)
        return self.projection(features)  # Shape: (B, feature_dim)
