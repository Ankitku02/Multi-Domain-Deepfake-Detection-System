import torch
import torch.nn as nn
from .cnn_branch import CNNBranch
from .vit_branch import ViTBranch
from .frequency_branch import FrequencyBranch
from .fusion_module import AttentionFusionModule

class MultiDomainDeepfakeDetector(nn.Module):
    """
    Main detector model combining:
    - CNN Branch (EfficientNet-B0)
    - ViT Branch (ViT-B/16)
    - Frequency Branch (Custom CNN)
    - Attention Fusion Module
    - Classification Head (MLP)
    """
    def __init__(self, cfg=None):
        super().__init__()
        
        # Load hyperparameters from config dictionary or use defaults
        cfg_model = cfg.get("model", {}) if cfg else {}
        
        cnn_name = cfg_model.get("cnn_name", "efficientnet_b0")
        cnn_feature_dim = cfg_model.get("cnn_feature_dim", 1280)
        
        vit_name = cfg_model.get("vit_name", "vit_base_patch16_224")
        vit_feature_dim = cfg_model.get("vit_feature_dim", 768)
        
        freq_feature_dim = cfg_model.get("freq_feature_dim", 128)
        
        fusion_dim = cfg_model.get("fusion_dim", 256)
        num_heads = cfg_model.get("num_heads", 4)
        dropout = cfg_model.get("dropout", 0.3)
        
        # Initialize Branches
        pretrained = cfg_model.get("pretrained", False)
        self.cnn_branch = CNNBranch(model_name=cnn_name, pretrained=pretrained, feature_dim=cnn_feature_dim)
        self.vit_branch = ViTBranch(model_name=vit_name, pretrained=pretrained, feature_dim=vit_feature_dim)
        self.frequency_branch = FrequencyBranch(in_channels=2, feature_dim=freq_feature_dim)
        
        # Initialize Fusion Module
        self.fusion = AttentionFusionModule(
            cnn_dim=cnn_feature_dim,
            vit_dim=vit_feature_dim,
            freq_dim=freq_feature_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Final Classifier Head (Outputs 2 logits: class 0=real, class 1=fake)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )

    def forward(self, face_img, freq_map):
        """
        Forward pass.
        Args:
            face_img (Tensor): Crop of the face, shape (B, 3, 224, 224)
            freq_map (Tensor): FFT/DCT log-magnitude maps, shape (B, 2, 224, 224)
        Returns:
            logits (Tensor): Raw class predictions, shape (B, 2)
            attn_weights (Tensor): Fusion attention weights, shape (B, num_heads, 3, 3)
        """
        # 1. Feature extraction from individual branches
        cnn_feats = self.cnn_branch(face_img)        # (B, cnn_feature_dim)
        vit_feats = self.vit_branch(face_img)        # (B, vit_feature_dim)
        freq_feats = self.frequency_branch(freq_map)  # (B, freq_feature_dim)
        
        # 2. Fusion
        fused_feats, attn_weights = self.fusion(cnn_feats, vit_feats, freq_feats)
        
        # 3. Final Classification
        logits = self.classifier(fused_feats)
        
        return logits, attn_weights
