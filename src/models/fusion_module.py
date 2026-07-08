import torch
import torch.nn as nn

class AttentionFusionModule(nn.Module):
    """
    Attention-based Fusion Module:
    Projects feature vectors from 3 domains (CNN, ViT, Frequency)
    into a common dimension, applies Multi-Head Self-Attention, and merges them.
    """
    def __init__(self, cnn_dim=1280, vit_dim=768, freq_dim=128, fusion_dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        
        # Linear projections to map all features to the common 'fusion_dim'
        self.proj_cnn = nn.Sequential(
            nn.Linear(cnn_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        self.proj_vit = nn.Sequential(
            nn.Linear(vit_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        self.proj_freq = nn.Sequential(
            nn.Linear(freq_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )
        
        # Self-attention module across the 3 modalities
        # Dimensions: (B, Seq_len=3, Embed_dim=fusion_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Output feature size after flattening (3 domains * fusion_dim)
        self.output_dim = fusion_dim * 3

    def forward(self, cnn_feats, vit_feats, freq_feats):
        # 1. Project all branches to common fusion_dim
        # Outputs shape: (B, fusion_dim)
        p_cnn = self.proj_cnn(cnn_feats)
        p_vit = self.proj_vit(vit_feats)
        p_freq = self.proj_freq(freq_feats)
        
        # 2. Stack into sequence: shape (B, 3, fusion_dim)
        stacked = torch.stack([p_cnn, p_vit, p_freq], dim=1)
        
        # 3. Apply Multi-Head Self-Attention
        # attn_output: (B, 3, fusion_dim)
        attn_output, attn_weights = self.self_attention(stacked, stacked, stacked)
        
        # Residual connection and normalization
        fused = self.norm(stacked + self.dropout(attn_output))
        
        # 4. Flatten the 3 modal tokens to preserve individual domain identity
        # Output shape: (B, 3 * fusion_dim)
        fused_flat = fused.view(fused.size(0), -1)
        
        return fused_flat, attn_weights
