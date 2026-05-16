"""
IRDAS Models — Disease-Specific Classification Branches
========================================================

Each disease gets its own branch with:
- CBAM attention (learns disease-specific attention patterns)
- Global Average Pooling (spatial compression)
- MC Dropout (uncertainty estimation at inference)
- Linear classifier (disease-specific output)

Each branch returns BOTH:
1. logits — for classification loss
2. feature embedding — for contrastive disentanglement loss

DR Branch: 5-class output (grades 0-4)
HR Branch: 1-class output (binary: HR present/absent)
"""

import torch
import torch.nn as nn
from models.cbam import CBAM


class DiseaseSpecificBranch(nn.Module):
    """
    One branch per disease. Each has its own CBAM and classification head.
    
    Returns both logits (for prediction) and feature embedding
    (for contrastive disentanglement loss).
    
    dropout_rate: MC Dropout — KEEP ACTIVE at inference for uncertainty estimation.
    
    Args:
        in_channels: Number of input feature channels (256 from FPN)
        num_classes: Number of output classes (5 for DR, 1 for HR)
        dropout_rate: Dropout probability for MC Dropout (default 0.3)
    """
    
    def __init__(self, in_channels, num_classes, dropout_rate=0.3):
        super().__init__()
        self.cbam     = CBAM(in_channels)
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.dropout  = nn.Dropout(p=dropout_rate)
        self.fc       = nn.Linear(in_channels, num_classes)
    
    def forward(self, fpn_feat):
        """
        Args:
            fpn_feat: (B, 256, 28, 28) — FPN output features
        
        Returns:
            logits: (B, num_classes) — classification logits
            feat:   (B, 256) — feature embedding for contrastive loss
        """
        x    = self.cbam(fpn_feat)          # spatial + channel attention
        feat = self.gap(x).flatten(1)       # (B, 256) — the embedding
        feat = self.dropout(feat)           # MC Dropout (active during inference too)
        logits = self.fc(feat)              # (B, num_classes)
        return logits, feat                 # return both for loss computation
