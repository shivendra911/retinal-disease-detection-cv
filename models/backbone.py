"""
IRDAS Models — EfficientNet-B0 Backbone with Multi-Scale Feature Extraction
============================================================================

Uses timm's EfficientNet-B0 with features_only=True to extract intermediate
feature maps at 3 spatial scales. These feed into the FPN for multi-scale fusion.

Feature map scales (for 224×224 input):
- P3: 28×28 × 40ch  (stride 8)  — fine-grained: microaneurysms, small lesions
- P4: 14×14 × 112ch (stride 16) — medium: hemorrhages, exudates
- P5: 7×7   × 320ch (stride 32) — coarse: optic disc, large structures

Why EfficientNet-B0?
- 5.3M parameters (fits T4 GPU with batch 32)
- 77.1% ImageNet top-1 (better than ResNet-50 at half the params)
- Compound scaling: balanced depth/width/resolution
- features_only mode makes FPN integration trivial
"""

import torch
import torch.nn as nn
import timm


class EfficientNetFPNBackbone(nn.Module):
    """
    EfficientNet-B0 with intermediate feature extraction for FPN.
    
    We hook into stages 2, 3, 4 to get features at 3 scales.
    P3: 28×28, 40 channels  (fine — microaneurysms, small lesions)
    P4: 14×14, 112 channels (mid — hemorrhages, exudates)
    P5: 7×7,   320 channels (coarse — optic disc, large structures)
    
    Args:
        pretrained: Whether to load ImageNet-pretrained weights
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=pretrained, features_only=True
        )
        # features_only=True returns list of feature maps from each stage
        # EfficientNet-B0 stages produce these channel dimensions:
        # Stage 0: 16ch, Stage 1: 24ch, Stage 2: 40ch, Stage 3: 112ch, Stage 4: 320ch
        # We take indices 2, 3, 4 corresponding to our P3, P4, P5
        self.out_channels = [40, 112, 320]
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        
        Returns:
            P3: (B, 40, 28, 28)   — fine-grained features
            P4: (B, 112, 14, 14)  — medium features
            P5: (B, 320, 7, 7)    — coarse features
        """
        features = self.backbone(x)
        P3 = features[2]   # stride 8,  28×28
        P4 = features[3]   # stride 16, 14×14
        P5 = features[4]   # stride 32, 7×7
        return P3, P4, P5
