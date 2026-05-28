"""
IRDAS Models — EfficientNet Backbone with Multi-Scale Feature Extraction
=========================================================================

Uses timm's EfficientNet with features_only=True to extract intermediate
feature maps at 3 spatial scales. These feed into the FPN for multi-scale fusion.

Default backbone: tf_efficientnet_b4_ns (NoisyStudent-pretrained, B4 variant).

Feature map scales (for 384×384 input with B4-NS):
- P3: 48×48 × 56ch  (stride 8)  — fine-grained: microaneurysms, small lesions
- P4: 24×24 × 160ch (stride 16) — medium: hemorrhages, exudates
- P5: 12×12 × 448ch (stride 32) — coarse: optic disc, large structures

Why EfficientNet-B4-NS?
- NoisyStudent pretraining (300M unlabeled images) → stronger transfer
- 19M parameters (fits 2×T4 GPUs with batch 16-32)
- 82.7% ImageNet top-1 (significantly better than B0's 77.1%)
- Compound scaling: balanced depth/width/resolution
- features_only mode makes FPN integration trivial
- Channel dims auto-detected via timm's feature_info API

The backbone_name parameter allows swapping to any timm-supported model
(e.g., 'efficientnet_b0', 'tf_efficientnet_b5_ns', 'convnext_base') without
code changes — channel dimensions are auto-detected.
"""

import torch
import torch.nn as nn
import timm
from typing import List, Tuple


class EfficientNetFPNBackbone(nn.Module):
    """
    EfficientNet backbone with intermediate feature extraction for FPN.
    
    We hook into stages 2, 3, 4 to get features at 3 scales.
    
    Default (tf_efficientnet_b4_ns at 384×384 input):
        P3: 48×48, 56 channels   (fine — microaneurysms, small lesions)
        P4: 24×24, 160 channels  (mid — hemorrhages, exudates)
        P5: 12×12, 448 channels  (coarse — optic disc, large structures)
    
    Channel dimensions are auto-detected from the backbone's feature_info,
    so any timm-compatible backbone can be used without code changes.
    
    Args:
        backbone_name: Name of the timm model to use as backbone.
                       Default: 'tf_efficientnet_b4_ns' (NoisyStudent B4).
        pretrained: Whether to load pretrained weights (ImageNet or NoisyStudent).
    """
    
    def __init__(
        self,
        backbone_name: str = 'tf_efficientnet_b4_ns',
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True
        )
        # features_only=True returns list of feature maps from each stage.
        # Auto-detect channel dimensions from timm's feature_info API.
        # For tf_efficientnet_b4_ns, feature_info.channels() = [24, 32, 56, 160, 448]
        # We take indices 2, 3, 4 corresponding to our P3, P4, P5.
        all_channels: List[int] = self.backbone.feature_info.channels()
        self.out_channels: List[int] = [all_channels[2], all_channels[3], all_channels[4]]
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract multi-scale features from the backbone.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
               For B4-NS default: (B, 3, 384, 384).
        
        Returns:
            P3: (B, C3, H/8, W/8)    — fine-grained features
                 B4-NS: (B, 56, 48, 48)
            P4: (B, C4, H/16, W/16)  — medium features
                 B4-NS: (B, 160, 24, 24)
            P5: (B, C5, H/32, W/32)  — coarse features
                 B4-NS: (B, 448, 12, 12)
        """
        features = self.backbone(x)
        P3 = features[2]   # stride 8
        P4 = features[3]   # stride 16
        P5 = features[4]   # stride 32
        return P3, P4, P5
