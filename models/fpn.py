"""
IRDAS Models — Feature Pyramid Network (FPN)
=============================================

Multi-scale feature fusion module.

Problem: Microaneurysms are 1-5 pixels. The optic disc is ~200 pixels.
A single-scale backbone cannot detect both effectively.

Solution: FPN takes features from 3 scales (P3, P4, P5) and fuses them
using a top-down pathway with lateral connections:

    P5 (7×7)  → upsample → add to P4 (14×14) → upsample → add to P3 (28×28)
    
The result is a single feature map at the finest scale (28×28) that contains
both fine-grained detail (from P3) and semantic context (from P5).

Output: (B, 256, 28, 28) — this feeds into both disease branches.

Reference: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """
    Feature Pyramid Network for multi-scale retinal feature fusion.
    
    Motivation: microaneurysms are 1-5px, optic disc is ~200px.
    Single-scale backbone cannot see both. FPN solves this.
    
    Architecture:
        1. Lateral 1×1 convolutions to unify channel dimensions
        2. Top-down pathway: upsample coarse → add to fine
        3. 3×3 smoothing convolution to remove aliasing artifacts
    
    Args:
        in_channels_list: List of input channel sizes [P3_ch, P4_ch, P5_ch]
                          For EfficientNet-B0: [40, 112, 320]
        out_channels: Unified output channels (256 is standard in FPN literature)
    """
    
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        
        # Lateral 1×1 convolutions to unify channel dimensions
        self.lat5 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lat4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lat3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        
        # 3×3 convolution to smooth after upsampling (removes aliasing)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, P3, P4, P5):
        """
        Top-down pathway with lateral connections.
        
        Args:
            P3: (B, 40, 28, 28)  — fine features
            P4: (B, 112, 14, 14) — medium features
            P5: (B, 320, 7, 7)   — coarse features
        
        Returns:
            out: (B, 256, 28, 28) — fused multi-scale features
        """
        # Top-down pathway: upsample coarse features and add to finer ones
        p5 = self.lat5(P5)                                         # (B, 256, 7, 7)
        p5_up = F.interpolate(p5, size=P4.shape[-2:], mode='nearest')  # → (B, 256, 14, 14)
        p4 = self.lat4(P4) + p5_up                                     # (B, 256, 14, 14)
        p4_up = F.interpolate(p4, size=P3.shape[-2:], mode='nearest')  # → (B, 256, 28, 28)
        p3 = self.lat3(P3) + p4_up                                     # (B, 256, 28, 28)
        
        # Smooth the final feature map
        out = self.relu(self.bn3(self.smooth3(p3)))                 # (B, 256, 28, 28)
        
        return out
