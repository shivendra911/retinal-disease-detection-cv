"""
IRDAS Models — CBAM (Convolutional Block Attention Module)
==========================================================

Dual attention mechanism: Channel Attention + Spatial Attention.

Channel Attention → "WHAT features matter" (which feature channels to emphasize)
Spatial Attention  → "WHERE to look" (which spatial locations are important)

Applied SEPARATELY in each disease branch:
- DR branch CBAM: learns to attend to microaneurysms, hemorrhages, exudates
- HR branch CBAM: learns to attend to AV nicking, vessel caliber changes

This per-branch attention is what enables disease-specific Grad-CAM++ heatmaps.

Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention: tells the network WHAT features matter.
    
    Mechanism:
    1. Global Average Pooling → captures average feature response
    2. Global Max Pooling → captures strongest feature response
    3. Shared MLP → learn channel relationships
    4. Sigmoid → produce channel-wise weights in [0, 1]
    5. Multiply with input → emphasize important channels
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for MLP bottleneck (default 16)
    """
    
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.shape
        avg = self.fc(self.avg_pool(x).view(b, c))
        mx  = self.fc(self.max_pool(x).view(b, c))
        scale = self.sigmoid(avg + mx).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Spatial Attention: tells the network WHERE to look.
    
    Mechanism:
    1. Average across all channels → (B, 1, H, W) average response map
    2. Max across all channels → (B, 1, H, W) strongest response map
    3. Concatenate → (B, 2, H, W)
    4. 7×7 convolution → spatial weight map
    5. Sigmoid → weights in [0, 1]
    6. Multiply with input → emphasize important spatial locations
    
    The output of this module's conv layer is used as the Grad-CAM++ target.
    
    Args:
        kernel_size: Convolution kernel size (7 is standard — captures local context)
    """
    
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Applied separately in each disease branch — each branch learns
    its OWN spatial and channel attention patterns:
    - DR branch: attends to microaneurysms, hemorrhage dots, exudates
    - HR branch: attends to arteriovenous nicking, vessel caliber changes
    
    The spatial attention's conv layer serves as the Grad-CAM++ target layer,
    enabling per-disease heatmap generation.
    
    Args:
        in_channels: Number of input feature channels
    """
    
    def __init__(self, in_channels):
        super().__init__()
        self.channel = ChannelAttention(in_channels)
        self.spatial = SpatialAttention()  # spatial.conv → Grad-CAM++ target
    
    def forward(self, x):
        x = self.channel(x)  # channel attention first
        x = self.spatial(x)  # then spatial attention
        return x
