"""
IRDAS Models — Auxiliary Vessel Segmentation Decoder
=====================================================

Lightweight U-Net style decoder for retinal vessel segmentation.

Purpose: AUXILIARY TASK ONLY — removed at inference time.
Forces the shared EfficientNet backbone to understand vascular anatomy.
Vessels are the substrate of both DR and HR — this is domain-justified.

Architecture:
    P5 (coarse) → upsample → concat P4 → conv → upsample → concat P3 → conv → upsample → 1ch mask

Channel dimensions are derived from the backbone's encoder_channels parameter,
so any backbone can be used without hardcoding channel sizes.

Target: Dice > 0.78 on DRIVE test set.
Training data: DRIVE dataset (40 images with pixel-level vessel masks).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class VesselDecoder(nn.Module):
    """
    Lightweight U-Net style decoder for retinal vessel segmentation.
    
    Auxiliary task only — removed at inference time.
    Forces shared EfficientNet encoder to understand vascular anatomy.
    Vessels are the substrate of both DR and HR — this is domain-justified.
    
    Target: Dice > 0.78 on DRIVE test set.
    
    Architecture:
        P5 → up_block → upsample → concat(P4) → up_block → upsample → concat(P3) → up_block → upsample → head
    
    Output spatial size is dynamically computed from the input tensor
    (P3 stride=8), so it works at any input resolution (224, 384, 512, etc.).
    
    Args:
        encoder_channels: List of channel sizes from backbone [P3_ch, P4_ch, P5_ch].
                          Auto-detected from backbone.out_channels.
                          B4-NS default: [56, 160, 448]
                          B0 legacy:     [40, 112, 320]
    """
    
    def __init__(self, encoder_channels: List[int]):
        super().__init__()
        # Unpack P3/P4/P5 channel dims from backbone
        p3_ch, p4_ch, p5_ch = encoder_channels
        
        # Decoder upsampling blocks — channel dims derived from encoder_channels
        mid_ch = p4_ch  # intermediate channel size matches P4 for balanced capacity
        self.up4 = self._up_block(p5_ch, mid_ch)              # P5 → mid_ch
        self.up3 = self._up_block(mid_ch + p4_ch, 64)         # concat(up4, P4) → 64ch
        self.up2 = self._up_block(64 + p3_ch, 32)             # concat(up3, P3) → 32ch
        
        # Final 1×1 conv to produce single-channel vessel probability map
        self.head = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def _up_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Double convolution block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, P3: torch.Tensor, P4: torch.Tensor, P5: torch.Tensor) -> torch.Tensor:
        """
        Args:
            P3: (B, P3_ch, H/8, W/8)   — fine features from backbone
            P4: (B, P4_ch, H/16, W/16) — medium features
            P5: (B, P5_ch, H/32, W/32) — coarse features
        
        Returns:
            vessel_pred: (B, 1, H, W) — vessel probability map at full input resolution.
                         Spatial size is dynamically inferred from P3 (stride 8).
        """
        # P5: H/32 → H/16
        x = self.up4(P5)
        x = F.interpolate(x, size=P4.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concat with P4 skip connection → H/16
        x = self.up3(torch.cat([x, P4], dim=1))
        x = F.interpolate(x, size=P3.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concat with P3 skip connection → H/8
        x = self.up2(torch.cat([x, P3], dim=1))
        
        # Upsample to full resolution — dynamically computed from P3 spatial dims
        # P3 is at stride 8, so full resolution = P3_spatial × 8
        full_h = P3.shape[2] * 8
        full_w = P3.shape[3] * 8
        x = F.interpolate(x, size=(full_h, full_w), mode='bilinear', align_corners=False)
        
        return self.sigmoid(self.head(x))  # (B, 1, H, W) vessel probability map
