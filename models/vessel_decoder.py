"""
IRDAS Models — Auxiliary Vessel Segmentation Decoder
=====================================================

Lightweight U-Net style decoder for retinal vessel segmentation.

Purpose: AUXILIARY TASK ONLY — removed at inference time.
Forces the shared EfficientNet backbone to understand vascular anatomy.
Vessels are the substrate of both DR and HR — this is domain-justified.

Architecture:
    P5 (7×7×320) → upsample → concat P4 → conv → upsample → concat P3 → conv → upsample → 1ch mask

Target: Dice > 0.78 on DRIVE test set.
Training data: DRIVE dataset (40 images with pixel-level vessel masks).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VesselDecoder(nn.Module):
    """
    Lightweight U-Net style decoder for retinal vessel segmentation.
    
    Auxiliary task only — removed at inference time.
    Forces shared EfficientNet encoder to understand vascular anatomy.
    Vessels are the substrate of both DR and HR — this is domain-justified.
    
    Target: Dice > 0.78 on DRIVE test set.
    
    Architecture:
        P5 → up_block → upsample → concat(P4) → up_block → upsample → concat(P3) → up_block → upsample → head
    
    Args:
        encoder_channels: List of channel sizes from backbone [P3_ch, P4_ch, P5_ch]
                          Default: [40, 112, 320] for EfficientNet-B0
    """
    
    def __init__(self, encoder_channels=[40, 112, 320]):
        super().__init__()
        # Decoder upsampling blocks
        self.up4 = self._up_block(320, 112)          # P5 → 112ch
        self.up3 = self._up_block(112 + 112, 64)     # concat(up4, P4) → 64ch
        self.up2 = self._up_block(64 + 40, 32)       # concat(up3, P3) → 32ch
        
        # Final 1×1 conv to produce single-channel vessel probability map
        self.head = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def _up_block(self, in_ch, out_ch):
        """Double convolution block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, P3, P4, P5):
        """
        Args:
            P3: (B, 40, 28, 28)   — fine features from backbone
            P4: (B, 112, 14, 14)  — medium features
            P5: (B, 320, 7, 7)    — coarse features
        
        Returns:
            vessel_pred: (B, 1, 224, 224) — vessel probability map
        """
        # P5: 7×7 → 14×14
        x = self.up4(P5)                                                           # (B, 112, 7, 7)
        x = F.interpolate(x, size=P4.shape[-2:], mode='bilinear', align_corners=False)  # (B, 112, 14, 14)
        
        # Concat with P4 skip connection → 14×14
        x = self.up3(torch.cat([x, P4], dim=1))                                   # (B, 64, 14, 14)
        x = F.interpolate(x, size=P3.shape[-2:], mode='bilinear', align_corners=False)  # (B, 64, 28, 28)
        
        # Concat with P3 skip connection → 28×28
        x = self.up2(torch.cat([x, P3], dim=1))                                   # (B, 32, 28, 28)
        
        # Upsample to full resolution
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)  # (B, 32, 224, 224)
        
        return self.sigmoid(self.head(x))  # (B, 1, 224, 224) vessel probability map
