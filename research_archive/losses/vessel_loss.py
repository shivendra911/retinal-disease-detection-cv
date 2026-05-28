"""
IRDAS Losses — Vessel Segmentation Loss
=========================================

Combined Dice + Binary Cross-Entropy loss for pixel-level vessel segmentation.

Why combined?
- Dice: Measures overlap between predicted and ground truth masks.
  Handles class imbalance naturally (vessels are only ~10% of pixels).
  But gradient can be noisy when prediction is very wrong.
  
- BCE: Per-pixel binary cross-entropy. More stable gradients than Dice alone.
  But doesn't handle imbalance well on its own.

Combined: stability of BCE + imbalance handling of Dice = best of both.

Used for training the auxiliary vessel decoder on DRIVE dataset.
"""

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE for vessel segmentation.
    
    Args:
        smooth: Smoothing factor to prevent division by zero in Dice (default 1.0)
    """
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred:   (B, 1, 224, 224) — predicted vessel probability map
            target: (B, 1, 224, 224) — ground truth vessel mask (binary)
        
        Returns:
            Scalar combined loss value
        """
        # Dice loss
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / \
                        (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        # BCE loss
        bce_loss = self.bce(pred, target)
        
        return dice_loss + bce_loss
