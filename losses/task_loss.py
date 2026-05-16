"""
IRDAS Losses — Task-Specific Classification Losses
====================================================

Focal Loss for DR grading (5-class ordinal classification).

Problem: APTOS has ~50% Grade 0, but only ~2% Grade 4.
Standard cross-entropy focuses on easy normal samples and ignores rare severe cases.

Solution: Focal loss down-weights easy (correctly classified) samples:
    FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

When the model is confident and correct (p_t → 1): loss → 0
When the model is wrong (p_t → 0): full loss applied

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for DR grading.
    
    Down-weights easy normal (grade 0) samples — forces model to focus
    on learning the rare severe grades (3, 4).
    
    Args:
        class_weights: Tensor of per-class weights (from dataset class imbalance)
        gamma: Focusing parameter (2.0 is standard). Higher = more focus on hard samples.
    """
    
    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 5) — raw DR grade logits
            targets: (B,) — ground truth DR grades 0-4
        
        Returns:
            Scalar focal loss value
        """
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights, reduction='none'
        )
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()
