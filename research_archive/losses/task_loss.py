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


class CoralOrdinalLoss(nn.Module):
    """
    CORAL (Consistent Rank Logits) ordinal loss for DR grading.
    
    DR grades 0→4 are ordinal — grade 3 is "closer to" grade 2 than grade 0.
    Standard cross-entropy treats all errors equally. CORAL penalises large
    rank errors more, perfectly matching QWK's weighting scheme.
    
    Uses coral-pytorch library. For K=5 classes, the model outputs K-1=4
    ordinal logits. Each logit represents P(grade > k).
    
    Label smoothing (ε) softens the binary ordinal targets to handle
    inter-grader disagreement at grade boundaries.
    
    Reference: Cao, Mirjalili, Raschka (2020). "Rank consistent ordinal
    regression for neural networks with application to age estimation."
    
    Args:
        num_classes: Number of ordinal classes (5 for DR grades 0-4)
        label_smoothing: Smoothing factor ε (0.1 recommended for DR)
    """
    def __init__(self, num_classes=5, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eps = label_smoothing
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, num_classes-1) — ordinal logits from CORAL head
            targets: (B,) — integer grade labels 0 to num_classes-1
        Returns:
            Scalar loss value
        """
        # Build ordinal binary targets: for grade k, targets are [1,1,...,1,0,...,0]
        # with k ones. Shape: (B, num_classes-1)
        levels = torch.arange(self.num_classes - 1, device=logits.device)
        ordinal_targets = (targets.unsqueeze(1) > levels.unsqueeze(0)).float()
        
        # Apply label smoothing to ordinal targets
        if self.eps > 0:
            ordinal_targets = ordinal_targets * (1 - self.eps) + (1 - ordinal_targets) * self.eps
        
        # Binary cross-entropy on each ordinal logit
        loss = F.binary_cross_entropy_with_logits(logits, ordinal_targets, reduction='mean')
        return loss


def ordinal_logits_to_class(logits):
    """
    Convert CORAL ordinal logits to class predictions.
    
    Each logit represents P(grade > k). The predicted class is the
    number of logits where sigmoid(logit) > 0.5.
    
    Args:
        logits: (B, K-1) ordinal logits
    Returns:
        (B,) integer class predictions
    """
    probas = torch.sigmoid(logits)
    return (probas > 0.5).sum(dim=1).long()
