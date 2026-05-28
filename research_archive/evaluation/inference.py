"""
IRDAS Evaluation — SOTA Inference Pipeline
============================================

Production inference with:
1. Test-Time Augmentation (TTA) — 8 geometric transforms averaged
2. Multi-fold ensemble — average predictions across K fold models  
3. QWK threshold optimisation — learned thresholds from validation set

TTA alone: +0.01-0.02 QWK (free, zero training cost)
Fold ensemble: +0.02-0.04 QWK
Threshold opt: +0.01 QWK
Combined: +0.03-0.05 QWK over single model without TTA
"""

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from typing import List, Optional
from losses.task_loss import ordinal_logits_to_class


def tta_augmentations():
    """
    8 geometric augmentations for Test-Time Augmentation.
    
    These are lossless geometric transforms — no information is created
    or destroyed, making the average strictly more informative than
    any single view.
    
    Returns:
        List of (name, transform_fn) tuples
    """
    return [
        ('identity',    lambda x: x),
        ('hflip',       lambda x: TF.hflip(x)),
        ('vflip',       lambda x: TF.vflip(x)),
        ('hflip+vflip', lambda x: TF.hflip(TF.vflip(x))),
        ('rot90',       lambda x: TF.rotate(x, 90)),
        ('rot180',      lambda x: TF.rotate(x, 180)),
        ('rot270',      lambda x: TF.rotate(x, 270)),
        ('hflip+rot90', lambda x: TF.hflip(TF.rotate(x, 90))),
    ]


def predict_with_tta(model, image_batch, n_tta=8, return_logits=False):
    """
    Test-Time Augmentation: average predictions over N augmented views.
    
    For each image, applies N geometric transforms, runs each through
    the model, and averages the ordinal logit probabilities.
    
    Args:
        model: Trained MSDNet model (eval mode)
        image_batch: (B, C, H, W) input tensor
        n_tta: Number of TTA augmentations (max 8)
        return_logits: If True, return raw averaged logits instead of classes
    
    Returns:
        If return_logits: (B, K-1) averaged ordinal logits
        Else: (B,) integer class predictions
    """
    model.eval()
    augs = tta_augmentations()[:n_tta]
    all_logits = []
    
    with torch.no_grad():
        for name, aug_fn in augs:
            augmented = aug_fn(image_batch)
            outputs = model(augmented)
            all_logits.append(outputs['dr_logits'])
    
    # Average ordinal logits across TTA passes
    avg_logits = torch.stack(all_logits).mean(dim=0)  # (B, K-1)
    
    if return_logits:
        return avg_logits
    
    return ordinal_logits_to_class(avg_logits)


def ensemble_predict(models, image_batch, n_tta=8, return_logits=False):
    """
    Multi-fold ensemble with TTA.
    
    For K fold models with N TTA augmentations each, averages K×N=K*8
    predictions per image. This is the full inference stack.
    
    Args:
        models: List of trained MSDNet models (one per fold)
        image_batch: (B, C, H, W) input tensor
        n_tta: Number of TTA augmentations per model
        return_logits: If True, return raw averaged logits
    
    Returns:
        If return_logits: (B, K-1) averaged ordinal logits
        Else: (B,) integer class predictions
    """
    all_logits = []
    
    for model in models:
        fold_logits = predict_with_tta(
            model, image_batch, n_tta=n_tta, return_logits=True
        )
        all_logits.append(fold_logits)
    
    avg_logits = torch.stack(all_logits).mean(dim=0)
    
    if return_logits:
        return avg_logits
    
    return ordinal_logits_to_class(avg_logits)


def ordinal_logits_to_continuous(logits):
    """
    Convert ordinal logits to continuous grade prediction.
    
    Used as input for QWK threshold optimization.
    Sum of sigmoid probabilities gives a continuous grade estimate.
    
    Args:
        logits: (B, K-1) ordinal logits
    Returns:
        (B,) continuous grade predictions (0.0 to K-1.0 range)
    """
    return torch.sigmoid(logits).sum(dim=1)


def full_inference_pipeline(models, dataloader, thresholds=None, 
                           n_tta=8, device='cuda'):
    """
    Complete SOTA inference pipeline.
    
    Pipeline: TTA × fold ensemble → continuous predictions → 
              optimized thresholds → final integer grades
    
    Args:
        models: List of trained fold models
        dataloader: Test DataLoader
        thresholds: Optimized QWK thresholds (from validation). 
                    If None, uses default [0.5, 1.5, 2.5, 3.5]
        n_tta: Number of TTA augmentations
        device: Device for inference
    
    Returns:
        all_preds: (N,) integer predictions
        all_continuous: (N,) continuous predictions (for analysis)
    """
    if thresholds is None:
        thresholds = [0.5, 1.5, 2.5, 3.5]
    
    all_continuous = []
    
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images = batch[0].to(device)
        else:
            images = batch.to(device)
        
        # Ensemble + TTA → averaged ordinal logits
        avg_logits = ensemble_predict(
            models, images, n_tta=n_tta, return_logits=True
        )
        
        # Convert to continuous predictions
        continuous = ordinal_logits_to_continuous(avg_logits)
        all_continuous.append(continuous.cpu().numpy())
    
    all_continuous = np.concatenate(all_continuous)
    
    # Apply optimized thresholds
    from evaluation.metrics import apply_optimized_thresholds
    all_preds = apply_optimized_thresholds(all_continuous, thresholds)
    
    return all_preds, all_continuous
