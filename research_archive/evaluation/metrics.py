"""
IRDAS Evaluation — Metrics Suite
==================================

All evaluation metrics used in the paper:
- Quadratic Weighted Kappa (QWK) — primary for DR grading
- AUC-ROC — primary for HR detection
- Expected Calibration Error (ECE) — model trustworthiness
- F1 Score — supplementary for both tasks
"""

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, roc_auc_score, f1_score,
    confusion_matrix, classification_report
)


def compute_qwk(y_true, y_pred):
    """
    Quadratic Weighted Kappa — primary metric for APTOS DR grading.
    
    Penalizes distant misclassifications more:
    - Predicting 0 when true is 4 → heavily penalized
    - Predicting 3 when true is 4 → lightly penalized
    
    Range: -1 to 1 (1 = perfect, 0 = random, <0 = worse than random)
    Target: > 0.89 for MSDNet
    """
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def compute_auc(y_true, y_prob):
    """
    AUC-ROC for HR binary detection.
    
    Measures ability to distinguish HR-positive from HR-negative
    across all classification thresholds.
    
    Range: 0 to 1 (1 = perfect, 0.5 = random)
    Target: > 0.91 for HR detection
    """
    return roc_auc_score(y_true, y_prob)


def compute_f1(y_true, y_pred, average='macro'):
    """
    F1 Score — harmonic mean of precision and recall.
    
    Args:
        average: 'macro' for multi-class (DR), 'binary' for HR
    """
    return f1_score(y_true, y_pred, average=average)


def compute_ece(y_prob, y_true, n_bins=15):
    """
    Expected Calibration Error — measures if confidence matches accuracy.
    
    A well-calibrated model saying "80% confident" should be correct 80% of the time.
    
    ECE < 0.05: well-calibrated (good for clinical use)
    ECE > 0.10: overconfident (dangerous in medical settings)
    
    Target: ECE < 0.06
    
    Args:
        y_prob: (N,) predicted probabilities for the positive class
        y_true: (N,) binary ground truth labels
        n_bins: Number of calibration bins
    
    Returns:
        ECE value (lower is better)
    """
    try:
        from netcal.metrics import ECE as NetCalECE
        ece = NetCalECE(n_bins)
        return ece.measure(y_prob, y_true)
    except ImportError:
        # Fallback manual implementation
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0
        for i in range(n_bins):
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece_val += bin_weight * abs(bin_acc - bin_conf)
        return ece_val


def get_confusion_matrix(y_true, y_pred, labels=None):
    """Generate confusion matrix for DR grading visualization."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def get_classification_report(y_true, y_pred, target_names=None):
    """Generate detailed classification report."""
    return classification_report(y_true, y_pred, target_names=target_names)


def optimize_qwk_thresholds(val_continuous_preds, val_labels, num_classes=5):
    """
    Optimize decision thresholds to maximize QWK on validation set.
    
    The model outputs continuous predictions (from ordinal regression).
    Default thresholds [0.5, 1.5, 2.5, 3.5] are suboptimal.
    Nelder-Mead optimisation finds thresholds that maximise QWK directly.
    
    Free +0.01 QWK improvement at zero computational cost.
    
    Args:
        val_continuous_preds: (N,) continuous grade predictions (float)
        val_labels: (N,) integer ground truth labels 0-4
        num_classes: Number of classes (5 for DR)
    
    Returns:
        optimized_thresholds: list of K-1 threshold values
    """
    from scipy.optimize import minimize
    
    def neg_qwk(thresholds):
        thresholds = sorted(thresholds)
        preds = np.zeros_like(val_continuous_preds, dtype=int)
        for i, t in enumerate(thresholds):
            preds[val_continuous_preds > t] = i + 1
        preds = np.clip(preds, 0, num_classes - 1)
        return -cohen_kappa_score(val_labels, preds, weights='quadratic')
    
    # Initial thresholds at midpoints
    x0 = [i + 0.5 for i in range(num_classes - 1)]
    result = minimize(neg_qwk, x0=x0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-4})
    return sorted(result.x.tolist())


def apply_optimized_thresholds(continuous_preds, thresholds):
    """
    Apply optimized thresholds to convert continuous predictions to classes.
    
    Args:
        continuous_preds: (N,) continuous grade predictions (float)
        thresholds: list of K-1 threshold values from optimize_qwk_thresholds
    
    Returns:
        (N,) integer class predictions
    """
    preds = np.zeros_like(continuous_preds, dtype=int)
    for i, t in enumerate(sorted(thresholds)):
        preds[continuous_preds > t] = i + 1
    return preds
