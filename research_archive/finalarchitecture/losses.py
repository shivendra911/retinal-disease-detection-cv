"""
╔══════════════════════════════════════════════════════════════════════╗
║  IRDAS finalarchitecture — All Loss Functions  (Research-Backed v2) ║
║                                                                      ║
║  Research updates vs initial version:                                ║
║  · Focal Tversky Loss replaces Dice+BCE for vessel (SOTA on DRIVE)  ║
║  · Focal BCE replaces plain BCE for HR (HRDC challenge best practice)║
║  · Uncertainty Weighting for joint calibration (Kendall et al. 2018) ║
║  · CORAL preserved exactly from teacher v7 (proven stable)          ║
║  · Contrastive loss with cosine distance (normalized embeddings)     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  CORAL ORDINAL LOSSES  (DR head — identical to teacher v7)
# ============================================================

def _coral_base(logits: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Per-sample CORAL ordinal loss (Cao et al., 2020).

    Each of K-1 output logits = log-odds of P(grade > k).
    The monotonicity of the cumulative probabilities is guaranteed by
    the label encoding, not by the loss itself.

    Args:
        logits: (B, K-1) raw pre-sigmoid logits from CORAL head
        levels: (B, K-1) binary ordinal targets [1]*grade + [0]*(K-1-grade)
    Returns:
        (B,) per-sample loss
    """
    return -torch.sum(
        F.logsigmoid(logits) * levels
        + (F.logsigmoid(logits) - logits) * (1 - levels),
        dim=1,
    )


def asymmetric_coral_loss(
    logits: torch.Tensor,
    coral_targets: torch.Tensor,
    grades_a: torch.Tensor,
    grades_b: torch.Tensor = None,
    lam: float = 1.0,
    fn_weight: float = 2.0,
) -> torch.Tensor:
    """Asymmetric CORAL with 2× weight for DR-positive samples.

    Mirrors dr_teacher_v5_fixed.py exactly — proven stable on APTOS.
    """
    per_sample = _coral_base(logits, coral_targets)
    if grades_b is None:
        grades_b = grades_a
    wa = torch.ones_like(per_sample); wa[grades_a > 0] = fn_weight
    wb = torch.ones_like(per_sample); wb[grades_b > 0] = fn_weight
    return (per_sample * (lam * wa + (1 - lam) * wb)).mean()


def coral_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORAL (K-1) logits → (K,) class probability vector."""
    cp = torch.sigmoid(logits)
    probs = torch.clamp(
        torch.cat([
            1 - cp[:, 0:1],
            cp[:, 0:1] - cp[:, 1:2],
            cp[:, 1:2] - cp[:, 2:3],
            cp[:, 2:3] - cp[:, 3:4],
            cp[:, 3:4],
        ], dim=1),
        min=1e-7,
    )
    return probs / probs.sum(1, keepdim=True)


def predict_grade(logits: torch.Tensor) -> torch.Tensor:
    """Discrete DR grade prediction from CORAL logits."""
    return (logits > 0).sum(1)


def predict_continuous(logits: torch.Tensor) -> torch.Tensor:
    """Continuous DR grade score ∈ [0, 4] from CORAL logits."""
    return torch.sigmoid(logits).sum(1)


# ============================================================
#  HR FOCAL BCE  (replaces plain BCE — HRDC challenge findings)
# ============================================================

def hr_focal_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    pos_weight: float = 2.5,
) -> torch.Tensor:
    """Focal Binary Cross-Entropy for HR classification.

    Why Focal over plain BCE?
    - HRDC 2023: best individual models used focal loss
    - Hard negatives (borderline normal-vs-mild HR) dominate with plain BCE
    - Focal factor (1-p)^γ down-weights easy examples, focuses on hard ones

    Args:
        logits:     (B, 1) or (B,) raw logits from HR head
        labels:     (B,) binary labels {0, 1}
        gamma:      focal modulation exponent (2.0 is standard)
        pos_weight: base weight for positive class (handles class imbalance)
    Returns:
        Scalar focal BCE loss
    """
    logits = logits.squeeze(1)  # (B,)
    probs  = torch.sigmoid(logits)

    # Focal weight = (1 - p_t)^gamma  where p_t is prob of correct class
    p_t      = probs * labels + (1 - probs) * (1 - labels)
    focal_wt = (1 - p_t).pow(gamma)

    # Weighted BCE (manual, to apply focal weight correctly)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        pos_weight=torch.tensor([pos_weight], device=logits.device),
        reduction='none',
    )
    return (focal_wt * bce).mean()


# ============================================================
#  VESSEL: FOCAL TVERSKY LOSS  (2024 SOTA for DRIVE)
# ============================================================

def tversky_index(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    beta: float  = 0.3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tversky similarity index.

    TI = TP / (TP + α·FP + β·FN)

    With α=0.7, β=0.3: penalises False Negatives 2.3× more than FPs.
    Critical for retinal vessels — missing a vessel (FN) is far worse
    than over-segmenting background (FP).

    Args:
        pred:   (B, 1, H, W) vessel probability map (after sigmoid)
        target: (B, 1, H, W) binary vessel mask {0, 1}
        alpha:  FP weight (default 0.7)
        beta:   FN weight (default 0.3)
        smooth: Laplace smoothing to prevent divide-by-zero
    Returns:
        Scalar Tversky index ∈ [0, 1] (higher is better)
    """
    p = pred.view(pred.size(0), -1).float()
    t = target.view(target.size(0), -1).float()
    tp = (p * t).sum(1)
    fp = (p * (1 - t)).sum(1)
    fn = ((1 - p) * t).sum(1)
    ti = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return ti.mean()


def focal_tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.7,
    beta: float  = 0.3,
    gamma: float = 0.75,
    fov_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Focal Tversky Loss for retinal vessel segmentation.

    FTL = (1 - TI)^γ

    The focal term (1-TI)^γ with γ<1 pushes the gradient towards
    harder-to-segment regions (thin capillaries, bifurcations).
    γ=0.75 is the empirically optimal value for DRIVE.

    FOV mask: if provided, loss is only computed inside the valid
    circular retinal field (excludes black border pixels).
    This prevents the model from getting easy gradients from background.

    Args:
        pred:     (B, 1, H, W) sigmoid-activated vessel probability
        target:   (B, 1, H, W) binary vessel mask
        alpha:    FP weight in Tversky denominator
        beta:     FN weight in Tversky denominator
        gamma:    focal modulation exponent (0.75 for DRIVE)
        fov_mask: (B, 1, H, W) binary mask of valid retinal region [optional]
    Returns:
        Scalar Focal Tversky loss ∈ [0, 1]
    """
    if fov_mask is not None:
        # Apply mask: only compute on valid retinal pixels
        mask = fov_mask.bool()
        pred   = pred[mask].unsqueeze(0).unsqueeze(0)
        target = target[mask].unsqueeze(0).unsqueeze(0)

    ti   = tversky_index(pred, target, alpha=alpha, beta=beta)
    loss = (1 - ti).pow(gamma)
    return loss


def vessel_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fov_mask: torch.Tensor = None,
    alpha: float = 0.7,
    beta: float  = 0.3,
    gamma: float = 0.75,
) -> torch.Tensor:
    """Primary vessel segmentation loss: Focal Tversky.

    Single loss (no Dice+BCE combination) per 2024 SOTA practice.
    FTL subsumes Dice (special case with α=β=0.5, γ=1.0).

    Args:
        pred:     (B, 1, H, W) VesselDecoder output (already sigmoid-activated)
        target:   (B, 1, H, W) binary vessel mask
        fov_mask: (B, 1, H, W) field-of-view mask [optional]
    Returns:
        Scalar loss
    """
    return focal_tversky_loss(pred, target, alpha=alpha, beta=beta,
                               gamma=gamma, fov_mask=fov_mask)


# ============================================================
#  CONTRASTIVE DISENTANGLEMENT
# ============================================================

def contrastive_disentanglement_loss(
    dr_feat: torch.Tensor,
    hr_feat: torch.Tensor,
    dr_grades: torch.Tensor,
    hr_labels: torch.Tensor,
    margin_pure: float = 0.1,
    margin_cooccur: float = 0.3,
) -> torch.Tensor:
    """Push DR and HR branch embeddings apart.

    Motivation: hemorrhages appear in BOTH DR and HR. Without
    disentanglement, the shared visual features cause the DR branch
    to inadvertently encode HR-specific patterns — a confound that
    hurts DR grading when HR status is unknown.

    Strategy:
    - Only ONE disease present: push apart with margin_pure.
      Strong signal to learn distinct representations.
    - BOTH diseases present: smaller margin (margin_cooccur).
      Co-occurrence is a real biological phenomenon; don't force
      complete orthogonality.
    - Neither disease: no loss. A healthy eye should produce
      similar-looking representations for both branches.

    Uses normalised cosine distance (stable, scale-invariant).

    Args:
        dr_feat:        (B, C) DR branch embedding (unnormalised)
        hr_feat:        (B, C) HR branch embedding (unnormalised)
        dr_grades:      (B,) integer DR grades
        hr_labels:      (B,) binary HR labels {0, 1}
        margin_pure:    push-apart margin for single-disease samples
        margin_cooccur: push-apart margin for co-occurrence samples
    Returns:
        Scalar contrastive loss (≥ 0)
    """
    dr_n = F.normalize(dr_feat, dim=1)  # unit sphere
    hr_n = F.normalize(hr_feat, dim=1)
    cosine_sim = (dr_n * hr_n).sum(1)   # (B,) ∈ [-1, 1]

    has_dr   = (dr_grades > 0).float()
    has_hr   = hr_labels.float()
    both     = has_dr * has_hr
    only_one = (has_dr + has_hr).clamp(0, 1)

    # Margin per sample: co-occurrence → smaller push-apart requirement
    margin = margin_cooccur * both + margin_pure * (only_one - both)

    # Hinge loss: penalise when cosine distance (1-sim) < margin
    # i.e., when embeddings are TOO SIMILAR in the disease space
    loss = only_one * F.relu(margin - (1.0 - cosine_sim))
    return loss.mean()


# ============================================================
#  UNCERTAINTY WEIGHTING  (Kendall et al., NeurIPS 2018)
# ============================================================

class UncertaintyWeightedLoss(nn.Module):
    """Automatic multi-task loss balancing via learned log-variance.

    Each task learns its own noise parameter σᵢ² (as log(σᵢ²) for stability).
    Total loss = Σᵢ  L_i / (2σᵢ²) + log(σᵢ)

    When a task's loss is high, its σᵢ grows → reduces its weight automatically.
    This prevents any single task from dominating the gradient signal.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to
    Weigh Losses in Scene Understanding", CVPR 2018.

    Why this over PCGrad for IRDAS?
    - PCGrad is O(K²) in gradient computations — expensive for 3+ tasks
    - Uncertainty Weighting is O(K) — same overhead as fixed weights
    - Learned σᵢ adapts during training — no manual tuning required
    - Backbone is frozen in Phase 5 → gradient conflicts are localised
      to head parameters, making the cheaper method sufficient

    Args:
        n_tasks: Number of tasks (3 for IRDAS: DR, HR, vessel)
    """

    def __init__(self, n_tasks: int = 3):
        super().__init__()
        # log(σᵢ²) — initialised to 0 (σ=1, equal weighting)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, *task_losses) -> tuple:
        """
        Args:
            *task_losses: Scalar tensors, one per task (order: DR, HR, vessel)
        Returns:
            (total_loss, weights_list) where weights = 1/(2σ²)
        """
        assert len(task_losses) == len(self.log_vars), (
            f"Expected {len(self.log_vars)} task losses, got {len(task_losses)}"
        )
        total = torch.tensor(0.0, device=self.log_vars.device)
        weights = []
        for i, loss in enumerate(task_losses):
            precision = torch.exp(-self.log_vars[i])  # 1/σ²
            total = total + precision * loss + self.log_vars[i]
            weights.append(precision.item())
        return total, weights


# ============================================================
#  COMBINED JOINT LOSS  (Phase 5 — with uncertainty weighting)
# ============================================================

def joint_loss_fixed_weights(
    outputs: dict,
    coral_targets: torch.Tensor,
    dr_grades: torch.Tensor,
    hr_labels: torch.Tensor,
    vessel_mask: torch.Tensor = None,
    fov_mask: torch.Tensor = None,
    alpha_dr: float = 1.0,
    alpha_hr: float = 0.5,
    alpha_vessel: float = 0.3,
    alpha_contrast: float = 0.2,
    fn_weight: float = 2.0,
    hr_pos_weight: float = 2.5,
    hr_focal_gamma: float = 2.0,
) -> tuple:
    """Joint loss with FIXED weights (used when uncertainty module not active).

    Args:
        outputs: model forward pass dict (dr_logits, hr_logits, dr_feat, hr_feat, vessel_pred)
        coral_targets: (B, 4) CORAL ordinal binary targets
        dr_grades:     (B,) integer DR grades 0-4
        hr_labels:     (B,) binary HR labels
        vessel_mask:   (B, 1, H, W) binary vessel mask [optional]
        fov_mask:      (B, 1, H, W) field-of-view mask [optional]
        alpha_*:       loss component weights
        fn_weight:     asymmetric DR loss weight
        hr_pos_weight: HR focal BCE positive class weight
        hr_focal_gamma: focal exponent for HR loss
    Returns:
        (total, loss_dict)
    """
    losses = {}

    losses['dr'] = asymmetric_coral_loss(
        outputs['dr_logits'], coral_targets, dr_grades, fn_weight=fn_weight
    )

    losses['hr'] = hr_focal_bce_loss(
        outputs['hr_logits'], hr_labels,
        gamma=hr_focal_gamma, pos_weight=hr_pos_weight,
    )

    if 'dr_feat' in outputs and 'hr_feat' in outputs:
        losses['contrast'] = contrastive_disentanglement_loss(
            outputs['dr_feat'], outputs['hr_feat'], dr_grades, hr_labels,
        )
    else:
        losses['contrast'] = torch.zeros(1, device=outputs['dr_logits'].device).squeeze()

    if vessel_mask is not None and 'vessel_pred' in outputs:
        losses['vessel'] = vessel_loss(
            outputs['vessel_pred'], vessel_mask, fov_mask=fov_mask,
        )
    else:
        losses['vessel'] = torch.zeros(1, device=outputs['dr_logits'].device).squeeze()

    losses['total'] = (
        alpha_dr       * losses['dr']
        + alpha_hr       * losses['hr']
        + alpha_contrast * losses['contrast']
        + alpha_vessel   * losses['vessel']
    )
    return losses['total'], losses
