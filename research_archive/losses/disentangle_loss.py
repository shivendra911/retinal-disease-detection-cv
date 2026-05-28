"""
IRDAS Losses — Contrastive Disentanglement Loss (NOVEL CONTRIBUTION)
=====================================================================

THIS IS THE CORE NOVEL CONTRIBUTION OF THE PAPER.

Problem:
    Both DR and HR cause retinal hemorrhages. Without this loss, the DR and HR
    branches learn nearly identical features for co-occurring cases. The model
    can't determine which disease caused a hemorrhage it sees.

Solution:
    Push DR and HR branch embeddings apart in feature space using a hinge loss
    on their cosine similarity.

Three cases handled explicitly:
    Case 1 (pure DR, no HR): cosine_similarity should be < margin_pure (0.1)
    Case 2 (pure HR, no DR): cosine_similarity should be < margin_pure (0.1)
    Case 3 (co-occurring):   cosine_similarity should be < margin_cooccur (0.3)
    Case 4 (neither):        no penalty (healthy images)

Why two different margins?
    Pure cases: branches should be very different (looking at different pathologies)
    Co-occurring: some feature sharing is justified (same vessels affected), but
                  not total overlap

No published retinal AI paper has this. This is the key novelty for journal submission.
"""

import torch
import torch.nn.functional as F


def contrastive_disentangle_loss(dr_feat, hr_feat, dr_label, hr_label,
                                  margin_pure=0.1, margin_cooccur=0.3):
    """
    Novel inter-disease contrastive disentanglement loss.
    
    Args:
        dr_feat:  (B, 256) — DR branch embedding
        hr_feat:  (B, 256) — HR branch embedding
        dr_label: (B,) — DR grade 0-4; 0 means no DR
        hr_label: (B,) — HR binary label; 0 means no HR
        margin_pure:    similarity ceiling for single-disease samples
        margin_cooccur: similarity ceiling for co-occurring samples
    
    Returns:
        Scalar loss value
    """
    # L2 normalize embeddings for stable cosine similarity
    dr_norm = F.normalize(dr_feat, dim=-1)
    hr_norm = F.normalize(hr_feat, dim=-1)
    cos_sim = (dr_norm * hr_norm).sum(dim=-1)  # (B,) in [-1, 1]
    
    # Masks for each case
    has_dr = (dr_label > 0).float()
    has_hr = (hr_label > 0).float()
    pure_dr   = has_dr * (1 - has_hr)      # DR only, no HR
    pure_hr   = has_hr * (1 - has_dr)      # HR only, no DR
    cooccur   = has_dr * has_hr            # both diseases present
    pure_mask = (pure_dr + pure_hr).clamp(max=1.0)  # any single disease
    
    # Hinge losses: penalize when similarity exceeds the margin
    L_pure    = pure_mask * F.relu(cos_sim - margin_pure)
    L_cooccur = cooccur   * F.relu(cos_sim - margin_cooccur)
    
    # Average over valid samples only (avoid division by zero)
    n_pure    = pure_mask.sum().clamp(min=1)
    n_cooccur = cooccur.sum().clamp(min=1)
    loss = L_pure.sum() / n_pure + L_cooccur.sum() / n_cooccur
    
    return loss
