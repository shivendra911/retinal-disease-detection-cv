"""
╔══════════════════════════════════════════════════════════════════════╗
║  IRDAS finalarchitecture — Multi-Task Model Assembly                 ║
║                                                                      ║
║  Thin assembly layer that imports from models/ and adds the          ║
║  multi-task head structure needed for IRDAS Phases 2–5.              ║
║                                                                      ║
║  Architecture (same as MSDNet in models/msdnet.py):                 ║
║    EfficientNet-B4-NS backbone                                        ║
║    → FPN (multi-scale fusion)                                        ║
║    → DR branch  (CBAM + GAP + Dropout + FC[4]) — CORAL ordinal      ║
║    → HR branch  (CBAM + GAP + Dropout + FC[1]) — binary HR           ║
║    → Vessel decoder (U-Net style, training only) — auxiliary         ║
║                                                                      ║
║  This module also provides the teacher-architecture variant          ║
║  (MSDNetTeacher) from dr_teacher_v5_fixed.py for loading             ║
║  weights into the multi-task model.                                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional, List

# Add project root to sys.path for importing models/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.backbone     import EfficientNetFPNBackbone
from models.fpn          import FPN
from models.cbam         import CBAM
from models.vessel_decoder import VesselDecoder


# ============================================================
# BUILDING BLOCKS  (shared between Teacher and Multi-Task)
# ============================================================

class GeMPooling(nn.Module):
    """Generalized Mean Pooling.

    Learnable power p interpolates between average pooling (p→1)
    and max pooling (p→∞). Default p=3 works well for fine-grained
    medical image features.

    Reference: Radenović et al., "Fine-Tuning CNN Image Retrieval", 2018
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(self.eps).pow(self.p), x.shape[-2:]
        ).pow(1.0 / self.p)


class DRBranch(nn.Module):
    """DR classification branch with CBAM attention + GeM pooling.

    Uses multi-sample dropout (MSD) during training for regularisation:
    K forward passes with different dropout masks, averaged.

    Returns both logits AND the feature embedding (before FC) for
    contrastive disentanglement loss.
    """

    def __init__(self, in_channels: int, coral_levels: int = 4,
                 dropout: float = 0.3, msd_k: int = 5):
        super().__init__()
        self.cbam    = CBAM(in_channels)
        self.pool    = GeMPooling()
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(in_channels, coral_levels)
        self.msd_k   = msd_k

    def forward(self, x: torch.Tensor):
        x    = self.cbam(x)
        feat = self.pool(x).flatten(1)        # (B, in_channels) — embedding
        if self.training:
            logits = torch.stack([
                self.fc(self.dropout(feat)) for _ in range(self.msd_k)
            ]).mean(0)
        else:
            logits = self.fc(feat)
        return logits, feat                   # (B, 4), (B, in_channels)


class HRBranch(nn.Module):
    """HR binary classification branch.

    CBAM + GAP + Dropout + FC(1). Returns logit + embedding for
    contrastive loss (same interface as DRBranch).
    """

    def __init__(self, in_channels: int, dropout: float = 0.3):
        super().__init__()
        self.cbam    = CBAM(in_channels)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor):
        x    = self.cbam(x)
        feat = self.gap(x).flatten(1)         # (B, in_channels) — embedding
        feat = self.dropout(feat)
        logits = self.fc(feat)                # (B, 1)
        return logits, feat                   # (B, 1), (B, in_channels)


# ============================================================
# FULL IRDAS MULTI-TASK MODEL
# ============================================================

class IRDASModel(nn.Module):
    """IRDAS Multi-Task Model — Phases 2-5.

    Combines:
    - EfficientNet-B4-NS backbone (shared)
    - FPN for multi-scale feature fusion
    - DR branch (CORAL ordinal, 4 logits → 5 grades)
    - HR branch (binary sigmoid, 1 logit)
    - Vessel decoder (U-Net style, training only)

    The vessel decoder is removed at inference (training_mode=False).
    Each disease branch has its own CBAM for disease-specific attention,
    enabling disentangled representations.

    Args:
        backbone_name: timm model name (default: tf_efficientnet_b4.ns_jft_in1k)
        pretrained: Download pretrained weights if True
        fpn_out_channels: FPN output channels (default 256)
        dropout_rate: Dropout probability for both branches
        coral_levels: CORAL ordinal levels = num_classes - 1 = 4
        msd_k: Multi-sample dropout passes for DR branch (training only)
        drop_path_rate: Stochastic depth rate for backbone
    """

    def __init__(
        self,
        backbone_name: str = 'tf_efficientnet_b4.ns_jft_in1k',
        pretrained: bool = True,
        fpn_out_channels: int = 256,
        dropout_rate: float = 0.3,
        coral_levels: int = 4,
        msd_k: int = 5,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        # Backbone with drop_path_rate if supported by timm
        try:
            self.backbone_module = timm.create_model(
                backbone_name, pretrained=pretrained,
                features_only=True, out_indices=(2, 3, 4),
                drop_path_rate=drop_path_rate,
            )
        except TypeError:
            # Some timm versions don't accept drop_path_rate in features_only mode
            self.backbone_module = timm.create_model(
                backbone_name, pretrained=pretrained,
                features_only=True, out_indices=(2, 3, 4),
            )
        ch = self.backbone_module.feature_info.channels()  # [C3, C4, C5]

        # FPN: multi-scale feature fusion
        self.fpn = FPN(in_channels_list=list(ch), out_channels=fpn_out_channels)

        # Disease-specific branches (each has its own CBAM)
        self.dr_branch = DRBranch(
            in_channels=fpn_out_channels,
            coral_levels=coral_levels,
            dropout=dropout_rate,
            msd_k=msd_k,
        )
        self.hr_branch = HRBranch(
            in_channels=fpn_out_channels,
            dropout=dropout_rate,
        )

        # Auxiliary vessel decoder (U-Net style, training only)
        self.vessel_decoder = VesselDecoder(encoder_channels=list(ch))

        self.training_mode = True   # controls vessel decoder activation

    @property
    def backbone(self):
        """Alias for compatibility with training scripts that freeze 'backbone'."""
        return self.backbone_module

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input fundus image

        Returns:
            Dict with:
                'dr_logits':   (B, 4)       CORAL ordinal logits
                'hr_logits':   (B, 1)       HR binary logit
                'dr_feat':     (B, 256)     DR branch embedding
                'hr_feat':     (B, 256)     HR branch embedding
                'vessel_pred': (B, 1, H, W) vessel map (training only)
        """
        # Multi-scale backbone features
        feats = self.backbone_module(x)
        P3, P4, P5 = feats[0], feats[1], feats[2]

        # FPN fusion → single feature map at P3 resolution
        fpn_feat = self.fpn(P3, P4, P5)  # (B, 256, H/8, W/8)

        # Disease-specific predictions
        dr_logits, dr_feat = self.dr_branch(fpn_feat)
        hr_logits, hr_feat = self.hr_branch(fpn_feat)

        outputs = {
            'dr_logits': dr_logits,
            'hr_logits': hr_logits,
            'dr_feat':   dr_feat,
            'hr_feat':   hr_feat,
        }

        # Auxiliary vessel segmentation (training only)
        if self.training_mode:
            outputs['vessel_pred'] = self.vessel_decoder(P3, P4, P5)

        return outputs

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for p in self.backbone_module.parameters():
            p.requires_grad = False
        print("  🧊 Backbone FROZEN")

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for p in self.backbone_module.parameters():
            p.requires_grad = True
        print("  🔥 Backbone UNFROZEN")

    def get_param_groups(self, head_lr: float, backbone_lr: float) -> list:
        """Get optimizer parameter groups with differential learning rates.

        Backbone uses lower LR to preserve pretrained features.
        Heads use higher LR for faster adaptation to new task.

        Args:
            head_lr: Learning rate for heads + FPN
            backbone_lr: Learning rate for backbone (typically 0.1× head LR)
        Returns:
            List of param group dicts for AdamW
        """
        head_params = (
            list(self.fpn.parameters())
            + list(self.dr_branch.parameters())
            + list(self.hr_branch.parameters())
            + list(self.vessel_decoder.parameters())
        )
        backbone_params = list(self.backbone_module.parameters())
        return [
            {'params': head_params,     'lr': head_lr},
            {'params': backbone_params, 'lr': backbone_lr},
        ]

    def predict_grade(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert CORAL logits to discrete DR grade (0-4)."""
        return (logits > 0).sum(1)

    def predict_continuous(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert CORAL logits to continuous grade score ∈ [0, 4]."""
        return torch.sigmoid(logits).sum(1)


# ============================================================
# TEACHER-COMPATIBLE MODEL (for loading dr_teacher_v5_fixed.py weights)
# ============================================================

class DWConv(nn.Module):
    """Depthwise + Pointwise conv block (used in BiFPN teacher)."""

    def __init__(self, ch: int):
        super().__init__()
        self.dw  = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw  = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class BiFPNLayer(nn.Module):
    """Single BiFPN layer with learned fusion weights (from teacher v7)."""

    def __init__(self, ch: int = 256, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.w_p4_td  = nn.Parameter(torch.ones(2))
        self.w_p3_out = nn.Parameter(torch.ones(2))
        self.w_p4_out = nn.Parameter(torch.ones(3))
        self.w_p5_out = nn.Parameter(torch.ones(2))
        self.conv_p4_td  = DWConv(ch)
        self.conv_p3_out = DWConv(ch)
        self.conv_p4_out = DWConv(ch)
        self.conv_p5_out = DWConv(ch)

    def _up(self, x, t):
        return F.interpolate(x, t.shape[-2:], mode='nearest')

    def _dn(self, x, t):
        return F.adaptive_avg_pool2d(x, t.shape[-2:])

    def forward(self, p3, p4, p5):
        w4  = F.relu(self.w_p4_td.clone());  w4  /= (w4.sum()  + self.eps)
        w3  = F.relu(self.w_p3_out.clone()); w3  /= (w3.sum()  + self.eps)
        w4o = F.relu(self.w_p4_out.clone()); w4o /= (w4o.sum() + self.eps)
        w5o = F.relu(self.w_p5_out.clone()); w5o /= (w5o.sum() + self.eps)
        p4_td  = self.conv_p4_td(w4[0]*p4 + w4[1]*self._up(p5, p4))
        p3_out = self.conv_p3_out(w3[0]*p3 + w3[1]*self._up(p4_td, p3))
        p4_out = self.conv_p4_out(w4o[0]*p4 + w4o[1]*p4_td + w4o[2]*self._dn(p3_out, p4))
        p5_out = self.conv_p5_out(w5o[0]*p5 + w5o[1]*self._dn(p4_out, p5))
        return p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    """Multi-layer BiFPN (teacher architecture)."""

    def __init__(self, in_ch: list, out_ch: int = 256, n: int = 2):
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.SiLU())
            for c in in_ch
        ])
        self.layers = nn.ModuleList([BiFPNLayer(out_ch) for _ in range(n)])

    def forward(self, p3r, p4r, p5r):
        p3, p4, p5 = self.lat[0](p3r), self.lat[1](p4r), self.lat[2](p5r)
        for layer in self.layers:
            p3, p4, p5 = layer(p3, p4, p5)
        return p3, p4, p5


class ChannelAttentionTeacher(nn.Module):
    """Channel Attention from teacher architecture (used in teacher CBAM)."""

    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        mid = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid, 1, bias=False), nn.ReLU(),
            nn.Conv2d(mid, in_planes, 1, bias=False)
        )

    def forward(self, x):
        return x * torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttentionTeacher(nn.Module):
    """Spatial Attention from teacher architecture."""

    def __init__(self, ks: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=(ks - 1) // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAMTeacher(nn.Module):
    """CBAM from teacher (conv-based channel attention — different from models/cbam.py)."""

    def __init__(self, p: int):
        super().__init__()
        self.ca = ChannelAttentionTeacher(p)
        self.sa = SpatialAttentionTeacher()

    def forward(self, x):
        return self.sa(self.ca(x))


class MSDNetTeacher(nn.Module):
    """Teacher model architecture — EXACTLY matches dr_teacher_v5_fixed.py.

    Used ONLY to load the trained teacher weights before task arithmetic.
    After weight loading, use IRDASModel for multi-task training.

    This class must remain byte-for-byte identical to the one in
    dr_teacher_v5_fixed.py or weight loading will fail.
    """

    CORAL_LEVELS = 4

    def __init__(self, backbone: str = 'tf_efficientnet_b4.ns_jft_in1k',
                 local_weights: str = '',
                 drop_path_rate: float = 0.2,
                 bifpn_channels: int = 256,
                 bifpn_layers: int = 2,
                 msd_k: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        if local_weights and os.path.exists(local_weights):
            self.backbone = timm.create_model(
                backbone, pretrained=False,
                features_only=True, out_indices=(2, 3, 4),
                drop_path_rate=drop_path_rate,
            )
            if local_weights.endswith('.safetensors'):
                from safetensors.torch import load_file
                sd = load_file(local_weights, device='cpu')
            else:
                sd = torch.load(local_weights, map_location='cpu')
                sd = sd.get('state_dict', sd.get('model', sd))
            self.backbone.load_state_dict(sd, strict=False)
            print("  ✅ Teacher backbone loaded from local weights")
        else:
            self.backbone = timm.create_model(
                backbone, pretrained=True,
                features_only=True, out_indices=(2, 3, 4),
                drop_path_rate=drop_path_rate,
            )

        ch = self.backbone.feature_info.channels()
        oc = bifpn_channels
        self.bifpn   = BiFPN(ch, oc, bifpn_layers)
        self.pool    = GeMPooling()
        self.cbam_p3 = CBAMTeacher(oc)
        self.cbam_p5 = CBAMTeacher(oc)
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(oc * 2, self.CORAL_LEVELS)
        self.msd_k   = msd_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        p3, _, p5 = self.bifpn(f[0], f[1], f[2])
        feat = torch.cat([
            self.pool(self.cbam_p3(p3)).flatten(1),
            self.pool(self.cbam_p5(p5)).flatten(1),
        ], 1)
        if self.training:
            return torch.stack([
                self.head(self.dropout(feat)) for _ in range(self.msd_k)
            ]).mean(0)
        return self.head(feat)


# ============================================================
# WEIGHT TRANSFER UTILITY
# ============================================================

def transfer_teacher_to_irdas(
    teacher_state_dict: dict,
    irdas_model: IRDASModel,
) -> dict:
    """Transfer DR teacher backbone weights into an IRDASModel.

    The teacher uses timm's backbone directly (self.backbone.*),
    while IRDASModel wraps it as self.backbone_module.*.

    We remap teacher backbone keys → IRDASModel backbone_module keys
    and load them with strict=False (heads will not match).

    Args:
        teacher_state_dict: State dict from MSDNetTeacher / swa_final.pth
        irdas_model: Target IRDASModel instance
    Returns:
        Dict of missing/unexpected keys for inspection
    """
    # Remap 'backbone.' prefix → 'backbone_module.'
    remapped = {}
    for k, v in teacher_state_dict.items():
        if k.startswith('backbone.'):
            new_key = 'backbone_module.' + k[len('backbone.'):]
            remapped[new_key] = v
        # Skip BiFPN, head, CBAM — not compatible with IRDASModel

    missing, unexpected = irdas_model.load_state_dict(remapped, strict=False)
    backbone_loaded = sum(1 for k in remapped if k in {
        n for n, _ in irdas_model.named_parameters()
    })
    print(f"  ✅ Transferred {backbone_loaded} backbone tensors from teacher")
    print(f"     Missing: {len(missing)} | Unexpected: {len(unexpected)}")
    return {'missing': missing, 'unexpected': unexpected}
