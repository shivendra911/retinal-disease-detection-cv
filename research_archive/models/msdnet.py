"""
IRDAS Models — MSDNet (Multi-Scale Disentangled Network)
=========================================================

Full model assembly: the core research artifact.

Architecture:
    Shared: EfficientNet-B4-NS → FPN (multi-scale features)
    DR branch: CBAM → GAP → Dropout → FC(K)  [K=5 for softmax, K=4 for CORAL ordinal]
    HR branch: CBAM → GAP → Dropout → FC(1)  [binary: HR present/absent]
    Vessel dec: U-Net decoder from P3/P4/P5   [auxiliary, training only]

Configurable:
    - Backbone: any timm model via config['backbone'] (default: tf_efficientnet_b4_ns)
    - Input resolution: any size (default: 384×384 for B4-NS)
    - DR classes: config['dr_num_classes'] — 5 for softmax, 4 (K-1) for CORAL ordinal
    - FPN channels, dropout rate

Novel component:
    Contrastive disentanglement loss applied to DR+HR branch embeddings.
    Forces disease-specific representations despite shared visual features
    (hemorrhages, which appear in both diseases).

Inference mode:
    - Vessel decoder is removed (training-only auxiliary task)
    - MC Dropout stays active for uncertainty estimation
    - predict_with_uncertainty() runs N forward passes and returns mean + std
    - predict_with_tta() applies 8 geometric augmentations for test-time averaging
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from models.backbone import EfficientNetFPNBackbone
from models.fpn import FPN
from models.disease_branches import DiseaseSpecificBranch
from models.vessel_decoder import VesselDecoder


class MSDNet(nn.Module):
    """
    Multi-Scale Disentangled Network for simultaneous
    Diabetic Retinopathy + Hypertensive Retinopathy detection.
    
    Default backbone: tf_efficientnet_b4_ns (NoisyStudent B4).
    Input resolution is configurable (no hardcoded spatial dimensions).
    
    Args:
        config: Dictionary with model hyperparameters from config.yaml.
                Required keys: pretrained, fpn_out_channels, dropout_rate
                Optional keys:
                    backbone (str): timm model name, default 'tf_efficientnet_b4_ns'
                    dr_num_classes (int): 5 for softmax, 4 for CORAL ordinal logits
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.backbone = EfficientNetFPNBackbone(
            backbone_name=config.get('backbone', 'tf_efficientnet_b4_ns'),
            pretrained=config.get('pretrained', True),
        )
        self.fpn = FPN(
            in_channels_list=self.backbone.out_channels,
            out_channels=config.get('fpn_out_channels', 256),
        )
        
        fpn_ch = config.get('fpn_out_channels', 256)
        dropout = config.get('dropout_rate', 0.3)
        
        # Disease-specific branches
        # dr_num_classes: total number of DR grades (5 for grades 0-4)
        # For CORAL ordinal regression, the FC head outputs K-1 logits
        # (each logit represents P(grade > k)), so we reduce by 1.
        self.dr_num_classes = config.get('dr_num_classes', 5)
        self.dr_ordinal = config.get('dr_ordinal', True)  # True = CORAL (K-1 logits)
        dr_output_dim = self.dr_num_classes - 1 if self.dr_ordinal else self.dr_num_classes
        self.dr_branch = DiseaseSpecificBranch(
            fpn_ch, num_classes=dr_output_dim, dropout_rate=dropout
        )
        self.hr_branch = DiseaseSpecificBranch(
            fpn_ch, num_classes=1, dropout_rate=dropout
        )
        
        # Auxiliary vessel decoder (training only)
        # Backbone out_channels auto-detected: e.g., [56, 160, 448] for B4-NS
        self.vessel_decoder = VesselDecoder(
            encoder_channels=self.backbone.out_channels
        )
        
        self.training_mode = True  # Controls vessel decoder activation
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete MSDNet.
        
        Args:
            x: Input tensor of shape (B, 3, H, W).
               Default: (B, 3, 384, 384) for B4-NS.
        
        Returns:
            Dictionary with:
            - dr_logits: (B, dr_num_classes) — DR grade logits
            - hr_logits: (B, 1) — HR binary logit
            - dr_feat:   (B, fpn_ch) — DR branch embedding (for contrastive loss)
            - hr_feat:   (B, fpn_ch) — HR branch embedding (for contrastive loss)
            - vessel_pred: (B, 1, H, W) — vessel mask (training only)
        """
        # Multi-scale feature extraction
        P3, P4, P5 = self.backbone(x)
        fpn_feat = self.fpn(P3, P4, P5)
        
        # Disease predictions (both branches use same FPN features)
        dr_logits, dr_feat = self.dr_branch(fpn_feat)
        hr_logits, hr_feat = self.hr_branch(fpn_feat)
        
        outputs = {
            'dr_logits': dr_logits,    # (B, dr_num_classes)
            'hr_logits': hr_logits,    # (B, 1)
            'dr_feat'  : dr_feat,      # (B, fpn_ch) — for contrastive loss
            'hr_feat'  : hr_feat,      # (B, fpn_ch) — for contrastive loss
        }
        
        # Auxiliary vessel segmentation — only during training
        if self.training_mode:
            vessel_pred = self.vessel_decoder(P3, P4, P5)
            outputs['vessel_pred'] = vessel_pred
        
        return outputs
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_passes: int = 30) -> Dict[str, torch.Tensor]:
        """
        MC Dropout inference for uncertainty estimation.
        
        Keeps model in train() mode to activate dropout, but disables
        vessel decoder (not needed at inference).
        
        Runs the same input through the model n_passes times with different
        dropout masks. The standard deviation across passes = uncertainty.
        
        Args:
            x: Input tensor (B, 3, H, W)
            n_passes: Number of stochastic forward passes (30 is standard)
        
        Returns:
            Dictionary with:
            - dr_mean: (B, dr_num_classes) — mean DR class probabilities
            - dr_uncertainty: (B,) — uncertainty per sample (std of predictions)
            - hr_mean: (B, 1) — mean HR probability
            - hr_uncertainty: (B,) — uncertainty per sample
        """
        self.train()  # activates dropout for stochastic inference
        self.training_mode = False  # skip vessel decoder at inference
        
        with torch.no_grad():
            dr_preds, hr_preds = [], []
            for _ in range(n_passes):
                out = self.forward(x)
                dr_preds.append(torch.softmax(out['dr_logits'], dim=-1))
                hr_preds.append(torch.sigmoid(out['hr_logits']))
            
            dr_stack = torch.stack(dr_preds)   # (n_passes, B, dr_num_classes)
            hr_stack = torch.stack(hr_preds)   # (n_passes, B, 1)
        
        self.training_mode = True  # restore for future training
        
        return {
            'dr_mean'       : dr_stack.mean(0),                 # (B, dr_num_classes)
            'dr_uncertainty': dr_stack.std(0).mean(-1),          # (B,) scalar per sample
            'hr_mean'       : hr_stack.mean(0),                 # (B, 1)
            'hr_uncertainty': hr_stack.std(0).squeeze(-1),      # (B,)
        }
    
    def predict_with_tta(
        self,
        x: torch.Tensor,
        n_tta: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Test-Time Augmentation (TTA) inference with geometric augmentations.
        
        Applies up to 8 geometric augmentations to the input, runs each
        through the model, and averages the DR logits across all views.
        This reduces variance from random orientation and improves QWK.
        
        The 8 augmentations (D4 dihedral group of the square):
            0: identity
            1: horizontal flip
            2: vertical flip
            3: horizontal + vertical flip (180° rotation equivalent)
            4: 90° rotation
            5: 180° rotation
            6: 270° rotation
            7: horizontal flip + 90° rotation
        
        Args:
            x: Input tensor (B, 3, H, W)
            n_tta: Number of augmentations to apply (1-8). Default 8 (full set).
                   Set to 1 for no augmentation (identity only).
        
        Returns:
            Dictionary with:
            - dr_logits: (B, dr_num_classes) — averaged DR logits across TTA views
            - hr_logits: (B, 1) — averaged HR logits across TTA views
        """
        self.eval()
        self.training_mode = False  # skip vessel decoder at inference
        
        # Define the 8 geometric augmentation functions and their inverses.
        # Each augment_fn transforms input before forward pass.
        # Each deaugment_fn is the inverse (not needed for classification logits,
        # but kept for consistency and potential segmentation TTA).
        def _identity(t: torch.Tensor) -> torch.Tensor:
            return t
        
        def _hflip(t: torch.Tensor) -> torch.Tensor:
            return t.flip(-1)
        
        def _vflip(t: torch.Tensor) -> torch.Tensor:
            return t.flip(-2)
        
        def _hvflip(t: torch.Tensor) -> torch.Tensor:
            return t.flip(-1).flip(-2)
        
        def _rot90(t: torch.Tensor) -> torch.Tensor:
            return t.rot90(1, [-2, -1])
        
        def _rot180(t: torch.Tensor) -> torch.Tensor:
            return t.rot90(2, [-2, -1])
        
        def _rot270(t: torch.Tensor) -> torch.Tensor:
            return t.rot90(3, [-2, -1])
        
        def _hflip_rot90(t: torch.Tensor) -> torch.Tensor:
            return t.flip(-1).rot90(1, [-2, -1])
        
        augment_fns = [
            _identity,      # 0: no change
            _hflip,         # 1: horizontal flip
            _vflip,         # 2: vertical flip
            _hvflip,        # 3: h+v flip
            _rot90,         # 4: 90° rotation
            _rot180,        # 5: 180° rotation
            _rot270,        # 6: 270° rotation
            _hflip_rot90,   # 7: hflip + 90° rotation
        ]
        
        # Clamp n_tta to valid range
        n_tta = max(1, min(n_tta, len(augment_fns)))
        
        with torch.no_grad():
            dr_logits_list = []
            hr_logits_list = []
            
            for i in range(n_tta):
                x_aug = augment_fns[i](x)
                out = self.forward(x_aug)
                dr_logits_list.append(out['dr_logits'])
                hr_logits_list.append(out['hr_logits'])
            
            # Average logits across all TTA views
            dr_logits_avg = torch.stack(dr_logits_list).mean(0)  # (B, dr_num_classes)
            hr_logits_avg = torch.stack(hr_logits_list).mean(0)  # (B, 1)
        
        self.training_mode = True  # restore for future training
        
        return {
            'dr_logits': dr_logits_avg,    # (B, dr_num_classes)
            'hr_logits': hr_logits_avg,    # (B, 1)
        }
