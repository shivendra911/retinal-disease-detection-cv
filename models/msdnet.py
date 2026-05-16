"""
IRDAS Models — MSDNet (Multi-Scale Disentangled Network)
=========================================================

Full model assembly: the core research artifact.

Architecture:
    Shared: EfficientNet-B0 → FPN (multi-scale features)
    DR branch: CBAM → GAP → Dropout → FC(5)  [grades 0-4]
    HR branch: CBAM → GAP → Dropout → FC(1)  [binary: HR present/absent]
    Vessel dec: U-Net decoder from P3/P4/P5   [auxiliary, training only]

Novel component:
    Contrastive disentanglement loss applied to DR+HR branch embeddings.
    Forces disease-specific representations despite shared visual features
    (hemorrhages, which appear in both diseases).

Inference mode:
    - Vessel decoder is removed (training-only auxiliary task)
    - MC Dropout stays active for uncertainty estimation
    - predict_with_uncertainty() runs N forward passes and returns mean + std
"""

import torch
import torch.nn as nn
from models.backbone import EfficientNetFPNBackbone
from models.fpn import FPN
from models.disease_branches import DiseaseSpecificBranch
from models.vessel_decoder import VesselDecoder


class MSDNet(nn.Module):
    """
    Multi-Scale Disentangled Network for simultaneous
    Diabetic Retinopathy + Hypertensive Retinopathy detection.
    
    Args:
        config: Dictionary with model hyperparameters from config.yaml
                Required keys: pretrained, fpn_out_channels, dropout_rate
    """
    
    def __init__(self, config):
        super().__init__()
        self.backbone = EfficientNetFPNBackbone(
            pretrained=config.get('pretrained', True)
        )
        self.fpn = FPN(
            in_channels_list=self.backbone.out_channels,
            out_channels=config.get('fpn_out_channels', 256)
        )
        
        fpn_ch = config.get('fpn_out_channels', 256)
        dropout = config.get('dropout_rate', 0.3)
        
        # Disease-specific branches
        self.dr_branch = DiseaseSpecificBranch(
            fpn_ch, num_classes=5, dropout_rate=dropout
        )
        self.hr_branch = DiseaseSpecificBranch(
            fpn_ch, num_classes=1, dropout_rate=dropout
        )
        
        # Auxiliary vessel decoder (training only)
        self.vessel_decoder = VesselDecoder(
            encoder_channels=self.backbone.out_channels
        )
        
        self.training_mode = True  # Controls vessel decoder activation
    
    def forward(self, x):
        """
        Forward pass through the complete MSDNet.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        
        Returns:
            Dictionary with:
            - dr_logits: (B, 5) — DR grade logits
            - hr_logits: (B, 1) — HR binary logit
            - dr_feat:   (B, 256) — DR branch embedding (for contrastive loss)
            - hr_feat:   (B, 256) — HR branch embedding (for contrastive loss)
            - vessel_pred: (B, 1, 224, 224) — vessel mask (training only)
        """
        # Multi-scale feature extraction
        P3, P4, P5 = self.backbone(x)
        fpn_feat = self.fpn(P3, P4, P5)
        
        # Disease predictions (both branches use same FPN features)
        dr_logits, dr_feat = self.dr_branch(fpn_feat)
        hr_logits, hr_feat = self.hr_branch(fpn_feat)
        
        outputs = {
            'dr_logits': dr_logits,    # (B, 5)
            'hr_logits': hr_logits,    # (B, 1)
            'dr_feat'  : dr_feat,      # (B, 256) — for contrastive loss
            'hr_feat'  : hr_feat,      # (B, 256) — for contrastive loss
        }
        
        # Auxiliary vessel segmentation — only during training
        if self.training_mode:
            vessel_pred = self.vessel_decoder(P3, P4, P5)
            outputs['vessel_pred'] = vessel_pred
        
        return outputs
    
    def predict_with_uncertainty(self, x, n_passes=30):
        """
        MC Dropout inference for uncertainty estimation.
        
        Keeps model in train() mode to activate dropout, but disables
        vessel decoder (not needed at inference).
        
        Runs the same input through the model n_passes times with different
        dropout masks. The standard deviation across passes = uncertainty.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
            n_passes: Number of stochastic forward passes (30 is standard)
        
        Returns:
            Dictionary with:
            - dr_mean: (B, 5) — mean DR class probabilities
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
            
            dr_stack = torch.stack(dr_preds)   # (n_passes, B, 5)
            hr_stack = torch.stack(hr_preds)   # (n_passes, B, 1)
        
        self.training_mode = True  # restore for future training
        
        return {
            'dr_mean'       : dr_stack.mean(0),                 # (B, 5)
            'dr_uncertainty': dr_stack.std(0).mean(-1),          # (B,) scalar per sample
            'hr_mean'       : hr_stack.mean(0),                 # (B, 1)
            'hr_uncertainty': hr_stack.std(0).squeeze(-1),      # (B,)
        }
