"""
╔══════════════════════════════════════════════════════════════════════╗
║  IRDAS Phase 2 — HR Specialist Training  (Kaggle-Ready)              ║
║                                                                      ║
║  Initialises from DR teacher (θ_base = swa_final.pth)               ║
║  Trains HR head + backbone on HRDC 2023 dataset                     ║
║  Saves θ_HR for task arithmetic merge                                ║
║                                                                      ║
║  Research-backed design:                                             ║
║  · Focal BCE (HRDC challenge best practice)                         ║
║  · Weighted sampler (class imbalance handling)                       ║
║  · Safe weight loading with shape-filter (no RuntimeError on mismatch)║
║  · Freeze → unfreeze backbone schedule                               ║
║  · EMA + SWA + Lookahead                                            ║
║  · Save top-5 checkpoints for ensemble                              ║
║  · Task vector computed and saved after training                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import gc
import time
import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, roc_auc_score, f1_score
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from finalarchitecture.config  import PATHS, MODEL_CFG, HR_CFG
from finalarchitecture.utils   import (
    set_seed, mem_checkpoint, aggressive_cleanup,
    Lookahead, ModelEMA, build_stage_scheduler, EpochLogger,
    save_checkpoint, print_banner,
)
from finalarchitecture.losses  import hr_focal_bce_loss
from finalarchitecture.datasets import build_hrdc_loaders
from finalarchitecture.models  import IRDASModel, safe_load_teacher_weights


# ============================================================
# VALIDATION
# ============================================================

def validate_hr(model, loader, device):
    """Evaluate HR classification: AUC, F1, accuracy."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            out    = model(imgs)
            all_logits.extend(out['hr_logits'].squeeze(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    probs      = 1 / (1 + np.exp(-all_logits))  # sigmoid
    preds      = (probs >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, probs) if len(np.unique(all_labels)) > 1 else 0.0
    f1  = f1_score(all_labels, preds, zero_division=0)
    acc = (preds == all_labels).mean()
    return {'auc': auc, 'f1': f1, 'acc': acc}


# ============================================================
# SINGLE EPOCH TRAIN
# ============================================================

def train_hr_epoch(model, loader, la, ema, scaler, device, accum_steps):
    """Train one epoch of HR specialist.

    Returns:
        mean loss (float)
    """
    model.train()
    model.training_mode = False  # disable vessel decoder (not needed here)
    total_loss = 0.0
    la.zero_grad(set_to_none=True)
    n_updates = 0

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            out  = model(imgs)
            loss = hr_focal_bce_loss(
                out['hr_logits'], labels,
                gamma=2.0,
                pos_weight=HR_CFG['hr_pos_weight'],
            ) / accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0 or step + 1 == len(loader):
            scaler.unscale_(la.opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(la.opt)
            scaler.update()
            la.sync()
            la.zero_grad(set_to_none=True)
            ema.update(model)
            n_updates += 1

    return total_loss / max(n_updates, 1)


# ============================================================
# TOP-K CHECKPOINT TRACKER
# ============================================================

class TopKCheckpoints:
    """Maintains top-K model checkpoints by a given metric (higher is better)."""

    def __init__(self, k: int, ckpt_dir: str, prefix: str, metric_name: str = 'f1'):
        self.k           = k
        self.ckpt_dir    = ckpt_dir
        self.prefix      = prefix
        self.metric_name = metric_name
        self.heap        = []  # list of (metric_value, filepath)

    def update(self, model, epoch: int, metric: float) -> bool:
        """Save checkpoint if this metric is in the top-K. Returns True if saved."""
        path = os.path.join(
            self.ckpt_dir, f"{self.prefix}_ep{epoch:03d}_{self.metric_name}{metric:.4f}.pth"
        )
        if len(self.heap) < self.k or metric > self.heap[0][0]:
            torch.save(model.state_dict(), path)
            self.heap.append((metric, path))
            self.heap.sort(key=lambda x: x[0])
            # Remove the worst if over limit
            if len(self.heap) > self.k:
                _, old_path = self.heap.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)
            return True
        return False

    def best_metric(self) -> float:
        return self.heap[-1][0] if self.heap else 0.0

    def best_path(self) -> str:
        return self.heap[-1][1] if self.heap else None


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(HR_CFG['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print_banner(
        "IRDAS Phase 2 — HR Specialist Training",
        GPU=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        Epochs=HR_CFG['total_epochs'],
        Image_size=f"{HR_CFG['image_size']}px",
        Batch=HR_CFG['batch_size'],
        Grad_accum=HR_CFG['grad_accum'],
        Effective_batch=HR_CFG['batch_size'] * HR_CFG['grad_accum'],
    )

    os.makedirs(PATHS['ckpt_dir'], exist_ok=True)
    logger = EpochLogger(os.path.join(PATHS['ckpt_dir'], 'hr_specialist_log.csv'))

    # ── Dataloaders ──────────────────────────────────────────
    print("  📂 Loading HRDC dataset...")
    train_loader, valid_loader = build_hrdc_loaders(
        PATHS['hrdc_imgs'],
        PATHS['hrdc_csv'],
        HR_CFG['image_size'],
        HR_CFG['batch_size'],
    )
    print(f"  Train: {len(train_loader.dataset)} | Val: {len(valid_loader.dataset)}")

    # ── Model — load teacher backbone as init ─────────────────
    print("\n  🏗  Building IRDASModel and loading DR teacher weights...")
    model = IRDASModel(
        backbone_name    = MODEL_CFG['backbone'],
        pretrained       = False,   # we load teacher weights below
        fpn_out_channels = MODEL_CFG['fpn_out_channels'],
        dropout_rate     = MODEL_CFG['dropout_rate'],
        coral_levels     = MODEL_CFG['coral_levels'],
        msd_k            = MODEL_CFG['msd_k'],
        drop_path_rate   = MODEL_CFG['drop_path_rate'],
    ).to(device)

    teacher_path = PATHS['dr_teacher_weights']
    if not os.path.exists(teacher_path):
        print(f"  ⚠️  Teacher weights not found at {teacher_path}")
        print("  Falling back to pretrained=True (ImageNet weights)")
        model = IRDASModel(
            backbone_name    = MODEL_CFG['backbone'],
            pretrained       = True,
            fpn_out_channels = MODEL_CFG['fpn_out_channels'],
            dropout_rate     = MODEL_CFG['dropout_rate'],
            coral_levels     = MODEL_CFG['coral_levels'],
            msd_k            = MODEL_CFG['msd_k'],
            drop_path_rate   = MODEL_CFG['drop_path_rate'],
        ).to(device)
    else:
        safe_load_teacher_weights(teacher_path, model, device='cpu')

    # Save θ_base before any HR training (needed for task vector)
    base_path = os.path.join(PATHS['ckpt_dir'], 'theta_base.pth')
    torch.save(model.state_dict(), base_path)
    print(f"  💾 θ_base saved to {base_path}")

    ema = ModelEMA(model, HR_CFG['ema_decay'])
    mem_checkpoint("post-model")

    # ── Freeze backbone ───────────────────────────────────────
    model.freeze_backbone()
    head_params = [p for p in model.parameters() if p.requires_grad]

    base_opt = torch.optim.AdamW(
        head_params,
        lr=HR_CFG['stage_lrs'][0][0],
        weight_decay=HR_CFG['weight_decay'],
    )
    la     = Lookahead(base_opt, HR_CFG['lookahead_k'], HR_CFG['lookahead_alpha'])
    scaler = torch.amp.GradScaler('cuda')

    backbone_unfrozen = False
    swa_model = None
    swa_sched = None

    stage_epochs = [HR_CFG['freeze_epochs'],
                    HR_CFG['total_epochs'] - HR_CFG['freeze_epochs']]
    scheduler = build_stage_scheduler(base_opt, stage_epochs[0], HR_CFG['warmup_epochs'])

    tracker = TopKCheckpoints(5, PATHS['ckpt_dir'], 'hr_specialist', 'f1')

    print(f"\n{'='*65}")
    print(f"  STARTING HR SPECIALIST TRAINING — {HR_CFG['total_epochs']} epochs")
    print(f"{'='*65}\n")

    for epoch in range(HR_CFG['total_epochs']):
        t0 = time.time()

        # ── Backbone unfreeze at freeze_epochs ────────────────
        if epoch == HR_CFG['freeze_epochs'] and not backbone_unfrozen:
            backbone_unfrozen = True
            model.unfreeze_backbone()
            head_lr, bb_lr = HR_CFG['stage_lrs'][1]
            base_opt.add_param_group({
                'params': list(model.backbone.parameters()),
                'lr': bb_lr,
                'weight_decay': HR_CFG['weight_decay'],
            })
            base_opt.param_groups[0]['lr'] = head_lr
            remaining = HR_CFG['total_epochs'] - epoch
            scheduler = build_stage_scheduler(base_opt, remaining, HR_CFG['warmup_epochs'])
            print(f"  🔥 Backbone unfrozen | head_lr={head_lr:.1e} bb_lr={bb_lr:.1e}")
            mem_checkpoint("unfreeze")

        # ── SWA activation ────────────────────────────────────
        if epoch == HR_CFG['swa_start'] and swa_model is None:
            print(f"\n  🌀 SWA activated at epoch {epoch+1}")
            swa_model = AveragedModel(model)
            swa_sched = SWALR(base_opt, swa_lr=HR_CFG['swa_lr'], anneal_epochs=5)
            mem_checkpoint("swa-init")

        accum = HR_CFG['grad_accum']
        train_loss = train_hr_epoch(model, train_loader, la, ema, scaler, device, accum)

        # Validate with EMA model
        ema.module.training_mode = False
        metrics = validate_hr(ema.module, valid_loader, device)
        ema.module.training_mode = True

        # SWA / scheduler step
        if epoch >= HR_CFG['swa_start'] and swa_model is not None:
            swa_model.update_parameters(model)
            swa_sched.step()
        elif scheduler is not None:
            scheduler.step()

        # Checkpoint top-5
        saved = tracker.update(ema.module, epoch + 1, metrics['f1'])

        lr_h = base_opt.param_groups[0]['lr']
        lr_b = base_opt.param_groups[1]['lr'] if backbone_unfrozen else 0.0
        elapsed = time.time() - t0
        star = " ⭐" if saved else ""

        logger.log(
            epoch + 1, "HR",
            loss=train_loss,
            auc=metrics['auc'],
            f1=metrics['f1'],
            acc=metrics['acc'],
            lr_h=lr_h, lr_b=lr_b,
        )
        print(
            f"  Ep [{epoch+1:03d}/{HR_CFG['total_epochs']}] "
            f"lr_h={lr_h:.2e} lr_b={lr_b:.2e} | "
            f"Loss={train_loss:.4f} | "
            f"AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} Acc={metrics['acc']:.4f} "
            f"[best F1={tracker.best_metric():.4f}]  {elapsed:.0f}s{star}"
        )
        mem_checkpoint(f"ep{epoch+1}")
        aggressive_cleanup()

    # ── Post-training: SWA BN update ─────────────────────────
    print("\n  📊 Running SWA BN calibration...")
    if swa_model is not None:
        # Manual BN update loop (avoids update_bn pitfalls with multi-return loaders)
        swa_model.train()
        swa_model.to(device)
        with torch.no_grad():
            for imgs, _ in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                swa_model(imgs)
        swa_path = os.path.join(PATHS['ckpt_dir'], 'hr_swa_final.pth')
        torch.save(swa_model.module.state_dict(), swa_path)
        print(f"  💾 SWA model saved to {swa_path}")

    # ── Save best EMA ─────────────────────────────────────────
    best_ema_path = os.path.join(PATHS['ckpt_dir'], 'hr_best_ema.pth')
    torch.save(ema.module.state_dict(), best_ema_path)

    # ── Compute task vector τ_HR = θ_HR − θ_base ─────────────
    print("\n  🔧 Computing HR task vector...")
    theta_base = torch.load(base_path, map_location='cpu')
    theta_hr   = torch.load(best_ema_path, map_location='cpu')
    tau_hr     = {k: (theta_hr[k].float() - theta_base[k].float())
                  for k in theta_base if k in theta_hr}
    tau_path = os.path.join(PATHS['ckpt_dir'], 'tau_hr.pth')
    torch.save(tau_hr, tau_path)

    print_banner(
        "HR Specialist Training COMPLETE",
        Best_F1=f"{tracker.best_metric():.4f}",
        Task_vector=tau_path,
        Top5_checkpoints=str(len(tracker.heap)),
    )


if __name__ == '__main__':
    main()
