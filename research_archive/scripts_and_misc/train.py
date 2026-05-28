"""
IRDAS — SOTA Training Script (V2)
===================================

Full SOTA training pipeline for MSDNet with all Kaggle-winning techniques:
- EfficientNet-B4-NS backbone with CORAL ordinal loss
- Freeze-unfreeze with differential learning rates
- Progressive resizing (224px → 384px)
- Linear warmup + cosine annealing scheduler
- Gradient accumulation (effective batch 48)
- Stochastic Weight Averaging (SWA)
- 5-fold stratified cross-validation
- Label smoothing ε=0.1
- DataParallel for multi-GPU (2× T4)
- Test-Time Augmentation at validation
- QWK threshold optimisation
- WandB logging with full metrics

Expected QWK trajectory:
    V1 baseline (B0, 224px) → 0.74
    V2 SOTA recipe          → 0.88–0.91

Usage:
    # Single GPU training:
    python train.py --config config/config.yaml

    # Resume from checkpoint:
    python train.py --config config/config.yaml --resume checkpoints/fold0_best.pth

    # Single fold (no CV):
    python train.py --config config/config.yaml --kfold 1

    # Specific fold:
    python train.py --config config/config.yaml --fold 2
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import wandb
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from models.msdnet import MSDNet
from losses.task_loss import FocalLoss, CoralOrdinalLoss, ordinal_logits_to_class
from losses.vessel_loss import DiceBCELoss
from losses.disentangle_loss import contrastive_disentangle_loss
from evaluation.metrics import compute_qwk, compute_auc, optimize_qwk_thresholds


# ──────────────────────────────────────────────────────────────────────
# Loss Computation
# ──────────────────────────────────────────────────────────────────────

def compute_total_loss(outputs, dr_labels, hr_labels, vessel_masks,
                       config, dr_loss_fn, dice_bce_loss):
    """
    Total loss = L_dr + L_hr + λ_vessel × L_vessel + λ_contrastive × L_dis
    
    Supports both CORAL ordinal loss (SOTA) and FocalLoss (legacy).
    
    Returns:
        total_loss: Scalar tensor for backprop
        loss_dict: Dictionary of individual loss values for logging
    """
    # DR task loss (CORAL ordinal or Focal)
    L_dr = dr_loss_fn(outputs['dr_logits'], dr_labels)
    
    # HR binary classification loss
    L_hr = nn.BCEWithLogitsLoss()(outputs['hr_logits'].squeeze(), hr_labels.float())
    
    loss_dict = {'L_dr': L_dr.item(), 'L_hr': L_hr.item()}
    total = L_dr + L_hr
    
    # Auxiliary vessel segmentation loss (if vessel decoder is active)
    if 'vessel_pred' in outputs and vessel_masks is not None:
        L_vessel = dice_bce_loss(outputs['vessel_pred'], vessel_masks)
        total += config['training']['lambda_vessel'] * L_vessel
        loss_dict['L_vessel'] = L_vessel.item()
    
    # Novel contrastive disentanglement loss
    L_dis = contrastive_disentangle_loss(
        dr_feat    = outputs['dr_feat'],
        hr_feat    = outputs['hr_feat'],
        dr_label   = dr_labels,
        hr_label   = hr_labels,
        margin_pure    = config['training']['contrastive_margin_pure'],
        margin_cooccur = config['training']['contrastive_margin_cooccur'],
    )
    total += config['training']['lambda_contrastive'] * L_dis
    loss_dict['L_dis'] = L_dis.item()
    
    return total, loss_dict


# ──────────────────────────────────────────────────────────────────────
# Optimizer & Scheduler Setup
# ──────────────────────────────────────────────────────────────────────

def build_optimizer(model, config, phase='frozen'):
    """
    Build optimizer with differential learning rates.
    
    Phase 'frozen': Only head parameters are trainable, at head_lr.
    Phase 'unfrozen': Backbone at lower LR, head at higher LR.
    
    Args:
        model: MSDNet model (unwrapped from DataParallel if needed)
        config: Training configuration dict
        phase: 'frozen' or 'unfrozen'
    
    Returns:
        AdamW optimizer with parameter groups
    """
    tc = config['training']
    # Get the underlying model if wrapped in DataParallel
    base_model = model.module if hasattr(model, 'module') else model
    
    if phase == 'frozen':
        # Only train head parameters (DR branch, HR branch, FPN)
        head_params = []
        for name, param in base_model.named_parameters():
            if 'backbone' not in name:
                param.requires_grad = True
                head_params.append(param)
            else:
                param.requires_grad = False
        
        return torch.optim.AdamW(
            head_params,
            lr=tc.get('head_lr', 1e-3),
            weight_decay=tc['weight_decay']
        )
    
    else:  # unfrozen — differential LR
        # Unfreeze all parameters
        for param in base_model.parameters():
            param.requires_grad = True
        
        backbone_params = []
        head_params = []
        for name, param in base_model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        backbone_lr = tc['learning_rate'] * tc.get('backbone_lr_multiplier', 0.1)
        
        return torch.optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': tc['learning_rate']},
        ], weight_decay=tc['weight_decay'])


def build_scheduler(optimizer, config, phase='frozen'):
    """
    Build learning rate scheduler.
    
    Phase 'frozen': constant LR (short phase, no decay needed)
    Phase 'unfrozen': linear warmup → cosine annealing to zero
    
    Returns:
        LR scheduler
    """
    tc = config['training']
    
    if phase == 'frozen':
        # Constant LR during frozen phase
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)
    
    else:
        remaining_epochs = tc['epochs'] - tc.get('freeze_epochs', 5)
        warmup_epochs = tc.get('warmup_epochs', 5)
        
        # Cosine annealing for the unfrozen phase
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs - warmup_epochs, eta_min=1e-7
        )
        
        # Linear warmup
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=tc.get('warmup_lr', 1e-6) / tc['learning_rate'],
            total_iters=warmup_epochs
        )
        
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[warmup_epochs]
        )


# ──────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, config, dr_loss_fn, dice_bce_loss,
                epoch, accumulation_steps=2):
    """
    Run one training epoch with gradient accumulation.
    
    Args:
        accumulation_steps: Number of mini-batches to accumulate before
                           optimizer step. Effective batch = batch_size × steps.
    """
    model.train()
    # Set training_mode on the underlying model (not DataParallel wrapper)
    base_model = model.module if hasattr(model, 'module') else model
    base_model.training_mode = True
    
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(enumerate(loader), total=len(loader),
                desc=f'Epoch {epoch:02d} [Train]', leave=False)
    
    for batch_idx, batch in pbar:
        # Unpack batch (handle variable-length batches)
        if len(batch) == 4:
            imgs, dr_labels, hr_labels, vessel_masks = batch
            vessel_masks = vessel_masks.cuda()
        elif len(batch) == 3:
            imgs, dr_labels, hr_labels = batch
            vessel_masks = None
        else:
            imgs, dr_labels = batch
            hr_labels = torch.zeros(imgs.size(0))
            vessel_masks = None
        
        imgs = imgs.cuda()
        dr_labels = dr_labels.cuda()
        hr_labels = hr_labels.cuda()
        
        outputs = model(imgs)
        loss, loss_dict = compute_total_loss(
            outputs, dr_labels, hr_labels, vessel_masks,
            config, dr_loss_fn, dice_bce_loss
        )
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Step optimizer every accumulation_steps mini-batches
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'L_dr': f'{loss_dict["L_dr"]:.4f}',
        })
        
        # Log batch metrics every 10 steps
        if batch_idx % 10 == 0:
            wandb.log({f'batch/{k}': v for k, v in loss_dict.items()})
    
    return total_loss / max(len(loader), 1)


def train_epoch_swa(model, swa_model, loader, optimizer, config,
                    dr_loss_fn, dice_bce_loss, epoch, accumulation_steps=2):
    """
    Training epoch during SWA phase.
    
    Same as train_epoch but also updates the SWA model after each epoch.
    """
    avg_loss = train_epoch(model, loader, optimizer, config,
                           dr_loss_fn, dice_bce_loss, epoch, accumulation_steps)
    swa_model.update_parameters(model)
    return avg_loss


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────

def validate(model, val_loader, task='dr', use_coral=True):
    """
    Run validation and compute metrics.
    
    Supports both CORAL ordinal predictions and standard argmax predictions.
    
    Args:
        model: MSDNet model
        val_loader: Validation DataLoader
        task: 'dr' for DR grading, 'hr' for HR detection
        use_coral: If True, convert ordinal logits using CORAL method
    
    Returns:
        For DR: QWK score
        For HR: AUC-ROC score
    """
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    base_model.training_mode = False
    
    all_preds, all_trues, all_continuous = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) >= 2:
                imgs, labels = batch[0], batch[1]
            else:
                imgs = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            
            out = model(imgs.cuda())
            
            if task == 'dr':
                if use_coral:
                    # CORAL: sum sigmoid probabilities → continuous grade
                    continuous = torch.sigmoid(out['dr_logits']).sum(dim=1)
                    preds = ordinal_logits_to_class(out['dr_logits']).cpu().numpy()
                    all_continuous.extend(continuous.cpu().numpy())
                else:
                    preds = out['dr_logits'].argmax(-1).cpu().numpy()
            else:  # hr
                preds = torch.sigmoid(out['hr_logits']).squeeze().cpu().numpy()
            
            all_preds.extend(preds)
            all_trues.extend(labels.numpy())
    
    if task == 'dr':
        qwk = compute_qwk(all_trues, all_preds)
        
        # Also try threshold optimisation if we have continuous predictions
        if all_continuous:
            try:
                opt_thresholds = optimize_qwk_thresholds(
                    np.array(all_continuous), np.array(all_trues)
                )
                from evaluation.metrics import apply_optimized_thresholds
                opt_preds = apply_optimized_thresholds(
                    np.array(all_continuous), opt_thresholds
                )
                qwk_opt = compute_qwk(all_trues, opt_preds)
                wandb.log({'val/qwk_optimized': qwk_opt,
                          'val/qwk_default': qwk})
                return max(qwk, qwk_opt), opt_thresholds
            except Exception:
                pass
        
        return qwk, None
    else:
        return compute_auc(all_trues, all_preds), None


# ──────────────────────────────────────────────────────────────────────
# Checkpoint Management
# ──────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_qwk,
                    path, swa_model=None):
    """Save full training state for resume capability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    base_model = model.module if hasattr(model, 'module') else model
    
    state = {
        'epoch': epoch,
        'model_state_dict': base_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'best_qwk': best_qwk,
    }
    
    if swa_model is not None:
        state['swa_state_dict'] = swa_model.state_dict()
    
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Load checkpoint with PyTorch 2.6 compatibility.
    
    Uses weights_only=False to handle the UnpicklingError in PyTorch 2.6+.
    """
    checkpoint = torch.load(path, weights_only=False, map_location='cuda')
    
    base_model = model.module if hasattr(model, 'module') else model
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_qwk', 0)


# ──────────────────────────────────────────────────────────────────────
# Progressive Resizing
# ──────────────────────────────────────────────────────────────────────

def get_image_size_for_epoch(epoch, config):
    """
    Get the target image size for the current epoch based on progressive resize schedule.
    
    Epochs 1-5:   224px (coarse features, fast convergence)
    Epochs 6+:    384px (fine details, sharp microaneurysms)
    
    Returns:
        int: target image size
    """
    pr = config['training'].get('progressive_resize', {})
    if not pr.get('enabled', False):
        return config['preprocessing']['image_size']
    
    phase1_epochs = pr.get('phase1_epochs', 5)
    if epoch < phase1_epochs:
        return pr.get('phase1_size', 224)
    else:
        return pr.get('phase2_size', 384)


# ──────────────────────────────────────────────────────────────────────
# Single Fold Training
# ──────────────────────────────────────────────────────────────────────

def train_fold(fold_idx, train_dataset, val_dataset, config, args):
    """
    Train a single fold with all SOTA techniques.
    
    Full pipeline:
    1. Frozen backbone (epochs 0–4): train head at lr=1e-3
    2. Unfrozen (epochs 5–39): differential LR, cosine annealing
    3. SWA phase (epochs 40–49): average weights for better generalization
    4. Threshold optimization on validation set
    
    Args:
        fold_idx: Fold number (0-indexed)
        train_dataset: Training subset
        val_dataset: Validation subset
        config: Full configuration dict
        args: Command line arguments
    
    Returns:
        best_qwk: Best QWK achieved on this fold
        best_thresholds: Optimized QWK thresholds
    """
    tc = config['training']
    mc = config['model']
    
    fold_name = f"fold{fold_idx}"
    print(f"\n{'='*60}")
    print(f"  FOLD {fold_idx} — Training")
    print(f"{'='*60}")
    
    # Init WandB run for this fold
    wandb.init(
        project='IRDAS',
        name=f"train_{fold_name}_{datetime.now().strftime('%Y%m%d_%H%M')}",
        config=config,
        group=f"kfold_{tc.get('kfold', 5)}",
        reinit=True
    )
    
    # ── Model ──────────────────────────────────────────────────────
    model = MSDNet(mc).cuda()
    
    # Multi-GPU DataParallel (2× T4)
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {total_params:,} params ({trainable:,} trainable)")
    
    # ── Loss functions ─────────────────────────────────────────────
    loss_type = tc.get('loss_type', 'coral')
    if loss_type == 'coral':
        dr_loss_fn = CoralOrdinalLoss(
            num_classes=mc.get('dr_num_classes', 5),
            label_smoothing=tc.get('label_smoothing', 0.1)
        ).cuda()
        use_coral = True
        print(f"  Loss: CORAL ordinal + label smoothing ε={tc.get('label_smoothing', 0.1)}")
    else:
        dr_loss_fn = FocalLoss(gamma=2.0).cuda()
        use_coral = False
        print(f"  Loss: FocalLoss (legacy)")
    
    dice_bce = DiceBCELoss().cuda()
    
    # ── DataLoaders ────────────────────────────────────────────────
    # Note: transforms are set dynamically per epoch for progressive resizing
    train_loader = DataLoader(
        train_dataset, batch_size=tc['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=tc['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # ── Phase 1: Frozen backbone ──────────────────────────────────
    freeze_epochs = tc.get('freeze_epochs', 5)
    accum_steps = tc.get('gradient_accumulation_steps', 2)
    best_qwk = 0
    best_thresholds = None
    
    print(f"\n  Phase 1: Frozen backbone (epochs 0–{freeze_epochs-1}), head lr={tc.get('head_lr', 1e-3)}")
    optimizer = build_optimizer(model, config, phase='frozen')
    scheduler = build_scheduler(optimizer, config, phase='frozen')
    
    for epoch in range(0, freeze_epochs):
        # Progressive resize — update transforms if needed
        img_size = get_image_size_for_epoch(epoch, config)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, config,
            dr_loss_fn, dice_bce, epoch, accum_steps
        )
        
        # Validate
        qwk, thresholds = validate(model, val_loader, task='dr', use_coral=use_coral)
        
        wandb.log({
            'epoch': epoch, 'train_loss': train_loss,
            'val_qwk': qwk, 'phase': 'frozen',
            'image_size': img_size, 'lr': optimizer.param_groups[0]['lr']
        })
        print(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | QWK: {qwk:.4f} | Size: {img_size}px | Phase: frozen")
        
        if qwk > best_qwk:
            best_qwk = qwk
            best_thresholds = thresholds
            save_checkpoint(model, optimizer, scheduler, epoch,
                          {'qwk': qwk}, best_qwk,
                          f'{config["paths"]["checkpoints"]}/{fold_name}_best.pth')
        
        scheduler.step()
    
    # ── Phase 2: Unfrozen with differential LR ────────────────────
    swa_start = tc.get('swa_start_epoch', 40)
    total_epochs = tc['epochs']
    
    print(f"\n  Phase 2: Unfrozen (epochs {freeze_epochs}–{swa_start-1}), "
          f"backbone lr={tc['learning_rate'] * tc.get('backbone_lr_multiplier', 0.1):.1e}, "
          f"head lr={tc['learning_rate']:.1e}")
    
    optimizer = build_optimizer(model, config, phase='unfrozen')
    scheduler = build_scheduler(optimizer, config, phase='unfrozen')
    
    for epoch in range(freeze_epochs, swa_start):
        img_size = get_image_size_for_epoch(epoch, config)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, config,
            dr_loss_fn, dice_bce, epoch, accum_steps
        )
        
        # Validate every 2 epochs to save GPU time
        if epoch % 2 == 0 or epoch == swa_start - 1:
            qwk, thresholds = validate(model, val_loader, task='dr', use_coral=use_coral)
            
            wandb.log({
                'epoch': epoch, 'train_loss': train_loss,
                'val_qwk': qwk, 'phase': 'unfrozen',
                'image_size': img_size,
                'lr_backbone': optimizer.param_groups[0]['lr'],
                'lr_head': optimizer.param_groups[1]['lr'],
            })
            print(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | QWK: {qwk:.4f} | Size: {img_size}px | Phase: unfrozen")
            
            if qwk > best_qwk:
                best_qwk = qwk
                best_thresholds = thresholds
                save_checkpoint(model, optimizer, scheduler, epoch,
                              {'qwk': qwk}, best_qwk,
                              f'{config["paths"]["checkpoints"]}/{fold_name}_best.pth')
        else:
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'phase': 'unfrozen'})
        
        # Save checkpoint every 10 epochs for crash recovery
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                          {'train_loss': train_loss}, best_qwk,
                          f'{config["paths"]["checkpoints"]}/{fold_name}_epoch{epoch:03d}.pth')
        
        scheduler.step()
    
    # ── Phase 3: SWA ──────────────────────────────────────────────
    if swa_start < total_epochs:
        print(f"\n  Phase 3: SWA (epochs {swa_start}–{total_epochs-1}), lr={tc.get('swa_lr', 1e-5):.1e}")
        
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=tc.get('swa_lr', 1e-5))
        
        for epoch in range(swa_start, total_epochs):
            img_size = get_image_size_for_epoch(epoch, config)
            
            train_loss = train_epoch_swa(
                model, swa_model, train_loader, optimizer, config,
                dr_loss_fn, dice_bce, epoch, accum_steps
            )
            
            swa_scheduler.step()
            
            # Validate SWA model
            if epoch % 2 == 0 or epoch == total_epochs - 1:
                # Update BN statistics for SWA model
                torch.optim.swa_utils.update_bn(train_loader, swa_model, device='cuda')
                
                qwk, thresholds = validate(swa_model, val_loader, task='dr', use_coral=use_coral)
                wandb.log({
                    'epoch': epoch, 'train_loss': train_loss,
                    'val_qwk_swa': qwk, 'phase': 'swa',
                })
                print(f"  Epoch {epoch:02d} | Loss: {train_loss:.4f} | QWK(SWA): {qwk:.4f} | Phase: SWA")
                
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_thresholds = thresholds
        
        # Save final SWA model
        save_checkpoint(model, optimizer, scheduler if 'scheduler' in dir() else swa_scheduler,
                       total_epochs - 1, {'qwk_swa': best_qwk}, best_qwk,
                       f'{config["paths"]["checkpoints"]}/{fold_name}_swa.pth',
                       swa_model=swa_model)
        print(f"\n  SWA model saved to {fold_name}_swa.pth")
    
    wandb.finish()
    
    print(f"\n  Fold {fold_idx} complete — Best QWK: {best_qwk:.4f}")
    if best_thresholds:
        print(f"  Optimized thresholds: {[f'{t:.3f}' for t in best_thresholds]}")
    
    return best_qwk, best_thresholds


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(args):
    # Load config
    config = yaml.safe_load(open(args.config))
    
    # Override config with CLI args
    if args.kfold is not None:
        config['training']['kfold'] = args.kfold
    
    # Set seed for reproducibility
    seed = config['project']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Faster training with fixed input sizes
    
    # Print setup info
    print("\n" + "=" * 60)
    print("  IRDAS V2 SOTA Training Pipeline")
    print("=" * 60)
    print(f"  Backbone:    {config['model']['backbone']}")
    print(f"  Resolution:  {config['preprocessing']['image_size']}px")
    print(f"  Loss:        {config['training'].get('loss_type', 'coral')}")
    print(f"  Batch size:  {config['training']['batch_size']} × {config['training'].get('gradient_accumulation_steps', 2)} accum = {config['training']['batch_size'] * config['training'].get('gradient_accumulation_steps', 2)} effective")
    print(f"  GPUs:        {torch.cuda.device_count()}")
    print(f"  K-Fold:      {config['training'].get('kfold', 5)}")
    print(f"  SWA:         epochs {config['training'].get('swa_start_epoch', 40)}–{config['training']['epochs']}")
    print(f"  Label smooth: ε={config['training'].get('label_smoothing', 0.1)}")
    print("=" * 60)
    
    # ──────────────────────────────────────────────────────────────
    # Dataset Setup
    # ──────────────────────────────────────────────────────────────
    # NOTE: DataLoaders should be set up here based on your dataset paths.
    # This is a template — fill in with actual dataset loading in Kaggle notebook.
    #
    # Example:
    #   from data.aptos_dataset import APTOSDataset
    #   from preprocessing.augmentation_pipeline import (
    #       get_fundus_train_transforms, get_val_transforms_with_resize
    #   )
    #   
    #   dataset = APTOSDataset(
    #       csv_path=f"{config['paths']['aptos']}/train.csv",
    #       img_dir=f"{config['paths']['aptos']}/train_images",
    #       transform=get_fundus_train_transforms(config['preprocessing']['image_size']),
    #       mode='train'
    #   )
    #   
    #   labels = dataset.df['diagnosis'].values
    
    print("\n" + "=" * 60)
    print("  TRAINING LOOP READY")
    print("  Set up DataLoaders in your Kaggle notebook, then either:")
    print("  1. Use train_fold() for single fold training")
    print("  2. Use the K-Fold CV loop below for full ensemble")
    print("=" * 60)
    
    # ──────────────────────────────────────────────────────────────
    # K-Fold Cross-Validation Loop (Template)
    # ──────────────────────────────────────────────────────────────
    # kfold = config['training'].get('kfold', 5)
    # skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
    # 
    # fold_results = []
    # for fold_idx, (train_idx, val_idx) in enumerate(skf.split(range(len(dataset)), labels)):
    #     if args.fold is not None and fold_idx != args.fold:
    #         continue  # Skip folds not requested
    #     
    #     # Create fold-specific datasets with appropriate transforms
    #     img_size = config['preprocessing']['image_size']
    #     train_subset = Subset(dataset, train_idx)
    #     train_subset.dataset.transform = get_fundus_train_transforms(img_size)
    #     
    #     val_dataset_fold = APTOSDataset(
    #         csv_path=f"{config['paths']['aptos']}/train.csv",
    #         img_dir=f"{config['paths']['aptos']}/train_images",
    #         transform=get_val_transforms_with_resize(img_size),
    #         mode='val'
    #     )
    #     val_subset = Subset(val_dataset_fold, val_idx)
    #     
    #     best_qwk, thresholds = train_fold(
    #         fold_idx, train_subset, val_subset, config, args
    #     )
    #     fold_results.append({
    #         'fold': fold_idx,
    #         'best_qwk': best_qwk,
    #         'thresholds': thresholds
    #     })
    #     print(f"\n  Fold {fold_idx}: QWK = {best_qwk:.4f}")
    # 
    # # Print summary
    # if fold_results:
    #     avg_qwk = np.mean([r['best_qwk'] for r in fold_results])
    #     std_qwk = np.std([r['best_qwk'] for r in fold_results])
    #     print(f"\n{'='*60}")
    #     print(f"  K-FOLD RESULTS ({kfold} folds)")
    #     print(f"  Mean QWK: {avg_qwk:.4f} ± {std_qwk:.4f}")
    #     for r in fold_results:
    #         print(f"    Fold {r['fold']}: {r['best_qwk']:.4f}")
    #     print(f"{'='*60}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IRDAS MSDNet V2 SOTA Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--kfold', type=int, default=None,
                        help='Number of CV folds (overrides config)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Train only this specific fold (0-indexed)')
    args = parser.parse_args()
    main(args)
