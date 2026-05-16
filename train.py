"""
IRDAS — Main Training Script
==============================

Entry point for training MSDNet. Handles:
- Config loading
- Dataset setup with train/val splits
- Model initialization
- Training loop with WandB logging
- Checkpoint saving (every 5 epochs + best model)
- Validation with QWK and AUC metrics
- Resume from checkpoint support

Usage (Kaggle):
    python train.py --config config/config.yaml

Usage (resume):
    python train.py --config config/config.yaml --resume checkpoints/epoch_20.pth
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import wandb
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, random_split

from models.msdnet import MSDNet
from losses.task_loss import FocalLoss
from losses.vessel_loss import DiceBCELoss
from losses.disentangle_loss import contrastive_disentangle_loss
from evaluation.metrics import compute_qwk, compute_auc


def compute_total_loss(outputs, dr_labels, hr_labels, vessel_masks,
                       config, focal_loss, dice_bce_loss):
    """
    Total loss = L_dr + L_hr + λ_vessel × L_vessel + λ_contrastive × L_dis
    
    Returns:
        total_loss: Scalar tensor for backprop
        loss_dict: Dictionary of individual loss values for logging
    """
    # Task losses
    L_dr = focal_loss(outputs['dr_logits'], dr_labels)
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


def train_epoch(model, loader, optimizer, config, focal_loss, dice_bce_loss, epoch):
    """Run one training epoch."""
    model.train()
    model.training_mode = True
    total_loss = 0
    
    for batch_idx, batch in enumerate(loader):
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
            config, focal_loss, dice_bce_loss
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Log batch metrics
        if batch_idx % 10 == 0:
            wandb.log({f'batch/{k}': v for k, v in loss_dict.items()})
    
    return total_loss / max(len(loader), 1)


def validate(model, val_loader, task='dr'):
    """Run validation and compute metrics."""
    model.eval()
    model.training_mode = False
    
    all_preds, all_trues = [], []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            out = model(imgs.cuda())
            
            if task == 'dr':
                preds = out['dr_logits'].argmax(-1).cpu().numpy()
            else:  # hr
                preds = torch.sigmoid(out['hr_logits']).squeeze().cpu().numpy()
            
            all_preds.extend(preds)
            all_trues.extend(labels.numpy())
    
    if task == 'dr':
        return compute_qwk(all_trues, all_preds)
    else:
        return compute_auc(all_trues, all_preds)


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, best_qwk, path):
    """Save full training state for resume capability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'best_qwk': best_qwk,
    }, path)


def main(args):
    # Load config
    config = yaml.safe_load(open(args.config))
    
    # Set seed for reproducibility
    torch.manual_seed(config['project']['seed'])
    np.random.seed(config['project']['seed'])
    
    # Init WandB
    wandb.init(
        project='IRDAS',
        name=f"train_{datetime.now().strftime('%Y%m%d_%H%M')}",
        config=config
    )
    
    # Initialize model
    model = MSDNet(config['model']).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MSDNet initialized: {total_params:,} parameters")
    
    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['T_max']
    )
    
    # Loss functions
    focal_loss = FocalLoss(gamma=2.0).cuda()
    dice_bce = DiceBCELoss().cuda()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_qwk = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_qwk = checkpoint['best_qwk']
        print(f"Resumed from epoch {start_epoch}, best QWK: {best_qwk:.4f}")
    
    # NOTE: DataLoaders should be set up here based on your dataset paths.
    # This is a template — fill in with actual dataset loading in Kaggle notebook.
    print("=" * 60)
    print("TRAINING LOOP READY")
    print("Set up DataLoaders in your Kaggle notebook, then call:")
    print("  train_epoch(model, train_loader, optimizer, config, focal_loss, dice_bce, epoch)")
    print("  validate(model, val_loader, task='dr')")
    print("=" * 60)
    
    # Training loop template
    # for epoch in range(start_epoch, config['training']['epochs']):
    #     train_loss = train_epoch(model, train_loader, optimizer,
    #                              config, focal_loss, dice_bce, epoch)
    #     
    #     # Validate every 3 epochs to save GPU time
    #     if epoch % 3 == 0 or epoch == config['training']['epochs'] - 1:
    #         qwk = validate(model, val_aptos_loader, task='dr')
    #         wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_qwk': qwk})
    #         print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | QWK: {qwk:.4f}")
    #         
    #         if qwk > best_qwk:
    #             best_qwk = qwk
    #             save_checkpoint(model, optimizer, scheduler, epoch,
    #                           {'qwk': qwk}, best_qwk, 'checkpoints/msdnet_best.pth')
    #     
    #     # Save checkpoint every 5 epochs for crash recovery
    #     if epoch % 5 == 0:
    #         save_checkpoint(model, optimizer, scheduler, epoch,
    #                        {'train_loss': train_loss}, best_qwk,
    #                        f'checkpoints/epoch_{epoch:03d}.pth')
    #     
    #     scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IRDAS MSDNet Training')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    main(args)
