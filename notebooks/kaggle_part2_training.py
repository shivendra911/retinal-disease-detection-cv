"""
============================================================================
IRDAS MSDNet — COMPLETE KAGGLE NOTEBOOK (Part 2: Training Loop)
============================================================================
Run AFTER Part 1 passes all checks.
"""

# ============================================================
# CELL 10: Create DataLoaders
# ============================================================
def create_dataloaders():
    """Set up train/val splits with class-balanced sampling + pre-caching."""
    df = pd.read_csv(CFG['aptos_csv'])

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=CFG['seed'])
    train_idx, val_idx = next(sss.split(df, df['diagnosis']))
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"Train class dist:\n{train_df['diagnosis'].value_counts().sort_index()}")

    # Pre-cache eliminates the CPU bottleneck (preprocessing done ONCE)
    precache = CFG.get('precache', True)
    train_ds = APTOSDataset(train_df, CFG['aptos_imgs'], get_train_aug(), precache=precache)
    val_ds = APTOSDataset(val_df, CFG['aptos_imgs'], get_val_aug(), precache=precache)

    # Weighted sampler for class imbalance
    class_counts = train_df['diagnosis'].value_counts().sort_index().values
    weights_per_class = 1.0 / class_counts
    sample_weights = [weights_per_class[label] for label in train_df['diagnosis'].values]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    nw = CFG.get('num_workers', 4)
    # With pre-caching, workers only do augmentation (very fast) — 0 workers is fine
    nw_actual = 0 if precache else nw
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              sampler=sampler, num_workers=nw_actual, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'],
                            shuffle=False, num_workers=nw_actual, pin_memory=True)

    vessel_loader = None
    if os.path.exists(CFG['drive_imgs']):
        drive_ds = DRIVEDataset(CFG['drive_imgs'], CFG['drive_masks'], get_vessel_aug('train'))
        vessel_loader = DataLoader(drive_ds, batch_size=4, shuffle=True, num_workers=0)
        print(f"DRIVE: {len(drive_ds)} vessel images loaded")

    return train_loader, val_loader, vessel_loader, train_ds.class_weights

# Uncomment:
# train_loader, val_loader, vessel_loader, class_weights = create_dataloaders()


# ============================================================
# CELL 11: Training Functions
# ============================================================
class Trainer:
    """Complete training manager with DataParallel, mixup, and checkpointing."""

    def __init__(self, model, class_weights, vessel_loader=None):
        self.model = model
        self.focal = FocalLoss(weights=class_weights.to(DEVICE),
                               label_smoothing=CFG.get('label_smoothing', 0.1)).to(DEVICE)
        self.dice_bce = DiceBCELoss().to(DEVICE)
        self.vessel_loader = vessel_loader
        self.vessel_iter = iter(vessel_loader) if vessel_loader else None

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=CFG['T_max'])

        self.best_qwk = 0
        self.history = []
        self.start_epoch = 0
        self.use_mixup = CFG.get('mixup_alpha', 0) > 0

    def _get_vessel_batch(self):
        if self.vessel_iter is None:
            return None
        try:
            return next(self.vessel_iter)
        except StopIteration:
            self.vessel_iter = iter(self.vessel_loader)
            return next(self.vessel_iter)

    def _get_model_attr(self, attr):
        """Get attribute from model, handling DataParallel wrapper."""
        m = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return getattr(m, attr)

    def _set_model_attr(self, attr, value):
        m = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        setattr(m, attr, value)

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        self._set_model_attr('use_vessel', self.vessel_loader is not None)
        total_loss, n_batches = 0, 0

        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            dr_labels = labels.to(DEVICE, non_blocking=True)
            hr_labels = torch.zeros(imgs.size(0)).to(DEVICE)

            # Mixup augmentation (better generalization + class imbalance handling)
            if self.use_mixup and np.random.rand() < 0.5:
                imgs, dr_a, dr_b, lam = mixup_data(imgs, dr_labels, CFG['mixup_alpha'])
                out = self.model(imgs)
                L_dr = self.focal.forward_mixup(out['dr_logits'], dr_a, dr_b, lam)
            else:
                out = self.model(imgs)
                L_dr = self.focal(out['dr_logits'], dr_labels)

            L_hr = F.binary_cross_entropy_with_logits(
                out['hr_logits'].squeeze(), hr_labels)
            total = L_dr + L_hr

            # Vessel loss (alternating batches from DRIVE)
            if 'vessel_pred' in out:
                vbatch = self._get_vessel_batch()
                if vbatch is not None:
                    v_imgs, v_masks = vbatch
                    v_imgs = v_imgs.to(DEVICE)
                    v_masks = v_masks.to(DEVICE)
                    v_out = self.model(v_imgs)
                    if 'vessel_pred' in v_out:
                        L_v = self.dice_bce(v_out['vessel_pred'], v_masks)
                        total = total + CFG['lambda_vessel'] * L_v

            # Contrastive disentanglement loss
            L_dis = contrastive_loss(out['dr_feat'], out['hr_feat'],
                                     dr_labels, hr_labels.long(),
                                     CFG['margin_pure'], CFG['margin_cooccur'])
            total = total + CFG['lambda_contrastive'] * L_dis

            # Check for NaN
            if torch.isnan(total):
                print(f"  ⚠️ NaN loss at batch {n_batches}! Skipping...")
                self.optimizer.zero_grad()
                continue

            self.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += total.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        self._set_model_attr('use_vessel', False)
        all_preds, all_trues = [], []
        for imgs, labels in loader:
            out = self.model(imgs.to(DEVICE))
            preds = out['dr_logits'].argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(labels.numpy())
        qwk = cohen_kappa_score(all_trues, all_preds, weights='quadratic')
        f1 = f1_score(all_trues, all_preds, average='macro')
        return qwk, f1

    def save_checkpoint(self, epoch, metrics, is_best=False):
        # Handle DataParallel: save the inner model's state_dict
        model_sd = self.model.module.state_dict() if isinstance(
            self.model, nn.DataParallel) else self.model.state_dict()
        state = {
            'epoch': epoch,
            'model_state_dict': model_sd,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_qwk': self.best_qwk,
            'metrics': metrics,
            'history': self.history,
            'config': CFG,
        }
        path = f"{CFG['ckpt_dir']}/epoch_{epoch:03d}.pth"
        torch.save(state, path)
        print(f"  💾 Saved: {path}")

        if is_best:
            best_path = f"{CFG['ckpt_dir']}/msdnet_best.pth"
            torch.save(state, best_path)
            print(f"  🏆 NEW BEST: {best_path} (QWK={metrics.get('qwk',0):.4f})")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        # Load into inner model if DataParallel
        target = self.model.module if isinstance(
            self.model, nn.DataParallel) else self.model
        target.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_qwk = ckpt['best_qwk']
        self.history = ckpt.get('history', [])
        print(f"Resumed from epoch {self.start_epoch}, best QWK: {self.best_qwk:.4f}")

    def train(self, train_loader, val_loader, resume_path=None):
        """Full training loop with validation, checkpointing, and early stopping."""
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

        print("=" * 60)
        print(f"TRAINING: epochs {self.start_epoch}-{CFG['epochs']-1}")
        print(f"  LR: {CFG['lr']}, Batch: {CFG['batch_size']}, GPUs: {N_GPUS}, Mixup: {self.use_mixup}")
        print("=" * 60)

        patience, no_improve = 15, 0

        for epoch in range(self.start_epoch, CFG['epochs']):
            t0 = time.time()

            if epoch < CFG['warmup_epochs']:
                warmup_lr = CFG['lr'] * (epoch + 1) / CFG['warmup_epochs']
                for pg in self.optimizer.param_groups:
                    pg['lr'] = warmup_lr

            train_loss = self.train_one_epoch(train_loader, epoch)
            elapsed = time.time() - t0

            # Validate every 3 epochs + first + last
            if epoch % 3 == 0 or epoch == CFG['epochs'] - 1 or epoch < 3:
                qwk, f1 = self.validate(val_loader)
                metrics = {'qwk': qwk, 'f1': f1, 'train_loss': train_loss}
                self.history.append({'epoch': epoch, **metrics, 'lr': self.optimizer.param_groups[0]['lr']})

                is_best = qwk > self.best_qwk
                if is_best:
                    self.best_qwk = qwk
                    no_improve = 0
                    self.save_checkpoint(epoch, metrics, is_best=True)
                else:
                    no_improve += 1

                print(f"Epoch {epoch:02d}/{CFG['epochs']-1} | "
                      f"Loss: {train_loss:.4f} | QWK: {qwk:.4f} | F1: {f1:.4f} | "
                      f"Best: {self.best_qwk:.4f} | {elapsed:.0f}s"
                      f"{' ★' if is_best else ''}")
            else:
                metrics = {'train_loss': train_loss}
                self.history.append({'epoch': epoch, **metrics, 'lr': self.optimizer.param_groups[0]['lr']})
                print(f"Epoch {epoch:02d}/{CFG['epochs']-1} | Loss: {train_loss:.4f} | {elapsed:.0f}s")

            # Periodic Checkpoint
            if epoch % CFG['checkpoint_every'] == 0 or epoch == CFG['epochs'] - 1:
                # Don't double-save if we just saved it as best
                if not (epoch % 3 == 0 and is_best):
                    self.save_checkpoint(epoch, metrics, is_best=False)

            # Step scheduler (after warmup)
            if epoch >= CFG['warmup_epochs']:
                self.scheduler.step()

            # Early stopping
            if no_improve >= patience:
                print(f"\n⚠️ Early stopping at epoch {epoch} (no improvement for {patience} validations)")
                self.save_checkpoint(epoch, metrics)
                break

        # Save training history
        hist_path = f"{CFG['output_dir']}/logs/training_history.json"
        with open(hist_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nHistory saved: {hist_path}")
        print(f"Best QWK: {self.best_qwk:.4f}")


# ============================================================
# CELL 12: RUN TRAINING
# ============================================================
# --- Uncomment and run this cell ---

# 1. Create data (with pre-caching — takes ~2 min, then training is FAST)
train_loader, val_loader, vessel_loader, class_weights = create_dataloaders()

# 2. Create model (auto-wraps with DataParallel if 2 GPUs)
model = build_model()

# 3. Create trainer (with mixup + label smoothing)
trainer = Trainer(model, class_weights, vessel_loader)

# 4. Train!
trainer.train(train_loader, val_loader)

# 5. (Optional) Resume from checkpoint if session crashed:
# trainer.train(train_loader, val_loader, resume_path=f"{CFG['ckpt_dir']}/epoch_025.pth")
