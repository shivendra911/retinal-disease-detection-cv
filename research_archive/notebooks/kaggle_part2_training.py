"""
============================================================================
IRDAS MSDNet — KAGGLE TRAINING NOTEBOOK (Part 2: Training Loop)
============================================================================
SOTA V2 Recipe — 3-Phase Training: Freeze → Unfreeze → SWA
Run AFTER Part 1 passes all checks.
"""

# ============================================================
# CELL 10: Create DataLoaders
# ============================================================
def create_dataloaders(train_df, val_df):
    """Set up train/val splits with class-balanced sampling + pre-caching."""
    
    train_ds = APTOSDataset(train_df, CFG['aptos_imgs'], get_train_aug(), 
                            precache=CFG['precache'])
    val_ds = APTOSDataset(val_df, CFG['aptos_imgs'], get_val_aug(), 
                          precache=CFG['precache'])
    
    # Weighted sampler to handle class imbalance
    sample_weights = [train_ds.class_weights[label] for label in train_df['diagnosis']]
    sampler = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(train_ds), replacement=True)
    
    # FIX: If precache is True, train_ds holds ~1.6GB in RAM. 
    # Using num_workers > 0 + persistent_workers=True forces PyTorch to instantly pickle 
    # and copy 1.6GB across processes -> causing instant GIL deadlock and memory spikes.
    nw = CFG.get('num_workers', 2)   # Always use workers regardless of precache
    pw = False                       # <-- CHANGE THIS TO FALSE
    pf = 2 if nw > 0 else None       # prefetch_factor
    
    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              sampler=sampler, num_workers=nw, pin_memory=True,
                              drop_last=True, persistent_workers=pw, prefetch_factor=pf)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False,
                            num_workers=nw, pin_memory=True,
                            persistent_workers=pw, prefetch_factor=pf)
    
    vessel_loader = None
    if CFG.get('use_vessel', True):
        drive_ds = DRIVEDataset(CFG['drive_imgs'], CFG['drive_masks'], get_vessel_aug('train'))
        # Using num_workers=0 to prevent multiprocessing lockouts when re-initializing iter() every 5 batches
        vessel_loader = DataLoader(drive_ds, batch_size=4, shuffle=True, num_workers=0)
        
    return train_loader, val_loader, vessel_loader, train_ds.class_weights


# ============================================================
# CELL 11: SOTA Trainer + Accuracy Boosters (EMA, Mixup)
# ============================================================

class EMA:
    """Exponential Moving Average of model weights.
    Maintains a smoothed copy of weights: ema_w = decay * ema_w + (1-decay) * model_w
    The EMA model generalizes better than the raw model (+0.5-1% QWK).
    Used for validation and final inference, not for training."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply(self, model):
        """Swap model weights with EMA weights (for validation)."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        """Restore original weights (after validation)."""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


def mixup_data(x, y, alpha=0.4):
    """Ordinal-aware Mixup: blend images and labels for regularization.
    With CORAL ordinal labels, Mixup creates soft inter-grade targets
    that teach the model grade boundaries (e.g., 1.7 = between grade 1 and 2)."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 (primary image dominates)
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


import itertools

class Trainer:
    """SOTA Training Manager with:
    - Phase 1: Frozen backbone (head_lr=1e-3)
    - Phase 2: Unfrozen with differential LR + cosine warmup
    - Phase 3: SWA (epoch 45+)
    - AMP mixed precision + gradient accumulation
    - EMA weight averaging + Mixup regularization
    - CORAL ordinal loss + QWK threshold optimization
    """

    def __init__(self, model, class_weights, vessel_loader=None):
        self.model = model

        # CORAL ordinal loss (SOTA) or FocalLoss (legacy)
        if CFG.get('dr_ordinal', True):
            self.dr_loss = CoralOrdinalLoss(
                num_classes=CFG['dr_num_classes'],
                label_smoothing=CFG['label_smoothing']
            ).to(DEVICE)
            self.use_coral = True
        else:
            self.dr_loss = FocalLoss(
                weights=class_weights.to(DEVICE),
                label_smoothing=CFG.get('label_smoothing', 0.0)
            ).to(DEVICE)
            self.use_coral = False

        self.dice_bce = DiceBCELoss().to(DEVICE)
        self.vessel_loader = vessel_loader
        self.vessel_iter = iter(vessel_loader) if vessel_loader else None

        # AMP — Mixed Precision: halves VRAM by using float16 for activations
        self.scaler = torch.amp.GradScaler('cuda')

        # EMA — smoothed model weights for better generalization
        self.ema = None
        if CFG.get('use_ema', False):
            self.ema = EMA(self._base_model() if isinstance(model, nn.DataParallel) else model,
                           decay=CFG.get('ema_decay', 0.999))
            print(f"✅ EMA enabled (decay={CFG['ema_decay']})")

        # Mixup — ordinal-aware image blending
        self.use_mixup = CFG.get('use_mixup', False)
        self.mixup_alpha = CFG.get('mixup_alpha', 0.4)
        if self.use_mixup:
            print(f"✅ Mixup enabled (alpha={self.mixup_alpha})")

        self.best_qwk = 0
        self.best_thresholds = None
        self.history = []
        self.start_epoch = 0
        self.accum_steps = CFG.get('gradient_accumulation', 2)

    def _base_model(self):
        """Get underlying model, unwrapping DataParallel if needed."""
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _get_vessel_batch(self):
        if self.vessel_loader is None:
            return None
        try:
            return next(self.vessel_iter)
        except StopIteration:
            self.vessel_iter = iter(self.vessel_loader)
            return next(self.vessel_iter)

    def _build_optimizer(self, phase='frozen'):
        """Build optimizer with differential learning rates.
        frozen: only head params at head_lr
        unfrozen: backbone at lr*0.1, head at lr"""
        base = self._base_model()
        if phase == 'frozen':
            for name, p in base.named_parameters():
                p.requires_grad = 'backbone' not in name
            head_params = [p for p in base.parameters() if p.requires_grad]
            return torch.optim.AdamW(head_params, lr=CFG['head_lr'],
                                     weight_decay=CFG['weight_decay'])
        else:
            for p in base.parameters():
                p.requires_grad = True
            backbone_params = [p for n, p in base.named_parameters() if 'backbone' in n]
            head_params = [p for n, p in base.named_parameters() if 'backbone' not in n]
            return torch.optim.AdamW([
                {'params': backbone_params, 'lr': CFG['lr'] * CFG['backbone_lr_mult']},
                {'params': head_params, 'lr': CFG['lr']},
            ], weight_decay=CFG['weight_decay'])

    def _build_scheduler(self, optimizer, total_epochs):
        """Linear warmup → cosine annealing."""
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6 / max(CFG['lr'], 1e-7),
            total_iters=CFG['warmup_epochs'])
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_epochs - CFG['warmup_epochs'], 1), eta_min=1e-7)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[CFG['warmup_epochs']])

    def train_one_epoch(self, loader, epoch, optimizer):
        """AMP + Mixup + EMA + gradient accumulation + decoupled vessel."""
        self.model.train()
        base = self._base_model()

        total_loss, n_batches = 0, 0
        optimizer.zero_grad()

        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(DEVICE, non_blocking=True)
            dr_labels = labels.to(DEVICE, non_blocking=True)

            # ── Mixup ──
            if self.use_mixup and epoch >= CFG.get('freeze_epochs', 5):
                imgs, dr_labels_a, dr_labels_b, lam = mixup_data(imgs, dr_labels, self.mixup_alpha)
            else:
                dr_labels_a, dr_labels_b, lam = dr_labels, dr_labels, 1.0

            # ── Step 1: All Forward Passes ──
            out = self.model(imgs, task='dr')
            
            v_out, v_masks = None, None
            if self.vessel_loader is not None:
                vbatch = self._get_vessel_batch()
                if vbatch is not None:
                    v_imgs, v_masks = vbatch[0].to(DEVICE), vbatch[1].to(DEVICE)
                    v_out = self.model(v_imgs, task='vessel')

            # ── Step 2: Combined Loss in AMP ──
            with torch.amp.autocast('cuda'):
                L_dr = lam * self.dr_loss(out['dr_logits'], dr_labels_a) + \
                       (1 - lam) * self.dr_loss(out['dr_logits'], dr_labels_b)
                
                combined_loss = L_dr / self.accum_steps
                
                if v_out is not None and 'vessel_pred' in v_out:
                    v_loss = (CFG['lambda_vessel'] * self.dice_bce(
                        v_out['vessel_pred'], v_masks)) / self.accum_steps
                    combined_loss = combined_loss + v_loss 

            # ── Step 3: Single Backward Pass ──
            if torch.isnan(combined_loss):
                print(f"  ⚠️ NaN at batch {batch_idx}! Skipping...")
                optimizer.zero_grad()
                continue

            self.scaler.scale(combined_loss).backward()
            batch_loss = combined_loss.item() * self.accum_steps

            # ── Step 4: Scaler step + EMA update ──
            if (batch_idx + 1) % self.accum_steps == 0 or (batch_idx + 1) == len(loader):
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

                if self.ema is not None:
                    self.ema.update(base)

            total_loss += batch_loss
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, loader):
        """Validate with EMA weights + CORAL predictions + QWK threshold optimization."""
        base = self._base_model()

        # Use EMA weights for validation (better generalization)
        if self.ema is not None:
            self.ema.apply(base)

        self.model.eval()

        all_preds, all_trues, all_continuous = [], [], []

        for imgs, labels in loader:
            out = self.model(imgs.to(DEVICE))
            if self.use_coral:
                preds = ordinal_logits_to_class(out['dr_logits']).cpu().numpy()
                continuous = torch.sigmoid(out['dr_logits']).sum(dim=1).cpu().numpy()
                all_continuous.extend(continuous)
            else:
                preds = out['dr_logits'].argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_trues.extend(labels.numpy())

        # Restore original weights after validation
        if self.ema is not None:
            self.ema.restore(base)

        qwk = cohen_kappa_score(all_trues, all_preds, weights='quadratic')
        f1 = f1_score(all_trues, all_preds, average='macro')
        thresholds = None

        # Try optimized thresholds for extra QWK boost
        if self.use_coral and all_continuous:
            try:
                thresholds = optimize_qwk_thresholds(
                    np.array(all_continuous), np.array(all_trues))
                opt_preds = apply_optimized_thresholds(
                    np.array(all_continuous), thresholds)
                qwk_opt = cohen_kappa_score(all_trues, opt_preds, weights='quadratic')
                if qwk_opt > qwk:
                    qwk = qwk_opt
            except Exception:
                pass

        return qwk, f1, thresholds

    def save_checkpoint(self, epoch, metrics, is_best=False, swa_model=None):
        model_sd = self._base_model().state_dict()
        state = {
            'epoch': epoch,
            'model_state_dict': model_sd,
            'best_qwk': self.best_qwk,
            'metrics': metrics,
            'history': self.history,
            'config': CFG,
            'thresholds': self.best_thresholds,
        }
        if swa_model is not None:
            state['swa_state_dict'] = swa_model.state_dict()
        fold_str = f"_fold{self.fold}" if hasattr(self, 'fold') else ""
        path = f"{CFG['ckpt_dir']}/epoch_{epoch:03d}{fold_str}.pth"
        torch.save(state, path)
        print(f"  💾 Saved: {path}")
        if is_best:
            best_path = f"{CFG['ckpt_dir']}/msdnet_best{fold_str}.pth"
            torch.save(state, best_path)
            print(f"  🏆 NEW BEST: {best_path} (QWK={metrics.get('qwk',0):.4f})")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self._base_model().load_state_dict(ckpt['model_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_qwk = ckpt['best_qwk']
        self.best_thresholds = ckpt.get('thresholds', None)
        self.history = ckpt.get('history', [])
        print(f"Resumed from epoch {self.start_epoch}, best QWK: {self.best_qwk:.4f}")

    def train(self, train_loader, val_loader, resume_path=None):
        """Full 3-phase SOTA training loop."""
        if resume_path and os.path.exists(resume_path):
            self.load_checkpoint(resume_path)

        freeze_epochs = CFG['freeze_epochs']
        swa_start = CFG['swa_start_epoch']
        total_epochs = CFG['epochs']

        print("=" * 60)
        print(f"TRAINING: SOTA V2+ Recipe (Maximum Accuracy)")
        print(f"  Backbone: {CFG['backbone']}")
        print(f"  Loss: {'CORAL ordinal' if self.use_coral else 'FocalLoss'}")
        print(f"  Resolution: {CFG['image_size']}px")
        print(f"  Batch: {CFG['batch_size']} × {self.accum_steps} accum = {CFG['batch_size'] * self.accum_steps} effective")
        print(f"  Phases: Frozen(0-{freeze_epochs-1}) → Unfrozen({freeze_epochs}-{swa_start-1}) → SWA({swa_start}-{total_epochs-1})")
        print(f"  EMA: {'✅ decay=' + str(CFG.get('ema_decay', 0)) if self.ema else '❌'}")
        print(f"  Mixup: {'✅ alpha=' + str(self.mixup_alpha) if self.use_mixup else '❌'}")
        print(f"  AMP: ✅ | GPUs: {N_GPUS}")
        print("=" * 60)

        patience, no_improve = 15, 0

        # ── PHASE 1: Frozen backbone ──────────────────────────
        print(f"\n🧊 Phase 1: Frozen backbone (epochs 0-{freeze_epochs-1}), head lr={CFG['head_lr']}")
        optimizer = self._build_optimizer('frozen')
        # Constant LR for frozen phase (short, no decay needed)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=1)

        for epoch in range(self.start_epoch, min(freeze_epochs, total_epochs)):
            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch, optimizer)
            elapsed = time.time() - t0

            qwk, f1, thresholds = self.validate(val_loader)
            metrics = {'qwk': qwk, 'f1': f1, 'train_loss': train_loss}
            self.history.append({'epoch': epoch, **metrics, 'phase': 'frozen',
                                'lr': optimizer.param_groups[0]['lr']})

            is_best = qwk > self.best_qwk
            if is_best:
                self.best_qwk = qwk
                self.best_thresholds = thresholds
                no_improve = 0
                self.save_checkpoint(epoch, metrics, is_best=True)
            else:
                no_improve += 1

            print(f"Epoch {epoch:02d}/{total_epochs-1} | Loss: {train_loss:.4f} | "
                  f"QWK: {qwk:.4f} | F1: {f1:.4f} | Best: {self.best_qwk:.4f} | "
                  f"{elapsed:.0f}s | 🧊 frozen{'  ★' if is_best else ''}")
            scheduler.step()

        # ── PHASE 2: Unfrozen with differential LR ──────────
        remaining = swa_start - freeze_epochs
        print(f"\n🔥 Phase 2: Unfrozen (epochs {freeze_epochs}-{swa_start-1}), "
              f"backbone lr={CFG['lr'] * CFG['backbone_lr_mult']:.1e}, head lr={CFG['lr']:.1e}")
        optimizer = self._build_optimizer('unfrozen')
        scheduler = self._build_scheduler(optimizer, remaining)

        for epoch in range(max(self.start_epoch, freeze_epochs), min(swa_start, total_epochs)):
            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch, optimizer)
            elapsed = time.time() - t0

            # Validate every 2 epochs + first + last of phase
            if epoch % 2 == 0 or epoch == swa_start - 1 or epoch == freeze_epochs:
                qwk, f1, thresholds = self.validate(val_loader)
                metrics = {'qwk': qwk, 'f1': f1, 'train_loss': train_loss}
                self.history.append({'epoch': epoch, **metrics, 'phase': 'unfrozen',
                                    'lr': optimizer.param_groups[-1]['lr']})

                is_best = qwk > self.best_qwk
                if is_best:
                    self.best_qwk = qwk
                    self.best_thresholds = thresholds
                    no_improve = 0
                    self.save_checkpoint(epoch, metrics, is_best=True)
                else:
                    no_improve += 1

                print(f"Epoch {epoch:02d}/{total_epochs-1} | Loss: {train_loss:.4f} | "
                      f"QWK: {qwk:.4f} | F1: {f1:.4f} | Best: {self.best_qwk:.4f} | "
                      f"{elapsed:.0f}s | 🔥 unfrozen{'  ★' if is_best else ''}")
            else:
                self.history.append({'epoch': epoch, 'train_loss': train_loss, 'phase': 'unfrozen',
                                    'lr': optimizer.param_groups[-1]['lr']})
                print(f"Epoch {epoch:02d}/{total_epochs-1} | Loss: {train_loss:.4f} | {elapsed:.0f}s | 🔥 unfrozen")

            # Periodic checkpoint for crash recovery
            if epoch % CFG['checkpoint_every'] == 0:
                self.save_checkpoint(epoch, {'train_loss': train_loss})

            scheduler.step()

            if no_improve >= patience:
                print(f"\n⚠️ Early stopping at epoch {epoch} (no improvement for {patience} validations)")
                self.save_checkpoint(epoch, {'train_loss': train_loss})
                break

        # ── PHASE 3: SWA ─────────────────────────────────────
        if swa_start < total_epochs and no_improve < patience:
            print(f"\n⚡ Phase 3: SWA (epochs {swa_start}-{total_epochs-1}), lr={CFG['swa_lr']:.1e}")
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(optimizer, swa_lr=CFG['swa_lr'])

            for epoch in range(max(self.start_epoch, swa_start), total_epochs):
                t0 = time.time()
                train_loss = self.train_one_epoch(train_loader, epoch, optimizer)
                elapsed = time.time() - t0

                swa_model.update_parameters(self.model)
                swa_scheduler.step()

                # Validate SWA model
                if epoch % 2 == 0 or epoch == total_epochs - 1:
                    # Update BN stats for SWA model
                    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=DEVICE)
                    # Temporarily swap model for validation
                    orig_model = self.model
                    self.model = swa_model
                    qwk, f1, thresholds = self.validate(val_loader)
                    self.model = orig_model

                    metrics = {'qwk': qwk, 'f1': f1, 'train_loss': train_loss}
                    self.history.append({'epoch': epoch, **metrics, 'phase': 'swa'})

                    is_best = qwk > self.best_qwk
                    if is_best:
                        self.best_qwk = qwk
                        self.best_thresholds = thresholds

                    print(f"Epoch {epoch:02d}/{total_epochs-1} | Loss: {train_loss:.4f} | "
                          f"QWK(SWA): {qwk:.4f} | F1: {f1:.4f} | Best: {self.best_qwk:.4f} | "
                          f"{elapsed:.0f}s | ⚡ SWA{'  ★' if is_best else ''}")
                else:
                    self.history.append({'epoch': epoch, 'train_loss': train_loss, 'phase': 'swa'})
                    print(f"Epoch {epoch:02d}/{total_epochs-1} | Loss: {train_loss:.4f} | {elapsed:.0f}s | ⚡ SWA")

            # Save SWA model
            self.save_checkpoint(total_epochs - 1, {'qwk': self.best_qwk, 'train_loss': train_loss},
                               is_best=False, swa_model=swa_model)
            print(f"  ⚡ SWA model saved")

        # Save training history
        hist_path = f"{CFG['output_dir']}/logs/training_history.json"
        with open(hist_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\nHistory saved: {hist_path}")
        print(f"Best QWK: {self.best_qwk:.4f}")
        if self.best_thresholds:
            print(f"Optimized thresholds: {[f'{t:.3f}' for t in self.best_thresholds]}")


# ============================================================
# CELL 12: RUN TRAINING (5-Fold Stratified CV)
# ============================================================

df = pd.read_csv(CFG['aptos_csv'])

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG['seed'])

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['diagnosis'])):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING FOLD {fold+1}/5")
    print(f"{'='*60}")
    
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    
    # 1. Create data
    train_loader, val_loader, vessel_loader, class_weights = create_dataloaders(train_df, val_df)
    
    # 2. Create model
    model = build_model()
    
    # 3. Create SOTA trainer
    trainer = Trainer(model, class_weights, vessel_loader)
    trainer.fold = fold  # Inject fold number for checkpoint naming
    
    # 4. Train with 3-phase SOTA recipe!
    try:
        trainer.train(train_loader, val_loader)
    except Exception as e:
        print(f"\n❌ FOLD {fold+1} CRASHED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        break
    
    # 5. Cleanup memory before next fold
    del model, trainer, train_loader, val_loader, vessel_loader
    import gc
    gc.collect()
    torch.cuda.empty_cache()
