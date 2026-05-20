"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Part 3: Loss Functions, Training Loop, Evaluation, XAI                    ║
║  Chapters 7-10 — Training the model and getting results                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CHAPTER 7: LOSS FUNCTIONS — Why 4 Different Losses?        ║
# ╚═══════════════════════════════════════════════════════════════╝
"""
Our total loss = L_dr + L_hr + 0.5 × L_vessel + 0.3 × L_dis

WHY 4 LOSSES?
Each loss teaches the model something different:

1. L_dr (Focal Loss)      → "learn to grade DR severity 0-4"
2. L_hr (BCE)             → "learn to detect HR present/absent"
3. L_vessel (Dice+BCE)    → "learn vascular anatomy" (auxiliary)
4. L_dis (Contrastive)    → "DR and HR branches must learn DIFFERENT things" (NOVEL)

WHAT IF we just used CrossEntropy for DR?
  CrossEntropy treats all samples equally. With 49% Grade 0 and 5% Grade 3,
  the model learns "always predict Grade 0" → 49% accuracy → useless.
  Focal Loss fixes this.
"""

class FocalLoss(nn.Module):
    """
    Focal Loss — Forces model to focus on HARD, RARE samples.

    Standard CrossEntropy: L = -log(p_t)
    Focal Loss:            L = -(1-p_t)^γ × log(p_t)

    The (1-p_t)^γ term is the KEY:
    - When model is CORRECT and CONFIDENT (p_t→1): (1-1)^2 = 0 → loss = 0 (ignore)
    - When model is WRONG (p_t→0): (1-0)^2 = 1 → full loss (focus here!)

    γ=2.0 is standard. Higher γ = more focus on hard samples.

    class_weights: additional weighting by inverse class frequency.
    Grade 3 (rare) gets ~9x the weight of Grade 0 (common).

    WHAT IF γ=0? Focal Loss = CrossEntropy (no focusing).
    WHAT IF γ=5? Too aggressive — model ignores easy samples entirely,
                 may become unstable.
    """
    def __init__(self, weights=None, gamma=2.0):
        super().__init__()
        self.weights = weights
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weights, reduction='none')
        pt = torch.exp(-ce)  # probability of correct class
        return (((1 - pt) ** self.gamma) * ce).mean()


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE for Vessel Segmentation.

    WHY COMBINED?
    - Dice: measures overlap. Good for imbalanced masks (vessels = ~10% of pixels).
            Formula: 2|A∩B| / (|A|+|B|). Range: 0=no overlap, 1=perfect.
            PROBLEM: gradient is noisy when prediction is very wrong.
    - BCE: per-pixel binary cross-entropy. Stable gradients.
           PROBLEM: doesn't handle imbalance well (99% background → predicts all-background).
    - Combined: stability of BCE + imbalance handling of Dice.

    smooth=1.0 prevents division by zero when both pred and target are all-zero.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        pf, tf = pred.view(-1), target.view(-1)
        inter = (pf * tf).sum()
        dice = 1 - (2*inter + self.smooth) / (pf.sum() + tf.sum() + self.smooth)
        return dice + self.bce(pred, target)


def contrastive_loss(dr_feat, hr_feat, dr_label, hr_label, m_pure=0.1, m_co=0.3):
    """
    ★ NOVEL CONTRIBUTION — Contrastive Disentanglement Loss ★

    THE PROBLEM:
    Both DR and HR cause hemorrhages. Without this loss, both branches learn
    "look for red spots" → identical features → model can't distinguish diseases.

    THE SOLUTION:
    Measure cosine similarity between DR and HR branch embeddings.
    If they're too similar → penalize.

    THREE CASES:
    Case 1 — Pure DR (no HR): cosine_sim should be < 0.1 (very different)
    Case 2 — Pure HR (no DR): cosine_sim should be < 0.1
    Case 3 — Both DR+HR:      cosine_sim should be < 0.3 (some overlap OK)
    Case 4 — Neither:         no penalty (healthy image)

    WHY TWO MARGINS?
    Pure cases: branches MUST look at different things (DR=lesions, HR=vessel caliber)
    Co-occurring: SOME feature sharing is justified (same vessels are affected by both)

    WHAT IF we skip this loss?
    Ablation study expected to show ~3-5% QWK drop and similar Grad-CAM++ heatmaps
    for both branches (proving disentanglement failed).
    """
    dr_n = F.normalize(dr_feat, dim=-1)  # L2 normalize for stable cosine sim
    hr_n = F.normalize(hr_feat, dim=-1)
    cos = (dr_n * hr_n).sum(-1)          # cosine similarity per sample

    has_dr = (dr_label > 0).float()
    has_hr = (hr_label > 0).float()
    pure = (has_dr * (1-has_hr) + has_hr * (1-has_dr)).clamp(max=1)  # either but not both
    co = has_dr * has_hr                                               # both present

    L_p = pure * F.relu(cos - m_pure)    # hinge: penalize only if sim > margin
    L_c = co * F.relu(cos - m_co)
    return L_p.sum() / pure.sum().clamp(min=1) + L_c.sum() / co.sum().clamp(min=1)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  CHAPTER 8: TRAINING LOOP                                  ║
# ╚═══════════════════════════════════════════════════════════════╝
"""
THE TRAINING PROCESS (what happens each epoch):

1. WARMUP (epochs 0-4):
   LR starts at 0 and linearly increases to 1e-4.
   WHY? The backbone has pretrained ImageNet weights. Hitting it with full LR
   immediately destroys those useful features ("catastrophic forgetting").

2. FOR EACH BATCH:
   a. Load 32 APTOS images + labels
   b. Forward pass → get predictions + embeddings
   c. Compute 4 losses (DR, HR, vessel, contrastive)
   d. Check for NaN → skip batch if unstable
   e. Backward pass (compute gradients)
   f. Gradient clipping (max_norm=1.0) — prevents exploding gradients
   g. Optimizer step (update weights)

3. VALIDATION (every 3 epochs):
   - Evaluate on held-out val set (no augmentation, no dropout)
   - Compute QWK and F1
   - If QWK improved → save as best model

4. CHECKPOINTING (every 5 epochs):
   - Save EVERYTHING: model + optimizer + scheduler + history
   - Can resume from any checkpoint after crash

5. SCHEDULER STEP (after warmup):
   - Cosine annealing: LR decays smoothly from 1e-4 to ~0
   - WHY cosine? Smoother than step decay. Model explores early, fine-tunes late.

6. EARLY STOPPING:
   - If QWK doesn't improve for 15 validation cycles → stop
   - Prevents overfitting and wasting GPU time
"""

def create_dataloaders():
    """Set up train/val splits with class-balanced sampling."""
    df = pd.read_csv(CFG['aptos_csv'])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=CFG['seed'])
    train_idx, val_idx = next(sss.split(df, df['diagnosis']))
    train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    train_ds = APTOSDataset(train_df, CFG['aptos_imgs'], get_train_aug())
    val_ds = APTOSDataset(val_df, CFG['aptos_imgs'], get_val_aug())

    # WeightedRandomSampler: oversample rare classes
    counts = train_df['diagnosis'].value_counts().sort_index().values
    w = 1.0 / counts
    sample_w = [w[l] for l in train_df['diagnosis'].values]
    sampler = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'],
                            shuffle=False, num_workers=2, pin_memory=True)

    vessel_loader = None
    if CFG.get('use_vessel', True) and os.path.exists(CFG['drive_imgs']):
        drive_ds = DRIVEDataset(CFG['drive_imgs'], CFG['drive_masks'], get_vessel_aug('train'))
        vessel_loader = DataLoader(drive_ds, batch_size=4, shuffle=True, num_workers=0)
        print(f"DRIVE: {len(drive_ds)} vessel images loaded")

    return train_loader, val_loader, vessel_loader, train_ds.class_weights


class Trainer:
    """Complete training manager."""

    def __init__(self, model, class_weights, vessel_loader=None):
        self.model = model
        self.focal = FocalLoss(weights=class_weights.to(DEVICE)).to(DEVICE)
        self.dice_bce = DiceBCELoss().to(DEVICE)
        self.vessel_loader = vessel_loader
        self.vessel_iter = iter(vessel_loader) if vessel_loader else None
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=CFG['T_max'])
        self.best_qwk, self.history, self.start_epoch = 0, [], 0

    def _get_vessel_batch(self):
        if self.vessel_iter is None: return None
        try: return next(self.vessel_iter)
        except StopIteration:
            self.vessel_iter = iter(self.vessel_loader)
            return next(self.vessel_iter)

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        total_loss, n = 0, 0
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            dr_labels = labels.to(DEVICE, non_blocking=True)
            hr_labels = torch.zeros(imgs.size(0)).to(DEVICE)
            out = self.model(imgs)
            L_dr = self.focal(out['dr_logits'], dr_labels)
            L_hr = F.binary_cross_entropy_with_logits(out['hr_logits'].squeeze(), hr_labels)
            loss = L_dr + L_hr
            if 'vessel_pred' in out:
                vb = self._get_vessel_batch()
                if vb:
                    vi, vm = vb[0].to(DEVICE), vb[1].to(DEVICE)
                    vo = self.model(vi)
                    if 'vessel_pred' in vo:
                        loss = loss + CFG['lambda_vessel'] * self.dice_bce(vo['vessel_pred'], vm)
            loss = loss + CFG['lambda_contrastive'] * contrastive_loss(
                out['dr_feat'], out['hr_feat'], dr_labels, hr_labels.long())
            if torch.isnan(loss):
                self.optimizer.zero_grad(); continue
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item(); n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        self.model.use_vessel = False
        preds, trues = [], []
        for imgs, labels in loader:
            out = self.model(imgs.to(DEVICE))
            preds.extend(out['dr_logits'].argmax(-1).cpu().numpy())
            trues.extend(labels.numpy())
        self.model.use_vessel = CFG.get('use_vessel', True)
        return cohen_kappa_score(trues, preds, weights='quadratic'), \
               f1_score(trues, preds, average='macro')

    def save_ckpt(self, epoch, metrics, is_best=False):
        state = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(),
                 'scheduler_state_dict': self.scheduler.state_dict(),
                 'best_qwk': self.best_qwk, 'metrics': metrics,
                 'history': self.history, 'config': CFG}
        torch.save(state, f"{CFG['ckpt_dir']}/epoch_{epoch:03d}.pth")
        if is_best:
            torch.save(state, f"{CFG['ckpt_dir']}/msdnet_best.pth")
            print(f"  🏆 NEW BEST saved (QWK={metrics.get('qwk',0):.4f})")

    def load_ckpt(self, path):
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_qwk = ckpt['best_qwk']
        self.history = ckpt.get('history', [])
        print(f"✅ Resumed from epoch {self.start_epoch}, best QWK: {self.best_qwk:.4f}")

    def train(self, train_loader, val_loader, resume_path=None):
        if resume_path and os.path.exists(resume_path):
            self.load_ckpt(resume_path)
        print(f"\n{'='*60}\n🚀 TRAINING START: epochs {self.start_epoch}→{CFG['epochs']-1}\n{'='*60}")
        no_improve = 0

        for epoch in range(self.start_epoch, CFG['epochs']):
            t0 = time.time()
            if epoch < CFG['warmup_epochs']:
                for pg in self.optimizer.param_groups:
                    pg['lr'] = CFG['lr'] * (epoch + 1) / CFG['warmup_epochs']

            loss = self.train_one_epoch(train_loader, epoch)
            elapsed = time.time() - t0

            if epoch % 3 == 0 or epoch == CFG['epochs']-1 or epoch < 3:
                qwk, f1 = self.validate(val_loader)
                metrics = {'qwk': qwk, 'f1': f1, 'loss': loss}
                self.history.append({'epoch': epoch, **metrics})
                is_best = qwk > self.best_qwk
                if is_best: self.best_qwk, no_improve = qwk, 0
                else: no_improve += 1
                print(f"Epoch {epoch:02d} | Loss:{loss:.4f} | QWK:{qwk:.4f} | "
                      f"F1:{f1:.4f} | Best:{self.best_qwk:.4f} | {elapsed:.0f}s"
                      f"{'  ★' if is_best else ''}")
                if epoch % CFG['checkpoint_every'] == 0:
                    self.save_ckpt(epoch, metrics, is_best)
            else:
                self.history.append({'epoch': epoch, 'loss': loss})
                print(f"Epoch {epoch:02d} | Loss:{loss:.4f} | {elapsed:.0f}s")

            if epoch >= CFG['warmup_epochs']: self.scheduler.step()
            if no_improve >= 15:
                print(f"⚠️ Early stopping at epoch {epoch}")
                self.save_ckpt(epoch, {'loss': loss}); break

        # Always save final
        self.save_ckpt(CFG['epochs']-1, {'loss': loss, 'best_qwk': self.best_qwk}, False)
        with open(f"{CFG['output_dir']}/logs/training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"\n🎉 Training complete! Best QWK: {self.best_qwk:.4f}")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  CHAPTER 9: SMOKE TEST + TRAINING EXECUTION                ║
# ╚═══════════════════════════════════════════════════════════════╝

def smoke_test():
    """CHECKPOINT: Must pass before training."""
    print("🔬 SMOKE TEST...")
    model = MSDNet().to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(2, 3, 224, 224).to(DEVICE)
    model.train(); out = model(x)
    assert out['dr_logits'].shape == (2,5), "DR shape wrong"
    assert out['hr_logits'].shape == (2,1), "HR shape wrong"
    print("  Forward ✓")
    dr_l = torch.tensor([0,3]).to(DEVICE)
    hr_l = torch.zeros(2).to(DEVICE)
    total = FocalLoss()(out['dr_logits'], dr_l) + \
            contrastive_loss(out['dr_feat'], out['hr_feat'], dr_l, hr_l.long())
    assert not torch.isnan(total), "NaN!"
    total.backward()
    print(f"  Loss={total:.4f}, Backward ✓")
    del model; torch.cuda.empty_cache(); gc.collect()
    print("✅ SMOKE TEST PASSED\n")

# ▶▶▶ UNCOMMENT AND RUN IN ORDER: ◀◀◀
# smoke_test()
# train_loader, val_loader, vessel_loader, cw = create_dataloaders()
# model = MSDNet().to(DEVICE)
# trainer = Trainer(model, cw, vessel_loader)
# trainer.train(train_loader, val_loader)
# # To resume: trainer.train(train_loader, val_loader, resume_path="checkpoints/epoch_025.pth")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  CHAPTER 10: EVALUATION + XAI                              ║
# ╚═══════════════════════════════════════════════════════════════╝
"""
After training, we generate all results for the paper:
1. Final metrics (QWK, F1, confusion matrix)
2. MC Dropout uncertainty analysis
3. Grad-CAM++ heatmaps per disease branch
4. Training curves

▶▶▶ UNCOMMENT EACH BLOCK AFTER TRAINING COMPLETES ◀◀◀
"""

# --- Block A: Final Evaluation ---
"""
model = MSDNet().to(DEVICE)
ckpt = torch.load(f"{CFG['ckpt_dir']}/msdnet_best.pth", map_location=DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
_, val_loader, _, _ = create_dataloaders()
model.eval(); model.use_vessel = False
preds, trues = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        out = model(imgs.to(DEVICE))
        preds.extend(out['dr_logits'].argmax(-1).cpu().numpy())
        trues.extend(labels.numpy())
qwk = cohen_kappa_score(trues, preds, weights='quadratic')
print(f"Final QWK: {qwk:.4f}")
print(f"Confusion Matrix:\\n{confusion_matrix(trues, preds)}")
json.dump({'qwk': float(qwk)}, open(f"{CFG['output_dir']}/final_results.json",'w'), indent=2)
"""

# --- Block B: MC Dropout Uncertainty ---
"""
unc_results = []
for imgs, labels in val_loader:
    u = model.predict_with_uncertainty(imgs.to(DEVICE), n=CFG['mc_passes'])
    for i in range(len(labels)):
        unc_results.append({'true': int(labels[i]),
            'pred': int(u['dr_mean'][i].argmax()), 'unc': float(u['dr_std'][i])})
udf = pd.DataFrame(unc_results)
print(f"Correct preds uncertainty: {udf[udf.true==udf.pred]['unc'].mean():.4f}")
print(f"Wrong preds uncertainty:   {udf[udf.true!=udf.pred]['unc'].mean():.4f}")
udf.to_csv(f"{CFG['output_dir']}/uncertainty.csv", index=False)
"""

# --- Block C: Grad-CAM++ ---
"""
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
cam_dr = GradCAMPlusPlus(model=model, target_layers=[model.dr_branch.cbam.spatial.conv])
cam_hr = GradCAMPlusPlus(model=model, target_layers=[model.hr_branch.cbam.spatial.conv])
df = pd.read_csv(CFG['aptos_csv'])
for grade in range(5):
    row = df[df.diagnosis==grade].iloc[0]
    raw = cv2.cvtColor(cv2.imread(f"{CFG['aptos_imgs']}/{row.id_code}.png"), cv2.COLOR_BGR2RGB)
    img = preprocess_fundus(raw)
    disp = cv2.resize(raw, (224,224)).astype(np.float32)/255
    t = get_val_aug()(image=img)['image'].unsqueeze(0).to(DEVICE)
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    ax[0].imshow(disp); ax[0].set_title(f'Grade {grade}')
    ax[1].imshow(show_cam_on_image(disp, cam_dr(input_tensor=t)[0], use_rgb=True)); ax[1].set_title('DR')
    ax[2].imshow(show_cam_on_image(disp, cam_hr(input_tensor=t)[0], use_rgb=True)); ax[2].set_title('HR')
    [a.axis('off') for a in ax]
    plt.savefig(f"{CFG['output_dir']}/xai_heatmaps/grade_{grade}.png", dpi=150); plt.close()
print("Heatmaps saved!")
"""

# --- Block D: Training Curves ---
"""
hist = json.load(open(f"{CFG['output_dir']}/logs/training_history.json"))
fig, (a1,a2) = plt.subplots(1,2,figsize=(14,5))
a1.plot([h['epoch'] for h in hist], [h['loss'] for h in hist], 'b-')
a1.set_title('Loss'); a1.set_xlabel('Epoch'); a1.grid(True, alpha=0.3)
qwk_h = [h for h in hist if 'qwk' in h]
a2.plot([h['epoch'] for h in qwk_h], [h['qwk'] for h in qwk_h], 'g-o')
a2.axhline(0.85, color='r', linestyle='--', label='Target')
a2.set_title('QWK'); a2.legend(); a2.grid(True, alpha=0.3)
plt.savefig(f"{CFG['output_dir']}/training_curves.png", dpi=150); plt.close()
print("Curves saved!")
"""

# --- Block E: Download Checklist ---
"""
print("\\n" + "="*60)
print("📥 DOWNLOAD THESE BEFORE SESSION ENDS:")
print("="*60)
for f in sorted(glob.glob(f"{CFG['ckpt_dir']}/*.pth") +
                glob.glob(f"{CFG['output_dir']}/*.json") +
                glob.glob(f"{CFG['output_dir']}/*.csv") +
                glob.glob(f"{CFG['output_dir']}/*.png") +
                glob.glob(f"{CFG['output_dir']}/xai_heatmaps/*.png")):
    print(f"  {f} ({os.path.getsize(f)/1e6:.1f}MB)")
"""

print("✅ All chapters loaded. Follow the ▶▶▶ UNCOMMENT ◀◀◀ markers in order.")
