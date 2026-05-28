"""
============================================================================
IRDAS MSDNet — KAGGLE TRAINING NOTEBOOK (Part 3: Evaluation + XAI)
============================================================================
SOTA V2 — CORAL predictions, TTA, QWK threshold optimization
Run AFTER training completes. Loads best checkpoint and generates all results.
"""

# ============================================================
# CELL 13 & 14: 5-Fold EMA + TTA Inference Ensemble
# ============================================================
"""
# --- Uncomment and run ---

# Recreate full val_loader (we will evaluate on the entire dataset for the final QWK)
# In Kaggle inference, this would be the test_loader
df = pd.read_csv(CFG['aptos_csv'])
_, val_loader, _, _ = create_dataloaders(df, df)

all_preds, all_trues, all_continuous = [], [], []

models = []
for fold in range(5):
    ckpt_path = f"{CFG['ckpt_dir']}/msdnet_best_fold{fold}.pth"
    if not os.path.exists(ckpt_path):
        print(f"Skipping fold {fold}, no checkpoint found.")
        continue
        
    model = build_model()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    
    # Load EMA weights if available, fallback to regular model state
    if 'swa_state_dict' in ckpt:
        model.load_state_dict(ckpt['swa_state_dict'])
        print(f"Fold {fold}: Loaded EMA weights (epoch {ckpt['epoch']})")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Fold {fold}: Loaded raw weights (epoch {ckpt['epoch']})")
        
    model.eval()
    model.use_vessel = False
    models.append(model)

print(f"Running ensemble inference with {len(models)} models and TTA...")

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        
        # Aggregate continuous predictions across all 5 models
        batch_continuous = torch.zeros(imgs.size(0)).to(DEVICE)
        
        for model in models:
            # TTA: average logits across 8 geometric transforms
            tta_out = model.predict_with_tta(imgs, n_tta=CFG['tta_passes'])
            continuous = torch.sigmoid(tta_out['dr_logits']).sum(dim=1)
            batch_continuous += continuous
            
        # Average the continuous predictions
        batch_continuous /= len(models)
        
        # Round the averaged continuous values to get the ordinal class
        preds = torch.round(batch_continuous).long().clamp(0, 4).cpu().numpy()
        
        all_preds.extend(preds)
        all_trues.extend(labels.numpy())
        all_continuous.extend(batch_continuous.cpu().numpy())

# Evaluate
qwk_ensemble = cohen_kappa_score(all_trues, all_preds, weights='quadratic')

# Optimized QWK (Nelder-Mead learned thresholds)
opt_thresholds = optimize_qwk_thresholds(np.array(all_continuous), np.array(all_trues))
opt_preds = apply_optimized_thresholds(np.array(all_continuous), opt_thresholds)
qwk_optimized = cohen_kappa_score(all_trues, opt_preds, weights='quadratic')

cm = confusion_matrix(all_trues, opt_preds, labels=[0,1,2,3,4])

print(f"\\nFINAL ENSEMBLE RESULTS:")
print(f"  QWK (TTA Ensemble):         {qwk_ensemble:.4f}")
print(f"  QWK (TTA + Opt Thresholds): {qwk_optimized:.4f}")
print(f"  Optimized thresholds:       {[f'{t:.3f}' for t in opt_thresholds]}")
print(f"  Confusion Matrix:\\n{cm}")
"""


# ============================================================
# CELL 15: MC Dropout Uncertainty Estimation
# ============================================================
"""
# --- Uncomment and run ---

base = model.module if isinstance(model, nn.DataParallel) else model
uncertainties = []

for imgs, labels in val_loader:
    unc = base.predict_with_uncertainty(imgs.to(DEVICE), n=CFG['mc_passes'])
    # CORAL: predict class from last-pass logits
    dr_pred = ordinal_logits_to_class(unc['dr_mean_logits']).cpu().numpy()
    dr_unc = unc['dr_std'].cpu().numpy()
    for i in range(len(labels)):
        uncertainties.append({
            'true': int(labels[i]),
            'pred': int(dr_pred[i]),
            'uncertainty': float(dr_unc[i]),
            'correct': int(labels[i]) == int(dr_pred[i])
        })

unc_df = pd.DataFrame(uncertainties)
print(f"MC Dropout Uncertainty Analysis (N={CFG['mc_passes']} passes):")
print(f"  Correct predictions — mean uncertainty: {unc_df[unc_df['correct']==True]['uncertainty'].mean():.4f}")
print(f"  Wrong predictions   — mean uncertainty: {unc_df[unc_df['correct']==False]['uncertainty'].mean():.4f}")
print(f"  Ratio (wrong/correct): {unc_df[unc_df['correct']==False]['uncertainty'].mean() / max(unc_df[unc_df['correct']==True]['uncertainty'].mean(), 1e-6):.2f}x")
unc_df.to_csv(f"{CFG['output_dir']}/uncertainty_results.csv", index=False)
print(f"Saved: {CFG['output_dir']}/uncertainty_results.csv")
"""


# ============================================================
# CELL 16: Grad-CAM++ Heatmaps
# ============================================================
"""
# --- Uncomment and run ---
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

base = model.module if isinstance(model, nn.DataParallel) else model
base.eval()
base.use_vessel = False

class ModelWrapper(torch.nn.Module):
    def __init__(self, model, key):
        super().__init__()
        self.model = model
        self.key = key
    def forward(self, x):
        return self.model(x)[self.key]

dr_target = [base.dr_branch.cbam.spatial.conv]
hr_target = [base.hr_branch.cbam.spatial.conv]
cam_dr = GradCAMPlusPlus(model=ModelWrapper(base, 'dr_logits'), target_layers=dr_target)
cam_hr = GradCAMPlusPlus(model=ModelWrapper(base, 'hr_logits'), target_layers=hr_target)

df = pd.read_csv(CFG['aptos_csv'])
os.makedirs(f"{CFG['output_dir']}/xai_heatmaps", exist_ok=True)
sz = CFG['image_size']

for grade in [0, 1, 2, 3, 4]:
    row = df[df['diagnosis'] == grade].iloc[0]
    img_path = os.path.join(CFG['aptos_imgs'], row['id_code'] + '.png')
    raw = cv2.imread(img_path)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = preprocess_fundus(raw, size=sz)
    display_img = cv2.resize(raw, (sz, sz)).astype(np.float32) / 255.0

    tensor = get_val_aug()(image=img)['image'].unsqueeze(0).to(DEVICE)

    hm_dr = cam_dr(input_tensor=tensor)[0]
    hm_hr = cam_hr(input_tensor=tensor)[0]
    ov_dr = show_cam_on_image(display_img, hm_dr, use_rgb=True)
    ov_hr = show_cam_on_image(display_img, hm_hr, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(display_img); axes[0].set_title(f'Grade {grade} Original')
    axes[1].imshow(ov_dr); axes[1].set_title('DR Attention (Grad-CAM++)')
    axes[2].imshow(ov_hr); axes[2].set_title('HR Attention (Grad-CAM++)')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{CFG['output_dir']}/xai_heatmaps/grade_{grade}.png", dpi=150)
    plt.close()
    print(f"  Grade {grade} heatmap saved ✓")

print(f"All heatmaps saved to {CFG['output_dir']}/xai_heatmaps/")
"""


# ============================================================
# CELL 17: Plot Training Curves (with Phase Markers)
# ============================================================
"""
# --- Uncomment and run ---
import matplotlib.pyplot as plt

with open(f"{CFG['output_dir']}/logs/training_history.json") as f:
    hist = json.load(f)

epochs = [h['epoch'] for h in hist]
losses = [h.get('train_loss', 0) for h in hist]
qwks = [h.get('qwk', None) for h in hist]
qwk_epochs = [e for e, q in zip(epochs, qwks) if q is not None]
qwk_vals = [q for q in qwks if q is not None]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Loss curve with phase markers
ax1.plot(epochs, losses, 'b-', linewidth=1.5, label='Train Loss')
ax1.axvline(x=CFG['freeze_epochs'], color='orange', linestyle='--', alpha=0.7, label='Unfreeze')
ax1.axvline(x=CFG['swa_start_epoch'], color='purple', linestyle='--', alpha=0.7, label='SWA Start')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss (3-Phase)')
ax1.legend(); ax1.grid(True, alpha=0.3)

# QWK curve
ax2.plot(qwk_epochs, qwk_vals, 'g-o', markersize=4, label='Val QWK')
ax2.axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Target (0.85)')
ax2.axhline(y=0.90, color='gold', linestyle='--', alpha=0.5, label='SOTA (0.90)')
ax2.axvline(x=CFG['freeze_epochs'], color='orange', linestyle='--', alpha=0.5, label='Unfreeze')
ax2.axvline(x=CFG['swa_start_epoch'], color='purple', linestyle='--', alpha=0.5, label='SWA Start')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('QWK'); ax2.set_title('Validation QWK')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CFG['output_dir']}/training_curves.png", dpi=150)
plt.show()
print("Training curves saved ✓")
"""


# ============================================================
# CELL 18: One-Click Results Download
# ============================================================
"""
# --- Uncomment and run ---
import shutil
from IPython.display import FileLink, display

zip_name = "IRDAS_V2_SOTA_Results"
os.makedirs(zip_name, exist_ok=True)

# Best weights
if os.path.exists(f"{CFG['ckpt_dir']}/msdnet_best.pth"):
    shutil.copy(f"{CFG['ckpt_dir']}/msdnet_best.pth", f"{zip_name}/msdnet_best.pth")

# Logs, metrics, plots
for src in [f"{CFG['output_dir']}/logs/training_history.json",
            f"{CFG['output_dir']}/final_results.json",
            f"{CFG['output_dir']}/training_curves.png",
            f"{CFG['output_dir']}/uncertainty_results.csv"]:
    if os.path.exists(src):
        shutil.copy(src, f"{zip_name}/{os.path.basename(src)}")

# Grad-CAM++ heatmaps
if os.path.exists(f"{CFG['output_dir']}/xai_heatmaps"):
    shutil.copytree(f"{CFG['output_dir']}/xai_heatmaps", f"{zip_name}/xai_heatmaps", dirs_exist_ok=True)

shutil.make_archive(zip_name, 'zip', zip_name)

print("✅ Done! Click the link below to download everything at once:")
display(FileLink(f"{zip_name}.zip"))
"""
