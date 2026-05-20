"""
============================================================================
IRDAS MSDNet — COMPLETE KAGGLE NOTEBOOK (Part 3: Evaluation + XAI)
============================================================================
Run AFTER training completes. Loads best checkpoint and generates all results.
"""

# ============================================================
# CELL 13: Load Best Model & Evaluate
# ============================================================
"""
# --- Uncomment and run ---

# Load best checkpoint
model = MSDNet().to(DEVICE)
ckpt = torch.load(f"{CFG['ckpt_dir']}/msdnet_best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
print(f"Loaded best model from epoch {ckpt['epoch']}, QWK: {ckpt['best_qwk']:.4f}")

# Final validation
_, val_loader, _, _ = create_dataloaders()
model.eval()
model.use_vessel = False

all_preds, all_trues, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        out = model(imgs.to(DEVICE))
        probs = torch.softmax(out['dr_logits'], -1).cpu()
        preds = probs.argmax(-1).numpy()
        all_probs.extend(probs.numpy())
        all_preds.extend(preds)
        all_trues.extend(labels.numpy())

qwk = cohen_kappa_score(all_trues, all_preds, weights='quadratic')
f1 = f1_score(all_trues, all_preds, average='macro')
cm = confusion_matrix(all_trues, all_preds, labels=[0,1,2,3,4])

print(f"\\nFINAL RESULTS:")
print(f"  QWK: {qwk:.4f}")
print(f"  F1:  {f1:.4f}")
print(f"  Confusion Matrix:\\n{cm}")

# Save results
results = {'qwk': float(qwk), 'f1': float(f1),
           'confusion_matrix': cm.tolist(), 'epoch': ckpt['epoch']}
with open(f"{CFG['output_dir']}/final_results.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {CFG['output_dir']}/final_results.json")
"""


# ============================================================
# CELL 14: MC Dropout Uncertainty Estimation
# ============================================================
"""
# --- Uncomment and run ---

model = MSDNet().to(DEVICE)
ckpt = torch.load(f"{CFG['ckpt_dir']}/msdnet_best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])

# Uncomment ONLY if you just restarted the session and val_loader is missing from RAM
# _, val_loader, _, _ = create_dataloaders()
uncertainties = []

for imgs, labels in val_loader:
    unc = model.predict_with_uncertainty(imgs.to(DEVICE), n=CFG['mc_passes'])
    dr_pred = unc['dr_mean'].argmax(-1).cpu().numpy()
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
print(f"  Correct predictions  — mean uncertainty: {unc_df[unc_df['correct']==True]['uncertainty'].mean():.4f}")
print(f"  Wrong predictions    — mean uncertainty: {unc_df[unc_df['correct']==False]['uncertainty'].mean():.4f}")
unc_df.to_csv(f"{CFG['output_dir']}/uncertainty_results.csv", index=False)
print(f"Saved: {CFG['output_dir']}/uncertainty_results.csv")
"""


# ============================================================
# CELL 15: Grad-CAM++ Heatmaps
# ============================================================
"""
# --- Uncomment and run ---
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt

model = MSDNet().to(DEVICE)
ckpt = torch.load(f"{CFG['ckpt_dir']}/msdnet_best.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
model.use_vessel = False

# pytorch_grad_cam expects a single tensor output, not a dict. 
# We wrap the model to return just the logits for the specific task.
class ModelWrapper(torch.nn.Module):
    def __init__(self, model, key):
        super().__init__()
        self.model = model
        self.key = key
    def forward(self, x):
        return self.model(x)[self.key]

dr_target = [model.dr_branch.cbam.spatial.conv]
hr_target = [model.hr_branch.cbam.spatial.conv]
cam_dr = GradCAMPlusPlus(model=ModelWrapper(model, 'dr_logits'), target_layers=dr_target)
cam_hr = GradCAMPlusPlus(model=ModelWrapper(model, 'hr_logits'), target_layers=hr_target)

# Get 5 sample images
df = pd.read_csv(CFG['aptos_csv'])
os.makedirs(f"{CFG['output_dir']}/xai_heatmaps", exist_ok=True)

for i, grade in enumerate([0, 1, 2, 3, 4]):
    row = df[df['diagnosis'] == grade].iloc[0]
    img_path = os.path.join(CFG['aptos_imgs'], row['id_code'] + '.png')
    raw = cv2.imread(img_path)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = preprocess_fundus(raw)
    display_img = cv2.resize(raw, (224, 224)).astype(np.float32) / 255.0

    tensor = get_val_aug()(image=img)['image'].unsqueeze(0).to(DEVICE)

    hm_dr = cam_dr(input_tensor=tensor)[0]
    hm_hr = cam_hr(input_tensor=tensor)[0]
    ov_dr = show_cam_on_image(display_img, hm_dr, use_rgb=True)
    ov_hr = show_cam_on_image(display_img, hm_hr, use_rgb=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(display_img); axes[0].set_title(f'Grade {grade} Original')
    axes[1].imshow(ov_dr); axes[1].set_title('DR Attention')
    axes[2].imshow(ov_hr); axes[2].set_title('HR Attention')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{CFG['output_dir']}/xai_heatmaps/grade_{grade}.png", dpi=150)
    plt.close()
    print(f"  Grade {grade} heatmap saved ✓")

print(f"All heatmaps saved to {CFG['output_dir']}/xai_heatmaps/")
"""


# ============================================================
# CELL 16: Plot Training Curves
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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(epochs, losses, 'b-', label='Train Loss')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Training Loss')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(qwk_epochs, qwk_vals, 'g-o', label='Val QWK')
ax2.axhline(y=0.85, color='r', linestyle='--', label='Target (0.85)')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('QWK'); ax2.set_title('Validation QWK')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CFG['output_dir']}/training_curves.png", dpi=150)
plt.show()
print("Training curves saved ✓")
"""


# ============================================================
# CELL 17: One-Click Results Download
# ============================================================
"""
# --- Uncomment and run ---
import os
import shutil
from IPython.display import FileLink, display

zip_name = "IRDAS_V2_Results"
os.makedirs(zip_name, exist_ok=True)

# 1. Grab the best weights
if os.path.exists(f"{CFG['ckpt_dir']}/msdnet_best.pth"):
    shutil.copy(f"{CFG['ckpt_dir']}/msdnet_best.pth", f"{zip_name}/msdnet_best.pth")

# 2. Grab the logs, metrics, and plots
if os.path.exists(f"{CFG['output_dir']}/logs/training_history.json"):
    shutil.copy(f"{CFG['output_dir']}/logs/training_history.json", f"{zip_name}/training_history.json")
if os.path.exists(f"{CFG['output_dir']}/final_results.json"):
    shutil.copy(f"{CFG['output_dir']}/final_results.json", f"{zip_name}/final_results.json")
if os.path.exists(f"{CFG['output_dir']}/training_curves.png"):
    shutil.copy(f"{CFG['output_dir']}/training_curves.png", f"{zip_name}/training_curves.png")
if os.path.exists(f"{CFG['output_dir']}/uncertainty_results.csv"):
    shutil.copy(f"{CFG['output_dir']}/uncertainty_results.csv", f"{zip_name}/uncertainty_results.csv")

# 3. Grab the Grad-CAM++ heatmaps
if os.path.exists(f"{CFG['output_dir']}/xai_heatmaps"):
    shutil.copytree(f"{CFG['output_dir']}/xai_heatmaps", f"{zip_name}/xai_heatmaps", dirs_exist_ok=True)

# Zip it all up!
shutil.make_archive(zip_name, 'zip', zip_name)

print("✅ Done! Click the link below to download everything at once:")
display(FileLink(f"{zip_name}.zip"))
"""
