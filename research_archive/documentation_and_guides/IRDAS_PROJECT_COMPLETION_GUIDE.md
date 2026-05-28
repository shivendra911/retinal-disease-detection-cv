# IRDAS — Project Completion & Tools Guide
## How to Efficiently Complete the Project & Deploy on AWS (Student Pack)

> This guide covers **which tool to use, when, and how** — from first commit to live deployment.
> Designed for efficiency: no wasted time, no wrong tools.

---

## Table of Contents

1. [Tool Stack Overview](#1-tool-stack-overview)
2. [Development Environment Setup](#2-development-environment-setup)
3. [Kaggle Workflow — GPU Training](#3-kaggle-workflow--gpu-training)
4. [VS Code + Local Development](#4-vs-code--local-development)
5. [Git Workflow](#5-git-workflow)
6. [WandB — Experiment Tracking](#6-wandb--experiment-tracking)
7. [Efficient Debugging Strategies](#7-efficient-debugging-strategies)
8. [AWS Deployment with Student Pack](#8-aws-deployment-with-student-pack)
9. [Daily Workflow Checklist](#9-daily-workflow-checklist)
10. [Time-Saving Tips](#10-time-saving-tips)

---

## 1. Tool Stack Overview

| Tool | Purpose | When to Use | Cost |
|------|---------|-------------|------|
| **Kaggle Notebooks** | GPU training, dataset access | All training & evaluation | Free (30h GPU/week) |
| **VS Code** | Code editing, debugging, git | All code writing & review | Free |
| **Git + GitHub** | Version control | Every session | Free |
| **WandB** | Experiment tracking | All training runs | Free tier |
| **Conda** | Python env management | Local development | Free |
| **Google AI Studio** | Gemini API key | Stage 3 only | Free tier |
| **AWS (Student Pack)** | Production deployment | Final phase only | $100 credit |
| **Claude/Gemini AI** | Code assistance | When stuck | Free/paid |
| **draw.io** | Architecture diagrams | Paper figures | Free |
| **Overleaf** | LaTeX paper writing | Paper phase | Free |

---

## 2. Development Environment Setup

### Local Machine (VS Code)

```powershell
# 1. Create conda environment
conda create -n irdas python=3.10 -y
conda activate irdas

# 2. Install CPU-only PyTorch (for local testing — no GPU needed locally)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install all project dependencies
pip install timm==0.9.12 albumentations==1.3.1 opencv-python==4.8.1 ^
            xgboost==2.0.1 shap==0.43.0 scikit-learn==1.3.2 ^
            pandas==2.1.1 numpy==1.24.0 matplotlib==3.8.0 ^
            seaborn==0.13.0 wandb==0.16.0 pyyaml

# 4. VS Code extensions to install
# - Python (Microsoft)
# - Pylance (type checking)
# - Jupyter (notebook support)
# - GitLens (git history)
# - YAML (config editing)
```

### VS Code Settings for This Project
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "~/anaconda3/envs/irdas/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

---

## 3. Kaggle Workflow — GPU Training

### When to Use Kaggle
- ALL model training (baseline, MSDNet, ablation experiments)
- Dataset exploration (large files)
- Evaluation & metric computation
- Grad-CAM heatmap generation

### Kaggle Session Checklist

**Start of session:**
```python
# Cell 1 — Always run first
!pip install timm==0.9.12 segmentation-models-pytorch==0.3.3 \
             albumentations==1.3.1 pytorch-grad-cam==1.4.8 \
             netcal==1.3.5 wandb==0.16.0 --quiet

import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Cell 2 — Mount your code
# Option A: Clone from GitHub
!git clone https://github.com/YOUR_USERNAME/retinal-disease-detection-cv.git
%cd retinal-disease-detection-cv

# Option B: Upload files manually via Kaggle UI
```

**End of session (ALWAYS DO THIS):**
```python
# 1. Save checkpoint to Kaggle output
import shutil, os
os.makedirs("/kaggle/working/outputs", exist_ok=True)

# 2. Download important files before session expires
# Files in /kaggle/working/ persist as "Output" — download from notebook page

# 3. Log session
save_session_log("training_run_2", {"epoch": 35, "qwk": 0.856}, 
                 notes="Stopped early, will resume")
```

### Managing GPU Quota (30h/week)
```
Monday:    4h — Primary training
Tuesday:   4h — Continue training  
Wednesday: 4h — Continue / evaluate
Thursday:  4h — Ablation experiment
Friday:    4h — Ablation experiment
Saturday:  0h — Quota resets! CPU-only work
Sunday:    0h — Paper writing / planning
            
Budget remaining: 10h buffer for retries
```

### Kaggle Tips
- **Save early, save often**: Checkpoints every 5 epochs
- **Use Kaggle Datasets**: Upload your code as a Kaggle Dataset for easy import
- **Pin package versions**: Exact versions prevent breakage
- **Internet OFF during training**: Turn off internet in notebook settings to get longer sessions (9h vs 12h limit)

---

## 4. VS Code + Local Development

### When to Use VS Code (NOT Kaggle)
- Writing and editing `.py` source files
- Code review and refactoring
- Git commits and pushes
- config.yaml editing
- Paper writing
- README/documentation updates
- Unit tests for loss functions and data loading (CPU only)

### Local Testing Strategy
```python
# Test your code locally with tiny fake data before uploading to Kaggle
import torch
from models.msdnet import MSDNet
import yaml

config = yaml.safe_load(open('config/config.yaml'))
model = MSDNet(config['model'])  # CPU mode
x = torch.randn(2, 3, 224, 224)
out = model(x)
print("DR:", out['dr_logits'].shape)  # Should be (2, 5)
print("HR:", out['hr_logits'].shape)  # Should be (2, 1)
print("Test passed!")
```

---

## 5. Git Workflow

### Initial Setup
```powershell
cd c:\Users\shive\projects\cv\retinal-disease-detection-cv
git init
git remote add origin https://github.com/YOUR_USERNAME/retinal-disease-detection-cv.git
git checkout -b main
git add .
git commit -m "[PHASE-0] Initialize project structure"
git push -u origin main
```

### Daily Workflow
```powershell
# 1. Start of day — pull latest
git pull origin main

# 2. Create feature branch for today's work
git checkout -b feat/phase3-backbone

# 3. Work, test, commit often
git add models/backbone.py models/fpn.py
git commit -m "[PHASE-3] Implement EfficientNet-B0 backbone with FPN hooks"

# 4. End of day — push and merge
git push origin feat/phase3-backbone
git checkout main
git merge feat/phase3-backbone
git push origin main
```

### After Kaggle Session
```powershell
# Download results from Kaggle, copy to project
# Update tracking document
git add IRDAS_ENHANCED_PLAN.md outputs/
git commit -m "[PHASE-6] Training run 1 complete — baseline QWK 0.837"
git push
```

---

## 6. WandB — Experiment Tracking

### Setup (One-Time)
1. Go to [wandb.ai](https://wandb.ai) → Create free account
2. Get API key from Settings → Copy
3. In Kaggle notebook: `!wandb login YOUR_API_KEY`

### What to Track

```python
# During training — log every batch
wandb.log({
    "batch/loss_dr": L_dr.item(),
    "batch/loss_hr": L_hr.item(),
    "batch/loss_vessel": L_vessel.item(),
    "batch/loss_contrastive": L_dis.item(),
    "batch/loss_total": total_loss.item(),
    "batch/learning_rate": optimizer.param_groups[0]['lr'],
})

# After each epoch — log metrics
wandb.log({
    "epoch/train_loss": avg_train_loss,
    "epoch/val_dr_qwk": qwk,
    "epoch/val_hr_auc": hr_auc,
    "epoch": epoch,
})

# Log images (preprocessing comparison, heatmaps)
wandb.log({
    "examples/preprocessing": [
        wandb.Image(before_img, caption="Before"),
        wandb.Image(after_img, caption="After CLAHE")
    ],
    "examples/gradcam_dr": wandb.Image(dr_heatmap, caption="DR attention"),
    "examples/gradcam_hr": wandb.Image(hr_heatmap, caption="HR attention"),
})
```

### WandB Dashboard to Create
- **Run comparison**: Overlay loss curves from baseline vs MSDNet
- **Ablation table**: Auto-generated from 6 experiment runs
- **Best model summary**: Auto-saved hyperparameters of best run

---

## 7. Efficient Debugging Strategies

### When Training Doesn't Converge

```python
# Debug checklist — run in order
# 1. Check data
for imgs, labels in train_loader:
    print(f"Image shape: {imgs.shape}, range: [{imgs.min():.2f}, {imgs.max():.2f}]")
    print(f"Labels: {labels[:10]}")
    break  # just check first batch

# 2. Check model output
model.eval()
with torch.no_grad():
    out = model(imgs[:2].cuda())
    print(f"DR logits: {out['dr_logits']}")  # should NOT be all same value
    print(f"HR logits: {out['hr_logits']}")

# 3. Check loss
loss, loss_dict = compute_total_loss(out, labels[:2].cuda(), ...)
print(f"Losses: {loss_dict}")  # no NaN, no Inf, reasonable magnitudes

# 4. Check gradients
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 100 or grad_norm == 0:
            print(f"WARNING: {name} grad_norm = {grad_norm}")
```

### Common Issues Quick Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss = NaN | Learning rate too high | Reduce to 1e-5, add gradient clipping |
| Loss doesn't decrease | Wrong labels or preprocessing | Print 10 samples, visually verify |
| QWK stuck at 0 | All predictions = same class | Check class weights, reduce dropout |
| CUDA OOM | Batch too large | Reduce batch_size to 16, use gradient accumulation |
| Heatmaps all same | Model not converged or wrong layer | Train longer, verify target_layer |

---

## 8. AWS Deployment with Student Pack

### What You Get with AWS Student Pack
- **$100 in AWS credits** via GitHub Student Developer Pack
- Valid for 1 year
- Enough for: EC2 (compute), S3 (storage), ECR (containers)

### Step 1: Activate AWS Credits

1. Go to [education.github.com/pack](https://education.github.com/pack)
2. Verify student status with LPU email
3. Find "AWS" in the pack → Click "Get access"
4. Create AWS account → Apply credits

### Step 2: Prepare Your Model for Deployment

```python
# export_model.py — Run this after training is complete
import torch
from models.msdnet import MSDNet
import yaml

# Load best model
config = yaml.safe_load(open('config/config.yaml'))
model = MSDNet(config['model'])
model.load_state_dict(torch.load('checkpoints/msdnet_best.pth', map_location='cpu'))
model.eval()
model.training_mode = False

# Remove vessel decoder (not needed at inference)
del model.vessel_decoder

# Save lighter inference model
torch.save(model.state_dict(), 'checkpoints/msdnet_inference.pth')
print(f"Inference model size: {os.path.getsize('checkpoints/msdnet_inference.pth') / 1e6:.1f} MB")
```

### Step 3: Create Flask API

```python
# app.py — Inference API
from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import io
from PIL import Image
from models.msdnet import MSDNet
from preprocessing.clahe_preprocess import preprocess_fundus
import yaml

app = Flask(__name__)

# Load model once at startup
config = yaml.safe_load(open('config/config.yaml'))
model = MSDNet(config['model'])
model.load_state_dict(torch.load('checkpoints/msdnet_inference.pth', map_location='cpu'))
model.eval()
model.training_mode = False

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_fundus(img)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    
    with torch.no_grad():
        result = model.predict_with_uncertainty(tensor, n_passes=30)
    
    dr_grade = result['dr_mean'].argmax(-1).item()
    dr_confidence = result['dr_mean'].max(-1).values.item()
    dr_uncertainty = result['dr_uncertainty'].item()
    hr_probability = result['hr_mean'].item()
    hr_uncertainty = result['hr_uncertainty'].item()
    
    return jsonify({
        'dr_grade': int(dr_grade),
        'dr_confidence': round(dr_confidence, 4),
        'dr_uncertainty': round(dr_uncertainty, 4),
        'hr_probability': round(hr_probability, 4),
        'hr_uncertainty': round(hr_uncertainty, 4),
        'referral_recommended': dr_grade >= 2 or hr_probability > 0.5,
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model': 'MSDNet v1.0'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Step 4: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

COPY models/ models/
COPY preprocessing/ preprocessing/
COPY config/ config/
COPY checkpoints/msdnet_inference.pth checkpoints/
COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

```text
# requirements_deploy.txt (minimal — no training deps)
torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
timm==0.9.12
flask==3.0.0
gunicorn==21.2.0
opencv-python-headless==4.8.1.78
numpy==1.24.0
pyyaml==6.0.1
```

### Step 5: Deploy to AWS EC2

```bash
# 1. Launch EC2 instance
# AWS Console → EC2 → Launch Instance
# - AMI: Amazon Linux 2023
# - Instance type: t3.medium (2 vCPU, 4GB RAM — $0.0416/hour)
# - Storage: 20 GB gp3
# - Security group: Allow ports 22 (SSH), 80 (HTTP), 5000 (API)
# - Key pair: Create and download .pem file

# 2. SSH into instance
ssh -i your-key.pem ec2-user@YOUR_PUBLIC_IP

# 3. Install Docker on EC2
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user
# Log out and back in for group change to take effect

# 4. Transfer project files (from local machine)
scp -i your-key.pem -r ./deploy/ ec2-user@YOUR_PUBLIC_IP:~/irdas/

# 5. Build and run container on EC2
cd ~/irdas
docker build -t irdas-api .
docker run -d -p 5000:5000 --name irdas irdas-api

# 6. Test the API
curl http://localhost:5000/health
# Expected: {"status":"healthy","model":"MSDNet v1.0"}

# Test with an image
curl -X POST -F "image=@test_fundus.png" http://YOUR_PUBLIC_IP:5000/predict
```

### Step 6: Add NGINX Reverse Proxy (Production)

```bash
# On EC2 instance
sudo yum install -y nginx

# Configure NGINX
sudo tee /etc/nginx/conf.d/irdas.conf << 'EOF'
server {
    listen 80;
    server_name YOUR_PUBLIC_IP;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 10M;  # Allow large retinal images
    }
}
EOF

sudo systemctl start nginx
sudo systemctl enable nginx
```

### AWS Cost Estimation (Student Pack $100)

| Resource | Usage | Monthly Cost | Duration |
|----------|-------|-------------|----------|
| EC2 t3.medium | 24/7 for demo | ~$30/month | 3 months |
| S3 (model storage) | ~500 MB | $0.02/month | 3 months |
| Data transfer | ~5 GB | $0.45/month | 3 months |
| **Total** | | **~$31/month** | **~3 months on $100** |

> **TIP:** Stop the EC2 instance when not demoing to save credits.
> Only run it for: paper reviewer demo, viva presentation, portfolio showcase.

### Step 7: Model Storage on S3

```bash
# Upload model to S3 for persistence
aws s3 mb s3://irdas-models
aws s3 cp checkpoints/msdnet_inference.pth s3://irdas-models/v1/
aws s3 cp checkpoints/triage_xgb.pkl s3://irdas-models/v1/

# Download on EC2 startup
aws s3 cp s3://irdas-models/v1/msdnet_inference.pth checkpoints/
```

---

## 9. Daily Workflow Checklist

### Coding Day (VS Code)
```
□ Pull latest from GitHub
□ Review IRDAS_ENHANCED_PLAN.md — what's next?
□ Create feature branch
□ Write code in VS Code
□ Test locally with dummy data (CPU)
□ Commit with [PHASE-X] prefix
□ Push to GitHub
□ Update checklist in IRDAS_ENHANCED_PLAN.md
```

### Training Day (Kaggle)
```
□ Check GPU quota remaining (Kaggle Settings)
□ Open notebook → Install deps → Verify GPU
□ Clone/pull latest code from GitHub
□ Run training / evaluation
□ Monitor WandB dashboard
□ Save checkpoints to /kaggle/working/
□ Download results and checkpoints
□ Save session log (JSON)
□ Commit results to GitHub
□ Update tracking document
```

### Paper Writing Day (CPU)
```
□ Open Overleaf project
□ Write one section at a time
□ Reference ablation_results.csv for tables
□ Create figures using matplotlib (local)
□ Proofread with Grammarly
□ Commit LaTeX source to GitHub
```

---

## 10. Time-Saving Tips

### 1. Pre-compute Preprocessed Images
```python
# Do preprocessing ONCE, save to disk
# Don't recompute CLAHE + Ben Graham every epoch
import os, cv2, numpy as np
from tqdm import tqdm

os.makedirs("data/preprocessed/aptos", exist_ok=True)
for fname in tqdm(os.listdir("data/raw/aptos/train_images")):
    img = cv2.imread(f"data/raw/aptos/train_images/{fname}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_fundus(img)
    np.save(f"data/preprocessed/aptos/{fname.split('.')[0]}.npy", img)
# This saves ~30% training time per epoch
```

### 2. Use Mixed Precision Training
```python
# Cuts GPU memory by ~40%, speeds up training by ~30% on T4
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for batch in loader:
    with autocast():  # FP16 forward pass
        outputs = model(imgs.cuda())
        loss, _ = compute_total_loss(outputs, ...)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### 3. Gradient Accumulation (If Batch Size Must Be Larger)
```python
# Simulate batch_size=64 with actual batch_size=16
accumulation_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(loader):
    loss, _ = compute_total_loss(model(batch), ...)
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Quick Validation (Don't Validate Every Epoch)
```python
# Validate every 3 epochs instead of every epoch — saves ~30% time
if epoch % 3 == 0 or epoch == config['training']['epochs'] - 1:
    metrics = validate(model, val_loader)
    wandb.log(metrics)
```

### 5. Cache SHAP Computation
```python
# SHAP is slow — compute once, save
import pickle
if os.path.exists("outputs/shap_values.pkl"):
    shap_values = pickle.load(open("outputs/shap_values.pkl", "rb"))
else:
    shap_values = explainer.shap_values(X)
    pickle.dump(shap_values, open("outputs/shap_values.pkl", "wb"))
```

---

## Keyboard Shortcuts (VS Code Productivity)

| Action | Shortcut |
|--------|----------|
| Open terminal | Ctrl + ` |
| Search in files | Ctrl + Shift + F |
| Go to file | Ctrl + P |
| Go to definition | F12 |
| Rename symbol | F2 |
| Toggle sidebar | Ctrl + B |
| Split editor | Ctrl + \\ |
| Git: Stage all | In Source Control: click + |
| Git: Commit | Ctrl + Enter (in commit message) |

---

## File Reference Map

| When you need... | Open this file |
|-----------------|----------------|
| What to do next | `IRDAS_ENHANCED_PLAN.md` |
| How something works | `IRDAS_LEARNING_GUIDE.md` |
| Exact code to write | `IRDAS_AGENT_README.md` |
| Tool/deploy help | `IRDAS_PROJECT_COMPLETION_GUIDE.md` (this file) |

---

*Project Completion Guide v1.0 — IRDAS for Shivendra Pratap*
*Work smart, not hard. Use the right tool for each task.*
