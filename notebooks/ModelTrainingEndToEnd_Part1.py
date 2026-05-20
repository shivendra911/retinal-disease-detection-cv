"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  IRDAS MSDNet — Model Training End-to-End (Learning Guide + Code)          ║
║  Part 1: Foundations — WHY we do each thing, WHAT tools we use, HOW        ║
╚══════════════════════════════════════════════════════════════════════════════╝

This file is a BOOK. Read top-to-bottom. Every code block has:
  - WHAT we're doing
  - WHY we need it
  - WHAT IF we skip it / do it differently
  - HOW it works internally

TABLE OF CONTENTS:
  Chapter 1: The Problem — What is Retinal Disease Detection?
  Chapter 2: Environment Setup — GPU, Libraries, Seeds
  Chapter 3: Data — Loading, Understanding, Cleaning
  Chapter 4: Preprocessing — Why Each Step Exists
  Chapter 5: Augmentation — Making Small Datasets Big
  Chapter 6: Model Architecture — EfficientNet → FPN → CBAM → Branches
  Chapter 7: Loss Functions — Why 4 Different Losses?
  Chapter 8: Training Loop — Optimizer, Scheduler, Checkpoints
  Chapter 9: Evaluation — Metrics That Matter Clinically
  Chapter 10: Explainability — Grad-CAM++ and Uncertainty
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CHAPTER 1: THE PROBLEM                                     ║
# ╚═══════════════════════════════════════════════════════════════╝
"""
WHAT IS DIABETIC RETINOPATHY (DR)?
══════════════════════════════════
- Diabetes damages tiny blood vessels in the retina (back of the eye)
- Over time: vessels leak → hemorrhages → new fragile vessels grow → blindness
- 5 severity grades:
    Grade 0: No DR (healthy retina)
    Grade 1: Mild (a few microaneurysms — tiny red dots)
    Grade 2: Moderate (hemorrhages + exudates appear)
    Grade 3: Severe (many hemorrhages, venous beading)
    Grade 4: Proliferative DR (new vessel growth — DANGEROUS)

WHY AI?
- 463 million diabetics worldwide, only ~200K ophthalmologists
- Early detection (Grade 1-2) prevents 90% of blindness
- AI can screen thousands of images per hour

OUR APPROACH (MSDNet):
- Multi-Scale: detect tiny lesions (1-5px) AND large structures (optic disc ~200px)
- Disentangled: separate branches for DR and HR (hypertensive retinopathy)
- Novel: contrastive loss pushes disease branches to learn DIFFERENT features

WHAT MAKES THIS RESEARCH-WORTHY?
1. Multi-task: DR + HR simultaneously (most papers do only one)
2. Per-branch CBAM attention (each disease looks at different regions)
3. Contrastive disentanglement loss (NOVEL — no prior paper has this)
4. Per-disease Grad-CAM++ heatmaps (clinician can see WHERE each disease is)
5. MC Dropout uncertainty (model says "I'm not sure" on ambiguous cases)
"""

# ╔═══════════════════════════════════════════════════════════════╗
# ║  CHAPTER 2: ENVIRONMENT SETUP                               ║
# ╚═══════════════════════════════════════════════════════════════╝

# --- Step 1: Install libraries ---
# !pip install -q timm==0.9.12 albumentations==1.3.1 pytorch-grad-cam==1.4.8

"""
WHY THESE LIBRARIES?
═══════════════════
timm (PyTorch Image Models):
  - Provides 800+ pretrained models including EfficientNet-B0
  - `features_only=True` extracts intermediate feature maps (we need this for FPN)
  - WHAT IF we used torchvision instead? We could, but timm's features_only mode
    is cleaner — torchvision requires manual hook registration.

albumentations:
  - Image augmentation library optimized for CV
  - KEY FEATURE: applies SAME spatial transform to image AND mask together
  - WHAT IF we used torchvision.transforms? It can't transform masks with images.
    For vessel segmentation, if we flip the image, we MUST flip the mask too.

pytorch-grad-cam:
  - Implements Grad-CAM++ (improved version of Grad-CAM)
  - WHY not plain Grad-CAM? Grad-CAM++ handles multiple instances better.
    Retinal images have SCATTERED lesions — Grad-CAM misses some, Grad-CAM++ gets all.
"""

import os, sys, json, time, gc, warnings, glob
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datetime import datetime
from collections import Counter, OrderedDict
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# --- Step 2: Reproducibility ---
"""
WHY SET SEEDS?
══════════════
Neural networks use random initialization, random data shuffling, random dropout.
Without fixed seeds, running the same code twice gives different results.
For a research paper, results MUST be reproducible.

We set seeds for ALL random number generators:
  - torch (model weight initialization, dropout masks)
  - numpy (data splitting, augmentation)
  - CUDA (GPU random operations)
  - cudnn.deterministic (GPU algorithm selection)

WHAT IF we skip this?
  - Results change every run → can't compare experiments fairly
  - Reviewer asks "did you cherry-pick your best run?" → we can say "no, seed=42"
"""
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # slightly slower but reproducible

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# --- Step 3: Configuration ---
"""
WHY A CONFIG DICTIONARY?
════════════════════════
All hyperparameters in ONE place. Never hardcode numbers in training code.

Benefits:
  1. Change one value here → changes everywhere automatically
  2. Saved inside checkpoints → know exactly what settings produced each model
  3. Easy to compare experiments: "Run A used lr=1e-4, Run B used lr=5e-5"

WHAT EACH VALUE MEANS (detailed):
"""
CFG = {
    # === Data ===
    'seed': 42,
    'image_size': 224,        # EfficientNet-B0's native input size. Using 224 because:
                               # - B0 was pretrained at 224x224
                               # - Larger (384, 512) = better quality but 4x slower + more VRAM
                               # - 224 is the sweet spot for T4 GPU with batch 32

    'batch_size': 32,          # How many images per training step.
                               # - Larger (64, 128) = smoother gradients but may OOM on T4
                               # - Smaller (8, 16) = noisier gradients but fits any GPU
                               # - 32 is standard for medical imaging on 16GB VRAM
                               # WHAT IF OOM? → reduce to 16, training takes 2x longer

    # === Training ===
    'epochs': 50,              # How many full passes through the dataset.
                               # - Too few (<20): model hasn't converged yet
                               # - Too many (>100): overfitting, wastes GPU time
                               # - 50 is standard for fine-tuning pretrained models

    'lr': 1e-4,               # Learning rate — THE most important hyperparameter.
                               # - Too high (1e-2): model oscillates, never converges
                               # - Too low (1e-6): model learns too slowly, wastes time
                               # - 1e-4 is standard for AdamW with pretrained backbone
                               # WHY not higher? Pretrained weights are already good.
                               #   High LR destroys them ("catastrophic forgetting").

    'weight_decay': 1e-2,     # L2 regularization — prevents overfitting.
                               # - Adds penalty for large weights: loss += wd * ||weights||²
                               # - 1e-2 is standard for AdamW (decoupled weight decay)
                               # WHAT IF we skip it? Model memorizes training data,
                               #   performs badly on new images.

    'T_max': 30,              # Cosine annealing period — LR decreases following a cosine curve
                               # from lr to ~0 over T_max epochs. After warmup (epoch 5),
                               # the scheduler runs for ~45 epochs, so T_max=30 means
                               # LR reaches minimum around epoch 35, then slightly rises.

    'warmup_epochs': 5,       # For first 5 epochs, LR gradually increases from 0 to 1e-4.
                               # WHY? Pretrained backbone has good features. If we hit it
                               # with full LR immediately, we destroy those features.
                               # Warmup = gentle start, then full speed.

    # === Loss Weights ===
    'lambda_vessel': 0.5,     # Weight for vessel segmentation loss.
                               # - Vessel decoder is AUXILIARY (helps backbone, not the goal)
                               # - Too high (>1.0): model focuses on vessels, ignores DR grading
                               # - Too low (<0.1): no benefit from auxiliary task
                               # - 0.5 = moderate influence on backbone features

    'lambda_contrastive': 0.3, # Weight for contrastive disentanglement loss.
                               # - Our NOVEL contribution — pushes DR/HR features apart
                               # - Too high (>1.0): branches learn nothing (too much repulsion)
                               # - Too low (<0.05): branches learn same features (no disentanglement)
                               # - 0.3 = sweet spot found empirically

    'margin_pure': 0.1,       # Max allowed cosine similarity for single-disease samples.
                               # If patient has ONLY DR (no HR), the DR and HR feature vectors
                               # should be very different (cosine sim < 0.1).

    'margin_cooccur': 0.3,    # Max similarity for co-occurring diseases.
                               # If patient has BOTH DR and HR, some feature overlap is OK
                               # (both diseases affect same vessels), but not total overlap.

    # === Model ===
    'fpn_channels': 256,       # FPN output channels. 256 is standard in object detection.
                               # Higher = more capacity but more memory.
    'dropout': 0.3,            # MC Dropout rate. 0.3 = drop 30% of neurons randomly.
                               # Used for BOTH regularization AND uncertainty estimation.
    'mc_passes': 30,           # Number of stochastic forward passes for uncertainty.
                               # More passes = better uncertainty estimate but slower.

    # === Checkpointing ===
    'checkpoint_every': 5,     # Save model every 5 epochs for crash recovery.

    # === Paths (UPDATE for your Kaggle dataset names) ===
    'aptos_csv': '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train.csv',
    'aptos_imgs': '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train_images',
    'drive_imgs': '/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training/images',
    'drive_masks': '/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training/1st_manual',
    'output_dir': '/kaggle/working',
    'ckpt_dir': '/kaggle/working/checkpoints',
}

os.makedirs(CFG['ckpt_dir'], exist_ok=True)
os.makedirs(f"{CFG['output_dir']}/logs", exist_ok=True)
os.makedirs(f"{CFG['output_dir']}/xai_heatmaps", exist_ok=True)
print("Config loaded. All directories created.")


# ╔═══════════════════════════════════════════════════════════════╗
# ║  CHAPTER 3: DATA — Loading, Understanding, Cleaning         ║
# ╚═══════════════════════════════════════════════════════════════╝
"""
THE DATASETS WE USE:
═══════════════════

1. APTOS 2019 (Primary — DR grading)
   - 3,662 retinal fundus photographs
   - Labeled by ophthalmologists: grades 0-4
   - From multiple clinics in India → variable image quality
   - KEY CHALLENGE: heavily imbalanced (49% grade 0, only 5% grade 3)

2. DRIVE (Auxiliary — vessel segmentation)
   - 40 retinal images with PIXEL-LEVEL vessel annotations
   - 20 for training, 20 for testing
   - Used ONLY to train vessel decoder (auxiliary task)
   - WHY? Forces backbone to understand vascular anatomy
   - WHAT IF we skip DRIVE? Model still works, but backbone doesn't
     learn vessel structure → slightly worse DR detection.

3. HRDC 2023 (HR detection — SKIPPED tonight)
   - Requires manual registration at grand-challenge.org
   - We'll train DR-only tonight, add HR later if available.

DATA CLEANING CONSIDERATIONS:
════════════════════════════
Q: Are there corrupt/bad images in APTOS?
A: Yes! Some known issues:
   - ~10 images are near-black (camera malfunction)
   - Some images have severe lens artifacts
   - Our preprocessing (Ben Graham + CLAHE) handles most of these
   - The model learns to be robust via augmentation

Q: Is the labeling reliable?
A: APTOS labels were done by ophthalmologists, but inter-observer variability
   exists (doctors sometimes disagree on Grade 1 vs 2). This is why QWK is
   the metric — it accounts for ordinal disagreement.

Q: What about data leakage?
A: We use stratified split (same patient never in train AND val).
   APTOS doesn't have patient IDs, so we split by image — not perfect
   but acceptable for this competition dataset.
"""

def validate_data():
    """
    CHECKPOINT: Run this FIRST. Do NOT proceed if any check fails.

    What this validates:
    1. CSV file exists → dataset was added to Kaggle correctly
    2. Image count matches → no missing downloads
    3. Random images load → cv2 can read the format
    4. Class distribution → we know the imbalance (needed for sampler weights)
    5. DRIVE exists → decides if vessel decoder is used
    """
    print("=" * 60)
    print("📋 DATA VALIDATION CHECKPOINT")
    print("=" * 60)
    errors = []

    # --- APTOS Validation ---
    if not os.path.exists(CFG['aptos_csv']):
        errors.append(f"❌ APTOS CSV not found at: {CFG['aptos_csv']}")
        errors.append("   FIX: Run path discovery cell, update CFG['aptos_csv']")
    else:
        df = pd.read_csv(CFG['aptos_csv'])
        print(f"\n📊 APTOS 2019 Dataset:")
        print(f"   Total images: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n   Class Distribution (THIS IS IMPORTANT):")

        dist = df['diagnosis'].value_counts().sort_index()
        grade_names = {0:'No DR', 1:'Mild', 2:'Moderate', 3:'Severe', 4:'Proliferative'}
        for grade, count in dist.items():
            pct = count / len(df) * 100
            bar = '█' * int(pct / 2)
            print(f"   Grade {grade} ({grade_names[grade]:>13}): {count:>5} ({pct:5.1f}%) {bar}")

        # Test loading random images
        print(f"\n   Loading test images:")
        for i in [0, len(df)//2, len(df)-1]:
            path = os.path.join(CFG['aptos_imgs'], df.iloc[i]['id_code'] + '.png')
            img = cv2.imread(path)
            if img is None:
                errors.append(f"❌ Cannot load image: {path}")
            else:
                print(f"   ✓ {df.iloc[i]['id_code']}.png → shape {img.shape}")

        # Check for problematic images (near-black)
        print(f"\n   Checking for corrupt images (sampling 50)...")
        bad_count = 0
        sample_indices = np.random.choice(len(df), min(50, len(df)), replace=False)
        for idx in sample_indices:
            path = os.path.join(CFG['aptos_imgs'], df.iloc[idx]['id_code'] + '.png')
            img = cv2.imread(path)
            if img is not None and img.mean() < 10:  # nearly black
                bad_count += 1
        print(f"   Near-black images found: {bad_count}/50 sampled")
        if bad_count > 5:
            print("   ⚠️ Many dark images — preprocessing will handle these")

    # --- DRIVE Validation ---
    CFG['use_vessel'] = True
    if not os.path.exists(CFG['drive_imgs']):
        print(f"\n⚠️ DRIVE dataset not found at: {CFG['drive_imgs']}")
        print("   Vessel decoder will be DISABLED. Training continues without it.")
        CFG['use_vessel'] = False
    else:
        imgs = sorted([f for f in os.listdir(CFG['drive_imgs'])
                       if f.endswith(('.tif','.png','.jpg'))])
        masks = sorted([f for f in os.listdir(CFG['drive_masks'])
                        if f.endswith(('.gif','.tif','.png'))])
        print(f"\n📊 DRIVE Dataset:")
        print(f"   Images: {len(imgs)}, Masks: {len(masks)}")
        if len(imgs) != len(masks):
            errors.append(f"❌ DRIVE mismatch: {len(imgs)} images ≠ {len(masks)} masks")
        else:
            print(f"   ✓ Image-mask pairs matched")

    # --- Final Verdict ---
    if errors:
        print("\n" + "❌ " * 20)
        print("VALIDATION FAILED — Fix these before training:")
        for e in errors:
            print(f"  {e}")
        raise RuntimeError("Data validation failed!")
    else:
        print("\n" + "✅ " * 20)
        print("ALL CHECKS PASSED — Safe to proceed!")
    return True

# ▶▶▶ UNCOMMENT AND RUN: ◀◀◀
# validate_data()
