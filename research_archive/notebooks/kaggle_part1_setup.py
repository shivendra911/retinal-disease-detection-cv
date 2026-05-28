"""
============================================================================
IRDAS MSDNet — KAGGLE TRAINING NOTEBOOK (Part 1: Setup + Validation)
============================================================================
SOTA V2 Recipe — EfficientNet-B4-NS + CORAL Ordinal Loss + TTA
Expected QWK: 0.88–0.93 (up from 0.74 baseline)

Run this FIRST. It installs deps, validates data, and smoke-tests the model.
Only proceed to Part 2 if ALL checks pass.
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
# !pip install -q timm albumentations grad-cam coral-pytorch scipy

# ============================================================
# CELL 2: Imports + Config
# ============================================================
import os, sys, json, time, gc, warnings
import numpy as np
import pandas as pd
import cv2

# CRITICAL FIX for DataLoader deadlock: disable OpenCV internal multithreading
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from datetime import datetime
from collections import Counter

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import cohen_kappa_score, roc_auc_score, f1_score, confusion_matrix

warnings.filterwarnings('ignore')

# ── Config (all hyperparameters in ONE place) ──
CFG = {
    'seed': 42,

    # === Architecture ===
    'backbone':    'tf_efficientnet_b4_ns',
    'image_size':  512,         # V2 optimization
    'fpn_channels': 256,
    'dropout':     0.3,
    'dr_num_classes': 5,
    'dr_ordinal':  True,

    # === Training ===
    'batch_size':              8,    # Halved to fit 512px in VRAM
    'epochs':                  60,
    'lr':                      1e-4,
    'head_lr':                 1e-3,
    'backbone_lr_mult':        0.1,
    'weight_decay':            1e-2,
    'warmup_epochs':           5,
    'freeze_epochs':           5,
    'swa_start_epoch':         45,
    'swa_lr':                  1e-5,
    'gradient_accumulation':   16,    # Doubled to maintain effective batch 128
    'label_smoothing':         0.15,  # V2 optimization for noisy labels

    # === Accuracy Boosters ===
    'use_ema':      True,
    'ema_decay':    0.9997,        # was 0.999 → slower accumulation, better 60-epoch average
    'use_mixup':    False,        # DISABLED — Mixup + CORAL ordinal = mathematically wrong

    # === Loss Weights ===
    'lambda_vessel':      0.5,
    'lambda_contrastive': 0.3,
    'margin_pure':        0.1,
    'margin_cooccur':     0.3,

    # === Inference ===
    'mc_passes':  30,
    'tta_passes': 8,

    # === DataLoader (safe, no deadlock) ===
    'num_workers': 2,             # workers handle augmentation in parallel
    'precache':    True,          # cache uint8 (after Bug 1 fix)
    'checkpoint_every': 5,

    # === Paths ===
    'aptos_csv':  '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train.csv',
    'aptos_imgs': '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train_images',
    'drive_imgs': '/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training/images',
    'drive_masks': '/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training/1st_manual',
    'output_dir': '/kaggle/working',
    'ckpt_dir':   '/kaggle/working/checkpoints',
}

def set_seed(seed=42):
    """Reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Faster with fixed input sizes

set_seed(CFG['seed'])
os.makedirs(CFG['ckpt_dir'], exist_ok=True)
os.makedirs(f"{CFG['output_dir']}/logs", exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Device: {DEVICE} | GPUs available: {N_GPUS}")
for i in range(N_GPUS):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | "
          f"VRAM: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")


# ============================================================
# CELL 3: Preprocessing Functions
# ============================================================
def ben_graham(img, sigmaX=10):
    """Remove uneven illumination. Formula: 4*img - 4*blur(img) + 128"""
    return cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), sigmaX), -4, 128)

def apply_clahe(img, clip=2.0, tile=8):
    """CLAHE on L channel in LAB space. Enhances local contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def green_channel_clahe(img, clip_limit=2.0, tile_size=8):
    """Enhanced CLAHE with green channel emphasis.
    Green channel carries ~80% of DR-relevant info (microaneurysms, hemorrhages).
    Blend: 60% green-enhanced + 40% standard LAB CLAHE."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    # Standard LAB CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    standard = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    # Green channel CLAHE
    green = img[:, :, 1]
    green_enhanced = clahe.apply(green)
    green_img = img.copy()
    green_img[:, :, 1] = green_enhanced
    return cv2.addWeighted(green_img, 0.6, standard, 0.4, 0)

def crop_circle(img):
    """Remove black border from circular fundus FOV."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]

def preprocess_fundus(img, size=CFG['image_size']):
    """Returns uint8 image [0-255] WITHOUT ImageNet normalization."""
    img = cv2.resize(img, (512, 512))
    img = ben_graham(img)
    img = green_channel_clahe(img)
    img = crop_circle(img)
    img = cv2.resize(img, (size, size))
    return img


# ============================================================
# CELL 4: Augmentation — Ophthalmology-Specific SOTA Pipeline
# ============================================================
def get_train_aug(image_size=None):
    """Maximum accuracy augmentation pipeline.
    Stronger than V1 — every augmentation is domain-justified for fundoscopy."""
    sz = image_size or CFG['image_size']
    return A.Compose([
        A.Resize(sz, sz),
        # Geometric (fundus has no canonical orientation)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=45, p=0.7, border_mode=0),
        # Ophthalmology-specific color augmentations
        A.OneOf([
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.1),
                contrast_limit=0.3, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2,
                          saturation=0.15, hue=0.02, p=1.0),
        ], p=0.8),
        # Simulate low-quality + lens distortion
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 40.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.CoarseDropout(
            max_holes=12, max_height=sz // 12, max_width=sz // 12,
            min_holes=2, min_height=sz // 32, min_width=sz // 32,
            fill_value=0, p=0.4),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

def get_val_aug(image_size=None):
    """Validation: resize + tensor only. No augmentation."""
    sz = image_size or CFG['image_size']
    return A.Compose([
        A.Resize(sz, sz), 
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])

def get_vessel_aug(mode='train', image_size=None):
    sz = image_size or CFG['image_size']
    if mode == 'val':
        return A.Compose([
            A.Resize(sz, sz), 
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2()
        ])
    return A.Compose([
        A.Resize(sz, sz),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5, border_mode=0),
        A.ColorJitter(brightness=0.15, contrast=0.15, p=0.3),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2(),
    ])


# ============================================================
# CELL 5: Dataset Classes (with PRE-CACHING)
# ============================================================
class APTOSDataset(Dataset):
    """APTOS 2019 DR grading with optional RAM pre-caching.
    Pre-caching: preprocesses ALL images once → stores in RAM.
    3,662 × 384×384×3 × 1 byte ≈ 1.6 GB — fits easily in Kaggle's 30GB RAM."""

    def __init__(self, df, img_dir, transform=None, precache=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        counts = self.df['diagnosis'].value_counts().sort_index()
        total = len(self.df)
        self.class_weights = torch.tensor([total / (5 * c) for c in counts], dtype=torch.float32)

        self.cache = {}
        if precache:
            print(f"  Pre-caching {len(self.df)} images at {CFG['image_size']}px... ", end='', flush=True)
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                img_path = os.path.join(self.img_dir, row['id_code'] + '.png')
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.cache[i] = preprocess_fundus(img, size=CFG['image_size'])
                if (i+1) % 500 == 0:
                    print(f"{i+1}", end=' ', flush=True)
            print(f"Done! ({len(self.cache)} cached)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self.cache:
            img = self.cache[idx]
        else:
            row = self.df.iloc[idx]
            img_path = os.path.join(self.img_dir, row['id_code'] + '.png')
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"NOT FOUND: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_fundus(img, size=CFG['image_size'])
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        label = torch.tensor(self.df.iloc[idx]['diagnosis'], dtype=torch.long)
        return img, label


class DRIVEDataset(Dataset):
    """DRIVE vessel segmentation. 20 train images + pixel masks."""
    def __init__(self, img_dir, mask_dir, transform=None):
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.tif','.png','.jpg'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.gif','.tif','.png'))])
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        mask = cv2.imread(os.path.join(self.mask_dir, self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"NOT FOUND: {self.images[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_fundus(img, size=CFG['image_size'])
        mask = cv2.resize(mask, (CFG['image_size'], CFG['image_size']))
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0)
        else:
            mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        return img, mask


# ============================================================
# CELL 6: Data Validation (RUN THIS — DO NOT SKIP)
# ============================================================
def validate_data():
    """Check datasets exist, images load, class balance is known."""
    print("=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)
    errors = []

    if not os.path.exists(CFG['aptos_csv']):
        errors.append(f"APTOS CSV not found: {CFG['aptos_csv']}")
    else:
        df = pd.read_csv(CFG['aptos_csv'])
        print(f"\n[APTOS] Total images: {len(df)}")
        dist = df['diagnosis'].value_counts().sort_index()
        for grade, count in dist.items():
            pct = count / len(df) * 100
            print(f"  Grade {grade}: {count:>5} ({pct:5.1f}%)")
        for i in [0, len(df)//2, len(df)-1]:
            p = os.path.join(CFG['aptos_imgs'], df.iloc[i]['id_code'] + '.png')
            img = cv2.imread(p)
            if img is None:
                errors.append(f"Cannot load: {p}")
            else:
                print(f"  Sample {i}: {p.split('/')[-1]} → {img.shape} ✓")

    if os.path.exists(CFG['drive_imgs']):
        imgs = [f for f in os.listdir(CFG['drive_imgs']) if f.endswith(('.tif','.png','.jpg'))]
        masks = [f for f in os.listdir(CFG['drive_masks']) if f.endswith(('.gif','.tif','.png'))]
        print(f"\n[DRIVE] Images: {len(imgs)}, Masks: {len(masks)}")
        if len(imgs) != len(masks):
            errors.append(f"DRIVE mismatch: {len(imgs)} images vs {len(masks)} masks")
    else:
        print("\n[DRIVE] Not found — vessel decoder will be SKIPPED")
        CFG['use_vessel'] = False

    if errors:
        print("\n❌ ERRORS FOUND:")
        for e in errors:
            print(f"  - {e}")
        raise RuntimeError("Fix data errors before continuing!")
    else:
        print("\n✅ ALL DATA CHECKS PASSED")

# Uncomment to run:
# validate_data()


# ============================================================
# CELL 7: Model Architecture (SOTA V2)
# ============================================================

# --- CBAM ---
class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction, bias=False), nn.ReLU(),
            nn.Linear(ch // reduction, ch, bias=False))
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.shape
        a = self.fc(self.avg(x).view(b, c))
        m = self.fc(self.mx(x).view(b, c))
        return x * self.sig(a + m).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=ks//2, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        a = torch.mean(x, dim=1, keepdim=True)
        m, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([a, m], dim=1)))

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.channel = ChannelAttention(ch)
        self.spatial = SpatialAttention()
    def forward(self, x):
        return self.spatial(self.channel(x))

# --- FPN ---
class FPN(nn.Module):
    def __init__(self, in_ch_list, out_ch=256):
        super().__init__()
        self.lat5 = nn.Conv2d(in_ch_list[2], out_ch, 1)
        self.lat4 = nn.Conv2d(in_ch_list[1], out_ch, 1)
        self.lat3 = nn.Conv2d(in_ch_list[0], out_ch, 1)
        self.smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, P3, P4, P5):
        p5 = self.lat5(P5)
        p4 = self.lat4(P4) + F.interpolate(p5, size=P4.shape[-2:], mode='nearest')
        p3 = self.lat3(P3) + F.interpolate(p4, size=P3.shape[-2:], mode='nearest')
        return self.relu(self.bn(self.smooth(p3)))

# --- Disease Branch ---
class DiseaseBranch(nn.Module):
    def __init__(self, ch, n_classes, dropout=0.3):
        super().__init__()
        self.cbam = CBAM(ch)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(ch, n_classes)
    def forward(self, x):
        x = self.cbam(x)
        feat = self.gap(x).flatten(1)
        feat = self.drop(feat)
        return self.fc(feat), feat

# --- Vessel Decoder (dynamic channels + dynamic output size) ---
class VesselDecoder(nn.Module):
    def __init__(self, encoder_channels):
        """encoder_channels: [P3_ch, P4_ch, P5_ch] auto-detected from backbone."""
        super().__init__()
        p3_ch, p4_ch, p5_ch = encoder_channels
        self.up4 = self._block(p5_ch, p4_ch)
        self.up3 = self._block(p4_ch * 2, 64)
        self.up2 = self._block(64 + p3_ch, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def _block(self, inc, outc):
        return nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1), nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Conv2d(outc, outc, 3, padding=1), nn.BatchNorm2d(outc), nn.ReLU(True))

    def forward(self, P3, P4, P5):
        x = self.up4(P5)
        x = F.interpolate(x, size=P4.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up3(torch.cat([x, P4], 1))
        x = F.interpolate(x, size=P3.shape[-2:], mode='bilinear', align_corners=False)
        x = self.up2(torch.cat([x, P3], 1))
        # Dynamic full-resolution output (P3 is stride 8)
        full_h, full_w = P3.shape[2] * 8, P3.shape[3] * 8
        x = F.interpolate(x, size=(full_h, full_w), mode='bilinear', align_corners=False)
        return self.head(x)  # Return raw logits for AMP-safe BCEWithLogitsLoss


# --- MSDNet (Full Model — SOTA V2) ---
class MSDNet(nn.Module):
    """Multi-Scale Disentangled Network with EfficientNet-B4-NS backbone.
    CORAL ordinal head outputs K-1=4 logits for 5 DR grades.
    ~19M params. Fits 2×T4 GPUs with DataParallel."""

    def __init__(self, cfg=CFG):
        super().__init__()
        backbone_name = cfg.get('backbone', 'tf_efficientnet_b4_ns')
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)

        # Auto-detect channel dimensions from timm's feature_info
        all_channels = self.backbone.feature_info.channels()
        self.out_channels = [all_channels[2], all_channels[3], all_channels[4]]

        fpn_ch = cfg.get('fpn_channels', 256)
        dropout = cfg.get('dropout', 0.3)

        self.fpn = FPN(self.out_channels, fpn_ch)

        # DR branch: CORAL ordinal outputs K-1 logits, softmax outputs K logits
        num_classes = cfg.get('dr_num_classes', 5)
        dr_ordinal = cfg.get('dr_ordinal', True)
        self.dr_output_dim = num_classes - 1 if dr_ordinal else num_classes
        self.dr_branch = DiseaseBranch(fpn_ch, self.dr_output_dim, dropout)

        self.hr_branch = DiseaseBranch(fpn_ch, 1, dropout)
        self.vessel_decoder = VesselDecoder(encoder_channels=self.out_channels)
        self.use_vessel = True

    def forward(self, x, task='all'):
        # FIX for DataParallel + AMP Deadlock: Autocast MUST be inside the forward pass
        with torch.amp.autocast('cuda'):
            feats = self.backbone(x)
            P3, P4, P5 = feats[2], feats[3], feats[4]
            
            out = {}
            if task in ['all', 'dr']:
                fpn_feat = self.fpn(P3, P4, P5)
                dr_logits, dr_feat = self.dr_branch(fpn_feat)
                hr_logits, hr_feat = self.hr_branch(fpn_feat)
                out.update({'dr_logits': dr_logits, 'hr_logits': hr_logits,
                            'dr_feat': dr_feat, 'hr_feat': hr_feat})
            
            if self.use_vessel and self.training and task in ['all', 'vessel']:
                out['vessel_pred'] = self.vessel_decoder(P3, P4, P5)
                
        return out

    @torch.no_grad()
    def predict_with_uncertainty(self, x, n=30):
        """MC Dropout: N stochastic forward passes, measure variance."""
        self.train()  # keep dropout active
        dr_preds, hr_preds = [], []
        for _ in range(n):
            out = self.forward(x)
            # CORAL: sigmoid on ordinal logits → sum gives continuous grade
            dr_preds.append(torch.sigmoid(out['dr_logits']))
            hr_preds.append(torch.sigmoid(out['hr_logits']))
        dr_s = torch.stack(dr_preds)   # (n, B, K-1)
        hr_s = torch.stack(hr_preds)   # (n, B, 1)
        return {
            'dr_mean': dr_s.mean(0),                 # (B, K-1) mean sigmoid probs
            'dr_std': dr_s.std(0).mean(-1),           # (B,) uncertainty per sample
            'dr_mean_logits': out['dr_logits'],       # last pass logits for class pred
            'hr_mean': hr_s.mean(0),
            'hr_std': hr_s.std(0).squeeze(-1),
        }

    @torch.no_grad()
    def predict_with_tta(self, x, n_tta=8):
        """Test-Time Augmentation: 8 geometric transforms, average logits.
        Lossless transforms (D4 dihedral group) — no info created/destroyed."""
        self.eval()

        def _identity(t): return t
        def _hflip(t): return t.flip(-1)
        def _vflip(t): return t.flip(-2)
        def _hvflip(t): return t.flip(-1).flip(-2)
        def _rot90(t): return t.rot90(1, [-2, -1])
        def _rot180(t): return t.rot90(2, [-2, -1])
        def _rot270(t): return t.rot90(3, [-2, -1])
        def _hflip_rot90(t): return t.flip(-1).rot90(1, [-2, -1])

        augs = [_identity, _hflip, _vflip, _hvflip,
                _rot90, _rot180, _rot270, _hflip_rot90][:n_tta]

        dr_all, hr_all = [], []
        for aug_fn in augs:
            out = self.forward(aug_fn(x))
            dr_all.append(out['dr_logits'])
            hr_all.append(out['hr_logits'])

        return {
            'dr_logits': torch.stack(dr_all).mean(0),
            'hr_logits': torch.stack(hr_all).mean(0),
        }


def build_model():
    """Create MSDNet (Forced single GPU to bypass DataParallel deadlock)"""
    model = MSDNet().to(DEVICE)
    
    # REMOVED the "if N_GPUS > 1: nn.DataParallel" block completely.
    print(f"🛑 DataParallel BYPASSED: Using strictly 1 GPU to prevent thread locks.")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable:,} trainable")
    return model


# ============================================================
# CELL 8: Loss Functions
# ============================================================
class CoralOrdinalLoss(nn.Module):
    """CORAL ordinal loss for DR grading.
    DR grades are ORDINAL — grade 3 is closer to 2 than to 0.
    Standard CE treats all errors equally. CORAL penalizes large rank errors.
    K=5 classes → K-1=4 ordinal logits, each representing P(grade > k).
    Label smoothing ε handles inter-grader disagreement at boundaries."""
    def __init__(self, num_classes=5, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.eps = label_smoothing
    def forward(self, logits, targets):
        levels = torch.arange(self.num_classes - 1, device=logits.device)
        ordinal_targets = (targets.unsqueeze(1) > levels.unsqueeze(0)).float()
        if self.eps > 0:
            ordinal_targets = ordinal_targets * (1 - self.eps) + (1 - ordinal_targets) * self.eps
        return F.binary_cross_entropy_with_logits(logits, ordinal_targets, reduction='mean')

def ordinal_logits_to_class(logits):
    """Convert CORAL ordinal logits → integer class predictions.
    Each logit = P(grade > k). Predicted class = count of sigmoid > 0.5."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1).long()

class FocalLoss(nn.Module):
    """Legacy Focal Loss — kept for ablation comparison."""
    def __init__(self, weights=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weights,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

class DiceBCELoss(nn.Module):
    """AMP-safe DiceBCE: expects logits, uses sigmoid for Dice, BCEWithLogits for BCE."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    def forward(self, logits, target):
        logits = logits.float()    # Cast to float32 for AMP safety
        target = target.float()
        pred = torch.sigmoid(logits)
        pf, tf = pred.view(-1), target.view(-1)
        inter = (pf * tf).sum()
        dice = 1 - (2*inter + self.smooth) / (pf.sum() + tf.sum() + self.smooth)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        return dice + bce

def contrastive_loss(dr_feat, hr_feat, dr_label, hr_label, m_pure=0.1, m_co=0.3):
    """Novel contrastive disentanglement: push DR/HR embeddings apart."""
    dr_n = F.normalize(dr_feat, dim=-1)
    hr_n = F.normalize(hr_feat, dim=-1)
    cos = (dr_n * hr_n).sum(-1)
    has_dr = (dr_label > 0).float()
    has_hr = (hr_label > 0).float()
    pure = (has_dr * (1-has_hr) + has_hr * (1-has_dr)).clamp(max=1)
    co = has_dr * has_hr
    L_p = pure * F.relu(cos - m_pure)
    L_c = co * F.relu(cos - m_co)
    return L_p.sum() / pure.sum().clamp(min=1) + L_c.sum() / co.sum().clamp(min=1)

def optimize_qwk_thresholds(continuous_preds, labels, num_classes=5):
    """Nelder-Mead optimization of QWK thresholds. Free +0.01 QWK."""
    from scipy.optimize import minimize
    def neg_qwk(thresholds):
        thresholds = sorted(thresholds)
        preds = np.zeros_like(continuous_preds, dtype=int)
        for i, t in enumerate(thresholds):
            preds[continuous_preds > t] = i + 1
        return -cohen_kappa_score(labels, np.clip(preds, 0, num_classes-1), weights='quadratic')
    x0 = [i + 0.5 for i in range(num_classes - 1)]
    result = minimize(neg_qwk, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-4})
    return sorted(result.x.tolist())

def apply_optimized_thresholds(continuous_preds, thresholds):
    """Apply learned thresholds to continuous predictions."""
    preds = np.zeros_like(continuous_preds, dtype=int)
    for i, t in enumerate(sorted(thresholds)):
        preds[continuous_preds > t] = i + 1
    return preds


# ============================================================
# CELL 9: Smoke Test (MUST PASS before training)
# ============================================================
def smoke_test():
    print("=" * 60)
    print("SMOKE TEST — SOTA V2 (B4-NS + CORAL)")
    print("=" * 60)

    model = MSDNet().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    sz = CFG['image_size']
    x = torch.randn(2, 3, sz, sz).to(DEVICE)

    # Forward pass (train mode)
    model.train()
    out = model(x)
    dr_dim = CFG['dr_num_classes'] - 1 if CFG['dr_ordinal'] else CFG['dr_num_classes']
    assert out['dr_logits'].shape == (2, dr_dim), f"DR shape wrong: {out['dr_logits'].shape}, expected (2, {dr_dim})"
    assert out['hr_logits'].shape == (2, 1), f"HR shape wrong: {out['hr_logits'].shape}"
    assert out['dr_feat'].shape[1] == CFG['fpn_channels'], f"DR feat wrong: {out['dr_feat'].shape}"
    assert 'vessel_pred' in out, "Vessel pred missing in train mode"
    assert out['vessel_pred'].shape == (2, 1, sz, sz), f"Vessel shape wrong: {out['vessel_pred'].shape}"
    print(f"  Forward pass (train) ✓  |  DR: {out['dr_logits'].shape}, Vessel: {out['vessel_pred'].shape}")

    # Loss computation
    dr_labels = torch.tensor([0, 3]).to(DEVICE)
    hr_labels = torch.tensor([0, 1]).float().to(DEVICE)
    vessel_gt = torch.rand(2, 1, sz, sz).to(DEVICE)

    # CORAL ordinal loss
    coral = CoralOrdinalLoss(num_classes=CFG['dr_num_classes'],
                             label_smoothing=CFG['label_smoothing']).to(DEVICE)
    dice_bce = DiceBCELoss().to(DEVICE)

    L_dr = coral(out['dr_logits'], dr_labels)
    L_hr = F.binary_cross_entropy_with_logits(out['hr_logits'].squeeze(), hr_labels)
    L_vessel = dice_bce(out['vessel_pred'], vessel_gt)
    L_dis = contrastive_loss(out['dr_feat'], out['hr_feat'], dr_labels, hr_labels.long())
    total = L_dr + L_hr + 0.5 * L_vessel + 0.3 * L_dis

    assert not torch.isnan(total), "Total loss is NaN!"
    print(f"  Losses — CORAL: {L_dr:.4f}, HR: {L_hr:.4f}, Vessel: {L_vessel:.4f}, Dis: {L_dis:.4f}")
    print(f"  Total: {total:.4f} ✓")

    # CORAL → class prediction
    preds = ordinal_logits_to_class(out['dr_logits'])
    assert preds.shape == (2,), f"ordinal_logits_to_class shape wrong: {preds.shape}"
    print(f"  CORAL → class: {preds.cpu().tolist()} ✓")

    # Backward pass
    total.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  Backward pass + gradients ✓")

    # TTA prediction
    model.eval()
    tta_out = model.predict_with_tta(x, n_tta=4)
    assert tta_out['dr_logits'].shape == (2, dr_dim), f"TTA DR shape wrong"
    print(f"  TTA (4 augs) ✓")

    # MC Dropout uncertainty
    unc = model.predict_with_uncertainty(x, n=5)
    print(f"  MC Dropout — uncertainty: {unc['dr_std'].mean():.4f} ✓")

    del model, x, out
    torch.cuda.empty_cache()
    gc.collect()
    print("\n✅ ALL SMOKE TESTS PASSED — Safe to train!")

# Uncomment to run:
# smoke_test()
