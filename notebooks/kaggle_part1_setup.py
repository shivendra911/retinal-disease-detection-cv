"""
============================================================================
IRDAS MSDNet — COMPLETE KAGGLE TRAINING NOTEBOOK (Part 1: Setup + Validation)
============================================================================
Run this FIRST. It installs deps, validates data, and smoke-tests the model.
Only proceed to Part 2 if ALL checks pass.
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
# !pip install -q timm albumentations grad-cam

# ============================================================
# CELL 2: Imports + Config
# ============================================================
import os, sys, json, time, gc, warnings
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
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
    'image_size': 384,            # ← HUGE BOOST: 224x224 destroys microaneurysms. 384x384 makes them visible.
    'batch_size': 32,             # ← Adjusted down for larger images to fit in 2x T4 VRAM
    'epochs': 50,
    'lr': 1e-4,                   # ← Scaled for batch size 32
    'weight_decay': 1e-2,
    'T_max': 30,
    'warmup_epochs': 5,
    'fpn_channels': 256,
    'dropout': 0.3,
    'lambda_vessel': 0.5,
    'lambda_contrastive': 0.3,
    'margin_pure': 0.1,
    'margin_cooccur': 0.3,
    'mc_passes': 30,
    'checkpoint_every': 5,
    'label_smoothing': 0.1,       
    'mixup_alpha': 0.0,           # ← DISABLED: Mixup destroys ordinality (a blend of Grade 0 and 4 is NOT Grade 2)
    'num_workers': 4,             
    'precache': True,             
    'backbone': 'efficientnet_b3',# ← UPGRADED: Much stronger feature extractor for medical images
    # Paths (matched to YOUR Kaggle dataset structure)
    'aptos_csv': '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train.csv',
    'aptos_imgs': '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train_images',
    'drive_imgs': '/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training/images',
    'drive_masks': '/kaggle/input/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction/DRIVE/training/1st_manual',
    'output_dir': '/kaggle/working',
    'ckpt_dir': '/kaggle/working/checkpoints',
}

def set_seed(seed=42):
    """Reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

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

def preprocess_fundus(img, size=224):
    """Full preprocessing pipeline: resize → Ben Graham → CLAHE → crop → normalize."""
    img = cv2.resize(img, (512, 512))
    img = ben_graham(img)
    img = apply_clahe(img)
    img = crop_circle(img)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return ((img - mean) / std).astype(np.float32)


# ============================================================
# CELL 4: Augmentation
# ============================================================
def get_train_aug():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.6, border_mode=0),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0, p=0.5),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        ToTensorV2(),
    ])

def get_val_aug():
    return A.Compose([ToTensorV2()])

def get_vessel_aug(mode='train'):
    if mode == 'val':
        return A.Compose([ToTensorV2()])
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5, border_mode=0),
        A.ColorJitter(brightness=0.15, contrast=0.15, p=0.3),
        ToTensorV2(),
    ])


# ============================================================
# CELL 5: Dataset Classes (with PRE-CACHING to eliminate CPU bottleneck)
# ============================================================
class APTOSDataset(Dataset):
    """APTOS 2019 DR grading with optional RAM pre-caching.
    Pre-caching: preprocesses ALL images once → stores in RAM.
    Result: DataLoader only does augmentation (fast) instead of
    imread+BenGraham+CLAHE+crop per batch (slow).
    3,662 × 224×224×3 × 4 bytes = ~2.1 GB — fits in Kaggle's 30GB RAM."""

    def __init__(self, df, img_dir, transform=None, precache=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        counts = self.df['diagnosis'].value_counts().sort_index()
        total = len(self.df)
        self.class_weights = torch.tensor([total / (5 * c) for c in counts], dtype=torch.float32)

        # Pre-cache all preprocessed images in RAM
        self.cache = {}
        if precache:
            print(f"  Pre-caching {len(self.df)} images... ", end='', flush=True)
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                img_path = os.path.join(self.img_dir, row['id_code'] + '.png')
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.cache[i] = preprocess_fundus(img)
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
            img = preprocess_fundus(img)
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
        img = preprocess_fundus(img)
        mask = cv2.resize(mask, (224, 224))
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

    # --- APTOS ---
    if not os.path.exists(CFG['aptos_csv']):
        errors.append(f"APTOS CSV not found: {CFG['aptos_csv']}")
    else:
        df = pd.read_csv(CFG['aptos_csv'])
        print(f"\n[APTOS] Total images: {len(df)}")
        dist = df['diagnosis'].value_counts().sort_index()
        for grade, count in dist.items():
            pct = count / len(df) * 100
            print(f"  Grade {grade}: {count:>5} ({pct:5.1f}%)")

        # Check 3 random images actually load
        for i in [0, len(df)//2, len(df)-1]:
            p = os.path.join(CFG['aptos_imgs'], df.iloc[i]['id_code'] + '.png')
            img = cv2.imread(p)
            if img is None:
                errors.append(f"Cannot load: {p}")
            else:
                print(f"  Sample {i}: {p.split('/')[-1]} → {img.shape} ✓")

    # --- DRIVE ---
    if os.path.exists(CFG['drive_imgs']):
        imgs = [f for f in os.listdir(CFG['drive_imgs']) if f.endswith(('.tif','.png','.jpg'))]
        masks = [f for f in os.listdir(CFG['drive_masks']) if f.endswith(('.gif','.tif','.png'))]
        print(f"\n[DRIVE] Images: {len(imgs)}, Masks: {len(masks)}")
        if len(imgs) != len(masks):
            errors.append(f"DRIVE mismatch: {len(imgs)} images vs {len(masks)} masks")
    else:
        print("\n[DRIVE] Not found — vessel decoder will be SKIPPED")
        CFG['use_vessel'] = False

    # --- Summary ---
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
# CELL 7: Model Architecture (everything in one place)
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

# --- Vessel Decoder ---
class VesselDecoder(nn.Module):
    def __init__(self, in_channels=[40, 112, 320]):
        super().__init__()
        self.up4 = self._block(in_channels[2], in_channels[1])
        self.up3 = self._block(in_channels[1] * 2, 64)
        self.up2 = self._block(64 + in_channels[0], 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.img_size = CFG['image_size']

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
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return torch.sigmoid(self.head(x))

# --- Mixup Augmentation (better class imbalance handling) ---
def mixup_data(x, y, alpha=0.2):
    """Mixup: blend two images and their labels for regularization.
    Creates soft labels → model learns smoother decision boundaries.
    Especially helps rare classes (Grade 3,4) by mixing with common ones."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


# --- MSDNet (Full Model) ---
class MSDNet(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        backbone_name = cfg.get('backbone', 'efficientnet_b0')
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        
        # Dynamic channels based on backbone
        if 'b3' in backbone_name:
            self.out_channels = [48, 136, 384]
        else:
            self.out_channels = [40, 112, 320]
            
        self.fpn = FPN(self.out_channels, cfg['fpn_channels'])
        self.dr_branch = DiseaseBranch(cfg['fpn_channels'], 5, cfg['dropout'])
        self.hr_branch = DiseaseBranch(cfg['fpn_channels'], 1, cfg['dropout'])
        self.vessel_decoder = VesselDecoder(in_channels=self.out_channels)
        self.use_vessel = True

    def forward(self, x):
        feats = self.backbone(x)
        P3, P4, P5 = feats[2], feats[3], feats[4]
        fpn_feat = self.fpn(P3, P4, P5)
        dr_logits, dr_feat = self.dr_branch(fpn_feat)
        hr_logits, hr_feat = self.hr_branch(fpn_feat)
        out = {'dr_logits': dr_logits, 'hr_logits': hr_logits,
               'dr_feat': dr_feat, 'hr_feat': hr_feat}
        if self.use_vessel and self.training:
            out['vessel_pred'] = self.vessel_decoder(P3, P4, P5)
        return out

    @torch.no_grad()
    def predict_with_uncertainty(self, x, n=30):
        self.train()
        self.use_vessel = False
        dr_preds, hr_preds = [], []
        for _ in range(n):
            out = self.forward(x)
            dr_preds.append(torch.softmax(out['dr_logits'], -1))
            hr_preds.append(torch.sigmoid(out['hr_logits']))
        self.use_vessel = True
        dr_s = torch.stack(dr_preds)
        hr_s = torch.stack(hr_preds)
        return {'dr_mean': dr_s.mean(0), 'dr_std': dr_s.std(0).mean(-1),
                'hr_mean': hr_s.mean(0), 'hr_std': hr_s.std(0).squeeze(-1)}


def build_model():
    """Create MSDNet and wrap with DataParallel if 2+ GPUs available."""
    model = MSDNet().to(DEVICE)
    if N_GPUS > 1:
        print(f"🚀 Using DataParallel across {N_GPUS} GPUs!")
        model = nn.DataParallel(model)
    else:
        print(f"Using single GPU")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    return model


# ============================================================
# CELL 8: Loss Functions
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, weights=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.label_smoothing = label_smoothing  # ← prevents overconfident predictions
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weights,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()

    def forward_mixup(self, logits, targets_a, targets_b, lam):
        """Focal loss for mixup: weighted sum of two target losses."""
        return lam * self.forward(logits, targets_a) + (1 - lam) * self.forward(logits, targets_b)

class DiceBCELoss(nn.Module):
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


# ============================================================
# CELL 9: Smoke Test (MUST PASS before training)
# ============================================================
def smoke_test():
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    model = MSDNet().to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    x = torch.randn(2, 3, 224, 224).to(DEVICE)

    # Forward pass (train mode)
    model.train()
    out = model(x)
    assert out['dr_logits'].shape == (2, 5), f"DR shape wrong: {out['dr_logits'].shape}"
    assert out['hr_logits'].shape == (2, 1), f"HR shape wrong: {out['hr_logits'].shape}"
    assert out['dr_feat'].shape == (2, 256), f"DR feat wrong: {out['dr_feat'].shape}"
    assert 'vessel_pred' in out, "Vessel pred missing in train mode"
    assert out['vessel_pred'].shape == (2, 1, 224, 224), f"Vessel shape wrong"
    print("  Forward pass (train) ✓")

    # Loss computation
    dr_labels = torch.tensor([0, 3]).to(DEVICE)
    hr_labels = torch.tensor([0, 1]).float().to(DEVICE)
    vessel_gt = torch.rand(2, 1, 224, 224).to(DEVICE)

    focal = FocalLoss().to(DEVICE)
    dice_bce = DiceBCELoss().to(DEVICE)

    L_dr = focal(out['dr_logits'], dr_labels)
    L_hr = F.binary_cross_entropy_with_logits(out['hr_logits'].squeeze(), hr_labels)
    L_vessel = dice_bce(out['vessel_pred'], vessel_gt)
    L_dis = contrastive_loss(out['dr_feat'], out['hr_feat'], dr_labels, hr_labels.long())
    total = L_dr + L_hr + 0.5*L_vessel + 0.3*L_dis

    assert not torch.isnan(total), "Total loss is NaN!"
    assert not torch.isinf(total), "Total loss is Inf!"
    print(f"  Losses — DR: {L_dr:.4f}, HR: {L_hr:.4f}, Vessel: {L_vessel:.4f}, Dis: {L_dis:.4f}")
    print(f"  Total: {total:.4f} ✓")

    # Backward pass
    total.backward()
    grad_ok = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  Backward pass + gradients ✓")

    # MC Dropout uncertainty
    model.eval()
    unc = model.predict_with_uncertainty(x, n=5)
    print(f"  MC Dropout — DR uncertainty: {unc['dr_std'].mean():.4f} ✓")

    del model, x, out
    torch.cuda.empty_cache()
    gc.collect()
    print("\n✅ ALL SMOKE TESTS PASSED — Safe to train!")

# Uncomment to run:
# smoke_test()
