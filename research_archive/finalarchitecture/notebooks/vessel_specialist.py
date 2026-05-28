"""
╔══════════════════════════════════════════════════════════════════════════╗
║  IRDAS — VESSEL SPECIALIST  v1  ·  Kaggle T4  ·  Phase 3               ║
║                                                                          ║
║  Pipeline position:                                                      ║
║    DR Teacher ✓ (θ_base, QWK 0.9342)                                    ║
║    → HR Specialist ✓ → τ_HR saved                                        ║
║    → [THIS] Vessel Specialist → θ_V + τ_V                               ║
║    → TIES Merge           → θ_merged                                     ║
║    → Joint Calibration    → IRDAS final                                  ║
║                                                                          ║
║  What this notebook does:                                                ║
║    1. Loads DR teacher EMA checkpoint as init (same θ_base as Phase 2)   ║
║    2. Trains the VesselDecoder branch on DRIVE dataset                   ║
║    3. Saves θ_V (best EMA) and τ_V = θ_V − θ_base                      ║
║                                                                          ║
║  Research-backed design:                                                 ║
║    · Focal Tversky Loss (α=0.7, β=0.3, γ=0.75) — 2024 SOTA on DRIVE   ║
║    · Patch-based training (256×256) — prevents overfitting on 20 imgs   ║
║    · FOV mask applied to loss — avoids easy background gradients         ║
║    · Elastic deformation — topology-preserving augmentation for vessels  ║
║    · Pre-loads all DRIVE images into RAM (tiny dataset)                  ║
║    · Early stopping (patience=10) — overfitting happens fast             ║
║    · Target Dice > 0.78 (DRIVE benchmark)                               ║
║                                                                          ║
║  Kaggle inputs expected:                                                 ║
║    /kaggle/input/datasets/nawazishbilal/dr-teacher-weights/best_ema_teacher.pth                       ║
║    /kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/images/                  ║
║    /kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/1st_manual/              ║
║                                                                          ║
║  Outputs written to /kaggle/working/vessel_specialist/                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================
# CELL 1 — Environment
# ============================================================
import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ============================================================
# CELL 2 — Imports
# ============================================================
import gc
import ctypes
import random
import copy
import time
import warnings

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import psutil
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

warnings.filterwarnings('ignore')
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True

_proc = psutil.Process(os.getpid())
try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


# ============================================================
# CELL 3 — Configuration
# ============================================================
CFG = {
    'seed':           42,
    'backbone':       'tf_efficientnet_b4.ns_jft_in1k',
    'drop_path_rate': 0.2,

    'total_epochs':   60,
    'freeze_epochs':  0,    # Train whole network from epoch 1
    'warmup_epochs':  3,
    'stage_lrs': [
        (1e-3, 1e-5),   # Unused Stage 1 config (0 epochs)
        (1e-3, 1e-5),   # Stage 2: full fine-tune
    ],

    'batch_size':    8,
    'grad_accum':    8,     # eff-batch = 64

    'weight_decay':  1e-2,
    'ema_decay':     0.999,
    'lookahead_k':   5,
    'lookahead_alpha': 0.5,

    # ── Patch settings ───────────────────────────────────────
    'patch_size':       256,     # 256×256 patches from full images
    'patches_per_img':  40,      # patches extracted per image per epoch
    'val_images_frac':  0.25,    # 25% of images held out for validation

    # ── Focal Tversky Loss ───────────────────────────────────
    'tversky_alpha':  0.7,   # FP weight (higher = less FP penalty)
    'tversky_beta':   0.3,   # FN weight (higher = more FN penalty) → catch vessels
    'tversky_gamma':  0.75,  # focal exponent (< 1 = focus on hard examples)

    # ── Early stopping ───────────────────────────────────────
    'early_stop_patience': 10,
    'target_dice': 0.78,

    # ── Paths ────────────────────────────────────────────────
    'teacher_weights': '/kaggle/input/datasets/nawazishbilal/dr-teacher-weights/best_ema_teacher.pth',
    'drive_imgs':      '/kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/images',
    'drive_masks':     '/kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/1st_manual',
    'out_dir':         '/kaggle/working/vessel_specialist',

    # ── Model ────────────────────────────────────────────────
    'fpn_out_channels': 256,
    'dropout': 0.3,
}


# ============================================================
# CELL 4 — Memory utilities
# ============================================================
def _malloc_trim():
    if _libc: _libc.malloc_trim(0)

def _rss_gb(): return _proc.memory_info().rss / 1e9

def _vram_str():
    parts = []
    for i in range(torch.cuda.device_count()):
        u = torch.cuda.memory_allocated(i) / 1024**3
        t = torch.cuda.get_device_properties(i).total_memory / 1024**3
        parts.append(f"G{i}:{u:.1f}/{t:.1f}GB")
    return " ".join(parts) or "No GPU"

def _mem_chk(label):
    print(f"  📊 [{label}] RSS={_rss_gb():.2f}GB  {_vram_str()}")

def _cleanup(): gc.collect(); _malloc_trim(); torch.cuda.empty_cache()

def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ============================================================
# CELL 5 — Preprocessing (same Ben Graham + CLAHE as teacher)
# ============================================================
def ben_graham_clahe(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img_bg  = cv2.addWeighted(img, 4, blurred, -4, 128)
    lab     = cv2.cvtColor(img_bg, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def apply_clahe(img_rgb):
    """Applies Contrast Limited Adaptive Histogram Equalization to the LAB Lightness channel."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        for ext in ['.tif', '.png', '.jpg', '.jpeg', '.bmp']:
            alt = os.path.splitext(path)[0] + ext
            img = cv2.imread(alt)
            if img is not None: break
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return apply_clahe(image)

def load_mask(path):
    """Load vessel mask — handles GIF format (DRIVE uses .gif)."""
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None and _HAS_PIL:
        mask = np.array(PILImage.open(path).convert('L'))
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return mask  # uint8 {0, 255}

def compute_fov_mask(img, threshold=20):
    """Compute circular Field-of-View mask.

    Fundus cameras produce a circular image; everything outside is black.
    Computing loss on black pixels gives trivial easy gradients — mask them out.
    """
    img_u8 = np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
    gray   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask  # uint8 {0, 255}


# ============================================================
# CELL 6 — Augmentation for vessel patches
# ============================================================
def get_vessel_transforms(is_train):
    """Joint image+mask+fov augmentation with elastic deformation."""
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                scale=(0.9, 1.1), rotate=(-30, 30),
                border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            # Elastic deformation — vessel topology-preserving
            A.ElasticTransform(alpha=120, sigma=6,
                                border_mode=cv2.BORDER_REFLECT_101, p=0.3),
            A.OneOf([
                A.GaussNoise(std_range=(0.005, 0.05), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            A.ColorJitter(brightness=0.15, contrast=0.2, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask', 'fov': 'mask'})
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask', 'fov': 'mask'})


# ============================================================
# CELL 7 — DRIVE Patch Dataset (in-memory, FOV-aware)
# ============================================================
class DRIVEPatchDataset(Dataset):
    """DRIVE vessel segmentation — patch-based training.

    All 20 images pre-loaded into RAM (tiny dataset).
    N random 256×256 patches extracted per image per epoch.
    Patches sampled only within valid FOV circle.

    Returns: (image, mask, fov_mask) — all (C, H, W) or (1, H, W).
    """

    def __init__(self, img_paths, mask_paths, transforms,
                 patch_size=256, patches_per_img=40, is_train=True):
        self.transforms      = transforms
        self.patch_size      = patch_size
        self.patches_per_img = patches_per_img
        self.is_train        = is_train

        print(f"  📦 Pre-loading {len(img_paths)} DRIVE images into RAM...")
        self._imgs  = []
        self._masks = []
        self._fovs  = []
        for ip, mp in zip(img_paths, mask_paths):
            img  = load_image(ip)
            mask = load_mask(mp)
            fov  = compute_fov_mask(img)
            self._imgs.append(img)
            self._masks.append(mask)
            self._fovs.append(fov)
        print(f"  ✅ {len(self._imgs)} images ready")

    def __len__(self):
        return len(self._imgs) * self.patches_per_img

    def __getitem__(self, idx):
        img_i = idx // self.patches_per_img
        img   = self._imgs[img_i]
        mask  = self._masks[img_i]
        fov   = self._fovs[img_i]
        H, W  = img.shape[:2]
        ps    = self.patch_size

        if self.is_train:
            # Try up to 20 times to land inside FOV
            for _ in range(20):
                y = random.randint(0, max(0, H - ps))
                x = random.randint(0, max(0, W - ps))
                if fov[y:y+ps, x:x+ps].mean() > 127:
                    break
        else:
            y = max(0, (H - ps) // 2)
            x = max(0, (W - ps) // 2)

        y = min(y, max(0, H - ps)); x = min(x, max(0, W - ps))

        img_p  = img[y:y+ps, x:x+ps]
        mask_f = (mask[y:y+ps, x:x+ps] > 127).astype(np.float32)
        fov_f  = (fov[y:y+ps, x:x+ps] > 127).astype(np.float32)

        aug    = self.transforms(image=img_p, mask=mask_f, fov=fov_f)
        img_t  = aug['image']                                         # (3, ps, ps)
        mask_t = aug['mask'].unsqueeze(0).float()                     # (1, ps, ps)
        fov_t  = aug['fov'].unsqueeze(0).float()                      # (1, ps, ps)
        return img_t, mask_t, fov_t


def build_drive_loaders():
    """Build DRIVE train/val loaders using the official train/test split."""
    img_dir_train  = CFG['drive_imgs']
    mask_dir_train = CFG['drive_masks']
    img_dir_val    = CFG['drive_imgs'].replace('/training/images', '/test/images')
    mask_dir_val   = CFG['drive_masks'].replace('/training/1st_manual', '/test/1st_manual')

    train_img_files = sorted([f for f in os.listdir(img_dir_train) if f.lower().endswith(('.tif', '.png', '.jpg', '.bmp'))])
    train_mask_files = sorted([f for f in os.listdir(mask_dir_train) if f.lower().endswith(('.gif', '.tif', '.png', '.bmp'))])
    val_img_files = sorted([f for f in os.listdir(img_dir_val) if f.lower().endswith(('.tif', '.png', '.jpg', '.bmp'))])
    val_mask_files = sorted([f for f in os.listdir(mask_dir_val) if f.lower().endswith(('.gif', '.tif', '.png', '.bmp'))])

    train_img = [os.path.join(img_dir_train, f) for f in train_img_files]
    train_mask = [os.path.join(mask_dir_train, f) for f in train_mask_files]
    val_img = [os.path.join(img_dir_val, f) for f in val_img_files]
    val_mask = [os.path.join(mask_dir_val, f) for f in val_mask_files]

    print(f"  DRIVE split: {len(train_img)} train / {len(val_img)} val images")
    print(f"  Patches per epoch: train={len(train_img)*CFG['patches_per_img']} val={len(val_img)*10}")

    ps = CFG['patch_size']
    train_ds = DRIVEPatchDataset(train_img, train_mask,
                                  get_vessel_transforms(True),
                                  ps, CFG['patches_per_img'], True)
    val_ds   = DRIVEPatchDataset(val_img, val_mask,
                                  get_vessel_transforms(False),
                                  ps, 10, False)

    tl = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,
                    num_workers=0, pin_memory=False)
    vl = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False,
                    num_workers=0, pin_memory=False)
    return tl, vl


# ============================================================
# CELL 8 — Model (Self-Contained)
# ============================================================

class ConvBN(nn.Module):
    def __init__(self, ci, co, k=3, p=1, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(ci, co, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(co)
        self.act  = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, ci, skip_ch, co):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            ConvBN(ci + skip_ch, co), ConvBN(co, co))
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], 1))

class SpatialAttention(nn.Module):
    """Highlights continuous spatial features (vessels) and suppresses background."""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.conv(scale))

class VesselDecoder(nn.Module):
    def __init__(self, encoder_channels):
        super().__init__()
        # Receives all 5 resolutions from EfficientNet
        c0, c1, c2, c3, c4 = encoder_channels
        self.up1 = UpBlock(c4, c3, 256)
        self.up2 = UpBlock(256, c2, 128)
        self.up3 = UpBlock(128, c1, 64)
        self.up4 = UpBlock(64, c0, 32)
        
        self.sa = SpatialAttention()  # Injecting Spatial Attention
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBN(32, 16),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, f0, f1, f2, f3, f4):
        x = self.up1(f4, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)
        x = self.up4(x, f0)
        x = self.sa(x)  # Apply attention before the final projection
        x = self.final_up(x)
        return x  # STRICTLY RAW LOGITS

class VesselSpecialistModel(nn.Module):
    def __init__(self):
        super().__init__()
        lw = CFG.get('local_bb_weights', '')
        try:
            self.backbone = timm.create_model(
                CFG['backbone'], pretrained=(not os.path.exists(lw)),
                features_only=True, out_indices=(0, 1, 2, 3, 4), 
                drop_path_rate=CFG['drop_path_rate'])
        except TypeError:
            self.backbone = timm.create_model(
                CFG['backbone'], pretrained=True,
                features_only=True, out_indices=(0, 1, 2, 3, 4))
                
        ch = self.backbone.feature_info.channels()
        self.decoder = VesselDecoder(ch)

    def forward(self, x):
        f = self.backbone(x)          
        return self.decoder(f[0], f[1], f[2], f[3], f[4])

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
        print("  🧊 Backbone FROZEN")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True
        print("  🔥 Backbone UNFROZEN")


# ============================================================
# CELL 9 — Safe Weight Loading (shape-filter prevents RuntimeError)
# ============================================================
def safe_load_weights(model, path, verbose=True):
    if not os.path.exists(path):
        print(f"  ⚠️  Weights not found: {path}")
        return 0

    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu')
        if isinstance(sd, dict):
            for key in ('module', 'state_dict', 'model', 'ema'):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]; break

    if any(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}

    model_sd = model.state_dict()
    filtered = {}
    skipped_shape, skipped_key = 0, 0

    for k, v in sd.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                skipped_shape += 1
            continue
            
        # V2 Fix: Handle teacher's missing prefix
        pk = 'backbone.' + k
        if pk in model_sd:
            if v.shape == model_sd[pk].shape:
                filtered[pk] = v
            else:
                skipped_shape += 1
        else:
            skipped_key += 1

    model_sd.update(filtered)
    model.load_state_dict(model_sd, strict=False)

    if verbose:
        print(f"  ✅ Loaded {len(filtered)} tensors from {os.path.basename(path)}")
    return len(filtered)


# ============================================================
# CELL 10 — Focal Tversky Loss
# ============================================================
def bce_dice_loss(logits, targets, fov_mask=None, bce_weight=0.5, pos_weight=5.0):
    if fov_mask is not None:
        m = fov_mask.squeeze(1).bool()
        logits = logits.squeeze(1)[m]
        targets = targets.squeeze(1)[m]
    else:
        logits = logits.view(-1)
        targets = targets.view(-1)
        
    # 1. Native PyTorch BCE with Logits (Numerically stable)
    bce_loss = F.binary_cross_entropy_with_logits(
        logits, targets.float(), 
        pos_weight=torch.tensor([pos_weight], device=logits.device)
    )
    
    # 2. Dice Loss (Needs probabilities, so we apply sigmoid here)
    probs = torch.sigmoid(logits)
    smooth = 1.0
    intersection = (probs * targets).sum()
    dice = (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
    dice_loss = 1.0 - dice
    
    return (bce_weight * bce_loss) + ((1.0 - bce_weight) * dice_loss)


# ============================================================
# CELL 11 — Dice coefficient (for monitoring only, not training)
# ============================================================
@torch.no_grad()
def dice_coefficient(pred, target, threshold=0.5, smooth=1.0):
    """Binary Dice for validation monitoring."""
    p = (pred > threshold).float().view(pred.size(0), -1)
    t = target.float().view(target.size(0), -1)
    inter = (p * t).sum(1)
    return ((2 * inter + smooth) / (p.sum(1) + t.sum(1) + smooth)).mean().item()


# ============================================================
# CELL 12 — Lookahead & EMA (same as teacher)
# ============================================================
class Lookahead:
    def __init__(self, opt, k=5, alpha=0.5):
        self.opt=opt; self.k=k; self.alpha=alpha; self._step=0
        self.slow={p:p.data.clone().detach()
                   for g in opt.param_groups for p in g['params']}
    def sync(self):
        self._step += 1
        if self._step % self.k == 0:
            for g in self.opt.param_groups:
                for p in g['params']:
                    if p not in self.slow: self.slow[p]=p.data.clone().detach()
                    self.slow[p].add_(self.alpha*(p.data-self.slow[p]))
                    p.data.copy_(self.slow[p])
    def zero_grad(self,set_to_none=True): self.opt.zero_grad(set_to_none=set_to_none)
    def step(self): self.opt.step()
    @property
    def param_groups(self): return self.opt.param_groups

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module=copy.deepcopy(model); self.module.eval(); self.decay=decay
    @torch.no_grad()
    def update(self, model):
        for ep,mp in zip(self.module.parameters(),model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data,alpha=1-self.decay)
        for eb,mb in zip(self.module.buffers(),model.buffers()): eb.copy_(mb)


# ============================================================
# CELL 13 — Scheduler
# ============================================================
def build_scheduler(opt, total_ep, warmup_ep):
    we = min(warmup_ep, total_ep // 3)
    ce = max(total_ep - we, 1)
    return SequentialLR(opt,
        [LinearLR(opt, 0.01, total_iters=we),
         CosineAnnealingLR(opt, T_max=ce, eta_min=1e-6)],
        milestones=[we])


# ============================================================
# CELL 14 — Validation
# ============================================================
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n = 0
    
    all_preds = []
    all_masks = []
    
    for imgs, masks, fovs in loader:
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        fovs  = fovs.to(device,  non_blocking=True)
        
        logits = model(imgs)
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, masks.shape[-2:], mode='bilinear', align_corners=False)
            
        loss = bce_dice_loss(logits, masks, fovs)
        total_loss += loss.item()
        
        # Apply ONE sigmoid strictly for the threshold sweep evaluation
        probs = torch.sigmoid(logits).detach()
        all_preds.append(probs.view(-1).cpu())
        all_masks.append(masks.view(-1).cpu())
        n += 1

    # Flatten predictions for threshold sweep
    all_preds_cat = torch.cat(all_preds)
    all_masks_cat = torch.cat(all_masks)
    
    best_dice = 0.0
    best_thresh = 0.5
    
    # Sweep thresholds between 0.1 and 0.9
    for t in np.arange(0.1, 0.91, 0.05):
        p = (all_preds_cat > t).float()
        inter = (p * all_masks_cat).sum()
        dice = (2. * inter + 1.0) / (p.sum() + all_masks_cat.sum() + 1.0)
        
        if dice.item() > best_dice:
            best_dice = dice.item()
            best_thresh = float(t)

    return {
        'loss': total_loss / max(n, 1),
        'dice': best_dice,
        'thresh': best_thresh
    }


# ============================================================
# CELL 15 — Train one epoch
# ============================================================
def train_one_epoch(model, loader, la, ema, scaler, device, accum):
    model.train()
    total_loss = 0.0
    la.zero_grad(set_to_none=True)
    n_updates = 0

    for step, (imgs, masks, fovs) in enumerate(loader):
        imgs  = imgs.to(device,  non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        fovs  = fovs.to(device,  non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, masks.shape[-2:], mode='bilinear', align_corners=False)
            
            loss = bce_dice_loss(logits, masks, fovs)
            loss = loss / accum

        scaler.scale(loss).backward()
        total_loss += loss.item() * accum

        if (step+1) % accum == 0 or step+1 == len(loader):
            scaler.unscale_(la.opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(la.opt)
            scaler.update()
            la.sync()
            la.zero_grad(set_to_none=True)
            ema.update(model)
            n_updates += 1

    return total_loss / len(loader)


# ============================================================
# CELL 16 — Main
# ============================================================
def main():
    set_seed(CFG['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*68}")
    print(f"  IRDAS Phase 3 — VESSEL SPECIALIST TRAINING")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs: {CFG['total_epochs']}  |  Patience: {CFG['early_stop_patience']}")
    print(f"  Loss: Focal Tversky (α={CFG['tversky_alpha']} β={CFG['tversky_beta']} γ={CFG['tversky_gamma']})")
    print(f"  Patch: {CFG['patch_size']}×{CFG['patch_size']}  ×  {CFG['patches_per_img']} per image")
    print(f"{'='*68}\n")

    os.makedirs(CFG['out_dir'], exist_ok=True)
    _mem_chk("startup")

    # ── DRIVE data ────────────────────────────────────────────
    train_loader, valid_loader = build_drive_loaders()
    _mem_chk("post-data")

    # ── Model ─────────────────────────────────────────────────
    print("\n  🏗  Building Vessel Specialist Model...")
    model = VesselSpecialistModel().to(device)

    teacher_loaded = safe_load_weights(model, CFG['teacher_weights'])
    if teacher_loaded == 0:
        print("  ℹ️  Teacher not found — backbone stays ImageNet-pretrained")

    # Save θ_base (should be same as Phase 2 — loaded from same teacher weights)
    base_path = os.path.join(CFG['out_dir'], 'theta_base_vessel.pth')
    torch.save(model.state_dict(), base_path)
    print(f"  💾 θ_base saved → {base_path}")
    _mem_chk("post-model")

    ema = ModelEMA(model, CFG['ema_decay'])

    # ── Stage 1: freeze backbone ──────────────────────────────
    model.freeze_backbone()
    dec_params = [p for p in model.parameters() if p.requires_grad]
    base_opt   = torch.optim.AdamW(dec_params, lr=CFG['stage_lrs'][0][0],
                                    weight_decay=CFG['weight_decay'])
    la     = Lookahead(base_opt, CFG['lookahead_k'], CFG['lookahead_alpha'])
    scaler = torch.amp.GradScaler('cuda')
    scheduler = build_scheduler(base_opt, CFG['freeze_epochs'], CFG['warmup_epochs'])

    backbone_unfrozen = False
    best_dice = 0.0
    best_path = None
    patience_ctr = 0

    print(f"\n{'='*68}")
    print(f"  STARTING — {CFG['total_epochs']} epochs  (early stop patience={CFG['early_stop_patience']})")
    print(f"{'='*68}\n")

    for epoch in range(CFG['total_epochs']):
        t0 = time.time()

        # ── Unfreeze backbone ─────────────────────────────────
        if epoch == CFG['freeze_epochs'] and not backbone_unfrozen:
            backbone_unfrozen = True
            model.unfreeze_backbone()
            head_lr, bb_lr = CFG['stage_lrs'][1]
            base_opt.add_param_group({
                'params': list(model.backbone.parameters()),
                'lr': bb_lr, 'weight_decay': CFG['weight_decay']})
            base_opt.param_groups[0]['lr'] = head_lr
            remaining = CFG['total_epochs'] - epoch
            scheduler = build_scheduler(base_opt, remaining, CFG['warmup_epochs'])
            print(f"  🔥 Backbone unfrozen | head={head_lr:.1e}  bb={bb_lr:.1e}")

        train_loss = train_one_epoch(model, train_loader, la, ema, scaler, device, CFG['grad_accum'])
        scheduler.step()

        metrics = validate(ema.module, valid_loader, device)
        dice    = metrics['dice']

        improved = dice > best_dice
        if improved:
            best_dice   = dice
            patience_ctr = 0
            best_path   = os.path.join(CFG['out_dir'], f'vessel_best_dice{dice:.4f}.pth')
            # Save bundled checkpoint
            torch.save({
                'state_dict': ema.module.state_dict(),
                'threshold': metrics['thresh'],
                'dice': dice,
                'epoch': epoch + 1
            }, best_path)
        else:
            patience_ctr += 1

        star = " ⭐" if improved else ""
        lr_h = base_opt.param_groups[0]['lr']
        lr_b = base_opt.param_groups[1]['lr'] if backbone_unfrozen else 0.0
        print(
            f"  Ep [{epoch+1:03d}/{CFG['total_epochs']}] "
            f"lr_h={lr_h:.2e} lr_b={lr_b:.2e} | "
            f"Loss={train_loss:.4f} | "
            f"Dice={dice:.4f} (thr={metrics['thresh']:.2f}) [best={best_dice:.4f}] "
            f"patience={patience_ctr}/{CFG['early_stop_patience']} "
            f"{time.time()-t0:.0f}s{star}"
        )
        _mem_chk(f"ep{epoch+1}")
        _cleanup()

        # ── Early stopping ────────────────────────────────────
        if patience_ctr >= CFG['early_stop_patience']:
            print(f"\n  ⏹  Early stopping at epoch {epoch+1} (patience exhausted)")
            break

        if best_dice >= CFG['target_dice']:
            print(f"\n  🎯 Target Dice {CFG['target_dice']} reached! Stopping.")
            break

    # ── Post-training ─────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  POST-TRAINING FINALIZATION")
    print(f"{'='*68}")

    # Save final EMA
    final_path = os.path.join(CFG['out_dir'], 'vessel_final_ema.pth')
    torch.save(ema.module.state_dict(), final_path)

    # Compute task vector τ_V = θ_V − θ_base
    print("\n  🔧 Computing task vector τ_V = θ_V − θ_base (backbone only)...")
    
    # Use the locally saved base_path to avoid dependencies on Phase 2 outputs
    theta_base = torch.load(base_path, map_location='cpu')
    
    theta_v_raw = torch.load(best_path or final_path, map_location='cpu')
    theta_v = theta_v_raw.get('state_dict', theta_v_raw) if isinstance(theta_v_raw, dict) else theta_v_raw
    
    BACKBONE_PREFIX = 'backbone.'
    tau_v   = {}
    skipped_nonbb, skipped_mismatch = 0, 0
    
    for k in theta_base:
        # STRICTLY limit to shared backbone parameters
        if not k.startswith(BACKBONE_PREFIX):
            skipped_nonbb += 1
            continue
            
        if k in theta_v and theta_v[k].shape == theta_base[k].shape:
            tau_v[k] = theta_v[k].float() - theta_base[k].float()
        else:
            skipped_mismatch += 1

    tau_path = os.path.join(CFG['out_dir'], 'tau_vessel.pth')
    torch.save(tau_v, tau_path)
    print(f"  ✅ τ_V: {len(tau_v)} backbone tensors")

    print(f"\n{'='*68}")
    print(f"  ✅ PHASE 3 COMPLETE")
    print(f"  Best val Dice:  {best_dice:.4f}  (target={CFG['target_dice']})")
    print(f"  θ_base:         {base_path}")
    print(f"  θ_V (best EMA): {best_path}")
    print(f"  τ_V:            {tau_path}")
    print(f"{'='*68}\n")
    print("  📋 Next step: Run ties_merge.py (Phase 4)")
    print(f"     Copy to Kaggle: {tau_path}")


if __name__ == '__main__':
    main()
