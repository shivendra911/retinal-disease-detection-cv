"""
╔══════════════════════════════════════════════════════════════════════════╗
║  IRDAS — HR SPECIALIST  v2  ·  Kaggle T4  ·  Phase 2                   ║
║                                                                          ║
║  v2 changes over v1 (every change has inline comment):                  ║
║    1. validate()         — threshold sweep, not hardcoded 0.5           ║
║    2. focal_bce_loss     — gamma 2→1, pos_weight 2.5→1.5, label smooth  ║
║    3. TopKCheckpoints    — saves threshold bundled with each .pth        ║
║    4. main() post-train  — best EMA from tracker.best_path() not final  ║
║    5. τ_HR               — backbone keys only, FPN/head excluded         ║
║    6. safe_load_weights  — backbone prefix stripping implemented         ║
║    7. MixUp              — regularisation for 605-sample HRDC dataset    ║
║    8. CFG                — freeze_epochs 5→20, bb_lr 3e-5→1e-5          ║
║    9. Self-training      — pseudo-label IDRiD (416 images) + fine-tune  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

# ============================================================
# CELL 1 — Environment lock
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
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import albumentations as A
from albumentations.pytorch import ToTensorV2

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

    # ── Training stages ──────────────────────────────────────
    # v2: freeze_epochs 5→20
    # With 605 samples, the backbone destabilises the output distribution the
    # moment it unfreezes — in v1 this caused F1=0 from ep15 onward (all
    # sigmoid outputs drifted below 0.5 while AUC kept rising to 0.75).
    # 20 epochs of head-only training lets the FC + FPN calibrate first.
    'freeze_epochs':  20,
    'total_epochs':   80,
    'warmup_epochs':  5,
    'stage_lrs': [
        (3e-4, 0.0),   # Stage 1: head only
        (1e-4, 1e-5),  # Stage 2: full — v2: backbone LR 3e-5→1e-5 for stability
    ],

    'batch_size':   16,
    'grad_accum':    4,
    'weight_decay': 1e-2,

    # ── SWA ─────────────────────────────────────────────────
    'swa_start':    68,    # scaled with total_epochs
    'swa_lr':       5e-6,

    # ── EMA / Lookahead ──────────────────────────────────────
    'ema_decay':        0.999,
    'lookahead_k':      5,
    'lookahead_alpha':  0.5,

    # ── Loss ─────────────────────────────────────────────────
    # v2: gamma 2→1, pos_weight 2.5→1.5
    # In v1 these two fought: focal down-weighted easy positives while
    # pos_weight up-weighted them. WeightedRandomSampler already handles
    # the 41% imbalance, so both values are reduced to stop the conflict.
    'focal_gamma':    1.0,
    'hr_pos_weight':  1.5,
    # v2: label_smoothing prevents overconfident logits on tiny dataset
    'label_smoothing': 0.05,

    # ── MixUp (v2: new) ──────────────────────────────────────
    'mixup_alpha': 0.3,
    'mixup_prob':  0.5,

    # ── Image ────────────────────────────────────────────────
    'image_size': 384,

    # ── Model architecture ───────────────────────────────────
    'fpn_out_channels': 256,
    'dropout': 0.4,   # v2: 0.3→0.4

    # ── Threshold sweep (v2: new) ────────────────────────────
    # In v1, threshold=0.5 caused F1=0 from ep15 onward — the model WAS
    # learning (AUC 0.34→0.75) but sigmoid outputs drifted below 0.5.
    'threshold_sweep': True,

    # ── Pseudo-labeling / self-training (v2: new) ────────────
    # IDRiD has no HR labels but is the same fundus domain. We run TTA
    # inference with the trained Phase 2 model, keep high-confidence
    # predictions as pseudo-labels, and fine-tune a second pass.
    'selftraining_conf':    0.80,   # sigmoid threshold for pseudo-label selection
    'selftraining_epochs':  15,
    'selftraining_lrs':     (5e-5, 5e-6),
    'selftraining_mixup':   0.2,

    # ── Paths ────────────────────────────────────────────────
    'teacher_weights':  '/kaggle/input/datasets/nawazishbilal/dr-teacher-weights/best_ema_teacher.pth',
    'local_bb_weights': '/kaggle/input/tf-efficientnet-b4-ns-weights/model.safetensors',

    # Phase 1 threshold — used as starting reference for the sweep
    'phase1_thresholds': '/kaggle/input/datasets/nawazishbilal/dr-teacher-weights/best_thresholds.npy',

    'hrdc_csv':  '/kaggle/input/datasets/nawazishbilal/hrdc-hypertensive-retinopathy-grading-challenge/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv',
    'hrdc_imgs': '/kaggle/input/datasets/nawazishbilal/hrdc-hypertensive-retinopathy-grading-challenge/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',

    # IDRiD — no HR labels; used for pseudo-labeling only
    'idrid_csv':  '/kaggle/input/datasets/nawazishbilal/b-disease-grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv',
    'idrid_imgs': '/kaggle/input/datasets/nawazishbilal/b-disease-grading/B. Disease Grading/1. Original Images/a. Training Set',

    'out_dir':    '/kaggle/working/hr_specialist',
    'topk_ckpts': 5,
    'val_split':  0.15,
}


# ============================================================
# CELL 4 — Memory Utilities
# ============================================================
def _malloc_trim():
    if _libc is not None:
        _libc.malloc_trim(0)

def _rss_gb():
    return _proc.memory_info().rss / 1e9

def _vram_str():
    parts = []
    for i in range(torch.cuda.device_count()):
        u = torch.cuda.memory_allocated(i) / 1024**3
        t = torch.cuda.get_device_properties(i).total_memory / 1024**3
        parts.append(f"G{i}:{u:.1f}/{t:.1f}GB")
    return " ".join(parts) if parts else "No GPU"

def _mem_chk(label):
    print(f"  📊 [{label}] RSS={_rss_gb():.2f} GB  |  {_vram_str()}")

def _cleanup():
    gc.collect(); _malloc_trim(); torch.cuda.empty_cache()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================================================
# CELL 5 — Preprocessing (Ben Graham + CLAHE — identical to teacher)
# ============================================================
def ben_graham_clahe(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img_bg  = cv2.addWeighted(img, 4, blurred, -4, 128)
    lab     = cv2.cvtColor(img_bg, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        for ext in ['.jpeg', '.jpg', '.png', '.JPG', '.PNG']:
            alt = os.path.splitext(path)[0] + ext
            img = cv2.imread(alt)
            if img is not None:
                break
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# CELL 6 — HRDC Defensive Column Detection
# ============================================================
def detect_hrdc_columns(df):
    fn_candidates = ['filename', 'image_id', 'image', 'img_id', 'id', 'file_name', 'Image']
    gr_candidates = ['grade', 'label', 'class', 'HR_grade', 'hr_grade', 'target', 'Hypertensive Retinopathy']
    fn_col = next((c for c in fn_candidates if c in df.columns), None)
    gr_col = next((c for c in gr_candidates if c in df.columns), None)
    if fn_col is None or gr_col is None:
        raise KeyError(
            f"Cannot detect HRDC columns!\n"
            f"Looked for filename in: {fn_candidates}\n"
            f"Looked for grade in: {gr_candidates}\n"
            f"Found columns: {list(df.columns)}"
        )
    print(f"  ✅ HRDC columns: filename='{fn_col}'  grade='{gr_col}'")
    return fn_col, gr_col

def resolve_hrdc_path(img_dir, fname):
    base = os.path.splitext(fname)[0] if '.' in fname else fname
    for ext in ['', '.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"HRDC image not found: {fname}")


# ============================================================
# CELL 7 — Augmentation
# ============================================================
def get_transforms(image_size, is_train):
    if is_train:
        hole_h = max(8, image_size // 12)
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                     scale=(0.80, 1.20), rotate=(-45, 45),
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1,
                          saturation=0.3, hue=0.02, p=0.5),
            A.OneOf([
                A.GaussNoise(std_range=(3.16/255.0, 10.0/255.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.Downscale(scale_range=(0.5, 0.75), p=1.0),
            ], p=0.4),
            A.CoarseDropout(num_holes_range=(4, 12),
                            hole_height_range=(hole_h, hole_h * 2),
                            hole_width_range=(hole_h, hole_h * 2),
                            fill=0, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_tta_transforms(image_size):
    """4-view TTA: original + h-flip + v-flip + hv-flip."""
    base = [A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()]
    return [
        A.Compose(base),
        A.Compose([A.HorizontalFlip(p=1.0)] + base),
        A.Compose([A.VerticalFlip(p=1.0)]   + base),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)] + base),
    ]


# ============================================================
# CELL 8 — HRDC Dataset
# ============================================================
class HRDCDataset(Dataset):
    def __init__(self, df, fn_col, gr_col, img_dir, transforms):
        self.df      = df.reset_index(drop=True)
        self.fn_col  = fn_col
        self.gr_col  = gr_col
        self.img_dir = img_dir
        self.tfms    = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = resolve_hrdc_path(self.img_dir, str(row[self.fn_col]))
        img   = load_image(path)
        img   = ben_graham_clahe(img)
        img   = self.tfms(image=img)['image']
        label = torch.tensor(1.0 if int(row[self.gr_col]) > 0 else 0.0,
                              dtype=torch.float32)
        return img, label


def build_hrdc_loaders(df, fn_col, gr_col):
    df['_bin'] = (df[gr_col] > 0).astype(int)
    train_df, valid_df = train_test_split(
        df, test_size=CFG['val_split'], stratify=df['_bin'], random_state=42)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    labels  = train_df['_bin'].values
    counts  = np.bincount(labels).astype(float)
    weights = (1.0 / counts[labels]).astype(np.float32)
    sampler = WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True)

    train_ds = HRDCDataset(train_df, fn_col, gr_col, CFG['hrdc_imgs'],
                            get_transforms(CFG['image_size'], True))
    valid_ds = HRDCDataset(valid_df, fn_col, gr_col, CFG['hrdc_imgs'],
                            get_transforms(CFG['image_size'], False))

    tl = DataLoader(train_ds, batch_size=CFG['batch_size'], sampler=sampler,
                    num_workers=0, pin_memory=False, drop_last=True)
    vl = DataLoader(valid_ds, batch_size=CFG['batch_size'] * 2, shuffle=False,
                    num_workers=0, pin_memory=False)
    return tl, vl, train_df, valid_df, fn_col, gr_col


# ============================================================
# CELL 9 — IDRiD Dataset (unlabeled — for pseudo-labeling only)
# ============================================================
class IDRiDUnlabeledDataset(Dataset):
    """IDRiD images without labels — used only for TTA inference.

    IDRiD has no HR labels. We generate pseudo-labels by running TTA
    inference with the trained Phase 2 model. High-confidence predictions
    (sigmoid > conf_thresh or < 1-conf_thresh) become training pseudo-labels.

    Why IDRiD is valid for this:
    - Same domain: retinal fundus photographs
    - Same preprocessing pipeline (Ben Graham + CLAHE)
    - DR and HR have real comorbidity (~35-40% of DR patients have HR)
    - We never use IDRiD images in the validation set
    """
    def __init__(self, img_dir, fnames, tta_tfms):
        self.img_dir  = img_dir
        self.fnames   = fnames
        self.tta_tfms = tta_tfms   # list of transform pipelines

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        # IDRiD images are named IDRiD_XX.jpg
        for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
            p = os.path.join(self.img_dir, fname + ext)
            if os.path.exists(p):
                path = p; break
        else:
            raise FileNotFoundError(f"IDRiD image not found: {fname}")

        raw_img = ben_graham_clahe(load_image(path))
        # Return all TTA views stacked: (n_tta, C, H, W)
        views = torch.stack([tfm(image=raw_img)['image'] for tfm in self.tta_tfms])
        return views, fname


# ============================================================
# CELL 10 — Combined Dataset for Self-Training
# ============================================================
class CombinedPseudoDataset(Dataset):
    """HRDC (labelled) + IDRiD (pseudo-labelled) combined training set.

    IDRiD samples are given sample_weight < 1 to reflect label uncertainty.
    Validation is ALWAYS on the original HRDC val split only — IDRiD never
    contaminates the validation set.
    """
    def __init__(self, hrdc_df, fn_col, gr_col,
                 pseudo_df, pseudo_fn_col, pseudo_gr_col):
        self.tfms = get_transforms(CFG['image_size'], is_train=True)

        self.entries = []   # (path, label, is_pseudo)

        for _, row in hrdc_df.iterrows():
            path  = resolve_hrdc_path(CFG['hrdc_imgs'], str(row[fn_col]))
            label = float(int(row[gr_col]) > 0)
            self.entries.append((path, label, False))

        for _, row in pseudo_df.iterrows():
            # IDRiD images stored as IDRiD_XX.jpg
            fname = str(row[pseudo_fn_col])
            path  = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG']:
                p = os.path.join(CFG['idrid_imgs'], fname + ext)
                if os.path.exists(p):
                    path = p; break
            if path is None:
                continue
            label = float(row[pseudo_gr_col])
            self.entries.append((path, label, True))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        path, label, _ = self.entries[idx]
        img = ben_graham_clahe(load_image(path))
        img = self.tfms(image=img)['image']
        return img, torch.tensor(label, dtype=torch.float32)

    def make_sampler(self):
        """WeightedRandomSampler: HRDC samples weighted 1.0, pseudo 0.5."""
        weights = []
        labels  = []
        for _, label, is_pseudo in self.entries:
            sample_w = 0.5 if is_pseudo else 1.0
            weights.append(sample_w)
            labels.append(int(label))
        # Additionally balance classes within each source
        labels  = np.array(labels)
        counts  = np.bincount(labels).astype(float)
        class_w = 1.0 / counts[labels]
        final_w = np.array(weights) * class_w
        final_w = torch.from_numpy(final_w.astype(np.float32))
        return WeightedRandomSampler(final_w, len(final_w), replacement=True)


# ============================================================
# CELL 11 — Model
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch // r, 1)
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, mid, bias=False), nn.ReLU(),
            nn.Linear(mid, ch, bias=False))
    def forward(self, x):
        b, c, _, _ = x.shape
        a = self.fc(self.ap(x).view(b, c))
        m = self.fc(self.mp(x).view(b, c))
        return x * torch.sigmoid(a + m).view(b, c, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=(ks - 1) // 2, bias=False)
    def forward(self, x):
        a = torch.mean(x, 1, keepdim=True)
        m, _ = torch.max(x, 1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([a, m], 1)))

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class FPN(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.lat5   = nn.Conv2d(in_ch[2], out_ch, 1)
        self.lat4   = nn.Conv2d(in_ch[1], out_ch, 1)
        self.lat3   = nn.Conv2d(in_ch[0], out_ch, 1)
        self.smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn     = nn.BatchNorm2d(out_ch)
        self.relu   = nn.ReLU(inplace=True)
    def forward(self, p3, p4, p5):
        p5u = F.interpolate(self.lat5(p5), size=p4.shape[-2:], mode='nearest')
        p4f = self.lat4(p4) + p5u
        p4u = F.interpolate(p4f, size=p3.shape[-2:], mode='nearest')
        return self.relu(self.bn(self.smooth(self.lat3(p3) + p4u)))

class HRBranch(nn.Module):
    def __init__(self, ch, dropout=0.4):
        super().__init__()
        self.cbam    = CBAM(ch)
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(ch, 1)
    def forward(self, x):
        feat = self.gap(self.cbam(x)).flatten(1)
        return self.fc(self.dropout(feat))

class HRSpecialistModel(nn.Module):
    def __init__(self):
        super().__init__()
        lw = CFG.get('local_bb_weights', '')
        self.backbone = timm.create_model(
            CFG['backbone'],
            pretrained=not os.path.exists(lw),
            features_only=True, out_indices=(2, 3, 4),
            drop_path_rate=CFG['drop_path_rate'])
        ch = self.backbone.feature_info.channels()
        oc = CFG['fpn_out_channels']
        self.fpn     = FPN(ch, oc)
        self.hr_head = HRBranch(oc, CFG['dropout'])

    def forward(self, x):
        f = self.backbone(x)
        return self.hr_head(self.fpn(f[0], f[1], f[2]))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("  🧊 Backbone FROZEN")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("  🔥 Backbone UNFROZEN")


# ============================================================
# CELL 12 — Safe Weight Loading (v2: backbone prefix stripping)
# ============================================================
def safe_load_weights(model, path, verbose=True):
    """v2: implements backbone prefix stripping that v1 only described."""
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
        # Direct match
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                filtered[k] = v
            else:
                skipped_shape += 1
            continue
        # v2: teacher saved backbone without 'backbone.' prefix — add it
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
        if skipped_shape: print(f"     ⚠️  Skipped {skipped_shape} shape-mismatched")
        if skipped_key:   print(f"     ℹ️  Skipped {skipped_key} unrecognised keys")
    return len(filtered)


# ============================================================
# CELL 13 — Loss (v2: calibrated gamma + label smoothing)
# ============================================================
def focal_bce_loss(logits, labels, gamma=1.0, pos_weight=1.5, label_smoothing=0.05):
    """v2: gamma 2→1, pos_weight 2.5→1.5, label smoothing added.

    Root cause of v1 collapse: focal gamma=2 and pos_weight=2.5 conflict
    when combined with WeightedRandomSampler. Sampler already rebalances
    batches. gamma=1 + pos_weight=1.5 gives gentle focal + mild upweighting
    without driving logits to extreme values that break the 0.5 decision boundary.
    """
    logits = logits.squeeze(1)
    labels_smooth = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing
    probs = torch.sigmoid(logits)
    p_t   = probs * labels + (1.0 - probs) * (1.0 - labels)
    focal = (1.0 - p_t).pow(gamma)
    bce   = F.binary_cross_entropy_with_logits(
        logits, labels_smooth,
        pos_weight=torch.tensor([pos_weight], device=logits.device),
        reduction='none')
    return (focal * bce).mean()


# ============================================================
# CELL 14 — MixUp (v2: new)
# ============================================================
def mixup_data(x, y, alpha=0.3):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


# ============================================================
# CELL 15 — Lookahead & EMA
# ============================================================
class Lookahead:
    def __init__(self, opt, k=5, alpha=0.5):
        self.opt   = opt; self.k = k; self.alpha = alpha; self._step = 0
        self.slow  = {p: p.data.clone().detach()
                      for g in opt.param_groups for p in g['params']}
    def sync(self):
        self._step += 1
        if self._step % self.k == 0:
            for g in self.opt.param_groups:
                for p in g['params']:
                    if p not in self.slow:
                        self.slow[p] = p.data.clone().detach()
                    self.slow[p].add_(self.alpha * (p.data - self.slow[p]))
                    p.data.copy_(self.slow[p])
    def zero_grad(self, set_to_none=True): self.opt.zero_grad(set_to_none=set_to_none)
    def step(self): self.opt.step()
    @property
    def param_groups(self): return self.opt.param_groups
    def state_dict(self): return self.opt.state_dict()
    def load_state_dict(self, sd): self.opt.load_state_dict(sd)

class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model); self.module.eval(); self.decay = decay
    @torch.no_grad()
    def update(self, model):
        for ep, mp in zip(self.module.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)
        for eb, mb in zip(self.module.buffers(), model.buffers()):
            eb.copy_(mb)


# ============================================================
# CELL 16 — Scheduler
# ============================================================
def build_scheduler(optimizer, stage_epochs, warmup_epochs):
    warmup_ep = min(warmup_epochs, stage_epochs // 3)
    cosine_ep = max(stage_epochs - warmup_ep, 1)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_ep)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_ep, eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_ep])


# ============================================================
# CELL 17 — Validation (v2: threshold sweep)
# ============================================================
@torch.no_grad()
def validate(model, loader, device, fixed_threshold=None):
    """v2: sweep thresholds 0.10→0.90 instead of hardcoded 0.5.

    In v1 hardcoded 0.5 caused F1=0 for 35 straight epochs because
    the sigmoid distribution shifted below 0.5 after backbone unfreeze.
    AUC kept rising (backbone WAS learning) but F1 was blind to it.
    Sweeping finds the real operating point.

    fixed_threshold: use at inference time with the saved threshold.
    """
    model.eval()
    logits_all, labels_all = [], []
    for imgs, labels in loader:
        logits_all.extend(model(imgs.to(device, non_blocking=True)).squeeze(1).cpu().numpy())
        labels_all.extend(labels.numpy())

    logits_all = np.array(logits_all)
    labels_all = np.array(labels_all)
    probs      = 1.0 / (1.0 + np.exp(-logits_all))

    auc = float(roc_auc_score(labels_all, probs)) if len(np.unique(labels_all)) > 1 else 0.5

    if fixed_threshold is not None:
        best_thresh = float(fixed_threshold)
    elif CFG['threshold_sweep']:
        # Load Phase 1 threshold as the starting reference
        ref_thresh = 0.5
        phase1_path = CFG.get('phase1_thresholds', '')
        if os.path.exists(phase1_path):
            ref_thresh = float(np.load(phase1_path)[0])
        else:
            try:
                # Fallback to local path if running locally
                ref_thresh = float(np.load('best_thresholds.npy')[0])
            except:
                ref_thresh = 0.4875941  # Known optimal Phase 1 threshold

        best_f1, best_thresh = -1.0, ref_thresh
        # Fine-grained sweep ±0.15 around the reference threshold
        sweep_start = max(0.05, ref_thresh - 0.15)
        sweep_end   = min(0.95, ref_thresh + 0.15)
        for t in np.arange(sweep_start, sweep_end, 0.01):
            f1_t = float(f1_score(labels_all, (probs >= t).astype(int), zero_division=0))
            if f1_t > best_f1:
                best_f1, best_thresh = f1_t, float(t)
    else:
        phase1_path = CFG.get('phase1_thresholds', '')
        if os.path.exists(phase1_path):
            best_thresh = float(np.load(phase1_path)[0])
        else:
            try:
                best_thresh = float(np.load('best_thresholds.npy')[0])
            except:
                best_thresh = 0.4875941

    preds = (probs >= best_thresh).astype(int)
    return {
        'auc':    auc,
        'f1':     float(f1_score(labels_all, preds, zero_division=0)),
        'acc':    float(accuracy_score(labels_all, preds)),
        'thresh': best_thresh,
    }


# ============================================================
# CELL 18 — Train one epoch (v2: MixUp)
# ============================================================
def train_one_epoch(model, loader, la, ema, scaler, device, accum,
                    mixup_alpha=None, mixup_prob=0.5):
    model.train()
    total_loss = 0.0
    la.zero_grad(set_to_none=True)
    n_updates = 0
    use_mu = (mixup_alpha is not None and mixup_alpha > 0)

    for step, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        do_mix = use_mu and random.random() < mixup_prob
        if do_mix:
            imgs, la_, lb, lam = mixup_data(imgs, labels, mixup_alpha)

        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            if do_mix:
                loss = (lam * focal_bce_loss(logits, la_,
                                              CFG['focal_gamma'], CFG['hr_pos_weight'],
                                              CFG['label_smoothing'])
                        + (1 - lam) * focal_bce_loss(logits, lb,
                                                      CFG['focal_gamma'], CFG['hr_pos_weight'],
                                                      CFG['label_smoothing']))
            else:
                loss = focal_bce_loss(logits, labels,
                                       CFG['focal_gamma'], CFG['hr_pos_weight'],
                                       CFG['label_smoothing'])
            loss = loss / accum

        scaler.scale(loss).backward()
        total_loss += loss.item() * accum

        if (step + 1) % accum == 0 or step + 1 == len(loader):
            scaler.unscale_(la.opt)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(la.opt)
            scaler.update()
            la.sync()
            la.zero_grad(set_to_none=True)
            ema.update(model)
            n_updates += 1

    return total_loss / max(n_updates, 1)


# ============================================================
# CELL 19 — TopK Checkpoint tracker (v2: saves threshold with model)
# ============================================================
class TopKCheckpoints:
    def __init__(self, k, ckpt_dir, prefix):
        self.k = k; self.ckpt_dir = ckpt_dir; self.prefix = prefix
        self.records = []

    def update(self, model, epoch, f1, threshold=0.5):
        path = os.path.join(
            self.ckpt_dir, f"{self.prefix}_ep{epoch:03d}_f1{f1:.4f}.pth")
        if len(self.records) < self.k or f1 > self.records[0][0]:
            # v2: bundle threshold so inference always knows the right cutoff
            torch.save({'state_dict': model.state_dict(),
                        'threshold': float(threshold),
                        'epoch': epoch, 'f1': float(f1)}, path)
            self.records.append((f1, path))
            self.records.sort(key=lambda x: x[0])
            if len(self.records) > self.k:
                _, worst = self.records.pop(0)
                try: os.remove(worst)
                except: pass
            return True
        return False

    def best_f1(self):   return self.records[-1][0] if self.records else 0.0
    def best_path(self): return self.records[-1][1] if self.records else None


# ============================================================
# CELL 20 — Pseudo-label generation (v2: new)
# ============================================================
@torch.no_grad()
def generate_pseudo_labels(model, device, conf_thresh):
    """Run 4-view TTA inference on all IDRiD images. Return high-confidence
    pseudo-label DataFrame and raw probabilities for logging.

    Only images with mean TTA sigmoid > conf_thresh (→ HR=1) or
    < 1-conf_thresh (→ HR=0) are kept. Uncertain images are discarded.
    This prevents noisy pseudo-labels from hurting self-training.
    """
    if not os.path.exists(CFG['idrid_csv']):
        print("  ⚠️  IDRiD CSV not found — skipping pseudo-labeling")
        return None

    idrid_df = pd.read_csv(CFG['idrid_csv'])
    # IDRiD CSV column: 'Image name' (e.g. IDRiD_01)
    fn_col_i = next((c for c in ['Image name', 'image name', 'filename', 'Image']
                     if c in idrid_df.columns), None)
    if fn_col_i is None:
        print(f"  ⚠️  Cannot detect IDRiD filename column. Found: {list(idrid_df.columns)}")
        return None

    fnames   = idrid_df[fn_col_i].tolist()
    tta_tfms = get_tta_transforms(CFG['image_size'])
    dataset  = IDRiDUnlabeledDataset(CFG['idrid_imgs'], fnames, tta_tfms)
    loader   = DataLoader(dataset, batch_size=4, shuffle=False,
                          num_workers=0, pin_memory=False)

    model.eval()
    results = []   # [(fname, mean_prob)]

    for views_batch, fname_batch in loader:
        # views_batch: (B, n_tta, C, H, W)
        B, n_tta, C, H, W = views_batch.shape
        views_flat = views_batch.view(B * n_tta, C, H, W).to(device, non_blocking=True)
        logits_flat = model(views_flat).squeeze(1)
        probs_flat  = torch.sigmoid(logits_flat).cpu().numpy()
        probs_mat   = probs_flat.reshape(B, n_tta)
        mean_probs  = probs_mat.mean(axis=1)
        for fname, prob in zip(fname_batch, mean_probs):
            results.append((fname, float(prob)))

    total = len(results)
    pseudo_rows = []
    for fname, prob in results:
        if prob >= conf_thresh:
            pseudo_rows.append({'pseudo_fname': fname, 'pseudo_label': 1, 'prob': prob})
        elif prob <= (1.0 - conf_thresh):
            pseudo_rows.append({'pseudo_fname': fname, 'pseudo_label': 0, 'prob': prob})

    kept = len(pseudo_rows)
    pos  = sum(r['pseudo_label'] for r in pseudo_rows)
    neg  = kept - pos

    print(f"  🔬 IDRiD pseudo-labels: {kept}/{total} kept "
          f"(conf>{conf_thresh:.2f})  → {pos} HR+ / {neg} HR−")

    if kept == 0:
        print("  ⚠️  No pseudo-labels passed confidence threshold — skipping self-training")
        return None

    return pd.DataFrame(pseudo_rows)


# ============================================================
# CELL 21 — Self-training phase (v2: new)
# ============================================================
def selftraining_phase(model, ema, hrdc_train_df, valid_loader,
                       pseudo_df, fn_col, gr_col, device, tracker):
    """Fine-tune on HRDC + pseudo-labeled IDRiD.

    Uses lower LRs than Phase 2 main training. Validation remains on the
    original HRDC validation split — IDRiD never enters the val set.
    Updates tracker if self-training yields a better F1.
    """
    epochs  = CFG['selftraining_epochs']
    head_lr, bb_lr = CFG['selftraining_lrs']

    print(f"\n{'='*68}")
    print(f"  SELF-TRAINING PHASE — {epochs} epochs")
    print(f"  HRDC train: {len(hrdc_train_df)} | IDRiD pseudo: {len(pseudo_df)}")
    print(f"  LR: head={head_lr:.1e}  bb={bb_lr:.1e}")
    print(f"{'='*68}\n")

    # Build combined dataset
    combined_ds = CombinedPseudoDataset(
        hrdc_train_df, fn_col, gr_col,
        pseudo_df, 'pseudo_fname', 'pseudo_label')
    sampler = combined_ds.make_sampler()
    loader  = DataLoader(combined_ds, batch_size=CFG['batch_size'],
                         sampler=sampler, num_workers=0,
                         pin_memory=False, drop_last=True)

    print(f"  Combined dataset size: {len(combined_ds)}")

    # Optimiser — both head and backbone at reduced LRs
    opt = torch.optim.AdamW([
        {'params': list(model.hr_head.parameters()) + list(model.fpn.parameters()),
         'lr': head_lr},
        {'params': list(model.backbone.parameters()),
         'lr': bb_lr},
    ], weight_decay=CFG['weight_decay'])
    la      = Lookahead(opt, CFG['lookahead_k'], CFG['lookahead_alpha'])
    scaler  = torch.amp.GradScaler('cuda')
    cosine  = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-7)

    for epoch in range(epochs):
        t0 = time.time()
        loss = train_one_epoch(model, loader, la, ema, scaler, device,
                               CFG['grad_accum'],
                               mixup_alpha=CFG['selftraining_mixup'],
                               mixup_prob=0.4)
        cosine.step()
        metrics = validate(ema.module, valid_loader, device)
        saved   = tracker.update(ema.module, 1000 + epoch + 1,
                                  metrics['f1'], metrics['thresh'])
        star    = " ⭐ NEW BEST" if saved else ""
        print(f"  ST Ep [{epoch+1:02d}/{epochs}] "
              f"Loss={loss:.4f} | "
              f"AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
              f"thr={metrics['thresh']:.2f} [best F1={tracker.best_f1():.4f}]  "
              f"{time.time()-t0:.0f}s{star}")
        _cleanup()


# ============================================================
# CELL 22 — Main Training Loop
# ============================================================
def main():
    set_seed(CFG['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*68}")
    print(f"  IRDAS Phase 2 — HR SPECIALIST  v2")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  RAM: {psutil.virtual_memory().total/1e9:.1f} GB")
    print(f"  Epochs: {CFG['total_epochs']}  |  eff-batch: {CFG['batch_size']*CFG['grad_accum']}")
    print(f"  Loss: Focal BCE (γ={CFG['focal_gamma']}, pos_w={CFG['hr_pos_weight']}, ls={CFG['label_smoothing']})")
    print(f"  MixUp α={CFG['mixup_alpha']} p={CFG['mixup_prob']}  |  Threshold: sweep")
    print(f"  Self-training: IDRiD pseudo-labels (conf>{CFG['selftraining_conf']})")
    print(f"{'='*68}\n")

    os.makedirs(CFG['out_dir'], exist_ok=True)
    _mem_chk("startup")

    # ── Load Phase 1 threshold as hint ───────────────────────
    phase1_hr_thresh = None
    p1t = CFG.get('phase1_thresholds')
    if p1t and os.path.exists(p1t):
        try:
            t = np.load(p1t, allow_pickle=True).item()
            phase1_hr_thresh = float(t.get('hr', t.get('HR', 0.5)))
            print(f"  ✅ Phase 1 HR threshold: {phase1_hr_thresh:.3f}")
        except Exception as e:
            print(f"  ⚠️  Phase 1 threshold load failed: {e}")

    # ── Load HRDC Data ────────────────────────────────────────
    print("\n  📂 Loading HRDC dataset...")
    raw_df = pd.read_csv(CFG['hrdc_csv'])
    fn_col, gr_col = detect_hrdc_columns(raw_df)
    train_loader, valid_loader, train_df, valid_df, fn_col, gr_col = \
        build_hrdc_loaders(raw_df, fn_col, gr_col)
    pos_rate = (raw_df[gr_col] > 0).mean()
    print(f"  Train: {len(train_df)} | Val: {len(valid_df)}")
    print(f"  HR positive rate: {pos_rate:.1%}")
    _mem_chk("post-data")

    # ── Build Model ───────────────────────────────────────────
    print("\n  🏗  Building HR Specialist Model...")
    model = HRSpecialistModel().to(device)

    n_loaded = safe_load_weights(model, CFG['teacher_weights'])
    if n_loaded == 0:
        print("  ℹ️  Loading ImageNet backbone weights as fallback")
        safe_load_weights(model, CFG['local_bb_weights'])

    # CRITICAL: save θ_base BEFORE any gradient update
    base_path = os.path.join(CFG['out_dir'], 'theta_base.pth')
    torch.save(model.state_dict(), base_path)
    print(f"  💾 θ_base saved → {base_path}")
    print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    _mem_chk("post-model")

    ema = ModelEMA(model, CFG['ema_decay'])

    # ── Optimizer — Stage 1: head only ───────────────────────
    model.freeze_backbone()
    head_params = [p for p in model.parameters() if p.requires_grad]
    base_opt  = torch.optim.AdamW(head_params, lr=CFG['stage_lrs'][0][0],
                                   weight_decay=CFG['weight_decay'])
    la        = Lookahead(base_opt, CFG['lookahead_k'], CFG['lookahead_alpha'])
    scaler    = torch.amp.GradScaler('cuda')
    scheduler = build_scheduler(base_opt, CFG['freeze_epochs'], CFG['warmup_epochs'])

    backbone_unfrozen = False
    swa_model = None
    swa_sched = None
    tracker   = TopKCheckpoints(CFG['topk_ckpts'], CFG['out_dir'], 'hr_specialist')

    print(f"\n{'='*68}")
    print(f"  STARTING — {CFG['total_epochs']} epochs")
    print(f"  Stage 1 (frozen): ep 1–{CFG['freeze_epochs']}")
    print(f"  Stage 2 (full):   ep {CFG['freeze_epochs']+1}–{CFG['swa_start']}")
    print(f"  SWA:              ep {CFG['swa_start']+1}–{CFG['total_epochs']}")
    print(f"{'='*68}\n")

    for epoch in range(CFG['total_epochs']):
        t0 = time.time()

        # ── Stage 2: unfreeze backbone ────────────────────────
        if epoch == CFG['freeze_epochs'] and not backbone_unfrozen:
            backbone_unfrozen = True
            model.unfreeze_backbone()
            head_lr, bb_lr = CFG['stage_lrs'][1]
            base_opt.add_param_group({
                'params':       list(model.backbone.parameters()),
                'lr':           bb_lr,
                'weight_decay': CFG['weight_decay']})
            base_opt.param_groups[0]['lr'] = head_lr
            remaining = CFG['total_epochs'] - epoch
            # v2: cosine only for backbone group — no warmup on pretrained weights
            scheduler = CosineAnnealingLR(base_opt, T_max=remaining, eta_min=1e-6)
            print(f"  🔥 Backbone unfrozen | head={head_lr:.1e}  bb={bb_lr:.1e}")
            _mem_chk("unfreeze")

        # ── SWA activation ────────────────────────────────────
        if epoch == CFG['swa_start'] and swa_model is None:
            print(f"\n  🌀 SWA activated at epoch {epoch+1}")
            swa_model = AveragedModel(model)
            swa_sched = SWALR(base_opt, swa_lr=CFG['swa_lr'], anneal_epochs=5)
            _mem_chk("swa-init")

        train_loss = train_one_epoch(
            model, train_loader, la, ema, scaler, device, CFG['grad_accum'],
            mixup_alpha=CFG['mixup_alpha'], mixup_prob=CFG['mixup_prob'])

        metrics = validate(ema.module, valid_loader, device)

        if epoch >= CFG['swa_start'] and swa_model is not None:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            scheduler.step()

        saved = tracker.update(ema.module, epoch + 1, metrics['f1'], metrics['thresh'])
        star  = " ⭐ NEW BEST" if saved else ""

        lr_h = base_opt.param_groups[0]['lr']
        lr_b = base_opt.param_groups[1]['lr'] if backbone_unfrozen else 0.0
        print(
            f"  Ep [{epoch+1:03d}/{CFG['total_epochs']}] "
            f"lr_h={lr_h:.2e} lr_b={lr_b:.2e} | "
            f"Loss={train_loss:.4f} | "
            f"AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
            f"thr={metrics['thresh']:.2f} Acc={metrics['acc']:.4f} "
            f"[best={tracker.best_f1():.4f}]  {time.time()-t0:.0f}s{star}"
        )
        _mem_chk(f"ep{epoch+1}")
        _cleanup()

    # ============================================================
    # POST-TRAINING: SWA BN calibration
    # ============================================================
    print(f"\n{'='*68}")
    print("  POST-TRAINING FINALIZATION")
    print(f"{'='*68}")

    if swa_model is not None:
        print("  🔧 SWA BN calibration...")
        swa_model.train(); swa_model.to(device)
        with torch.no_grad():
            for imgs, _ in train_loader:
                swa_model(imgs.to(device, non_blocking=True))
        swa_path = os.path.join(CFG['out_dir'], 'hr_swa_final.pth')
        torch.save(swa_model.module.state_dict(), swa_path)
        print(f"  💾 SWA saved → {swa_path}")
        del swa_model; _cleanup()

    # ============================================================
    # SELF-TRAINING: pseudo-label IDRiD → fine-tune
    # ============================================================
    print(f"\n{'='*68}")
    print("  SELF-TRAINING — IDRiD PSEUDO-LABELING")
    print(f"{'='*68}")

    # Load best Phase 2 model for pseudo-label generation
    best_ckpt  = torch.load(tracker.best_path(), map_location='cpu')
    best_sd    = best_ckpt['state_dict']
    best_thresh_before_st = best_ckpt['threshold']

    model.load_state_dict(best_sd)
    ema.module.load_state_dict(best_sd)  # sync EMA to best checkpoint
    print(f"  📥 Loaded best checkpoint (F1={tracker.best_f1():.4f}, "
          f"thresh={best_thresh_before_st:.2f}) for inference")

    pseudo_df = generate_pseudo_labels(model, device, CFG['selftraining_conf'])

    if pseudo_df is not None and len(pseudo_df) > 0:
        selftraining_phase(model, ema, train_df, valid_loader,
                           pseudo_df, fn_col, gr_col, device, tracker)
    else:
        print("  ℹ️  No pseudo-labels generated — skipping self-training")

    # ============================================================
    # SAVE OUTPUTS — always from tracker.best_path() (v2 fix)
    # ============================================================
    print(f"\n{'='*68}")
    print("  SAVING FINAL ARTIFACTS")
    print(f"{'='*68}")

    # v2: load best checkpoint from tracker — NOT ema.module (which is just
    # the final state). This correctly picks the best epoch across both
    # Phase 2 main training AND self-training.
    final_ckpt   = torch.load(tracker.best_path(), map_location='cpu')
    final_sd     = final_ckpt['state_dict']
    final_thresh = final_ckpt['threshold']
    final_epoch  = final_ckpt['epoch']
    final_f1     = final_ckpt['f1']

    best_ema_path = os.path.join(CFG['out_dir'], 'hr_best_ema.pth')
    torch.save(final_sd, best_ema_path)
    print(f"  💾 hr_best_ema.pth  → ep{final_epoch} F1={final_f1:.4f} thresh={final_thresh:.3f}")

    thresh_path = os.path.join(CFG['out_dir'], 'hr_threshold.npy')
    np.save(thresh_path, np.array({'hr': final_thresh}))
    print(f"  💾 hr_threshold.npy → {final_thresh:.3f}")

    # v2: τ_HR = backbone keys only (FPN/head excluded)
    # Task arithmetic is only defined on shared parameters — the backbone
    # that both HR and Vessel specialists diverged from the same θ_base.
    # Including FPN/head deltas would corrupt the Phase 4 TIES merge.
    print(f"\n  🔧 Computing τ_HR = θ_HR − θ_base (backbone only)...")
    theta_base = torch.load(base_path, map_location='cpu')
    theta_hr   = final_sd   # already in memory — no extra disk I/O

    BACKBONE_PREFIX = 'backbone.'
    tau_hr = {}
    skipped_nonbb, skipped_mismatch = 0, 0
    for k in theta_base:
        if not k.startswith(BACKBONE_PREFIX):
            skipped_nonbb += 1; continue
        if k in theta_hr and theta_hr[k].shape == theta_base[k].shape:
            tau_hr[k] = theta_hr[k].float() - theta_base[k].float()
        else:
            skipped_mismatch += 1

    tau_path = os.path.join(CFG['out_dir'], 'tau_hr.pth')
    torch.save(tau_hr, tau_path)
    print(f"  ✅ τ_HR: {len(tau_hr)} backbone tensors")
    print(f"     Excluded: {skipped_nonbb} non-backbone (FPN/HR-head), "
          f"{skipped_mismatch} shape-mismatch")

    del theta_base, theta_hr, final_ckpt, best_ckpt
    gc.collect()

    print(f"\n{'='*68}")
    print(f"  ✅ PHASE 2 COMPLETE")
    print(f"  Best val F1:    {tracker.best_f1():.4f}  (threshold={final_thresh:.3f})")
    print(f"  θ_base:         {base_path}")
    print(f"  θ_HR (EMA):     {best_ema_path}")
    print(f"  HR threshold:   {thresh_path}")
    print(f"  τ_HR:           {tau_path}")
    print(f"  Top-{CFG['topk_ckpts']} ckpts:  {[r[1] for r in tracker.records]}")
    print(f"{'='*68}\n")
    print("  📋 Upload to Kaggle before Phase 3:")
    print(f"     {tau_path}")
    print(f"     {best_ema_path}")
    print(f"     {thresh_path}")


# ============================================================
# CELL 23 — Run
# ============================================================
if __name__ == '__main__':
    main()