"""
╔══════════════════════════════════════════════════════════════════════╗
║  IRDAS — DR TEACHER  v6  ·  Kaggle T4 (single GPU)                  ║
║                                                                      ║
║  v5 → v6 fixes:                                                     ║
║  v5 solved RAM (stable ~6 GB). But QWK stuck at 0.33:              ║
║                                                                      ║
║  BUG: CosineAnnealingLR decayed LR to eta_min=1e-6 at end of      ║
║  Stage 1. New schedulers for Stage 2+ read the optimizer's          ║
║  *current* LR (1e-6) as their max_lr. So the model was frozen       ║
║  from epoch 14 onwards — 52 epochs of zero learning.                ║
║                                                                      ║
║  v6 fixes:                                                           ║
║  1. LR RESET at every stage transition (head_lr, backbone_lr)       ║
║  2. Per-stage WARMUP (3 epochs linear ramp) + cosine decay          ║
║  3. Redesigned 3-stage schedule: 256→384→512                        ║
║     (128px too coarse for DR features; fewer transitions = fewer    ║
║      cold starts; better use of 12hr Kaggle budget)                 ║
║  4. Per-class prediction logging for diagnostic                     ║
║                                                                      ║
║  All v5 memory fixes preserved.                                     ║
║  Expected QWK: 0.85+ (was 0.33 in v5 due to LR bug)               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ============================================================
# CELL 1 — Environment Lock (run first, before any imports)
# ============================================================
import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# Prevents OpenMP/BLAS from spawning threads that bloat RSS

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
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from scipy.optimize import minimize
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')
cv2.setNumThreads(0)         # Prevent OpenCV from spawning threads
torch.backends.cudnn.benchmark = True

_proc = psutil.Process(os.getpid())

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None

def _malloc_trim():
    """Ask glibc to return freed pages to the OS."""
    if _libc is not None:
        _libc.malloc_trim(0)

def _rss_mb() -> float:
    return _proc.memory_info().rss / 1e6

def _rss_gb() -> float:
    return _proc.memory_info().rss / 1e9

def _vram_str():
    """GPU VRAM usage string."""
    parts = []
    for i in range(torch.cuda.device_count()):
        u = torch.cuda.memory_allocated(i) / 1024**3
        t = torch.cuda.get_device_properties(i).total_memory / 1024**3
        parts.append(f"G{i}:{u:.1f}/{t:.1f}GB")
    return "  ".join(parts)

def _mem_checkpoint(label):
    """Print a labeled memory checkpoint."""
    print(f"  📊 [{label}] RSS={_rss_gb():.2f} GB  |  {_vram_str()}")

def _aggressive_cleanup():
    """Full memory cleanup — call after every epoch."""
    gc.collect()
    _malloc_trim()
    torch.cuda.empty_cache()

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================================================
# CELL 3 — Configuration
# ============================================================
# v6 schedule: 3 stages (256→384→512) with LR reset + warmup.
#
# Time budget (single T4):
#   Stage 1 (256px, 10 ep × ~20s):    ~200s
#   Stage 2 (384px, 40 ep × ~88s):  ~3,520s
#   Stage 3 (512px, 25 ep × ~180s): ~4,500s
#   Total: ~8,220s ≈ 2.3 hours  (well within 12hr limit)

NUM_CLASSES  = 5
CORAL_LEVELS = 4

CFG = {
    'seed': 42,
    'backbone': 'tf_efficientnet_b4.ns_jft_in1k',
    'drop_path_rate': 0.2,

    # (start_epoch, end_epoch, image_size, batch_size, grad_accum)
    # effective batch = batch_size × accum (single GPU)
    'stages': [
        (0,  10, 256, 16, 8),    # Phase 1: Warmup (frozen backbone)
        (10, 50, 384,  8, 16),   # Phase 2: Main training (unfrozen)
        (50, 75, 512,  4, 32),   # Phase 3: Fine-tune + SWA
    ],

    'cache_size': 512,
    'total_epochs':  75,
    'freeze_epochs': 10,         # Entire Phase 1 is frozen
    'warmup_epochs': 3,          # v6 FIX: warmup at start of each stage

    'head_lr':      1e-3,
    'backbone_lr':  1e-4,
    'weight_decay': 1e-2,

    'swa_start': 65,             # SWA for last 10 epochs of Phase 3
    'swa_lr':    5e-6,

    'bifpn_channels': 256,
    'bifpn_layers':   2,
    'msd_k':   5,
    'dropout': 0.3,

    'mixup_prob':  0.5,
    'mixup_alpha': 0.4,
    'fn_weight':    2.0,
    'loss_alpha':   0.5,
    'label_smooth': 0.10,

    'lookahead_k':     5,
    'lookahead_alpha': 0.5,
    'ema_decay': 0.9997,

    'local_weights': '/kaggle/input/tf-efficientnet-b4-ns-weights/model.safetensors',
    'aptos_csv':  '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train.csv',
    'aptos_imgs': '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train_images',
    'ckpt_dir':   '/kaggle/working/checkpoints',
}


# ============================================================
# CELL 4 — Preprocessing
# ============================================================
def ben_graham_clahe(img):
    """Ben Graham's preprocessing + CLAHE.
    Removes uneven illumination and enhances local contrast."""
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img_bg  = cv2.addWeighted(img, 4, blurred, -4, 128)
    lab     = cv2.cvtColor(img_bg, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def compute_lds_weights(labels, num_classes=5, sigma=1.0):
    """Label Distribution Smoothing weights for imbalanced classes."""
    counts   = np.bincount(labels, minlength=num_classes).astype(float)
    x        = np.arange(-2, 3)
    kernel   = np.exp(-x**2 / (2*sigma**2)); kernel /= kernel.sum()
    smoothed = np.maximum(np.convolve(counts, kernel, mode='same'), 1.0)
    w        = 1.0 / smoothed[labels]
    return w / w.max()


# ============================================================
# CELL 5 — Augmentation Transforms
# ============================================================
def get_transforms(image_size, is_train):
    """Get albumentations transforms.
    A.Resize() handles downscaling from the 512px cache."""
    if is_train:
        gmax = max(10, image_size // 10)
        gmin = max(6,  image_size // 20)
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                     scale=(0.85, 1.15), rotate=(-30, 30),
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.6),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.ColorJitter(saturation=0.2, p=0.5),
            A.OneOf([
                A.GaussNoise(std_range=(3.16/255.0, 7.07/255.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            A.GridDropout(ratio=0.2, unit_size_range=(gmin, gmax),
                          random_offset=True, p=0.4),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ============================================================
# CELL 6 — Dataset + Cache (FIXED: single-resolution)
# ============================================================
class DRDataset(Dataset):
    """DR grading dataset with shared cache lookup.

    FIX: Cache is now a flat dict {id_code: np.ndarray} at a single
    resolution (512px). A.Resize() in the transform pipeline handles
    downscaling to the current stage's target size.

    .copy() is kept because some albumentations transforms (e.g.
    HorizontalFlip via OpenCV) can modify arrays in-place.
    """

    def __init__(self, df, img_dir, transforms, shared_cache=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transforms = transforms
        self._cache = shared_cache   # flat dict, not nested

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = str(self.df.iloc[idx]['id_code'])
        grade   = int(self.df.iloc[idx]['diagnosis'])

        if self._cache is not None:
            img = self._cache[id_code].copy()
        else:
            path = os.path.join(self.img_dir, id_code + '.png')
            img  = ben_graham_clahe(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

        img   = self.transforms(image=img)['image']
        coral = torch.tensor([1]*grade + [0]*(CORAL_LEVELS-grade), dtype=torch.float32)
        return img, coral, torch.tensor(grade, dtype=torch.long)


def build_shared_cache(df, img_dir):
    """Build a SINGLE-resolution cache at 512px.

    FIX: Previously cached at 4 resolutions (128/256/384/512) =
    ~5.2 GB. Now caches only at 512px = ~2.8 GB.
    A.Resize() in the transform pipeline handles downscaling.
    """
    cache_sz = CFG['cache_size']
    t0 = time.time()
    n = len(df)
    print(f"\n📦 Building cache ({n} imgs × {cache_sz}px) | RSS {_rss_gb():.1f} GB")

    cache = {}
    for i in range(n):
        id_code = str(df.iloc[i]['id_code'])
        path    = os.path.join(img_dir, id_code + '.png')
        raw     = cv2.imread(path)
        if raw is None:
            raw = cv2.imread(path.replace('.png', '.jpeg'))
        img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        del raw
        img = cv2.resize(img, (cache_sz, cache_sz), interpolation=cv2.INTER_AREA)
        img = ben_graham_clahe(img)
        cache[id_code] = img
        del img
        if (i+1) % 600 == 0:
            print(f"  {i+1}/{n} | {time.time()-t0:.0f}s | RSS {_rss_gb():.1f} GB")

    total_mb = sum(v.nbytes for v in cache.values()) / 1e6
    print(f"  ✅ Cache ready — {total_mb:.0f} MB ({len(cache)} images) | RSS {_rss_gb():.1f} GB\n")
    return cache


def build_loaders(train_df, valid_df, image_size, batch_size,
                  lds_weights, shared_cache=None):
    """Build train/val DataLoaders.

    FIX: No num_gpus multiplier — batch_size is the actual per-step
    batch size for the single GPU.
    """
    sampler = WeightedRandomSampler(
        torch.from_numpy(lds_weights).float(), len(lds_weights), True)
    train_ds = DRDataset(train_df, CFG['aptos_imgs'],
                         get_transforms(image_size, True), shared_cache)
    valid_ds = DRDataset(valid_df, CFG['aptos_imgs'],
                         get_transforms(image_size, False), shared_cache)
    return (
        DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                   num_workers=0, pin_memory=False, drop_last=True),
        DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False,
                   num_workers=0, pin_memory=False),
    )


# ============================================================
# CELL 7 — Model Architecture
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        mid = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid, 1, bias=False), nn.ReLU(),
            nn.Conv2d(mid, in_planes, 1, bias=False))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=(ks-1)//2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAM(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.ca = ChannelAttention(p)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class DWConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class BiFPNLayer(nn.Module):
    def __init__(self, ch=256, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.w_p4_td  = nn.Parameter(torch.ones(2))
        self.w_p3_out = nn.Parameter(torch.ones(2))
        self.w_p4_out = nn.Parameter(torch.ones(3))
        self.w_p5_out = nn.Parameter(torch.ones(2))
        self.conv_p4_td  = DWConv(ch)
        self.conv_p3_out = DWConv(ch)
        self.conv_p4_out = DWConv(ch)
        self.conv_p5_out = DWConv(ch)

    def _up(self, x, t):
        return F.interpolate(x, t.shape[-2:], mode='nearest')

    def _dn(self, x, t):
        return F.adaptive_avg_pool2d(x, t.shape[-2:])

    def forward(self, p3, p4, p5):
        w4  = F.relu(self.w_p4_td.clone());  w4  = w4  / (w4.sum()  + self.eps)
        w3  = F.relu(self.w_p3_out.clone()); w3  = w3  / (w3.sum()  + self.eps)
        w4o = F.relu(self.w_p4_out.clone()); w4o = w4o / (w4o.sum() + self.eps)
        w5o = F.relu(self.w_p5_out.clone()); w5o = w5o / (w5o.sum() + self.eps)
        p4_td  = self.conv_p4_td(w4[0]*p4  + w4[1]*self._up(p5, p4))
        p3_out = self.conv_p3_out(w3[0]*p3 + w3[1]*self._up(p4_td, p3))
        p4_out = self.conv_p4_out(w4o[0]*p4 + w4o[1]*p4_td + w4o[2]*self._dn(p3_out, p4))
        p5_out = self.conv_p5_out(w5o[0]*p5 + w5o[1]*self._dn(p4_out, p5))
        return p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    def __init__(self, in_ch, out_ch=256, n=2):
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.SiLU())
            for c in in_ch
        ])
        self.layers = nn.ModuleList([BiFPNLayer(out_ch) for _ in range(n)])

    def forward(self, p3r, p4r, p5r):
        p3, p4, p5 = self.lat[0](p3r), self.lat[1](p4r), self.lat[2](p5r)
        for layer in self.layers:
            p3, p4, p5 = layer(p3, p4, p5)
        return p3, p4, p5


class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(self.eps).pow(self.p), x.shape[-2:]
        ).pow(1.0 / self.p)


class MSDNetTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        lw = CFG.get('local_weights', '')
        if os.path.exists(lw):
            self.backbone = timm.create_model(
                CFG['backbone'], pretrained=False,
                features_only=True, out_indices=(2, 3, 4),
                drop_path_rate=CFG['drop_path_rate'])
            if lw.endswith('.safetensors'):
                from safetensors.torch import load_file
                sd = load_file(lw, device='cpu')
            else:
                sd = torch.load(lw, map_location='cpu')
                sd = sd.get('state_dict', sd.get('model', sd))
            self.backbone.load_state_dict(sd, strict=False)
            print("  ✅ Backbone loaded from local weights")
        else:
            print("  ⚠️  Local weights not found — downloading from timm hub")
            self.backbone = timm.create_model(
                CFG['backbone'], pretrained=True,
                features_only=True, out_indices=(2, 3, 4),
                drop_path_rate=CFG['drop_path_rate'])

        ch = self.backbone.feature_info.channels()
        oc = CFG['bifpn_channels']
        self.bifpn    = BiFPN(ch, oc, CFG['bifpn_layers'])
        self.pool     = GeMPooling()
        self.cbam_p3  = CBAM(oc)
        self.cbam_p5  = CBAM(oc)
        self.dropout  = nn.Dropout(CFG['dropout'])
        self.head     = nn.Linear(oc * 2, CORAL_LEVELS)

    def forward(self, x):
        f = self.backbone(x)
        p3, p4, p5 = self.bifpn(f[0], f[1], f[2])
        feat = torch.cat([
            self.pool(self.cbam_p3(p3)).flatten(1),
            self.pool(self.cbam_p5(p5)).flatten(1)
        ], 1)
        if self.training:
            # Multi-sample dropout: average K forward passes with different dropout masks
            return torch.stack([
                self.head(self.dropout(feat)) for _ in range(CFG['msd_k'])
            ]).mean(0)
        return self.head(feat)


# ============================================================
# CELL 8 — Loss Functions
# ============================================================
def _coral_base(logits, levels):
    return -torch.sum(
        F.logsigmoid(logits) * levels +
        (F.logsigmoid(logits) - logits) * (1 - levels), dim=1)


def asymmetric_coral_loss(logits, coral_targets, grades_a,
                           grades_b=None, lam=1.0, fn_weight=2.0):
    per_sample = _coral_base(logits, coral_targets)
    if grades_b is None:
        grades_b = grades_a
    wa = torch.ones_like(per_sample); wa[grades_a > 0] = fn_weight
    wb = torch.ones_like(per_sample); wb[grades_b > 0] = fn_weight
    return (per_sample * (lam * wa + (1 - lam) * wb)).mean()


def coral_to_probs(logits):
    cp = torch.sigmoid(logits)
    probs = torch.clamp(torch.cat([
        1 - cp[:, 0:1],
        cp[:, 0:1] - cp[:, 1:2],
        cp[:, 1:2] - cp[:, 2:3],
        cp[:, 2:3] - cp[:, 3:4],
        cp[:, 3:4]
    ], 1), min=1e-7)
    return probs / probs.sum(1, keepdim=True)


def soft_kappa_loss(logits, ts):
    probs = coral_to_probs(logits)
    n   = NUM_CLASSES
    idx = torch.arange(n, device=logits.device, dtype=torch.float32)
    wm  = ((idx.unsqueeze(0) - idx.unsqueeze(1))**2) / ((n - 1)**2)
    O   = torch.einsum('bi,bj->ij', ts, probs); O = O / (O.sum() + 1e-7)
    E   = torch.outer(ts.sum(0), probs.sum(0));  E = E / (E.sum() + 1e-7)
    return (wm * O).sum() / ((wm * E).sum() + 1e-7)


def _dist_smooth(grades, eps):
    idx  = torch.arange(NUM_CLASSES, device=grades.device, dtype=torch.float32)
    dist = torch.abs(idx.unsqueeze(0) - grades.unsqueeze(1).float())
    sw   = torch.exp(-dist); sw = sw / sw.sum(1, keepdim=True)
    return F.one_hot(grades, NUM_CLASSES).float() * (1 - eps) + sw * eps


def combined_loss(logits, coral_targets, grades_a, grades_b, lam):
    a, eps = CFG['loss_alpha'], CFG['label_smooth']
    cl = asymmetric_coral_loss(
        logits, coral_targets, grades_a, grades_b, lam, CFG['fn_weight'])
    sa = _dist_smooth(grades_a, eps)
    s  = lam * sa + (1 - lam) * _dist_smooth(grades_b, eps) if lam < 1.0 else sa
    return a * cl + (1 - a) * soft_kappa_loss(logits, s)


def ordinal_mixup(imgs, coral_targets, grades, prob=0.5, alpha=0.4):
    if random.random() > prob:
        return imgs, coral_targets, grades, grades, 1.0
    lam = max(float(np.random.beta(alpha, alpha)), 0.5)
    B, g = imgs.size(0), grades.cpu().numpy()
    mix_idx = np.arange(B)
    for i in range(B):
        cands = np.where((np.abs(g - g[i]) <= 1) & (np.arange(B) != i))[0]
        if len(cands):
            mix_idx[i] = np.random.choice(cands)
    t = torch.from_numpy(mix_idx).to(imgs.device)
    return (lam * imgs + (1 - lam) * imgs[t],
            lam * coral_targets + (1 - lam) * coral_targets[t],
            grades, grades[t], lam)


# ============================================================
# CELL 9 — Training Utilities (Lookahead, EMA)
# ============================================================
class Lookahead:
    """Lookahead optimizer wrapper (Zhang et al., 2019)."""
    def __init__(self, opt, k=5, alpha=0.5):
        self.opt = opt
        self.k = k
        self.alpha = alpha
        self._step = 0
        self.slow = {p: p.data.clone().detach()
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

    def zero_grad(self, set_to_none=True):
        self.opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.opt.param_groups

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd)


class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.9997):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ep, mp in zip(self.module.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)
        for eb, mb in zip(self.module.buffers(), model.buffers()):
            eb.copy_(mb)


def predict_grade(logits):
    return (logits > 0).sum(1).cpu().numpy()


def predict_continuous(logits):
    return torch.sigmoid(logits).sum(1).cpu().numpy()


def optimize_thresholds(cont_preds, targets):
    def neg_qwk(t):
        p = np.digitize(cont_preds, np.sort(t)).clip(0, NUM_CLASSES - 1)
        try:
            return -cohen_kappa_score(targets, p, weights='quadratic')
        except Exception:
            return 0.0
    r = minimize(neg_qwk, [0.5, 1.5, 2.5, 3.5], method='Nelder-Mead',
                 options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000})
    t = np.sort(r.x)
    print(f"  Optimised thresholds: {t.round(4)}")
    return t


def tta_predict(model, loader, device):
    """3-fold TTA: original + H-flip + V-flip."""
    model.eval()
    all_cont, all_tgt = [], []
    with torch.no_grad():
        for imgs, _, grades in loader:
            imgs = imgs.to(device, non_blocking=True)
            avg = (model(imgs) +
                   model(torch.flip(imgs, [3])) +
                   model(torch.flip(imgs, [2]))) / 3
            all_cont.extend(predict_continuous(avg))
            all_tgt.extend(grades.numpy())
    return np.array(all_cont), np.array(all_tgt)


# ============================================================
# CELL 10 — Train / Validate One Epoch
# ============================================================
def train_one_epoch(model, loader, la, ema, scaler, device, accum_steps):
    model.train()
    total_loss = 0.0
    la.zero_grad(set_to_none=True)
    n_updates = 0
    rss_start = _rss_mb()

    for step, (imgs, coral_targets, grades) in enumerate(loader):
        imgs          = imgs.to(device, non_blocking=True)
        coral_targets = coral_targets.to(device, non_blocking=True)
        grades        = grades.to(device, non_blocking=True)

        imgs, coral_targets, grades_a, grades_b, lam = ordinal_mixup(
            imgs, coral_targets, grades, CFG['mixup_prob'], CFG['mixup_alpha'])

        with torch.amp.autocast('cuda'):
            logits = model(imgs)
            loss   = combined_loss(logits, coral_targets, grades_a, grades_b, lam)
            loss   = loss / accum_steps

        scaler.scale(loss).backward()
        total_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0 or step + 1 == len(loader):
            scaler.unscale_(la.opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(la.opt)
            scaler.update()
            la.sync()
            la.zero_grad(set_to_none=True)
            ema.update(model)
            n_updates += 1

    rss_end = _rss_mb()
    print(f"   train RSS delta: {rss_end - rss_start:+.0f} MB")
    return total_loss / max(n_updates, 1)


def validate_one_epoch(model, loader, device):
    model.eval()
    all_disc, all_cont, all_tgt = [], [], []
    val_loss = 0.0
    rss_start = _rss_mb()

    with torch.no_grad():
        for imgs, coral_targets, grades in loader:
            imgs          = imgs.to(device, non_blocking=True)
            coral_targets = coral_targets.to(device, non_blocking=True)
            grades_dev    = grades.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                logits    = model(imgs)
                val_loss += asymmetric_coral_loss(
                    logits, coral_targets, grades_dev,
                    fn_weight=CFG['fn_weight']).item()
            all_disc.extend(predict_grade(logits))
            all_cont.extend(predict_continuous(logits))
            all_tgt.extend(grades.numpy())

    rss_end = _rss_mb()
    print(f"   val RSS delta: {rss_end - rss_start:+.0f} MB")
    qwk = cohen_kappa_score(all_tgt, all_disc, weights='quadratic')
    return qwk, val_loss / len(loader), np.array(all_cont), np.array(all_tgt), np.array(all_disc)


# ============================================================
# CELL 11 — Main Training Loop (v6: LR reset + warmup + 3 stages)
# ============================================================
def _build_stage_scheduler(optimizer, stage_epochs, warmup_epochs):
    """Build a warmup + cosine decay scheduler for one stage.

    v6 FIX: Each stage gets a FRESH scheduler that starts from the
    optimizer's current LR (which we explicitly reset before calling
    this function). This prevents the LR collapse bug from v5.
    """
    warmup_ep = min(warmup_epochs, stage_epochs // 3)
    cosine_ep = max(stage_epochs - warmup_ep, 1)

    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_ep)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_ep, eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_ep])


def _log_class_distribution(all_disc, all_tgt):
    """Print per-class prediction counts vs ground truth."""
    pred_counts = Counter(all_disc)
    true_counts = Counter(all_tgt)
    print(f"   Class dist — True: {dict(sorted(true_counts.items()))}")
    print(f"                Pred: {dict(sorted(pred_counts.items()))}")


def main():
    set_seed(CFG['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*65}")
    print(f"  DR TEACHER v6 — LR-FIXED + MEMORY-FIXED BUILD")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  System RAM: {psutil.virtual_memory().total/1e9:.1f} GB")
    print(f"  Stages: {' → '.join(str(s[2])+'px' for s in CFG['stages'])}")
    print(f"{'='*65}\n")

    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    _mem_checkpoint("startup")

    # ── Data split ───────────────────────────────────────────────
    full_df = pd.read_csv(CFG['aptos_csv'])
    train_df, valid_df = train_test_split(
        full_df, test_size=0.15, stratify=full_df['diagnosis'], random_state=42)
    lds_weights = compute_lds_weights(train_df['diagnosis'].values)

    print(f"  Train: {len(train_df)} | Val: {len(valid_df)}")
    print(f"  Class distribution: {dict(train_df['diagnosis'].value_counts().sort_index())}")

    # ── Cache (single resolution) ────────────────────────────────
    shared_cache = build_shared_cache(
        pd.concat([train_df, valid_df], ignore_index=True), CFG['aptos_imgs'])
    _aggressive_cleanup()
    _mem_checkpoint("post-cache")

    # ── Model (NO DataParallel) ──────────────────────────────────
    model = MSDNetTeacher().to(device)
    ema   = ModelEMA(model, CFG['ema_decay'])

    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n  Model: {total_p:.1f}M params")
    _mem_checkpoint("post-model")

    # ── Optimizer (head only initially) ──────────────────────────
    head_params = [p for n, p in model.named_parameters() if 'backbone' not in n]
    base_opt    = torch.optim.AdamW(
        head_params, lr=CFG['head_lr'], weight_decay=CFG['weight_decay'])
    la     = Lookahead(base_opt, CFG['lookahead_k'], CFG['lookahead_alpha'])
    scaler = torch.amp.GradScaler('cuda')

    for p in model.backbone.parameters():
        p.requires_grad = False
    print(f"  🧊 Backbone FROZEN for first {CFG['freeze_epochs']} epochs")
    _mem_checkpoint("post-optimizer")

    # ── SWA: DEFERRED ────────────────────────────────────────────
    swa_model = None
    swa_sched = None

    best_qwk, best_thresholds = 0.0, np.array([0.5, 1.5, 2.5, 3.5])
    backbone_unfrozen = False
    current_stage = -1
    train_loader = valid_loader = scheduler = None

    print(f"\n{'='*65}")
    print(f"  STARTING TRAINING — {CFG['total_epochs']} epochs")
    print(f"{'='*65}\n")

    for epoch in range(CFG['total_epochs']):
        t_ep = time.time()

        # ── Stage transition ─────────────────────────────────────
        stage_idx = next(i for i, (s, e, *_) in enumerate(CFG['stages']) if s <= epoch < e)
        if stage_idx != current_stage:
            current_stage = stage_idx
            _, _, img_sz, bs, accum = CFG['stages'][stage_idx]
            eff = bs * accum
            print(f"\n{'─'*65}")
            print(f"  📐 Stage {stage_idx+1} | {img_sz}px | bs={bs} accum={accum} eff={eff}")
            print(f"{'─'*65}")

            # Free old loaders before creating new ones
            del train_loader, valid_loader
            _aggressive_cleanup()

            train_loader, valid_loader = build_loaders(
                train_df, valid_df, img_sz, bs,
                lds_weights, shared_cache=shared_cache)

            # ── v6 FIX: RESET LR before creating scheduler ──────
            # Without this, CosineAnnealingLR inherits the decayed
            # LR from the previous stage (1e-6) as its max_lr.
            base_opt.param_groups[0]['lr'] = CFG['head_lr']
            if backbone_unfrozen and len(base_opt.param_groups) > 1:
                base_opt.param_groups[1]['lr'] = CFG['backbone_lr']
            bb_lr_str = f"  backbone={base_opt.param_groups[1]['lr']:.1e}" if backbone_unfrozen else ""
            print(f"  🔄 LR reset: head={base_opt.param_groups[0]['lr']:.1e}{bb_lr_str}")

            stage_epochs = CFG['stages'][stage_idx][1] - CFG['stages'][stage_idx][0]
            scheduler = _build_stage_scheduler(
                base_opt, stage_epochs, CFG['warmup_epochs'])
            _mem_checkpoint(f"stage-{stage_idx+1}-init")

        # ── Backbone unfreeze ────────────────────────────────────
        if epoch == CFG['freeze_epochs'] and not backbone_unfrozen:
            backbone_unfrozen = True
            torch.cuda.empty_cache()
            print(f"\n  🔥 Backbone UNFROZEN at epoch {epoch+1}")
            for p in model.backbone.parameters():
                p.requires_grad = True
            base_opt.add_param_group({
                'params': list(model.backbone.parameters()),
                'lr': CFG['backbone_lr'],
                'weight_decay': CFG['weight_decay']
            })
            # Rebuild scheduler with backbone param group included
            remaining = CFG['stages'][current_stage][1] - epoch
            scheduler = _build_stage_scheduler(
                base_opt, remaining, CFG['warmup_epochs'])
            _mem_checkpoint("unfreeze")

        # ── SWA activation (DEFERRED from startup) ───────────────
        if epoch == CFG['swa_start'] and swa_model is None:
            print(f"\n  🌀 SWA activated at epoch {epoch+1}")
            swa_model = AveragedModel(model)
            swa_sched = SWALR(base_opt, swa_lr=CFG['swa_lr'], anneal_epochs=5)
            _mem_checkpoint("swa-init")

        accum = CFG['stages'][current_stage][4]
        phase_str = ("🧊 Frozen"    if epoch < CFG['freeze_epochs'] else
                     "🌀 SWA"       if epoch >= CFG['swa_start']    else
                     "🔥 Fine-tune")

        # ── Train ────────────────────────────────────────────────
        train_loss = train_one_epoch(model, train_loader, la, ema, scaler, device, accum)

        # ── Validate (using EMA model directly — NO DataParallel wrapper) ──
        qwk, val_loss, val_cont, val_tgt, val_disc = validate_one_epoch(
            ema.module, valid_loader, device)

        # ── Per-class diagnostic (every 10 epochs) ────────────────
        if epoch % 10 == 0:
            _log_class_distribution(val_disc.tolist(), val_tgt.astype(int).tolist())

        # ── Threshold optimization ───────────────────────────────
        if epoch >= 20 and epoch % 5 == 0:
            opt_t = optimize_thresholds(val_cont, val_tgt)
            opt_p = np.digitize(val_cont, opt_t).clip(0, NUM_CLASSES - 1)
            opt_qwk = cohen_kappa_score(val_tgt, opt_p, weights='quadratic')
            if opt_qwk > qwk:
                qwk = opt_qwk
                best_thresholds = opt_t.copy()

        # ── SWA / scheduler step ─────────────────────────────────
        if epoch >= CFG['swa_start'] and swa_model is not None:
            swa_model.update_parameters(model)
            swa_sched.step()
        elif scheduler is not None:
            scheduler.step()

        # ── Checkpointing ────────────────────────────────────────
        star = ""
        if qwk > best_qwk and epoch < CFG['swa_start']:
            best_qwk = qwk
            best_thresholds = best_thresholds.copy()
            torch.save(model.state_dict(), f"{CFG['ckpt_dir']}/best_dr_teacher.pth")
            torch.save(ema.module.state_dict(), f"{CFG['ckpt_dir']}/best_ema_teacher.pth")
            np.save(f"{CFG['ckpt_dir']}/best_thresholds.npy", best_thresholds)
            star = "  ⭐ NEW BEST"

        # ── Epoch logging ────────────────────────────────────────
        lr_h = base_opt.param_groups[0]['lr']
        lr_b = base_opt.param_groups[1]['lr'] if backbone_unfrozen else 0.0
        elapsed = time.time() - t_ep
        print(
            f"  Ep [{epoch+1:03d}/{CFG['total_epochs']}] {phase_str}  "
            f"lr_h={lr_h:.2e} lr_b={lr_b:.2e}  |  "
            f"TrLoss={train_loss:.4f} VaLoss={val_loss:.4f}  |  "
            f"EMA QWK={qwk:.4f} (best={best_qwk:.4f})  "
            f"[{_vram_str()}]  |  {elapsed:.0f}s{star}"
        )
        _mem_checkpoint(f"epoch-{epoch+1}-end")

        # ── AGGRESSIVE CLEANUP every epoch (FIX #5) ─────────────
        _aggressive_cleanup()

    # ══════════════════════════════════════════════════════════════
    # POST-TRAINING
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  POST-TRAINING FINALIZATION")
    print(f"{'='*65}")

    # ── Update SWA batch norms ───────────────────────────────────
    if swa_model is not None:
        swa_ds = DRDataset(train_df, CFG['aptos_imgs'],
                           get_transforms(CFG['stages'][-1][2], False),
                           shared_cache)
        swa_loader = DataLoader(swa_ds, batch_size=CFG['stages'][-1][3],
                                num_workers=0, pin_memory=False)
        torch.optim.swa_utils.update_bn(swa_loader, swa_model, device=device)
        torch.save(swa_model.module.state_dict(), f"{CFG['ckpt_dir']}/swa_final.pth")
        del swa_ds, swa_loader

    torch.save(ema.module.state_dict(), f"{CFG['ckpt_dir']}/ema_final.pth")

    # ── Final TTA evaluation ─────────────────────────────────────
    _, val_loader_f = build_loaders(
        train_df, valid_df, CFG['stages'][-1][2],
        CFG['stages'][-1][3], lds_weights, shared_cache)
    cp, tgt = tta_predict(ema.module, val_loader_f, device)
    opt_t   = optimize_thresholds(cp, tgt)
    opt_p   = np.digitize(cp, opt_t).clip(0, NUM_CLASSES - 1)
    tta_qwk = cohen_kappa_score(tgt, opt_p, weights='quadratic')
    np.save(f"{CFG['ckpt_dir']}/optimal_thresholds.npy", opt_t)

    print(f"\n{'='*65}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"  Best pre-SWA EMA QWK:  {best_qwk:.4f}")
    print(f"  Final TTA QWK:         {tta_qwk:.4f}")
    print(f"  RSS final:             {_rss_gb():.1f} GB")
    print(f"{'='*65}\n")


# ============================================================
# CELL 12 — Run Training
# ============================================================
if __name__ == '__main__':
    main()
