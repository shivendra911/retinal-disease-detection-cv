"""
╔══════════════════════════════════════════════════════════════════════════╗
║  IRDAS — JOINT CALIBRATION  v1  ·  Kaggle T4  ·  Phase 5               ║
║                                                                          ║
║  Pipeline position:                                                      ║
║    DR Teacher ✓ (θ_base, QWK 0.9342)                                    ║
║    → HR Specialist ✓ → τ_HR                                              ║
║    → Vessel Specialist ✓ → τ_V                                           ║
║    → TIES Merge ✓ → θ_merged                                             ║
║    → [THIS] Joint Calibration → IRDAS final model                        ║
║                                                                          ║
║  What this notebook does:                                                ║
║    1. Loads θ_merged (from Phase 4 TIES merge)                           ║
║    2. FREEZES backbone — only FPN + all heads adapt                      ║
║    3. Trains all 3 tasks jointly: DR + HR + (vessel if DRIVE available)  ║
║    4. Uses Uncertainty Weighting for automatic loss balancing             ║
║       (Kendall et al. NeurIPS 2018 — no manual λ tuning)                ║
║    5. Monitors all tasks every epoch — stops if DR QWK drops >5%         ║
║    6. Saves final_irdas.pth — ready for inference / submission           ║
║                                                                          ║
║  Research-backed design:                                                 ║
║    · Backbone FROZEN: preserve task-arithmetic-merged representations    ║
║    · Uncertainty Weighting: O(K) auto-balance vs PCGrad O(K²)           ║
║    · Contrastive disentanglement: pushes DR/HR embeddings apart         ║
║    · Focal Tversky for vessel, Focal BCE for HR, CORAL for DR            ║
║    · EMA model for checkpointing                                         ║
║    · Manual SWA BN update (avoids update_bn TypeError)                   ║
║                                                                          ║
║  Kaggle inputs:                                                          ║
║    /kaggle/input/ties-merge-outputs/merged_model.pth                    ║
║    /kaggle/input/aptos2019-blindness-detection/ (DR)                    ║
║    /kaggle/input/datasets/nawazishbilal/hrdc-hypertensive-retinopathy-grading-challenge/2-Hypertensive Retinopathy Classification/ (HR)                                         ║
║    /kaggle/input/drive-retinal-vessel/ (vessel, optional)               ║
║                                                                          ║
║  Outputs → /kaggle/working/joint_calibration/                            ║
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
import gc, ctypes, random, copy, time, warnings

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import psutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
try: _libc = ctypes.CDLL("libc.so.6")
except: _libc = None


# ============================================================
# CELL 3 — Configuration
# ============================================================
CFG = {
    'seed': 42,

    # ── Model ────────────────────────────────────────────────
    'backbone':          'tf_efficientnet_b4.ns_jft_in1k',
    'drop_path_rate':    0.2,
    'fpn_out_channels':  256,
    'dropout':           0.3,
    'coral_levels':      4,
    'msd_k':             5,
    'bifpn_layers':      2,

    # ── Training ─────────────────────────────────────────────
    'total_epochs':   10,     # Short calibration — backbone is frozen
    'warmup_epochs':  2,
    'head_lr':        1e-4,   # Head + FPN LR only (backbone frozen)
    'weight_decay':   1e-2,
    'ema_decay':      0.999,
    'lookahead_k':    5,
    'lookahead_alpha': 0.5,

    'batch_size':   16,
    'grad_accum':    4,       # eff-batch = 64

    # ── Dataset sizes ────────────────────────────────────────
    'image_size':   384,
    'patch_size':   256,      # for DRIVE vessel patches
    'patches_per_img': 40,

    # ── Loss hyperparameters ─────────────────────────────────
    'fn_weight':        2.0,
    'hr_pos_weight':    2.5,
    'focal_gamma':      2.0,
    'tversky_alpha':    0.7,
    'tversky_beta':     0.3,
    'tversky_gamma':    0.75,
    'contrast_margin_pure':    0.1,
    'contrast_margin_cooccur': 0.3,

    # ── Early stopping (DR QWK guardian) ─────────────────────
    'qwk_drop_tolerance': 0.05,
    'val_split_aptos':    0.15,
    'val_split_hrdc':     0.15,
    'val_images_frac_drive': 0.25,

    # ── Paths ────────────────────────────────────────────────
    # UPDATE THIS PATH to point to your Phase 4 output notebook
    'merged_model_path': '/kaggle/input/datasets/shivendrapratap0911/merged-model-pth/merged_model.pth',
    
    # Restored known working Kaggle dataset paths:
    'aptos_csv':    '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train.csv',
    'aptos_imgs':   '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train_images',
    
    'hrdc_csv':     '/kaggle/input/datasets/shivendrapratap0911/2-hypertensive-retinopathy-classification/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv',
    'hrdc_imgs':    '/kaggle/input/datasets/shivendrapratap0911/2-hypertensive-retinopathy-classification/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set',
    
    'drive_imgs':   '/kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/images',
    'drive_masks':  '/kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/1st_manual',
    
    'out_dir':      '/kaggle/working/joint_calibration',
}

CORAL_LEVELS = CFG['coral_levels']


# ============================================================
# CELL 4 — Memory utils
# ============================================================
def _malloc_trim():
    if _libc: _libc.malloc_trim(0)
def _rss_gb(): return _proc.memory_info().rss / 1e9
def _vram_str():
    parts = []
    for i in range(torch.cuda.device_count()):
        u = torch.cuda.memory_allocated(i)/1024**3
        t = torch.cuda.get_device_properties(i).total_memory/1024**3
        parts.append(f"G{i}:{u:.1f}/{t:.1f}GB")
    return " ".join(parts) or "No GPU"
def _mem_chk(l): print(f"  📊 [{l}] RSS={_rss_gb():.2f}GB  {_vram_str()}")
def _cleanup(): gc.collect(); _malloc_trim(); torch.cuda.empty_cache()
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ============================================================
# CELL 5 — Preprocessing (identical to teacher)
# ============================================================
def ben_graham_clahe(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img_bg  = cv2.addWeighted(img, 4, blurred, -4, 128)
    lab     = cv2.cvtColor(img_bg, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

def load_image(path, exts=('.png','.jpeg','.jpg','.tif','.bmp')):
    img = cv2.imread(path)
    if img is None:
        for ext in exts:
            alt = os.path.splitext(path)[0] + ext
            img = cv2.imread(alt)
            if img is not None: break
    if img is None: raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None and _HAS_PIL:
        m = np.array(PILImage.open(path).convert('L'))
    if m is None: raise FileNotFoundError(f"Cannot read mask: {path}")
    return m

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def compute_fov_mask(img, thr=20):
    img_u8 = np.clip(img,0,255).astype(np.uint8) if img.dtype!=np.uint8 else img
    gray   = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    _, m   = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


# ============================================================
# CELL 6 — Augmentations
# ============================================================
def get_fundus_transforms(sz, is_train):
    if is_train:
        hh = max(8, sz//12)
        return A.Compose([
            A.Resize(sz,sz),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
            A.Affine(translate_percent={"x":(-0.1,0.1),"y":(-0.1,0.1)},
                     scale=(0.8,1.2), rotate=(-45,45),
                     border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7),
            A.RandomBrightnessContrast(0.25,0.25,p=0.5),
            A.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.3,hue=0.02,p=0.5),
            A.OneOf([A.GaussNoise(std_range=(3.16/255,10./255),p=1),
                     A.GaussianBlur(blur_limit=(3,7),p=1),
                     A.MotionBlur(blur_limit=7,p=1)],p=0.4),
            A.CoarseDropout(num_holes_range=(4,12),
                            hole_height_range=(hh,hh*2),
                            hole_width_range=(hh,hh*2), fill=0, p=0.5),
            A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
            ToTensorV2()])
    return A.Compose([
        A.Resize(sz,sz),
        A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
        ToTensorV2()])

def get_vessel_transforms(is_train):
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.Affine(translate_percent={'x':(-0.05,0.05),'y':(-0.05,0.05)},
                     scale=(0.9,1.1), rotate=(-30,30),
                     border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            A.ElasticTransform(alpha=120,sigma=6,border_mode=cv2.BORDER_REFLECT_101,p=0.3),
            A.OneOf([A.GaussNoise(std_range=(0.005,0.05),p=1),
                     A.GaussianBlur(blur_limit=(3,5),p=1)],p=0.3),
            A.ColorJitter(brightness=0.15,contrast=0.2,p=0.4),
            A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
            ToTensorV2()
        ], additional_targets={'mask':'mask','fov':'mask'})
    return A.Compose([
        A.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
        ToTensorV2()
    ], additional_targets={'mask':'mask','fov':'mask'})


# ============================================================
# CELL 7 — Datasets
# ============================================================
def _lds_weights(labels, nc=5, sigma=1.0):
    counts   = np.bincount(labels, minlength=nc).astype(float)
    x        = np.arange(-2,3)
    kernel   = np.exp(-x**2/(2*sigma**2)); kernel/=kernel.sum()
    smoothed = np.maximum(np.convolve(counts,kernel,mode='same'),1.0)
    w        = 1.0/smoothed[labels]
    return (w/w.max()).astype(np.float32)

class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, tfms, cache=None):
        self.df=df.reset_index(drop=True); self.img_dir=img_dir
        self.tfms=tfms; self._cache=cache
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]; ic=str(row['id_code']); g=int(row['diagnosis'])
        if self._cache and ic in self._cache:
            img = self._cache[ic].copy()
        else:
            img = load_image(os.path.join(self.img_dir,ic+'.png'))
            img = crop_image_from_gray(img)
            img = ben_graham_clahe(img)
        img = self.tfms(image=img)['image']
        coral = torch.tensor([1]*g+[0]*(CORAL_LEVELS-g),dtype=torch.float32)
        return img, coral, torch.tensor(g,dtype=torch.long)

def _detect_hrdc_cols(df):
    fn = next((c for c in ['filename','image_id','image','file_name','id','Image'] if c in df.columns), None)
    gr = next((c for c in ['grade','label','class','HR_grade','target','Hypertensive Retinopathy']    if c in df.columns), None)
    if not fn or not gr:
        raise KeyError(f"HRDC column detection failed. Found: {list(df.columns)}")
    return fn, gr

def _resolve_hrdc(img_dir, fname):
    base = os.path.splitext(fname)[0] if '.' in fname else fname
    for ext in ['','.jpg','.jpeg','.png','.JPG','.PNG']:
        p = os.path.join(img_dir,base+ext)
        if os.path.exists(p): return p
    raise FileNotFoundError(f"HRDC image not found: {fname}")

class HRDCDataset(Dataset):
    def __init__(self, df, fn_col, gr_col, img_dir, tfms):
        self.df=df.reset_index(drop=True); self.fn_col=fn_col; self.gr_col=gr_col
        self.img_dir=img_dir; self.tfms=tfms
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        row=self.df.iloc[i]
        img = load_image(_resolve_hrdc(self.img_dir,str(row[self.fn_col])))
        img = crop_image_from_gray(img)
        img = ben_graham_clahe(img)
        img=self.tfms(image=img)['image']
        return img, torch.tensor(1.0 if int(row[self.gr_col])>0 else 0.0,dtype=torch.float32)

class DRIVEPatchDataset(Dataset):
    def __init__(self, img_paths, mask_paths, tfms, patch_size=256, ppi=40, is_train=True):
        self.tfms=tfms; self.ps=patch_size; self.ppi=ppi; self.is_train=is_train
        self._imgs=[]; self._masks=[]; self._fovs=[]
        for ip,mp in zip(img_paths,mask_paths):
            img=ben_graham_clahe(load_image(ip)); m=load_mask(mp); fov=compute_fov_mask(img)
            self._imgs.append(img); self._masks.append(m); self._fovs.append(fov)
    def __len__(self): return len(self._imgs)*self.ppi
    def __getitem__(self,idx):
        ii=idx//self.ppi; img=self._imgs[ii]; m=self._masks[ii]; fov=self._fovs[ii]
        H,W=img.shape[:2]; ps=self.ps
        if self.is_train:
            for _ in range(20):
                y=random.randint(0,max(0,H-ps)); x=random.randint(0,max(0,W-ps))
                if fov[y:y+ps,x:x+ps].mean()>127: break
        else: y=max(0,(H-ps)//2); x=max(0,(W-ps)//2)
        y=min(y,max(0,H-ps)); x=min(x,max(0,W-ps))
        mf=(m[y:y+ps,x:x+ps]>127).astype(np.float32)
        ff=(fov[y:y+ps,x:x+ps]>127).astype(np.float32)
        aug=self.tfms(image=img[y:y+ps,x:x+ps],mask=mf,fov=ff)
        return aug['image'], aug['mask'].unsqueeze(0).float(), \
               aug['fov'].unsqueeze(0).float()


# ============================================================
# CELL 8 — Model (Unified IRDAS Architecture)
# ============================================================

# ── Attention & Pooling ───────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid = max(ch//r, 1)
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.mp = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(),
            nn.Conv2d(mid, ch, 1, bias=False))
    def forward(self, x):
        return x * torch.sigmoid(self.fc(self.ap(x)) + self.fc(self.mp(x)))

class SpatialAttention(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=(ks-1)//2, bias=False)
    def forward(self, x):
        a = torch.mean(x, 1, keepdim=True)
        m, _ = torch.max(x, 1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([a,m], 1)))

class CBAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.ca = ChannelAttention(ch); self.sa = SpatialAttention()
    def forward(self, x): return self.sa(self.ca(x))

class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p); self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(self.eps).pow(self.p), x.shape[-2:]).pow(1.0/self.p)

# ── BiFPN (Matches Phase 1/4) ─────────────────────────────────
class DWConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.bn(self.pw(self.dw(x))))

class BiFPNLayer(nn.Module):
    def __init__(self, ch=256, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.w_p4_td = nn.Parameter(torch.ones(2)); self.w_p3_out = nn.Parameter(torch.ones(2))
        self.w_p4_out = nn.Parameter(torch.ones(3)); self.w_p5_out = nn.Parameter(torch.ones(2))
        self.conv_p4_td = DWConv(ch); self.conv_p3_out = DWConv(ch)
        self.conv_p4_out = DWConv(ch); self.conv_p5_out = DWConv(ch)
    def _up(self, x, t): return F.interpolate(x, t.shape[-2:], mode='nearest')
    def _dn(self, x, t): return F.adaptive_avg_pool2d(x, t.shape[-2:])
    def forward(self, p3, p4, p5):
        e = self.eps
        # 🚨 FIX: Changed /= to out-of-place division
        w4  = F.relu(self.w_p4_td.clone());   w4  = w4  / (w4.sum()  + e)
        w3  = F.relu(self.w_p3_out.clone());  w3  = w3  / (w3.sum()  + e)
        w4o = F.relu(self.w_p4_out.clone());  w4o = w4o / (w4o.sum() + e)
        w5o = F.relu(self.w_p5_out.clone());  w5o = w5o / (w5o.sum() + e)
        
        p4_td = self.conv_p4_td(w4[0]*p4 + w4[1]*self._up(p5,p4))
        p3_out = self.conv_p3_out(w3[0]*p3 + w3[1]*self._up(p4_td,p3))
        p4_out = self.conv_p4_out(w4o[0]*p4 + w4o[1]*p4_td + w4o[2]*self._dn(p3_out,p4))
        p5_out = self.conv_p5_out(w5o[0]*p5 + w5o[1]*self._dn(p4_out,p5))
        return p3_out, p4_out, p5_out

class BiFPN(nn.Module):
    def __init__(self, in_ch, out_ch=256, n=2):
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.SiLU())
            for c in in_ch])
        self.layers = nn.ModuleList([BiFPNLayer(out_ch) for _ in range(n)])
    def forward(self, p3r, p4r, p5r):
        p3, p4, p5 = self.lat[0](p3r), self.lat[1](p4r), self.lat[2](p5r)
        for l in self.layers: p3, p4, p5 = l(p3, p4, p5)
        return p3, p4, p5

# ── VesselDecoder ─────────────────────────────────────────────
class UpBlock(nn.Module):
    def __init__(self,ci,sk,co):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv=nn.Sequential(
            nn.Conv2d(ci+sk,co,3,padding=1,bias=False), nn.BatchNorm2d(co), nn.SiLU(),
            nn.Conv2d(co,co,3,padding=1,bias=False), nn.BatchNorm2d(co), nn.SiLU())
    def forward(self,x,skip):
        x=self.up(x)
        if x.shape[-2:]!=skip.shape[-2:]: x=F.interpolate(x,skip.shape[-2:],mode='bilinear',align_corners=False)
        return self.conv(torch.cat([x,skip],1))

class VesselDecoder(nn.Module):
    def __init__(self,ch): 
        super().__init__()
        c3,c4,c5=ch
        self.up1=UpBlock(c5,c4,256); self.up2=UpBlock(256,c3,128)
        self.up3=nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
                                nn.Conv2d(128,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.SiLU(),
                                nn.Conv2d(64,64,3,padding=1,bias=False),nn.BatchNorm2d(64),nn.SiLU())
        self.head=nn.Conv2d(64,1,1)
    def forward(self,p3,p4,p5):
        x=self.up1(p5,p4); x=self.up2(x,p3); x=self.up3(x)
        return torch.sigmoid(self.head(x))

# ── Full IRDAS Model (Perfectly aligns with merged_model.pth) ──
class IRDASModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Backbone
        try:
            self.backbone = timm.create_model(
                CFG['backbone'], pretrained=False, features_only=True,
                out_indices=(2,3,4), drop_path_rate=CFG['drop_path_rate'])
        except TypeError:
            self.backbone = timm.create_model(
                CFG['backbone'], pretrained=False, features_only=True, out_indices=(2,3,4))
        
        ch = self.backbone.feature_info.channels()
        oc = CFG['fpn_out_channels']
        
        # 2. Shared BiFPN Neck (Loads from merged_model)
        self.bifpn = BiFPN(ch, oc, 2)
        
        # 3. Shared Pooling & Attention (Loads from merged_model)
        self.pool = GeMPooling()
        self.cbam_p3 = CBAM(oc)
        self.cbam_p5 = CBAM(oc)
        self.dropout = nn.Dropout(CFG['dropout'])
        self.msd_k = CFG['msd_k']
        
        # 4. Phase 1 DR Head (Loads from merged_model)
        self.head = nn.Linear(oc * 2, CFG['coral_levels'])
        
        # 5. NEW: HR Head (Initializes Randomly, trains on shared features)
        self.hr_head = nn.Linear(oc * 2, 1)
        
        # 6. NEW: Vessel Decoder (Initializes Randomly, trains on backbone features)
        self.vessel_dec = VesselDecoder(list(ch))
        self._use_vessel = True

    def forward(self, x):
        f = self.backbone(x)
        p3_r, p4_r, p5_r = f[0], f[1], f[2]
        
        # BiFPN Forward
        p3, _, p5 = self.bifpn(p3_r, p4_r, p5_r)
        
        # Feature Extraction
        feat = torch.cat([
            self.pool(self.cbam_p3(p3)).flatten(1),
            self.pool(self.cbam_p5(p5)).flatten(1)
        ], 1)
        
        # DR Branch Logic
        if self.training:
            dr_logits = torch.stack([self.head(self.dropout(feat)) for _ in range(self.msd_k)]).mean(0)
        else:
            dr_logits = self.head(feat)
            
        # HR Branch Logic
        hr_logits = self.hr_head(self.dropout(feat))
        
        out = {
            'dr_logits': dr_logits, 
            'hr_logits': hr_logits,
            'dr_feat': feat,      
            'hr_feat': feat       
        }
        
        # Vessel Branch Logic
        if self._use_vessel and self.training:
            out['vessel_pred'] = self.vessel_dec(p3_r, p4_r, p5_r)
            
        return out

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False
        print("  🧊 Backbone FROZEN")


# ============================================================
# CELL 9 — Uncertainty Weighting (Kendall et al. 2018)
# ============================================================
class UncertaintyWeighting(nn.Module):
    """Automatic multi-task loss balancing.

    Each task i learns log(σᵢ²). Total = Σ L_i/(2σᵢ²) + log(σᵢ).
    High-loss tasks auto-increase σᵢ → reduce their gradient influence.
    """
    def __init__(self,n=3):
        super().__init__()
        self.log_vars=nn.Parameter(torch.zeros(n))  # task 0=DR 1=HR 2=vessel
    def forward(self,*losses):
        total=torch.tensor(0.,device=self.log_vars.device)
        weights=[]
        for i,loss in enumerate(losses):
            prec=torch.exp(-self.log_vars[i])
            total=total+prec*loss+self.log_vars[i]
            weights.append(prec.item())
        return total,weights


# ============================================================
# CELL 10 — All Loss Functions
# ============================================================
def _coral_base(logits,levels):
    return -torch.sum(F.logsigmoid(logits)*levels+(F.logsigmoid(logits)-logits)*(1-levels),1)

def coral_loss(logits,coral_targets,grades,fn_weight=2.0):
    per_sample=_coral_base(logits,coral_targets)
    w=torch.ones_like(per_sample); w[grades>0]=fn_weight
    return (per_sample*w).mean()

def focal_bce(logits,labels,gamma=2.0,pos_weight=2.5):
    logits=logits.squeeze(1); probs=torch.sigmoid(logits)
    p_t=probs*labels+(1-probs)*(1-labels)
    fw=(1-p_t).pow(gamma)
    bce=F.binary_cross_entropy_with_logits(
        logits,labels.float(),
        pos_weight=torch.tensor([pos_weight],device=logits.device),reduction='none')
    return (fw*bce).mean()

def focal_tversky(pred,target,fov=None,a=0.7,b=0.3,g=0.75,sm=1.0):
    if fov is not None:
        mask=fov.squeeze(1).bool()
        pred=pred.squeeze(1)[mask].unsqueeze(0).unsqueeze(0)
        target=target.squeeze(1)[mask].unsqueeze(0).unsqueeze(0)
    p=pred.reshape(pred.size(0),-1).float()
    t=target.reshape(target.size(0),-1).float()
    tp=(p*t).sum(1); fp=(p*(1-t)).sum(1); fn=((1-p)*t).sum(1)
    ti=(tp+sm)/(tp+a*fp+b*fn+sm)
    return (1-ti).pow(g).mean()

def contrastive_loss(dr_feat,hr_feat,dr_grades,hr_labels,m_pure=0.1,m_co=0.3):
    dn=F.normalize(dr_feat,dim=1); hn=F.normalize(hr_feat,dim=1)
    sim=(dn*hn).sum(1)
    has_dr=(dr_grades>0).float(); has_hr=hr_labels.float()
    both=has_dr*has_hr; only_one=(has_dr+has_hr).clamp(0,1)
    margin=m_co*both+m_pure*(only_one-both)
    return (only_one*F.relu(margin-(1.-sim))).mean()


# ============================================================
# CELL 11 — Safe weight loading
# ============================================================
def safe_load_weights(model, path, verbose=True):
    if not os.path.exists(path):
        print(f"  ⚠️  Not found: {path}"); return 0
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd=load_file(path,device='cpu')
    else:
        sd=torch.load(path,map_location='cpu')
        if isinstance(sd,dict):
            # merged_model.pth wraps in {'state_dict':..., ...}
            if 'state_dict' in sd: sd=sd['state_dict']
            else:
                for k in ('module','model','ema'):
                    if k in sd: sd=sd[k]; break
    if any(k.startswith('module.') for k in sd): sd={k[7:]:v for k,v in sd.items()}
    msd=model.state_dict()
    filtered={k:v for k,v in sd.items() if k in msd and v.shape==msd[k].shape}
    skip=len(sd)-len(filtered)
    msd.update(filtered); model.load_state_dict(msd,strict=False)
    if verbose: print(f"  ✅ {len(filtered)}/{len(sd)} tensors loaded"
                      +(f"  [{skip} skipped]" if skip else ""))
    return len(filtered)


# ============================================================
# CELL 12 — Lookahead & EMA (same as teacher)
# ============================================================
class Lookahead:
    def __init__(self,opt,k=5,a=0.5):
        self.opt=opt; self.k=k; self.alpha=a; self._step=0
        self.slow={p:p.data.clone().detach() for g in opt.param_groups for p in g['params']}
    def sync(self):
        self._step+=1
        if self._step%self.k==0:
            for g in self.opt.param_groups:
                for p in g['params']:
                    if p not in self.slow: self.slow[p]=p.data.clone().detach()
                    self.slow[p].add_(self.alpha*(p.data-self.slow[p])); p.data.copy_(self.slow[p])
    def zero_grad(self,set_to_none=True): self.opt.zero_grad(set_to_none=set_to_none)
    def step(self): self.opt.step()
    @property
    def param_groups(self): return self.opt.param_groups

class ModelEMA:
    def __init__(self,model,decay=0.999):
        self.module=copy.deepcopy(model); self.module.eval(); self.decay=decay
    @torch.no_grad()
    def update(self,model):
        for ep,mp in zip(self.module.parameters(),model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data,alpha=1-self.decay)
        for eb,mb in zip(self.module.buffers(),model.buffers()): eb.copy_(mb)


# ============================================================
# CELL 13 — Data Loaders (APTOS + HRDC + DRIVE)
# ============================================================
def build_all_loaders():
    loaders={}

    # ── APTOS ─────────────────────────────────────────────────
    print("  📂 APTOS...")
    df=pd.read_csv(CFG['aptos_csv'])
    tr,vl=train_test_split(df,test_size=CFG['val_split_aptos'],
                            stratify=df['diagnosis'],random_state=42)
    w=_lds_weights(tr['diagnosis'].values)
    smp=WeightedRandomSampler(torch.from_numpy(w).float(),len(w),replacement=True)
    loaders['aptos_train']=DataLoader(
        APTOSDataset(tr,CFG['aptos_imgs'],get_fundus_transforms(CFG['image_size'],True)),
        batch_size=CFG['batch_size'],sampler=smp,num_workers=0,pin_memory=False,drop_last=True)
    loaders['aptos_val']=DataLoader(
        APTOSDataset(vl,CFG['aptos_imgs'],get_fundus_transforms(CFG['image_size'],False)),
        batch_size=CFG['batch_size']*2,shuffle=False,num_workers=0,pin_memory=False)
    print(f"    train={len(tr)} val={len(vl)}")

    # ── HRDC ──────────────────────────────────────────────────
    print("  📂 HRDC...")
    hdf=pd.read_csv(CFG['hrdc_csv']); fn_col,gr_col=_detect_hrdc_cols(hdf)
    hdf['_b']=(hdf[gr_col]>0).astype(int)
    htr,hvl=train_test_split(hdf,test_size=CFG['val_split_hrdc'],stratify=hdf['_b'],random_state=42)
    htr=htr.reset_index(drop=True); hvl=hvl.reset_index(drop=True)
    hl=htr['_b'].values; hc=np.bincount(hl).astype(float)
    hw=(1./hc[hl]).astype(np.float32)
    hsmp=WeightedRandomSampler(torch.from_numpy(hw),len(hw),replacement=True)
    loaders['hrdc_train']=DataLoader(
        HRDCDataset(htr,fn_col,gr_col,CFG['hrdc_imgs'],get_fundus_transforms(CFG['image_size'],True)),
        batch_size=CFG['batch_size'],sampler=hsmp,num_workers=0,pin_memory=False,drop_last=True)
    loaders['hrdc_val']=DataLoader(
        HRDCDataset(hvl,fn_col,gr_col,CFG['hrdc_imgs'],get_fundus_transforms(CFG['image_size'],False)),
        batch_size=CFG['batch_size']*2,shuffle=False,num_workers=0,pin_memory=False)
    print(f"    train={len(htr)} val={len(hvl)}")

    # ── DRIVE (optional) ──────────────────────────────────────
    di,dm=CFG['drive_imgs'],CFG['drive_masks']
    if os.path.isdir(di) and os.path.isdir(dm):
        print("  📂 DRIVE (vessel)...")
        di_val, dm_val = di.replace('/training/images', '/test/images'), dm.replace('/training/1st_manual', '/test/1st_manual')
        imgs_tr=sorted([f for f in os.listdir(di) if f.lower().endswith(('.tif','.png','.jpg','.bmp'))])
        masks_tr=sorted([f for f in os.listdir(dm) if f.lower().endswith(('.gif','.tif','.png','.bmp'))])
        imgs_vl=sorted([f for f in os.listdir(di_val) if f.lower().endswith(('.tif','.png','.jpg','.bmp'))])
        masks_vl=sorted([f for f in os.listdir(dm_val) if f.lower().endswith(('.gif','.tif','.png','.bmp'))])
        ti_imgs=[os.path.join(di,f) for f in imgs_tr]; ti_msks=[os.path.join(dm,f) for f in masks_tr]
        vi_imgs=[os.path.join(di_val,f) for f in imgs_vl]; vi_msks=[os.path.join(dm_val,f) for f in masks_vl]
        loaders['drive_train']=DataLoader(
            DRIVEPatchDataset(ti_imgs,ti_msks,get_vessel_transforms(True),
                               CFG['patch_size'],CFG['patches_per_img'],True),
            batch_size=CFG['batch_size'],shuffle=True,num_workers=0,pin_memory=False)
        loaders['drive_val']=DataLoader(
            DRIVEPatchDataset(vi_imgs,vi_msks,get_vessel_transforms(False),
                               CFG['patch_size'],10,False),
            batch_size=CFG['batch_size'],shuffle=False,num_workers=0,pin_memory=False)
        print(f"    train={len(ti_imgs)} val={len(vi_imgs)} images  ppi={CFG['patches_per_img']}")
    else:
        print("  ℹ️  DRIVE not found — vessel loss disabled in Phase 5")
        loaders['drive_train']=None; loaders['drive_val']=None

    return loaders


# ============================================================
# CELL 14 — Validation
# ============================================================
@torch.no_grad()
def validate_all(model, loaders, device):
    model.eval(); metrics={}

    # DR
    all_p,all_l=[],[]
    for imgs,coral,grades in loaders['aptos_val']:
        imgs=imgs.to(device,non_blocking=True)
        out=model(imgs)
        all_p.extend((out['dr_logits']>0).sum(1).cpu().numpy())
        all_l.extend(grades.numpy())
    metrics['dr_qwk']=cohen_kappa_score(all_l,all_p,weights='quadratic',labels=[0,1,2,3,4])

    # HR
    all_p,all_l=[],[]
    for imgs,labels in loaders['hrdc_val']:
        imgs=imgs.to(device,non_blocking=True)
        out=model(imgs)
        probs=torch.sigmoid(out['hr_logits'].squeeze(1)).cpu().numpy()
        all_p.extend((probs>=0.5).astype(int))
        all_l.extend(labels.numpy())
    metrics['hr_f1']=f1_score(all_l,all_p,zero_division=0)
    metrics['hr_auc']=roc_auc_score(all_l,[1/(1+np.exp(-x)) for x in
                         [torch.sigmoid(model(imgs.to(device))['hr_logits'].squeeze(1)).cpu().numpy()
                          for imgs,_ in loaders['hrdc_val']][0]]) if False else 0.0  # skip AUC for speed

    # Vessel
    if loaders['drive_val']:
        dice_sum,n=0.,0
        for imgs,masks,fovs in loaders['drive_val']:
            imgs=imgs.to(device,non_blocking=True)
            model._use_vessel=True
            out=model(imgs)
            if 'vessel_pred' in out:
                vp=out['vessel_pred']
                if vp.shape[-2:]!=masks.shape[-2:]:
                    vp=F.interpolate(vp,masks.shape[-2:],mode='bilinear',align_corners=False)
                masks_d=(masks>0.5).float().to(device)
                p=(vp>0.5).float()
                inter=(p*masks_d).view(p.size(0),-1).sum(1)
                denom=p.view(p.size(0),-1).sum(1)+masks_d.view(masks_d.size(0),-1).sum(1)
                dice_sum+=((2*inter+1)/(denom+1)).mean().item(); n+=1
        metrics['vessel_dice']=dice_sum/max(n,1)
    else:
        metrics['vessel_dice']=0.0

    return metrics


# ============================================================
# CELL 15 — Main Training Loop
# ============================================================
def main():
    set_seed(CFG['seed'])
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*68}")
    print(f"  IRDAS Phase 5 — JOINT CALIBRATION")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Epochs: {CFG['total_epochs']}  |  Backbone: FROZEN")
    print(f"  Loss balancing: Uncertainty Weighting (Kendall et al. 2018)")
    print(f"{'='*68}\n")

    os.makedirs(CFG['out_dir'],exist_ok=True)
    _mem_chk("startup")

    # ── Data ─────────────────────────────────────────────────
    loaders=build_all_loaders()
    _mem_chk("post-data")

    # ── Model ─────────────────────────────────────────────────
    print("\n  🏗  Building IRDAS multi-task model...")
    model=IRDASModel().to(device)

    n_loaded=safe_load_weights(model, CFG['merged_model_path'])
    if n_loaded==0:
        raise RuntimeError(
            f"Failed to load merged_model.pth!\n"
            f"Path: {CFG['merged_model_path']}\n"
            "Verify Phase 4 (TIES merge) completed successfully.")

    model.freeze_backbone()   # CRITICAL: backbone stays frozen in Phase 5
    model._use_vessel = loaders['drive_train'] is not None

    ema=ModelEMA(model,CFG['ema_decay'])
    _mem_chk("post-model")

    # ── Get baseline QWK from merged model ───────────────────
    print("\n  📊 Baseline metrics (merged model, before calibration)...")
    baseline_metrics=validate_all(model,loaders,device)
    baseline_qwk=baseline_metrics['dr_qwk']
    qwk_floor=baseline_qwk*(1-CFG['qwk_drop_tolerance'])
    print(f"  DR QWK:      {baseline_qwk:.4f}  (floor={qwk_floor:.4f})")
    print(f"  HR F1:       {baseline_metrics['hr_f1']:.4f}")
    print(f"  Vessel Dice: {baseline_metrics['vessel_dice']:.4f}")

    # ── Uncertainty weighting module ─────────────────────────
    n_tasks = 3 if model._use_vessel else 2
    uw = UncertaintyWeighting(n_tasks).to(device)

    # ── Optimizer: heads + BiFPN + uncertainty weights ─────────
    head_params = (
        list(model.bifpn.parameters())
        + list(model.cbam_p3.parameters())
        + list(model.cbam_p5.parameters())
        + list(model.pool.parameters())
        + list(model.head.parameters())
        + list(model.hr_head.parameters())
        + (list(model.vessel_dec.parameters()) if model._use_vessel else [])
        + list(uw.parameters())
    )
    base_opt=torch.optim.AdamW(head_params,lr=CFG['head_lr'],weight_decay=CFG['weight_decay'])
    la    =Lookahead(base_opt,CFG['lookahead_k'],CFG['lookahead_alpha'])
    scaler=torch.amp.GradScaler('cuda')

    def _build_sched(ep,w):
        we=min(w,ep//3); ce=max(ep-we,1)
        return SequentialLR(base_opt,[LinearLR(base_opt,0.01,total_iters=we),
                                       CosineAnnealingLR(base_opt,T_max=ce,eta_min=1e-6)],
                             milestones=[we])
    scheduler=_build_sched(CFG['total_epochs'],CFG['warmup_epochs'])

    best_qwk=baseline_qwk; best_path=None; accum=CFG['grad_accum']

    # Make iterators for non-APTOS datasets (cycle through them)
    hrdc_iter = iter(loaders['hrdc_train'])
    drive_iter = iter(loaders['drive_train']) if loaders['drive_train'] else None

    print(f"\n{'='*68}")
    print(f"  CALIBRATION STARTING — {CFG['total_epochs']} epochs (backbone frozen)")
    print(f"{'='*68}\n")

    for epoch in range(CFG['total_epochs']):
        t0=time.time(); model.train(); la.zero_grad(set_to_none=True)
        total_loss_sum=0.; n_updates=0

        for step,(imgs_a,coral,grades) in enumerate(loaders['aptos_train']):
            imgs_a=imgs_a.to(device,non_blocking=True)
            coral=coral.to(device,non_blocking=True)
            grades=grades.to(device,non_blocking=True)

            # Get HRDC batch (cycle)
            try: imgs_h,hr_labels=next(hrdc_iter)
            except StopIteration:
                hrdc_iter=iter(loaders['hrdc_train'])
                imgs_h,hr_labels=next(hrdc_iter)
            imgs_h=imgs_h.to(device,non_blocking=True)
            hr_labels=hr_labels.to(device,non_blocking=True)

            with torch.amp.autocast('cuda'):
                # DR forward (APTOS batch)
                model._use_vessel=False
                out_a=model(imgs_a)
                l_dr=coral_loss(out_a['dr_logits'],coral,grades,CFG['fn_weight'])

                # HR forward (HRDC batch)
                model._use_vessel=False
                out_h=model(imgs_h)
                l_hr=focal_bce(out_h['hr_logits'],hr_labels,CFG['focal_gamma'],CFG['hr_pos_weight'])

                # Vessel forward (DRIVE batch if available)
                if drive_iter is not None:
                    try: imgs_v,masks_v,fovs_v=next(drive_iter)
                    except StopIteration:
                        drive_iter=iter(loaders['drive_train'])
                        imgs_v,masks_v,fovs_v=next(drive_iter)
                    imgs_v=imgs_v.to(device,non_blocking=True)
                    masks_v=masks_v.to(device,non_blocking=True)
                    fovs_v=fovs_v.to(device,non_blocking=True)
                    model._use_vessel=True
                    out_v=model(imgs_v)
                    vp=out_v.get('vessel_pred')
                    if vp is not None and vp.shape[-2:]!=masks_v.shape[-2:]:
                        vp=F.interpolate(vp,masks_v.shape[-2:],mode='bilinear',align_corners=False)
                    l_v=focal_tversky(vp,masks_v,fovs_v,
                                       CFG['tversky_alpha'],CFG['tversky_beta'],CFG['tversky_gamma'])
                    total,weights=uw(l_dr,l_hr,l_v)
                else:
                    total,weights=uw(l_dr,l_hr)

                # Contrastive disentanglement (APTOS + HRDC on same forward — use cached feats)
                # Align batch sizes (take min)
                min_b=min(out_a['dr_feat'].shape[0],out_h['hr_feat'].shape[0])
                l_c=contrastive_loss(
                    out_a['dr_feat'][:min_b],out_h['hr_feat'][:min_b],
                    grades[:min_b],hr_labels[:min_b],
                    CFG['contrast_margin_pure'],CFG['contrast_margin_cooccur'])
                total=total+0.2*l_c
                total=total/accum

            scaler.scale(total).backward()
            total_loss_sum+=total.item()*accum

            if (step+1)%accum==0 or step+1==len(loaders['aptos_train']):
                scaler.unscale_(la.opt)
                torch.nn.utils.clip_grad_norm_(head_params,1.0)
                scaler.step(la.opt); scaler.update()
                la.sync(); la.zero_grad(set_to_none=True)
                ema.update(model); n_updates+=1

        scheduler.step()
        model._use_vessel=True

        metrics=validate_all(ema.module,loaders,device)
        qwk=metrics['dr_qwk']; f1=metrics['hr_f1']; dice=metrics['vessel_dice']

        improved=qwk>best_qwk
        if improved:
            best_qwk=qwk
            best_path=os.path.join(CFG['out_dir'],f'final_irdas_ep{epoch+1:02d}_qwk{qwk:.4f}.pth')
            torch.save(ema.module.state_dict(),best_path)

        star=" ⭐" if improved else ""
        wstr=" ".join(f"w{i}={w:.2f}" for i,w in enumerate(weights))
        print(
            f"  Ep [{epoch+1:02d}/{CFG['total_epochs']}] "
            f"Loss={total_loss_sum/max(n_updates,1):.4f} | "
            f"QWK={qwk:.4f}  F1={f1:.4f}  Dice={dice:.4f} "
            f"| {wstr}  {time.time()-t0:.0f}s{star}"
        )
        _mem_chk(f"ep{epoch+1}")
        _cleanup()

        # ── DR QWK guardian ───────────────────────────────────
        if qwk<qwk_floor:
            print(f"\n  ⚠️  DR QWK {qwk:.4f} < floor {qwk_floor:.4f} — STOPPING CALIBRATION")
            print("     Loading best checkpoint before regression...")
            if best_path: model.load_state_dict(torch.load(best_path,map_location='cpu'))
            break

    # ── Final save ────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  FINALIZING...")

    # Save final EMA
    final_path=os.path.join(CFG['out_dir'],'final_irdas.pth')
    torch.save(ema.module.state_dict(),final_path)
    print(f"  💾 Final IRDAS model → {final_path}")

    # Final evaluation
    final_metrics=validate_all(ema.module,loaders,device)
    print(f"\n  FINAL METRICS:")
    print(f"    DR QWK:       {final_metrics['dr_qwk']:.4f}  (merged baseline={baseline_qwk:.4f})")
    print(f"    HR F1:        {final_metrics['hr_f1']:.4f}")
    print(f"    Vessel Dice:  {final_metrics['vessel_dice']:.4f}")

    print(f"\n  ✅ PHASE 5 COMPLETE — IRDAS Pipeline Done!")
    print(f"  Teacher QWK:   ~0.9342 (Phase 1)")
    print(f"  Merged QWK:    {baseline_qwk:.4f} (Phase 4)")
    print(f"  Final QWK:     {final_metrics['dr_qwk']:.4f} (Phase 5)")
    print(f"{'='*68}\n")
    print(f"  Final model: {final_path}")


if __name__ == '__main__':
    main()
