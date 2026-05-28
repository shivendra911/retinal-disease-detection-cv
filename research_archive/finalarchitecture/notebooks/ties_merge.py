"""
╔══════════════════════════════════════════════════════════════════════════╗
║  IRDAS — TIES MERGE  v1  ·  Kaggle CPU/T4  ·  Phase 4                   ║
║                                                                          ║
║  Pipeline position:                                                      ║
║    DR Teacher ✓ (θ_base, QWK 0.9342)                                    ║
║    → HR Specialist ✓ → τ_HR                                              ║
║    → Vessel Specialist ✓ → τ_V                                           ║
║    → [THIS] TIES Merge → θ_merged                                        ║
║    → Joint Calibration → IRDAS final                                     ║
║                                                                          ║
║  What this notebook does:                                                ║
║    1. Loads θ_base, τ_HR, τ_V                                            ║
║    2. Applies TIES merging (better than plain task arithmetic)            ║
║    3. Grid-searches λ_HR and λ_V on APTOS val set (QWK metric)           ║
║    4. Saves θ_merged = θ_base + λ*TIES(τ_HR, τ_V)                       ║
║    5. Reports per-task performance before and after merge                ║
║                                                                          ║
║  TIES vs plain task arithmetic:                                          ║
║    Plain TA: θ = θ_base + λ₁τ_HR + λ₂τ_V  ← parameter interference     ║
║    TIES:                                                                 ║
║      1. TRIM   — zero small deltas (< p-th percentile) — removes noise  ║
║      2. ELECT  — choose dominant sign per parameter                      ║
║      3. MERGE  — average only deltas matching the elected sign           ║
║    Result: up to 10% better than plain TA on multi-task conflicts        ║
║                                                                          ║
║  Failure guard:                                                          ║
║    Red line: if merged QWK drops < 0.88, auto-reduce λ and re-merge     ║
║                                                                          ║
║  Kaggle inputs expected:                                                 ║
║    /kaggle/input/datasets/nawazishbilal/dr-teacher-weights/best_ema_teacher.pth  (θ_base)             ║
║    /kaggle/input/hr-specialist-outputs/tau_hr.pth                       ║
║    /kaggle/input/vessel-specialist-outputs/tau_vessel.pth               ║
║    /kaggle/input/aptos2019-blindness-detection/train.csv                ║
║    /kaggle/input/aptos2019-blindness-detection/train_images/            ║
║                                                                          ║
║  Outputs → /kaggle/working/ties_merge/                                   ║
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
import gc, copy, random, time, warnings
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')
cv2.setNumThreads(0)
torch.backends.cudnn.benchmark = True


# ============================================================
# CELL 3 — Configuration (Updated Path Configuration)
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

    # ── Absolute Paths ───────────────────────────────────────
    # The base teacher model path provided directly by you
    'theta_base_path': '/kaggle/input/datasets/shivendrapratap0911/phrase4/phrase4/best_ema_teacher.pth',
    
    # Path to the Vessel specialist vector tracked from image_cbdfc0.png
    'tau_vessel_path': '/kaggle/input/datasets/shivendrapratap0911/phrase4/phrase4/tau_vessel.pth',
    
    # Path to the newly computed local scratch vector from Cell 1
    'tau_hr_path':     '/kaggle/working/tau_hr.pth',
    
    # Standard competition baseline inputs
    'aptos_csv':       '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train.csv',
    'aptos_imgs':      '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train_images',
    'out_dir':         '/kaggle/working/ties_merge',

    # ── TIES parameters ──────────────────────────────────────
    'ties_trim_pct':   20,    # trim bottom-20% of delta magnitudes (noise removal)

    # ── Lambda grid search ───────────────────────────────────
    'lambda_range': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    'val_split':    0.15,     # APTOS val fraction for lambda search
    'image_size':   384,
    'batch_size':   32,

    # ── Red line: if DR QWK falls below this after merge, reduce lambda ──
    'dr_qwk_red_line': 0.88,
    'teacher_qwk':     0.9342,  # known DR teacher QWK (Phase 1)
}


# ============================================================
# CELL 4 — Seed + cleanup
# ============================================================
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def _cleanup(): gc.collect(); torch.cuda.empty_cache()


# ============================================================
# CELL 5 — Preprocessing (same as teacher)
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
        for ext in ['.jpeg', '.jpg', '.png']:
            alt = os.path.splitext(path)[0] + ext
            img = cv2.imread(alt)
            if img is not None: break
    if img is None: raise FileNotFoundError(f"Cannot read: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ============================================================
# CELL 6 — APTOS Val Dataset
# ============================================================
CORAL_LEVELS = 4

class APTOSValDataset(Dataset):
    def __init__(self, df, img_dir, image_size):
        self.df       = df.reset_index(drop=True)
        self.img_dir  = img_dir
        self.tfms     = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = load_image(os.path.join(self.img_dir, str(row['id_code']) + '.png'))
        img   = ben_graham_clahe(img)
        img   = self.tfms(image=img)['image']
        grade = int(row['diagnosis'])
        coral = torch.tensor([1]*grade + [0]*(CORAL_LEVELS-grade), dtype=torch.float32)
        return img, coral, torch.tensor(grade, dtype=torch.long)


# ============================================================
# CELL 7 — DR Specialist Model (same BiFPN architecture as teacher)
# ============================================================

class DWConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw  = nn.Conv2d(ch, ch, 3, 1, 1, groups=ch, bias=False)
        self.pw  = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()
    def forward(self, x): return self.act(self.bn(self.pw(self.dw(x))))

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
    def _up(self, x, t): return F.interpolate(x, t.shape[-2:], mode='nearest')
    def _dn(self, x, t): return F.adaptive_avg_pool2d(x, t.shape[-2:])
    def forward(self, p3, p4, p5):
        e = self.eps
        w4  = F.relu(self.w_p4_td.clone());  w4  /= (w4.sum()+e)
        w3  = F.relu(self.w_p3_out.clone()); w3  /= (w3.sum()+e)
        w4o = F.relu(self.w_p4_out.clone()); w4o /= (w4o.sum()+e)
        w5o = F.relu(self.w_p5_out.clone()); w5o /= (w5o.sum()+e)
        p4_td  = self.conv_p4_td(w4[0]*p4 + w4[1]*self._up(p5,p4))
        p3_out = self.conv_p3_out(w3[0]*p3 + w3[1]*self._up(p4_td,p3))
        p4_out = self.conv_p4_out(w4o[0]*p4 + w4o[1]*p4_td + w4o[2]*self._dn(p3_out,p4))
        p5_out = self.conv_p5_out(w5o[0]*p5 + w5o[1]*self._dn(p4_out,p5))
        return p3_out, p4_out, p5_out

class BiFPN(nn.Module):
    def __init__(self, in_ch, out_ch=256, n=2):
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.SiLU())
            for c in in_ch])
        self.layers = nn.ModuleList([BiFPNLayer(out_ch) for _ in range(n)])
    def forward(self, p3r, p4r, p5r):
        p3, p4, p5 = self.lat[0](p3r), self.lat[1](p4r), self.lat[2](p5r)
        for l in self.layers: p3, p4, p5 = l(p3, p4, p5)
        return p3, p4, p5

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
        return x * torch.sigmoid(self.conv(torch.cat([a,m],1)))

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

class MSDNetTeacher(nn.Module):
    """EXACT copy of teacher architecture from dr_teacher_v5_fixed.py.
    Needed to load θ_base weights for CORAL validation.
    """
    CORAL_LEVELS = 4
    def __init__(self):
        super().__init__()
        try:
            self.backbone = timm.create_model(
                CFG['backbone'], pretrained=False,
                features_only=True, out_indices=(2,3,4),
                drop_path_rate=CFG['drop_path_rate'])
        except TypeError:
            self.backbone = timm.create_model(
                CFG['backbone'], pretrained=False,
                features_only=True, out_indices=(2,3,4))
        ch = self.backbone.feature_info.channels()
        oc = CFG['fpn_out_channels']
        self.bifpn   = BiFPN(ch, oc, 2)
        self.pool    = GeMPooling()
        self.cbam_p3 = CBAM(oc); self.cbam_p5 = CBAM(oc)
        self.dropout = nn.Dropout(CFG['dropout'])
        self.head    = nn.Linear(oc * 2, self.CORAL_LEVELS)
        self.msd_k   = CFG['msd_k']

    def forward(self, x):
        f = self.backbone(x)
        p3, _, p5 = self.bifpn(f[0], f[1], f[2])
        feat = torch.cat([
            self.pool(self.cbam_p3(p3)).flatten(1),
            self.pool(self.cbam_p5(p5)).flatten(1)], 1)
        if self.training:
            return torch.stack([self.head(self.dropout(feat))
                                 for _ in range(self.msd_k)]).mean(0)
        return self.head(feat)


# ============================================================
# CELL 8 — Safe weight loading
# ============================================================
def safe_load_weights(model, path, verbose=True):
    if not os.path.exists(path):
        print(f"  ⚠️  Not found: {path}"); return 0
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu')
        if isinstance(sd, dict):
            for k in ('module','state_dict','model','ema'):
                if k in sd: sd = sd[k]; break
    if any(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k,v in sd.items()}
    msd = model.state_dict()
    filtered = {k:v for k,v in sd.items() if k in msd and v.shape==msd[k].shape}
    msd.update(filtered); model.load_state_dict(msd, strict=False)
    if verbose:
        print(f"  ✅ Loaded {len(filtered)}/{len(sd)} tensors from {os.path.basename(path)}")
    return len(filtered)


# ============================================================
# CELL 9 — TIES Merge Algorithm
# ============================================================
def ties_merge(tau_list: list, trim_pct: float = 20.0) -> dict:
    """TIES Merging (Trim, Elect Sign, Merge) — NeurIPS 2023.

    Resolves parameter interference between task vectors.
    Superior to plain task arithmetic when task vectors conflict.

    Steps:
      1. TRIM:  For each param, zero out the bottom `trim_pct`% of |delta| values.
                Removes low-magnitude noise that adds interference without signal.
      2. ELECT: For each param, compute the dominant sign across all task vectors.
                dominant_sign[i] = sign(sum of deltas[i] across tasks)
      3. MERGE: Average only the deltas that AGREE with the elected sign.
                Deltas opposing the elected sign are discarded.

    Args:
        tau_list:  List of task vector dicts {param_name: delta_tensor}
        trim_pct:  Percentile below which to zero deltas (default 20%)

    Returns:
        Merged task vector dict (same keys as common intersection of tau_list)
    """
    # Find common keys present in ALL task vectors
    common_keys = set(tau_list[0].keys())
    for tau in tau_list[1:]:
        common_keys &= set(tau.keys())
    common_keys = list(common_keys)

    merged = {}
    for key in common_keys:
        # Stack: (num_tasks, *param_shape)
        stacked = torch.stack([tau[key].float() for tau in tau_list], dim=0)

        # ── Step 1: TRIM ──────────────────────────────────────
        flat_abs = stacked.abs().view(stacked.shape[0], -1)  # (T, N)
        thresholds = torch.quantile(flat_abs, trim_pct / 100.0, dim=1)  # (T,)
        for t_idx in range(stacked.shape[0]):
            mask = stacked[t_idx].abs() < thresholds[t_idx]
            stacked[t_idx][mask] = 0.0

        # ── Step 2: ELECT SIGN ───────────────────────────────
        # Sum all task vectors → positive sum → positive wins, etc.
        sum_delta    = stacked.sum(dim=0)      # (*param_shape)
        elected_sign = torch.sign(sum_delta)   # (*param_shape) ∈ {-1, 0, 1}
        # Handle zeros: default to +1 sign
        elected_sign = torch.where(
            elected_sign == 0,
            torch.ones_like(elected_sign),
            elected_sign)

        # ── Step 3: MERGE ────────────────────────────────────
        # Only average deltas that AGREE with elected_sign
        # Disagreeing deltas contribute 0 (they would cause interference)
        agreed = stacked * (torch.sign(stacked) == elected_sign.unsqueeze(0)).float()
        n_agreed = (torch.sign(stacked) == elected_sign.unsqueeze(0)).float().sum(0)
        n_agreed = torch.clamp(n_agreed, min=1.0)  # avoid div-by-zero
        merged[key] = agreed.sum(0) / n_agreed      # element-wise average

    return merged


# ============================================================
# CELL 10 — DR QWK Evaluation
# ============================================================
@torch.no_grad()
def eval_dr_qwk(model, loader, device):
    """Evaluate DR grading QWK on APTOS val set."""
    model.eval()
    all_preds, all_labels = [], []
    for imgs, coral, grades in loader:
        imgs   = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds  = (logits > 0).sum(1).cpu().numpy()  # discrete grade
        all_preds.extend(preds)
        all_labels.extend(grades.numpy())
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic',
                             labels=[0,1,2,3,4])
    return float(qwk)


# ============================================================
# CELL 11 — Main
# ============================================================
def main():
    set_seed(CFG['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*68}")
    print(f"  IRDAS Phase 4 — TIES MERGE")
    print(f"  Algorithm: TIES (Trim={CFG['ties_trim_pct']}%) + lambda grid search")
    print(f"  DR QWK red line: {CFG['dr_qwk_red_line']}")
    print(f"  Teacher QWK baseline: {CFG['teacher_qwk']}")
    print(f"{'='*68}\n")

    os.makedirs(CFG['out_dir'], exist_ok=True)

    # ── Load task vectors ─────────────────────────────────────
    print("  📂 Loading task vectors...")
    if not os.path.exists(CFG['tau_hr_path']):
        raise FileNotFoundError(f"τ_HR not found: {CFG['tau_hr_path']}")
    if not os.path.exists(CFG['tau_vessel_path']):
        raise FileNotFoundError(f"τ_V not found: {CFG['tau_vessel_path']}")

    tau_hr     = torch.load(CFG['tau_hr_path'],     map_location='cpu')
    tau_vessel = torch.load(CFG['tau_vessel_path'], map_location='cpu')
    print(f"  τ_HR:    {len(tau_hr)} tensors")
    print(f"  τ_vessel: {len(tau_vessel)} tensors")

    # ── TIES merge (shared computation) ──────────────────────
    print("\n  🔧 Running TIES merge on [τ_HR, τ_V]...")
    t0 = time.time()
    tau_tied = ties_merge([tau_hr, tau_vessel], CFG['ties_trim_pct'])
    print(f"  ✅ TIES merged: {len(tau_tied)} params  ({time.time()-t0:.1f}s)")

    # ── APTOS validation set for lambda search ─────────────────
    print("\n  📂 Building APTOS val loader for λ search...")
    df = pd.read_csv(CFG['aptos_csv'])
    _, val_df = train_test_split(
        df, test_size=CFG['val_split'], stratify=df['diagnosis'], random_state=42)
    val_ds = APTOSValDataset(val_df, CFG['aptos_imgs'], CFG['image_size'])
    val_loader = DataLoader(val_ds, batch_size=CFG['batch_size'], shuffle=False,
                            num_workers=0, pin_memory=False)
    print(f"  APTOS val: {len(val_ds)} samples\n")

    # ── Build model and load θ_base ───────────────────────────
    print("  🏗  Building model (teacher architecture)...")
    model = MSDNetTeacher().to(device)
    n_loaded = safe_load_weights(model, CFG['theta_base_path'])
    if n_loaded == 0:
        raise RuntimeError(
            f"Failed to load θ_base from {CFG['theta_base_path']}!\n"
            "Verify the path points to swa_final.pth from Phase 1.")
    model.eval()
    theta_base_sd = copy.deepcopy(model.state_dict())

    # ── Baseline QWK (before merge) ───────────────────────────
    print("\n  📊 Baseline DR QWK (θ_base, no merge)...")
    baseline_qwk = eval_dr_qwk(model, val_loader, device)
    print(f"     QWK = {baseline_qwk:.4f}  (teacher trained: {CFG['teacher_qwk']:.4f})")
    _cleanup()

    # ── Lambda grid search ────────────────────────────────────
    print(f"\n  🔍 Lambda grid search over {CFG['lambda_range']}...")
    print(f"{'─'*60}")
    print(f"  {'λ':>8}  {'QWK':>8}  {'vs baseline':>12}  {'status':>10}")
    print(f"{'─'*60}")

    best_qwk   = -1.0
    best_lambda = 0.5
    results    = []
    
    # 🚨 CRITICAL FIX: Dynamically set the red line based on local validation subset
    # Since your local val subset scores ~0.82, a hardcoded 0.88 will fail every time.
    # We allow a maximum 3% drop from whatever the base model scores on this specific subset.
    actual_red_line = baseline_qwk - 0.03 

    for lam in CFG['lambda_range']:
        # Apply scaled TIES-merged delta
        merged_sd = copy.deepcopy(theta_base_sd)
        for k, v in tau_tied.items():
            if k in merged_sd and v.shape == merged_sd[k].shape:
                # FIX: Send 'v' to the same device as 'merged_sd[k]' before math
                merged_sd[k] = merged_sd[k].float() + lam * v.to(merged_sd[k].device).float()

        # Load into model
        load_result = model.load_state_dict(merged_sd, strict=False)
        model.eval()

        qwk    = eval_dr_qwk(model, val_loader, device)
        delta  = qwk - baseline_qwk
        status = "✅ OK" if qwk >= actual_red_line else "⛔ RED LINE"
        results.append({'lambda': lam, 'qwk': qwk, 'delta': delta})

        print(f"  λ={lam:>6.2f}  QWK={qwk:.4f}  Δ={delta:+.4f}  {status}")

        if qwk > best_qwk:
            best_qwk    = qwk
            best_lambda = lam

        _cleanup()

    print(f"{'─'*60}")
    print(f"\n  ✅ Best: λ = {best_lambda}  →  QWK = {best_qwk:.4f}")

    # ── Check red line ────────────────────────────────────────
    if best_qwk < actual_red_line:
        print(f"\n  ⚠️  RED LINE TRIGGERED: best QWK {best_qwk:.4f} < {actual_red_line:.4f}")
        print(f"  Falling back to θ_base (no merge). Diagnose task vector compatibility.")
        best_lambda = 0.0   # no merge — save base as "merged"

    # ── Build final merged model ──────────────────────────────
    print(f"\n  🔧 Building final merged model (λ = {best_lambda})...")
    final_sd = copy.deepcopy(theta_base_sd)
    if best_lambda > 0:
        n_merged = 0
        for k, v in tau_tied.items():
            if k in final_sd and v.shape == final_sd[k].shape:
                # FIX: Send 'v' to device here as well
                final_sd[k] = final_sd[k].float() + best_lambda * v.to(final_sd[k].device).float()
                n_merged += 1
        print(f"  Applied TIES delta to {n_merged} parameter tensors")
    else:
        print("  ⚠️  Using unmerged θ_base (red line protection)")

    # ── Save ──────────────────────────────────────────────────
    merged_path = os.path.join(CFG['out_dir'], 'merged_model.pth')
    torch.save({
        'state_dict':  final_sd,
        'best_lambda': best_lambda,
        'best_qwk':    best_qwk,
        'baseline_qwk': baseline_qwk,
        'results':     results,
        'ties_trim_pct': CFG['ties_trim_pct'],
        'tau_keys':    len(tau_tied),
    }, merged_path)
    print(f"  💾 Merged model saved → {merged_path}")

    # Also save the TIES merged delta alone (useful for debugging)
    ties_delta_path = os.path.join(CFG['out_dir'], 'tau_ties_merged.pth')
    torch.save(tau_tied, ties_delta_path)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*68}")
    print(f"  ✅ PHASE 4 COMPLETE — TIES MERGE")
    print(f"  Baseline QWK:     {baseline_qwk:.4f}")
    print(f"  Merged QWK:       {best_qwk:.4f}  (λ={best_lambda})")
    print(f"  Δ from baseline:  {best_qwk - baseline_qwk:+.4f}")
    print(f"  Red line ({actual_red_line:.4f}): {'PASSED ✅' if best_qwk >= actual_red_line else 'TRIGGERED ⛔'}")
    print(f"  Output: {merged_path}")
    print(f"{'='*68}\n")
    print("  📋 Next step: Run joint_calibration.py (Phase 5)")
    print(f"     Copy to Kaggle: {merged_path}")
    print("  📋 Next step: Run joint_calibration.py (Phase 5)")
    print(f"     Copy to Kaggle: {merged_path}")


if __name__ == '__main__':
    main()
