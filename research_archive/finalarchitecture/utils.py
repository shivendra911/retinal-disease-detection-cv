"""
╔══════════════════════════════════════════════════════════════════════╗
║  IRDAS finalarchitecture — Shared Utilities                          ║
║                                                                      ║
║  Memory management, EMA, Lookahead, logging helpers.                 ║
║  Ported from dr_teacher_v5_fixed.py + extended for multi-task.       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import gc
import ctypes
import random
import copy
import time

import numpy as np
import torch
import psutil

_proc = psutil.Process(os.getpid())

try:
    _libc = ctypes.CDLL("libc.so.6")
except OSError:
    _libc = None


# ============================================================
# MEMORY UTILITIES
# ============================================================

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
    return "  ".join(parts) if parts else "No GPU"


def mem_checkpoint(label: str):
    """Print a labeled memory checkpoint."""
    print(f"  📊 [{label}] RSS={_rss_gb():.2f} GB  |  {_vram_str()}")


def aggressive_cleanup():
    """Full memory cleanup — call after every epoch."""
    gc.collect()
    _malloc_trim()
    torch.cuda.empty_cache()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# LOOKAHEAD OPTIMIZER WRAPPER
# ============================================================

class Lookahead:
    """Lookahead optimizer wrapper (Zhang et al., 2019).

    Maintains 'slow weights' that are periodically synced with
    the fast weights from the inner optimizer, improving convergence
    stability on noisy loss landscapes like medical imaging.
    """

    def __init__(self, opt, k: int = 5, alpha: float = 0.5):
        self.opt   = opt
        self.k     = k
        self.alpha = alpha
        self._step = 0
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

    def zero_grad(self, set_to_none: bool = True):
        self.opt.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.opt.step()

    @property
    def param_groups(self):
        return self.opt.param_groups

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, sd):
        self.opt.load_state_dict(sd)


# ============================================================
# EXPONENTIAL MOVING AVERAGE
# ============================================================

class ModelEMA:
    """Exponential Moving Average of model parameters.

    EMA weights are used for validation and final checkpointing.
    They provide smoother, more stable predictions than raw weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay  = decay

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for ep, mp in zip(self.module.parameters(), model.parameters()):
            ep.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)
        for eb, mb in zip(self.module.buffers(), model.buffers()):
            eb.copy_(mb)


# ============================================================
# LR SCHEDULER BUILDER
# ============================================================

def build_stage_scheduler(optimizer, stage_epochs: int, warmup_epochs: int):
    """Build warmup + cosine decay scheduler for one training stage.

    Each stage gets a FRESH scheduler. Always reset optimizer LRs
    to the desired stage LRs before calling this function.

    Args:
        optimizer: AdamW (inner optimizer, not Lookahead wrapper)
        stage_epochs: Total epochs in this stage
        warmup_epochs: Linear warmup epochs
    """
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR, LinearLR, SequentialLR
    )
    warmup_ep = min(warmup_epochs, stage_epochs // 3)
    cosine_ep = max(stage_epochs - warmup_ep, 1)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_ep)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_ep, eta_min=1e-6)
    return SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_ep])


# ============================================================
# METRIC LOGGING
# ============================================================

class EpochLogger:
    """Simple per-epoch metric logger that prints and optionally saves CSV."""

    def __init__(self, log_path: str = None):
        self.rows = []
        self.log_path = log_path

    def log(self, epoch: int, phase: str, **metrics):
        row = {'epoch': epoch, 'phase': phase, **metrics}
        self.rows.append(row)
        parts = [f"Ep[{epoch:03d}] {phase}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print("  " + "  |  ".join(parts))
        if self.log_path:
            self._save()

    def _save(self):
        import pandas as pd
        pd.DataFrame(self.rows).to_csv(self.log_path, index=False)


# ============================================================
# WEIGHT LOADING UTILITY
# ============================================================

def load_weights(model: torch.nn.Module, path: str, strict: bool = False) -> int:
    """Load weights with automatic handling of DataParallel / SWA prefixes.

    Returns the number of successfully loaded keys.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Weights not found: {path}")

    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path, device='cpu')
    else:
        sd = torch.load(path, map_location='cpu')
        # Unwrap common wrapper prefixes
        if isinstance(sd, dict):
            sd = sd.get('module', sd.get('state_dict', sd.get('model', sd)))

    # Strip DataParallel prefix
    if any(k.startswith('module.') for k in sd):
        sd = {k[7:]: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"  ✅ Loaded weights from: {os.path.basename(path)}")
    if missing:
        print(f"     Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"     Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    return len(sd) - len(missing)


# ============================================================
# CHECKPOINT SAVE / LOAD
# ============================================================

def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    ema: ModelEMA = None,
    optimizer=None,
    epoch: int = 0,
    metrics: dict = None,
):
    """Save model checkpoint with optional EMA, optimizer state, metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        'epoch':   epoch,
        'model':   model.state_dict(),
        'metrics': metrics or {},
    }
    if ema is not None:
        ckpt['ema'] = ema.module.state_dict()
    if optimizer is not None:
        inner = optimizer.opt if isinstance(optimizer, Lookahead) else optimizer
        ckpt['optimizer'] = inner.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: torch.nn.Module, ema: ModelEMA = None):
    """Load checkpoint, returning epoch and metrics dict."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if ema is not None and 'ema' in ckpt:
        ema.module.load_state_dict(ckpt['ema'])
    return ckpt.get('epoch', 0), ckpt.get('metrics', {})


# ============================================================
# TRAINING SUMMARY PRINTER
# ============================================================

def print_banner(title: str, **info):
    width = 65
    print(f"\n{'='*width}")
    print(f"  {title}")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"{'='*width}\n")
