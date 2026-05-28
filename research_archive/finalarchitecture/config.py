"""
╔══════════════════════════════════════════════════════════════════════╗
║  IRDAS finalarchitecture — Centralized Configuration                 ║
║                                                                      ║
║  All hyperparameters for Phases 2–5 live here.                       ║
║  Adjust paths before running on your environment.                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ============================================================
# PATH CONFIGURATION  (adjust for your environment)
# ============================================================

PATHS = {
    # ── Kaggle input paths ──────────────────────────────────
    'aptos_csv':     '/kaggle/input/aptos2019-blindness-detection/train.csv',
    'aptos_imgs':    '/kaggle/input/aptos2019-blindness-detection/train_images',

    'hrdc_csv':      '/kaggle/input/hrdc-2023/train.csv',
    'hrdc_imgs':     '/kaggle/input/hrdc-2023/images',

    'drive_imgs':    '/kaggle/input/drive-retinal-vessel/training/images',
    'drive_masks':   '/kaggle/input/drive-retinal-vessel/training/1st_manual',

    # ── Saved DR Teacher weights (from dr_teacher_v5_fixed.py) ─
    # Use swa_final.pth for best performance; fallback to best_ema_teacher.pth
    'dr_teacher_weights': '/kaggle/working/checkpoints/swa_final.pth',

    # ── Output paths ────────────────────────────────────────
    'ckpt_dir':           '/kaggle/working/finalarchitecture/checkpoints',
    'log_dir':            '/kaggle/working/finalarchitecture/logs',
}


# ============================================================
# SHARED MODEL CONFIGURATION
# ============================================================

MODEL_CFG = {
    'backbone':           'tf_efficientnet_b4.ns_jft_in1k',
    'drop_path_rate':     0.2,
    'fpn_out_channels':   256,
    'dropout_rate':       0.3,
    'dr_num_classes':     5,         # DR grades 0–4
    'coral_levels':       4,         # CORAL K-1 ordinal levels
    'bifpn_channels':     256,
    'bifpn_layers':       2,
    'msd_k':              5,         # Multi-sample dropout passes
}


# ============================================================
# PHASE 2 — HR SPECIALIST TRAINING
# ============================================================

HR_CFG = {
    'seed':              42,
    'image_size':        384,
    'total_epochs':      50,
    'freeze_epochs':     5,           # Freeze backbone for first 5 epochs
    'warmup_epochs':     3,

    # Per-stage LRs: (head_lr, backbone_lr)
    'stage_lrs': [
        (3e-4, 0),        # Stage 1: head only (backbone frozen)
        (3e-4, 3e-5),     # Stage 2: full fine-tuning
    ],

    'batch_size':        16,
    'grad_accum':        4,           # Effective batch = 64
    'weight_decay':      1e-2,

    'swa_start':         40,
    'swa_lr':            5e-6,

    'ema_decay':         0.999,
    'lookahead_k':       5,
    'lookahead_alpha':   0.5,

    # Loss weights
    'hr_pos_weight':     2.5,        # upweight HR-positive samples (class imbalance)

    # Checkpoint
    'save_prefix':       'hr_specialist',
}


# ============================================================
# PHASE 3 — VESSEL SPECIALIST TRAINING
# ============================================================

VESSEL_CFG = {
    'seed':              42,
    'image_size':        384,
    'mask_size':         384,        # vessel mask output size
    'total_epochs':      60,
    'freeze_epochs':     5,
    'warmup_epochs':     3,

    'stage_lrs': [
        (3e-4, 0),        # Stage 1: decoder + FPN only (backbone frozen)
        (3e-4, 3e-5),     # Stage 2: full fine-tuning
    ],

    'batch_size':        8,          # DRIVE has only 40 images — small batches
    'grad_accum':        8,
    'weight_decay':      1e-2,

    'swa_start':         48,
    'swa_lr':            5e-6,

    'ema_decay':         0.999,
    'lookahead_k':       5,
    'lookahead_alpha':   0.5,

    # Loss weights
    'dice_weight':       0.7,        # Dice loss weight
    'bce_weight':        0.3,        # BCE loss weight

    # Target metric
    'target_dice':       0.78,       # DRIVE test target

    # Checkpoint
    'save_prefix':       'vessel_specialist',
}


# ============================================================
# PHASE 4 — TASK ARITHMETIC MERGE
# ============================================================

MERGE_CFG = {
    # Task vector weights — λ₁ and λ₂ in:
    # θ_final = θ_base + λ₁·τ_HR + λ₂·τ_vessel
    'lambda_hr':          0.5,
    'lambda_vessel':      0.5,

    # Grid search range for λ optimization
    'lambda_search_range': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
    'lambda_search_metric': 'qwk',   # optimize for DR QWK on val set

    # Checkpoint
    'save_prefix':        'merged',
}


# ============================================================
# PHASE 5 — JOINT CALIBRATION
# ============================================================

JOINT_CFG = {
    'seed':              42,
    'image_size':        384,
    'total_epochs':      10,
    'warmup_epochs':     2,

    # Backbone FROZEN — only heads + FPN adapt
    'head_lr':           1e-4,
    'weight_decay':      1e-2,

    'batch_size':        16,
    'grad_accum':        4,

    'ema_decay':         0.999,

    # Loss weights for joint training
    'alpha_dr':          1.0,        # DR CORAL loss weight
    'alpha_hr':          0.5,        # HR BCE loss weight
    'alpha_vessel':      0.3,        # Vessel segmentation loss weight
    'alpha_contrast':    0.2,        # Contrastive disentanglement loss weight

    # CORAL loss settings (same as teacher)
    'fn_weight':         2.0,
    'label_smooth':      0.05,
    'loss_alpha':        1.0,        # pure CORAL

    # Contrastive loss settings
    'contrastive_margin_pure':    0.1,    # margin when diseases don't co-occur
    'contrastive_margin_cooccur': 0.3,    # margin when they co-occur

    # Checkpoint
    'save_prefix':       'final_irdas',
}
