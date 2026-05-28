# IRDAS — finalarchitecture/notebooks/

## Kaggle Notebook Pipeline (Phases 2–5)

Each file is **fully self-contained** — all model code, utilities, and loss functions are inline.
No imports from the project's `models/` or `data/` directories. Upload directly to Kaggle.

---

## Pipeline Overview

```
Phase 1  ✓  dr_teacher_v5_fixed.py  →  swa_final.pth  (QWK 0.9342)
              ↓
Phase 2      hr_specialist.py        →  tau_hr.pth  +  hr_best_ema.pth
              ↓
Phase 3      vessel_specialist.py    →  tau_vessel.pth  +  vessel_best_ema.pth
              ↓
Phase 4      ties_merge.py           →  merged_model.pth  (TIES algorithm)
              ↓
Phase 5      joint_calibration.py   →  final_irdas.pth  ← final model
```

---

## Notebooks

| File | Phase | Dataset | Key Output |
|------|-------|---------|-----------|
| `hr_specialist.py` | 2 | HRDC 2023 | `tau_hr.pth`, `hr_best_ema.pth` |
| `vessel_specialist.py` | 3 | DRIVE | `tau_vessel.pth`, `vessel_best_ema.pth` |
| `ties_merge.py` | 4 | APTOS (val only) | `merged_model.pth` |
| `joint_calibration.py` | 5 | APTOS + HRDC + DRIVE | `final_irdas.pth` |

---

## Kaggle Dataset Dependencies

### Phase 2 — HR Specialist
```
/kaggle/input/datasets/nawazishbilal/dr-teacher-weights/best_ema_teacher.pth      ← upload Phase 1 output
/kaggle/input/datasets/nawazishbilal/hrdc-hypertensive-retinopathy-grading-challenge/2-Hypertensive Retinopathy Classification/2-Groundtruths/HRDC Hypertensive Retinopathy Classification Training Labels.csv
/kaggle/input/datasets/nawazishbilal/hrdc-hypertensive-retinopathy-grading-challenge/2-Hypertensive Retinopathy Classification/1-Images/1-Training Set/
```

### Phase 3 — Vessel Specialist
```
/kaggle/input/datasets/nawazishbilal/dr-teacher-weights/best_ema_teacher.pth      ← same teacher weights
/kaggle/input/hr-specialist-outputs/theta_base.pth  ← (optional, for matching θ_base)
/kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/images/
/kaggle/input/datasets/zionfuo/drive2004/DRIVE/training/1st_manual/
```

### Phase 4 — TIES Merge
```
/kaggle/input/datasets/nawazishbilal/dr-teacher-weights/best_ema_teacher.pth      ← θ_base
/kaggle/input/hr-specialist-outputs/tau_hr.pth      ← from Phase 2
/kaggle/input/vessel-specialist-outputs/tau_vessel.pth  ← from Phase 3
/kaggle/input/aptos2019-blindness-detection/train.csv
/kaggle/input/aptos2019-blindness-detection/train_images/
```

### Phase 5 — Joint Calibration
```
/kaggle/input/ties-merge-outputs/merged_model.pth   ← from Phase 4
/kaggle/input/aptos2019-blindness-detection/        ← DR data
/kaggle/input/datasets/nawazishbilal/hrdc-hypertensive-retinopathy-grading-challenge/2-Hypertensive Retinopathy Classification/                            ← HR data
/kaggle/input/datasets/zionfuo/drive2004/DRIVE/                 ← vessel data (optional)
```

---

## Key Design Decisions (Research-Backed)

### Phase 2 — HR Specialist
| Decision | Reason |
|----------|--------|
| Focal BCE (γ=2) | HRDC 2023 challenge — best models used focal loss |
| Weighted sampler | ~30% HR positive — handles class imbalance |
| Freeze backbone 5 ep | Prevent destroying teacher features before HR head stabilises |
| Top-5 checkpoint tracking | HRDC F1 is noisy — ensemble best checkpoints |
| Manual SWA BN update loop | `update_bn()` throws TypeError with multi-return DataLoaders |
| Task vector τ_HR = θ_HR − θ_base | Saved immediately after training |

### Phase 3 — Vessel Specialist
| Decision | Reason |
|----------|--------|
| Focal Tversky (α=0.7,β=0.3,γ=0.75) | 2024 SOTA for DRIVE — penalises FN 2.3× vs FP |
| Patch training (256×256) | DRIVE has only 20 training images — patches create ~800 samples/epoch |
| FOV mask applied to loss | Exclude black border — prevents trivial easy gradients |
| Elastic deformation | Vessel topology preservation during augmentation |
| Early stopping (patience=10) | Overfitting happens extremely fast on 20 images |

### Phase 4 — TIES Merge
| Decision | Reason |
|----------|--------|
| TIES > plain task arithmetic | Resolves parameter interference via sign-consensus |
| Trim 20% smallest deltas | Removes noisy low-magnitude updates before merging |
| Lambda grid search [0.1–1.0] | Per-task optimal scaling |
| DR QWK red line = 0.88 | If merge collapses DR, fall back to θ_base automatically |

### Phase 5 — Joint Calibration
| Decision | Reason |
|----------|--------|
| Backbone FROZEN | Preserve task-arithmetic representations |
| Uncertainty Weighting | Auto-balances 3 tasks — no manual λ, O(K) vs PCGrad O(K²) |
| DR QWK guardian | If QWK drops > 5% from baseline, stop immediately |
| 10 epochs only | Short calibration prevents catastrophic forgetting |

---

## Failure Modes Protected Against

| Risk | Protection |
|------|-----------|
| `strict=False` doesn't guard shape mismatches | `safe_load_weights()` shape-filters before loading |
| HRDC CSV column naming varies | `detect_hrdc_cols()` tries all known variants |
| DRIVE images can't be found | Multiple extension fallbacks (.tif/.png/.jpg/.bmp) |
| DRIVE GIF masks unreadable by cv2 | PIL fallback in `load_mask()` |
| `update_bn` TypeError with dict loaders | Manual BN calibration loop everywhere |
| Task arithmetic collapses DR (Phase 4) | Red line check + automatic fallback to θ_base |
| Joint calibration forgets DR (Phase 5) | QWK guardian stops training if floor breached |
| HR class imbalance suppresses positive preds | Weighted sampler + focal loss |
| DRIVE overfitting on 20 images | Patch training + early stopping + heavy augmentation |
