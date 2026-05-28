"""
╔══════════════════════════════════════════════════════════════════════╗
║  IRDAS finalarchitecture — Datasets (Research-Backed v2)             ║
║                                                                      ║
║  Research updates:                                                   ║
║  · DRIVE uses patch-based training (256×256) to handle 20 imgs      ║
║  · DRIVE FOV mask computed and returned for masked loss              ║
║  · Elastic deformation added for vessel topology learning            ║
║  · HRDC defensive column detection (handles dataset format variants) ║
║  · safe_load_weights: shape-match filter prevents RuntimeError       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import cv2
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# ============================================================
# CORAL CONSTANTS
# ============================================================

CORAL_LEVELS = 4   # K-1 ordinal levels for 5-grade DR
NUM_CLASSES  = 5   # DR grades 0–4


# ============================================================
# SHARED PREPROCESSING  (Ben Graham + CLAHE — matches teacher v7)
# ============================================================

def ben_graham_clahe(img: np.ndarray) -> np.ndarray:
    """Ben Graham illumination normalization + CLAHE on L channel.

    CRITICAL: must be IDENTICAL to dr_teacher_v5_fixed.py to ensure
    consistent train/inference preprocessing.

    Raises:
        ValueError: if img is None or wrong dtype
    """
    if img is None:
        raise ValueError("ben_graham_clahe received None image")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img_bg  = cv2.addWeighted(img, 4, blurred, -4, 128)
    lab     = cv2.cvtColor(img_bg, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def load_fundus_image(path: str, preprocess: bool = True) -> np.ndarray:
    """Load fundus image, try multiple extensions if needed.

    Args:
        path: Primary path to try (usually .png)
        preprocess: Whether to apply Ben Graham + CLAHE

    Returns:
        RGB uint8 numpy array

    Raises:
        FileNotFoundError: if no readable image found at any path variant
    """
    tried = [path]
    img = cv2.imread(path)
    if img is None:
        for ext in ['.jpeg', '.jpg', '.tif', '.bmp']:
            alt = os.path.splitext(path)[0] + ext
            tried.append(alt)
            img = cv2.imread(alt)
            if img is not None:
                break

    if img is None:
        raise FileNotFoundError(
            f"Could not read image. Tried: {tried}"
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if preprocess:
        img = ben_graham_clahe(img)
    return img


# ============================================================
# AUGMENTATION TRANSFORMS
# ============================================================

def get_fundus_transforms(image_size: int, is_train: bool) -> A.Compose:
    """v7 teacher augmentation pipeline (strong train, identity val).

    Matches dr_teacher_v5_fixed.py exactly for consistency.
    """
    if is_train:
        hole_h = max(8, image_size // 12)
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.80, 1.20), rotate=(-45, 45),
                border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.7,
            ),
            A.RandomBrightnessContrast(0.25, 0.25, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1,
                          saturation=0.3, hue=0.02, p=0.5),
            A.OneOf([
                A.GaussNoise(std_range=(3.16 / 255.0, 10.0 / 255.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.Downscale(scale_range=(0.5, 0.75), p=1.0),
            ], p=0.4),
            A.CoarseDropout(
                num_holes_range=(4, 12),
                hole_height_range=(hole_h, hole_h * 2),
                hole_width_range=(hole_h, hole_h * 2),
                fill=0, p=0.5,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_vessel_patch_transforms(patch_size: int, is_train: bool) -> A.Compose:
    """Augmentation for vessel segmentation patches.

    Applies identical transforms to image AND mask (spatial consistency).
    Adds elastic deformation — critical for vessel topology learning.
    Does NOT add CoarseDropout or ColorJitter to mask (image-only).

    Args:
        patch_size: Square patch size (256 recommended for DRIVE)
        is_train: Training or validation transform
    """
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1), rotate=(-30, 30),
                border_mode=cv2.BORDER_REFLECT_101, p=0.5,
            ),
            # Elastic deformation — vessel topology-preserving
            A.ElasticTransform(
                alpha=120, sigma=6,
                border_mode=cv2.BORDER_REFLECT_101, p=0.3,
            ),
            # Image-only augmentations (no effect on mask)
            A.OneOf([
                A.GaussNoise(std_range=(0.005, 0.05), p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.3),
            A.ColorJitter(brightness=0.15, contrast=0.2, p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),  # simulate illumination
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets={'mask': 'mask', 'fov': 'mask'})
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask', 'fov': 'mask'})


# ============================================================
# LABEL DISTRIBUTION SMOOTHING WEIGHTS (APTOS sampler)
# ============================================================

def compute_lds_weights(
    labels: np.ndarray, num_classes: int = 5, sigma: float = 1.0
) -> np.ndarray:
    """LDS class-balanced sampling weights — identical to teacher v7."""
    counts   = np.bincount(labels, minlength=num_classes).astype(float)
    x        = np.arange(-2, 3)
    kernel   = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel  /= kernel.sum()
    smoothed = np.maximum(np.convolve(counts, kernel, mode='same'), 1.0)
    w        = 1.0 / smoothed[labels]
    return (w / w.max()).astype(np.float32)


# ============================================================
# APTOS 2019 — DR grading with CORAL ordinal targets
# ============================================================

class APTOSDataset(Dataset):
    """APTOS 2019 Diabetic Retinopathy Grading.

    Returns CORAL ordinal targets + raw integer grade.
    Supports shared image cache (same as teacher v7).

    Args:
        df:           DataFrame with ['id_code', 'diagnosis']
        img_dir:      Path to image directory
        transforms:   Albumentations Compose
        shared_cache: Optional {id_code: np.ndarray} pre-built cache
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        transforms: A.Compose,
        shared_cache: dict = None,
    ):
        self.df          = df.reset_index(drop=True)
        self.img_dir     = img_dir
        self.transforms  = transforms
        self._cache      = shared_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        id_code = str(row['id_code'])
        grade   = int(row['diagnosis'])

        if self._cache is not None:
            img = self._cache[id_code].copy()
        else:
            img = load_fundus_image(
                os.path.join(self.img_dir, id_code + '.png')
            )

        img   = self.transforms(image=img)['image']
        coral = torch.tensor(
            [1] * grade + [0] * (CORAL_LEVELS - grade), dtype=torch.float32
        )
        return img, coral, torch.tensor(grade, dtype=torch.long)


def build_aptos_cache(
    df: pd.DataFrame, img_dir: str, cache_size: int = 512
) -> dict:
    """Build APTOS image cache at a single resolution (same as teacher)."""
    n = len(df)
    print(f"\n📦 Building APTOS cache ({n} imgs × {cache_size}px)")
    t0    = time.time()
    cache = {}
    for i in range(n):
        id_code = str(df.iloc[i]['id_code'])
        img     = load_fundus_image(
            os.path.join(img_dir, id_code + '.png')
        )
        img             = cv2.resize(img, (cache_size, cache_size), interpolation=cv2.INTER_AREA)
        cache[id_code]  = img
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n} | {time.time()-t0:.0f}s")
    mb = sum(v.nbytes for v in cache.values()) / 1e6
    print(f"  ✅ APTOS cache ready: {mb:.0f} MB\n")
    return cache


# ============================================================
# HRDC 2023 — Hypertensive Retinopathy (binary classification)
# ============================================================

def _detect_hrdc_columns(df: pd.DataFrame) -> tuple:
    """Defensively detect filename and label columns in HRDC CSV.

    HRDC datasets have inconsistent column names across download sources.
    This function tries all known variants and raises a clear error if none found.

    Returns:
        (filename_col, grade_col) column name strings
    Raises:
        KeyError: with helpful message listing actual columns available
    """
    filename_candidates = ['filename', 'image_id', 'image', 'img_id', 'id', 'file_name']
    grade_candidates    = ['grade', 'label', 'class', 'HR_grade', 'hr_grade', 'target']

    fn_col = next((c for c in filename_candidates if c in df.columns), None)
    gr_col = next((c for c in grade_candidates    if c in df.columns), None)

    if fn_col is None or gr_col is None:
        raise KeyError(
            f"Cannot find required columns in HRDC CSV.\n"
            f"Looking for filename col in: {filename_candidates}\n"
            f"Looking for grade col in: {grade_candidates}\n"
            f"Actual columns: {list(df.columns)}\n"
            f"Please set the correct column names in datasets.py."
        )
    return fn_col, gr_col


def _resolve_hrdc_path(img_dir: str, fname: str) -> str:
    """Try multiple path variants to find an HRDC image.

    HRDC images come in .jpg and .png; CSV may or may not include extension.

    Args:
        img_dir: Image directory
        fname:   Filename from CSV (may lack extension)
    Returns:
        Resolved absolute path
    Raises:
        FileNotFoundError: if no variant exists
    """
    tried = []
    base = os.path.splitext(fname)[0] if '.' in fname else fname
    for ext in ['', '.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
        candidate = os.path.join(img_dir, base + ext)
        tried.append(candidate)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"HRDC image not found. Tried:\n" + "\n".join(f"  {p}" for p in tried)
    )


class HRDCDataset(Dataset):
    """HRDC 2023 Hypertensive Retinopathy — binary classification.

    Handles inconsistent CSV column names and image extensions defensively.
    Returns (image_tensor, hr_binary_label).

    Args:
        img_dir:    Path to HRDC image directory
        labels_csv: Path to CSV (columns auto-detected)
        transforms: Albumentations transform
    """

    def __init__(self, img_dir: str, labels_csv: str, transforms: A.Compose):
        df               = pd.read_csv(labels_csv)
        self.fn_col, self.gr_col = _detect_hrdc_columns(df)
        self.df          = df.reset_index(drop=True)
        self.img_dir     = img_dir
        self.transforms  = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        fname = str(row[self.fn_col])
        path  = _resolve_hrdc_path(self.img_dir, fname)
        img   = load_fundus_image(path)
        img   = self.transforms(image=img)['image']

        label = torch.tensor(
            1.0 if int(row[self.gr_col]) > 0 else 0.0,
            dtype=torch.float32,
        )
        return img, label


# ============================================================
# DRIVE — Retinal Vessel Segmentation (PATCH-BASED)
# ============================================================

def _compute_fov_mask(img: np.ndarray, threshold: int = 20) -> np.ndarray:
    """Compute circular Field-Of-View mask for a fundus image.

    The fundus camera captures a circular region of the retina.
    Outside this circle, all pixels are black (invalid).
    Including them in the loss gives easy gradients from background.

    Method: threshold on grayscale > threshold → binary FOV mask.

    Args:
        img:       (H, W, 3) RGB image (post-preprocessing, may be float or uint8)
        threshold: pixel brightness threshold for valid region
    Returns:
        (H, W) binary uint8 mask (255 = valid, 0 = invalid)
    """
    if img.dtype != np.uint8:
        img_u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        img_u8 = img
    gray  = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Morphological closing to fill small holes inside valid region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask  # uint8, {0, 255}


class DRIVEPatchDataset(Dataset):
    """DRIVE vessel segmentation — PATCH-BASED training.

    DRIVE has only 20 training images. Training on full images overfits.
    Solution: extract N random 256×256 patches per image per epoch.
    This gives ~800 unique samples per epoch (20 imgs × 40 patches).

    Each patch includes:
    - image tensor: (3, H, W) normalised
    - mask tensor:  (1, H, W) binary {0, 1}
    - fov tensor:   (1, H, W) binary FOV mask (for masked loss)

    Args:
        img_dir:        Directory with DRIVE images (.tif or .png)
        mask_dir:       Directory with vessel masks (.gif or .png)
        transforms:     Albumentations Compose (applied to full image first)
        patch_size:     Square patch size to extract (default 256)
        patches_per_img: Number of random patches to sample per image per epoch
        is_train:       If False, returns centre-crop instead of random patch
    """

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        transforms: A.Compose,
        patch_size: int = 256,
        patches_per_img: int = 40,
        is_train: bool = True,
    ):
        self.img_paths  = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.tif', '.png', '.jpg', '.bmp'))
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if f.lower().endswith(('.gif', '.tif', '.png', '.bmp'))
        ])
        assert len(self.img_paths) == len(self.mask_paths), (
            f"Image/mask count mismatch: {len(self.img_paths)} vs {len(self.mask_paths)}"
        )
        self.transforms      = transforms
        self.patch_size      = patch_size
        self.patches_per_img = patches_per_img
        self.is_train        = is_train

        # Pre-load all images into memory (DRIVE is tiny — 20 images × ~300KB each)
        print(f"  📦 Pre-loading DRIVE: {len(self.img_paths)} images...")
        self._imgs  = []
        self._masks = []
        self._fovs  = []
        for ip, mp in zip(self.img_paths, self.mask_paths):
            img  = load_fundus_image(ip)           # RGB, preprocessed
            mask = self._load_mask(mp)              # H×W uint8 {0,255}
            fov  = _compute_fov_mask(img)           # H×W uint8 {0,255}
            self._imgs.append(img)
            self._masks.append(mask)
            self._fovs.append(fov)
        print(f"  ✅ DRIVE loaded: {len(self._imgs)} images")

    @staticmethod
    def _load_mask(path: str) -> np.ndarray:
        """Load vessel mask, handling GIF and standard formats."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            try:
                from PIL import Image as PILImage
                mask = np.array(PILImage.open(path).convert('L'))
            except Exception as e:
                raise FileNotFoundError(f"Cannot load mask {path}: {e}")
        return mask  # uint8

    def __len__(self):
        return len(self._imgs) * self.patches_per_img

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_img
        img     = self._imgs[img_idx]
        mask    = self._masks[img_idx]
        fov     = self._fovs[img_idx]

        H, W    = img.shape[:2]
        ps      = self.patch_size

        if self.is_train:
            # Random crop — but only from within the valid FOV region
            # Try up to 20 times to find a patch with >50% valid pixels
            for _ in range(20):
                y = random.randint(0, H - ps)
                x = random.randint(0, W - ps)
                fov_patch = fov[y:y+ps, x:x+ps]
                if fov_patch.mean() > 127:  # >50% valid
                    break
        else:
            # Centre crop for validation
            y = max(0, (H - ps) // 2)
            x = max(0, (W - ps) // 2)

        img_p  = img[y:y+ps, x:x+ps]
        mask_p = mask[y:y+ps, x:x+ps]
        fov_p  = fov[y:y+ps, x:x+ps]

        # Normalise mask and FOV to float32 [0, 1]
        mask_f = (mask_p > 127).astype(np.float32)
        fov_f  = (fov_p  > 127).astype(np.float32)

        augmented = self.transforms(image=img_p, mask=mask_f, fov=fov_f)
        img_t  = augmented['image']               # (3, ps, ps)
        mask_t = torch.from_numpy(augmented['mask']).unsqueeze(0).float()  # (1, ps, ps)
        fov_t  = torch.from_numpy(augmented['fov']).unsqueeze(0).float()   # (1, ps, ps)

        return img_t, mask_t, fov_t


# ============================================================
# DATALOADER BUILDERS
# ============================================================

def build_aptos_loaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    img_dir: str,
    shared_cache: dict = None,
) -> tuple:
    """APTOS train/val loaders with LDS weighted sampling.

    Returns:
        (train_loader, valid_loader, lds_weights)
    """
    lds_weights = compute_lds_weights(train_df['diagnosis'].values)
    sampler     = WeightedRandomSampler(
        torch.from_numpy(lds_weights).float(), len(lds_weights), replacement=True
    )
    train_ds = APTOSDataset(
        train_df, img_dir,
        get_fundus_transforms(image_size, is_train=True),
        shared_cache,
    )
    valid_ds = APTOSDataset(
        valid_df, img_dir,
        get_fundus_transforms(image_size, is_train=False),
        shared_cache,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, valid_loader, lds_weights


def build_hrdc_loaders(
    img_dir: str,
    labels_csv: str,
    image_size: int,
    batch_size: int,
    val_split: float = 0.15,
) -> tuple:
    """HRDC train/val loaders with stratified split and weighted sampling.

    Returns:
        (train_loader, valid_loader)
    """
    df = pd.read_csv(labels_csv)
    _, gr_col = _detect_hrdc_columns(df)
    df['binary'] = (df[gr_col] > 0).astype(int)

    train_df, valid_df = train_test_split(
        df, test_size=val_split,
        stratify=df['binary'], random_state=42,
    )
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # Weighted sampler to handle HR class imbalance
    labels  = train_df['binary'].values
    counts  = np.bincount(labels).astype(float)
    weights = 1.0 / counts[labels]
    weights = (weights / weights.max()).astype(np.float32)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights), len(weights), replacement=True
    )

    def _make_hrdc(df_split, is_train):
        ds = HRDCDataset(
            img_dir, labels_csv=labels_csv,
            transforms=get_fundus_transforms(image_size, is_train=is_train),
        )
        ds.df = df_split  # override with split subset
        return ds

    train_loader = DataLoader(
        _make_hrdc(train_df, True), batch_size=batch_size, sampler=sampler,
        num_workers=0, pin_memory=False, drop_last=True,
    )
    valid_loader = DataLoader(
        _make_hrdc(valid_df, False), batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, valid_loader


def build_drive_loaders(
    img_dir: str,
    mask_dir: str,
    patch_size: int = 256,
    batch_size: int = 8,
    patches_per_img: int = 40,
    val_fraction: float = 0.25,
) -> tuple:
    """DRIVE patch-based train/val loaders.

    Splits image-level (not patch-level) to prevent data leakage.
    Validation returns centre patches, not random patches.

    Returns:
        (train_loader, valid_loader)
    """
    img_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(('.tif', '.png', '.jpg', '.bmp'))
    ])
    n_val    = max(1, int(len(img_files) * val_fraction))
    random.seed(42)
    val_set  = set(random.sample(img_files, n_val))
    train_fs = [f for f in img_files if f not in val_set]
    val_fs   = [f for f in img_files if f in val_set]

    print(f"  DRIVE split: {len(train_fs)} train / {len(val_fs)} val images")

    # Build mask file list to match image list
    mask_files = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(('.gif', '.tif', '.png', '.bmp'))
    ])

    def _filter_dir(all_imgs, all_masks, keep_set):
        """Return parallel (img_paths, mask_paths) for images in keep_set."""
        pairs = []
        for img_f, msk_f in zip(all_imgs, all_masks):
            if img_f in keep_set or os.path.basename(img_f) in keep_set:
                pairs.append((img_f, msk_f))
        return pairs

    # We build separate DRIVEPatchDataset instances for train/val
    # pointing to filtered subsets via custom img_paths attribute
    train_ds = _DRIVESubset(
        img_dir, mask_dir, img_files=train_fs,
        transforms=get_vessel_patch_transforms(patch_size, is_train=True),
        patch_size=patch_size, patches_per_img=patches_per_img, is_train=True,
    )
    valid_ds = _DRIVESubset(
        img_dir, mask_dir, img_files=val_fs,
        transforms=get_vessel_patch_transforms(patch_size, is_train=False),
        patch_size=patch_size, patches_per_img=10, is_train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    return train_loader, valid_loader


class _DRIVESubset:
    """Internal helper — DRIVE dataset restricted to a specific list of image files."""

    def __init__(
        self,
        img_dir: str,
        mask_dir: str,
        img_files: list,
        transforms: A.Compose,
        patch_size: int = 256,
        patches_per_img: int = 40,
        is_train: bool = True,
    ):
        # Find matching mask files (sort both, assume 1:1 correspondence)
        all_mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(('.gif', '.tif', '.png', '.bmp'))
        ])
        all_img_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.tif', '.png', '.jpg', '.bmp'))
        ])
        assert len(all_img_files) == len(all_mask_files), (
            f"DRIVE img/mask count mismatch: {len(all_img_files)} vs {len(all_mask_files)}"
        )
        # Build index from filename → position
        img_idx = {f: i for i, f in enumerate(all_img_files)}
        self._selected_pairs = [
            (os.path.join(img_dir, f), os.path.join(mask_dir, all_mask_files[img_idx[f]]))
            for f in img_files if f in img_idx
        ]

        # Pre-load into memory
        self._imgs  = []
        self._masks = []
        self._fovs  = []
        for ip, mp in self._selected_pairs:
            img  = load_fundus_image(ip)
            mask = DRIVEPatchDataset._load_mask(mp)
            fov  = _compute_fov_mask(img)
            self._imgs.append(img)
            self._masks.append(mask)
            self._fovs.append(fov)

        self.transforms      = transforms
        self.patch_size      = patch_size
        self.patches_per_img = patches_per_img
        self.is_train        = is_train

    def __len__(self):
        return len(self._imgs) * self.patches_per_img

    def __getitem__(self, idx):
        img_i  = idx // self.patches_per_img
        img    = self._imgs[img_i]
        mask   = self._masks[img_i]
        fov    = self._fovs[img_i]
        H, W   = img.shape[:2]
        ps     = self.patch_size

        if self.is_train:
            for _ in range(20):
                y = random.randint(0, max(0, H - ps))
                x = random.randint(0, max(0, W - ps))
                if fov[y:y+ps, x:x+ps].mean() > 127:
                    break
        else:
            y = max(0, (H - ps) // 2)
            x = max(0, (W - ps) // 2)

        y = min(y, max(0, H - ps))
        x = min(x, max(0, W - ps))

        img_p  = img[y:y+ps, x:x+ps]
        mask_f = (mask[y:y+ps, x:x+ps] > 127).astype(np.float32)
        fov_f  = (fov[y:y+ps, x:x+ps] > 127).astype(np.float32)

        augmented = self.transforms(image=img_p, mask=mask_f, fov=fov_f)
        img_t  = augmented['image']
        mask_t = torch.from_numpy(augmented['mask']).unsqueeze(0).float()
        fov_t  = torch.from_numpy(augmented['fov']).unsqueeze(0).float()
        return img_t, mask_t, fov_t
