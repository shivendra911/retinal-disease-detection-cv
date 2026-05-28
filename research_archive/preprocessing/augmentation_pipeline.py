"""
IRDAS Preprocessing — Data Augmentation Pipeline
=================================================

Medical image augmentation using Albumentations.
Each augmentation is chosen with clinical justification:

- Geometric transforms: fundus images can be rotated/flipped freely
  (no canonical orientation for left/right eye)
- Color transforms: simulate varying camera quality across clinics
- Noise/blur: simulate low-quality rural clinic cameras
- Elastic transforms: simulate slight retinal shape variation

IMPORTANT: Augmentation is applied AFTER preprocessing (CLAHE etc.)
but BEFORE conversion to tensor. The preprocessing normalizes the image
to float32 with ImageNet stats — augmentation works on this normalized image.

Note: For vessel segmentation (DRIVE dataset), the same spatial transforms
must be applied to both image AND mask. Use get_vessel_transforms() for this.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms():
    """
    Training augmentation pipeline for disease classification.
    
    Applied to APTOS and HRDC images during training.
    Aggressive augmentation to compensate for small medical datasets.
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # === Geometric (fundus has no canonical orientation) ===
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15,
            rotate_limit=30, p=0.6,
            border_mode=0  # zero-pad borders
        ),
        
        # === Color (simulate different cameras/lighting) ===
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.1, hue=0.0,  # no hue shift — retinal colors are diagnostic
            p=0.5
        ),
        
        # === Noise & Blur (simulate low-quality images) ===
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # === Elastic (slight retinal shape variation) ===
        A.ElasticTransform(alpha=120, sigma=6, p=0.3),
        
        # === Simulate cataract / media opacity ===
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.4
        ),
        
        # === Convert to PyTorch tensor ===
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Validation/test transforms — NO augmentation, only tensor conversion.
    
    Returns:
        Albumentations Compose object (ToTensorV2 only)
    """
    return A.Compose([
        ToTensorV2(),
    ])


def get_vessel_transforms(mode='train'):
    """
    Augmentation for vessel segmentation (DRIVE dataset).
    
    CRITICAL: Both image AND mask must receive the same spatial transforms.
    Albumentations handles this automatically when you pass mask= to __call__.
    
    Color/noise transforms only apply to the image, not the mask.
    
    Args:
        mode: 'train' or 'val'
    
    Returns:
        Albumentations Compose object
    """
    if mode == 'val':
        return A.Compose([ToTensorV2()])
    
    return A.Compose([
        # Spatial transforms (applied to both image and mask)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.1,
            rotate_limit=20, p=0.5,
            border_mode=0
        ),
        
        # Image-only transforms (NOT applied to mask)
        A.ColorJitter(brightness=0.15, contrast=0.15, p=0.3),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        
        ToTensorV2(),
    ])


def get_fundus_train_transforms(image_size=384):
    """
    SOTA ophthalmology-specific augmentation pipeline for DR grading.
    
    Domain-justified augmentations that simulate real-world fundus image
    variations seen across different clinics and cameras:
    - CLAHE variation: simulates varying preprocessing quality
    - Brightness/darkness: simulates poorly dilated pupils
    - Motion blur: simulates unfocused cameras
    - Downscale: simulates low-resolution rural clinic cameras
    - Geometric: fundus has no canonical orientation
    
    NO Mixup/CutMix — these are destructive for ordinal classification.
    Mixed fundus images don't correspond to any real disease state.
    
    Args:
        image_size: Target image size (384 or 512)
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        # === Resize for progressive training ===
        A.Resize(image_size, image_size),
        
        # === Geometric (fundus has no canonical orientation) ===
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15,
            rotate_limit=30, p=0.6,
            border_mode=0
        ),
        
        # === Ophthalmology-specific color augmentations ===
        A.OneOf([
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3, 0.1),  # simulate poorly dilated pupil (darker)
                contrast_limit=0.3, p=1.0
            ),
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.1, hue=0.0,  # no hue shift — retinal colors are diagnostic
                p=1.0
            ),
        ], p=0.7),
        
        # === Simulate low-quality images ===
        A.OneOf([
            A.GaussNoise(var_limit=(5.0, 30.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),  # unfocused camera
        ], p=0.3),
        
        # === Simulate varying image quality across clinics ===
        A.Downscale(scale_min=0.5, scale_max=0.9, p=0.15),
        
        # === Elastic (slight retinal shape variation) ===
        A.ElasticTransform(alpha=120, sigma=6, p=0.2),
        
        # === Coarse dropout — simulate partial occlusion ===
        A.CoarseDropout(
            max_holes=8, max_height=image_size // 16, max_width=image_size // 16,
            min_holes=1, min_height=image_size // 32, min_width=image_size // 32,
            fill_value=0, p=0.3
        ),
        
        # === Convert to PyTorch tensor ===
        ToTensorV2(),
    ])


def get_val_transforms_with_resize(image_size=384):
    """Validation transforms with resize — no augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2(),
    ])

