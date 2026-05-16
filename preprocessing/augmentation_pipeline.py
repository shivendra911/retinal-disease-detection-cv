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
