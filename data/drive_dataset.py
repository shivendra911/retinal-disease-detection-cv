"""
IRDAS Data — DRIVE (Digital Retinal Images for Vessel Extraction)
==================================================================

40 images with pixel-level vessel segmentation masks.
Used ONLY to train the auxiliary vessel decoder branch.
This teaches the backbone to understand vascular anatomy.

Kaggle dataset ID: search "DRIVE retinal vessel segmentation"
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from preprocessing.clahe_preprocess import preprocess_fundus


class DRIVEDataset(Dataset):
    """
    Digital Retinal Images for Vessel Extraction.
    
    40 images with pixel-level vessel segmentation masks.
    Used ONLY to train the auxiliary vessel decoder branch.
    
    Args:
        img_dir: Path to DRIVE images
        mask_dir: Path to vessel segmentation masks
        transform: Albumentations transform (must handle image+mask)
    """
    
    def __init__(self, img_dir, mask_dir, transform=None):
        self.images = sorted([f for f in os.listdir(img_dir) 
                              if f.endswith(('.tif', '.png', '.jpg'))])
        self.masks  = sorted([f for f in os.listdir(mask_dir) 
                              if f.endswith(('.gif', '.tif', '.png'))])
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img  = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        mask = cv2.imread(os.path.join(self.mask_dir, self.masks[idx]),
                          cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise FileNotFoundError(f"Image not found: {self.images[idx]}")
        
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = preprocess_fundus(img)
        mask = cv2.resize(mask, (224, 224))
        mask = (mask > 127).astype(np.float32)  # binarize: 0 or 1
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0)  # (1, 224, 224)
        else:
            mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        
        return img, mask
