"""
IRDAS Data — APTOS 2019 Diabetic Retinopathy Dataset
======================================================

Primary training dataset for DR grading.
Labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR
Primary metric: Quadratic Weighted Kappa (QWK)

Kaggle dataset ID: aptos2019-blindness-detection
Total images: 3,662 (train)

Class distribution (heavily imbalanced):
  Grade 0: ~49% | Grade 1: ~10% | Grade 2: ~27% | Grade 3: ~5% | Grade 4: ~9%
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from preprocessing.clahe_preprocess import preprocess_fundus


class APTOSDataset(Dataset):
    """
    APTOS 2019 Diabetic Retinopathy Grading Dataset.
    
    Args:
        csv_path: Path to train.csv (columns: id_code, diagnosis)
        img_dir: Path to train_images directory
        transform: Albumentations transform pipeline
        mode: 'train' or 'val' (affects augmentation)
        use_preprocessed: If True, load pre-computed .npy files instead of raw images
    """
    
    def __init__(self, csv_path, img_dir, transform=None, mode='train',
                 use_preprocessed=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        self.use_preprocessed = use_preprocessed
        
        # Compute class weights for handling imbalance
        counts = self.df['diagnosis'].value_counts().sort_index()
        total = len(self.df)
        self.class_weights = torch.tensor(
            [total / (5 * c) for c in counts], dtype=torch.float32
        )
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.use_preprocessed:
            # Load pre-computed preprocessed image (saves ~30% training time)
            npy_path = os.path.join(self.img_dir, row['id_code'] + '.npy')
            img = np.load(npy_path)
        else:
            # Load and preprocess on-the-fly
            img_path = os.path.join(self.img_dir, row['id_code'] + '.png')
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_fundus(img)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        label = torch.tensor(row['diagnosis'], dtype=torch.long)
        return img, label
