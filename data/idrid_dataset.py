"""
IRDAS Data — IDRiD (Indian Diabetic Retinopathy Image Dataset)
===============================================================

Cross-dataset generalization test — Indian clinic images.
DO NOT use for training. Test-only dataset.

Collected from actual Indian clinics — real-world quality variance.
DR grades: 0-4 (same scale as APTOS)

This dataset proves your model generalizes to real Indian clinical conditions.
The QWK on IDRiD is reported as "cross-dataset generalization" in the paper.
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from preprocessing.clahe_preprocess import preprocess_fundus


class IDRiDDataset(Dataset):
    """
    Indian Diabetic Retinopathy Image Dataset.
    
    TEST-ONLY. Never train on this.
    
    Args:
        img_dir: Path to IDRiD images
        labels_csv: Path to labels CSV
        transform: Albumentations transform (val transforms only)
    """
    
    def __init__(self, img_dir, labels_csv, transform=None):
        self.df = pd.read_csv(labels_csv)
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.img_dir, row['Image name'] + '.jpg'))
        if img is None:
            raise FileNotFoundError(f"Image not found: {row['Image name']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_fundus(img)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, torch.tensor(row['Retinopathy grade'], dtype=torch.long)
