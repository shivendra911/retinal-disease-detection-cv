"""
IRDAS Data — HRDC 2023 (Hypertensive Retinopathy Detection Challenge)
======================================================================

Second disease dataset — HR detection.
Labels: 0=Normal, 1=Mild HR, 2=Moderate HR
Converted to binary: 0=No HR, 1=HR present

~1,200 images total.

IMPORTANT: Shivendra must register at grand-challenge.org to download.
After download, upload to Kaggle as private dataset named 'hrdc-2023'.
"""

import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from preprocessing.clahe_preprocess import preprocess_fundus


class HRDCDataset(Dataset):
    """
    Hypertensive Retinopathy Detection Challenge 2023.
    
    Args:
        img_dir: Path to HRDC images
        labels_csv: Path to labels CSV
        transform: Albumentations transform pipeline
        binary: If True, convert to binary (HR present/absent). Default True.
    """
    
    def __init__(self, img_dir, labels_csv, transform=None, binary=True):
        self.df = pd.read_csv(labels_csv)
        self.img_dir = img_dir
        self.transform = transform
        self.binary = binary
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(os.path.join(self.img_dir, row['filename']))
        if img is None:
            raise FileNotFoundError(f"Image not found: {row['filename']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_fundus(img)
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        if self.binary:
            label = torch.tensor(1 if row['grade'] > 0 else 0, dtype=torch.float32)
        else:
            label = torch.tensor(row['grade'], dtype=torch.long)
        
        return img, label
