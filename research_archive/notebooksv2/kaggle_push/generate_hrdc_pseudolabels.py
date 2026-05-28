"""
╔══════════════════════════════════════════════════════════════════════╗
║  IRDAS — Phase 2: Generate HRDC Pseudo-Labels (Kaggle Script)        ║
║                                                                      ║
║  Takes the trained DR Teacher (swa_final.pth) and generates          ║
║  Diabetic Retinopathy pseudolabels for the HRDC dataset.             ║
║                                                                      ║
║  Output: hrdc_dr_pseudolabels.csv                                    ║
║  Contains original HR grade + teacher's DR continuous & discrete     ║
║  predictions.                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import cv2
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

# ============================================================
# CONFIGURATION
# ============================================================
CFG = {
    'backbone': 'tf_efficientnet_b4.ns_jft_in1k',
    'image_size': 512,  # Use the final fine-tuned resolution
    'batch_size': 16,
    'coral_levels': 4,
    'num_classes': 5,
    
    # Pre-trained teacher weights
    'weights_path': '/kaggle/working/checkpoints/swa_final.pth',
    
    # HRDC Dataset paths on Kaggle
    'hrdc_csv': '/kaggle/input/hrdc-2023/train.csv',  # Adjust this path based on actual Kaggle dataset
    'hrdc_imgs': '/kaggle/input/hrdc-2023/images/',   # Adjust this path based on actual Kaggle dataset
    
    'output_csv': '/kaggle/working/hrdc_dr_pseudolabels.csv'
}

# ============================================================
# PREPROCESSING
# ============================================================
def ben_graham_clahe(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(3):
        img[:, :, i] = clahe.apply(img[:, :, i])
    return img

class HRDCInferenceDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # In case filename is missing extension in CSV
        fname = str(row['filename'])
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            fname += '.jpg'
            
        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path)
        
        # Fallback if image not found
        if img is None:
            # Return a blank image (or handle gracefully in loader)
            img = np.zeros((CFG['image_size'], CFG['image_size'], 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (CFG['image_size'], CFG['image_size']))
            img = ben_graham_clahe(img)
            
        if self.transform:
            img = self.transform(image=img)['image']
            
        return img, row['filename']

def get_inference_transform(image_size):
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ============================================================
# MODEL DEFINITION (CORAL)
# ============================================================
class CoralEfficientNet(nn.Module):
    def __init__(self, backbone_name, levels):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, 1)
        self.register_buffer('bias', torch.arange(levels, dtype=torch.float32))

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        # Broadcasting: [B, 1] + [Levels] -> [B, Levels]
        return logits + self.bias

def predict_continuous(logits):
    """Convert CORAL logits to continuous expected grade [0, 4.0]"""
    probs = torch.sigmoid(logits)
    expected_grade = torch.sum(probs, dim=1)
    return expected_grade.cpu().numpy()

# ============================================================
# INFERENCE LOGIC (TTA)
# ============================================================
def tta_predict(model, loader, device):
    """Full D4 dihedral TTA (8 augmentations) for maximum precision."""
    model.eval()
    all_cont = []
    all_fnames = []
    
    print(f"Starting TTA inference on {len(loader.dataset)} images...")
    start_time = time.time()
    
    with torch.no_grad():
        for i, (imgs, fnames) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            preds = []
            
            # Identity + 3 rotations
            for k in range(4):
                rotated = torch.rot90(imgs, k, [2, 3])
                preds.append(model(rotated))
                # + horizontal flip of each rotation
                preds.append(model(torch.flip(rotated, [3])))
                
            avg_logits = sum(preds) / len(preds)
            cont_preds = predict_continuous(avg_logits)
            
            all_cont.extend(cont_preds)
            all_fnames.extend(fnames)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {i+1}/{len(loader)} batches ({elapsed:.1f}s)")
                
    return all_fnames, np.array(all_cont)

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading HRDC dataset from: {CFG['hrdc_csv']}")
    df = pd.read_csv(CFG['hrdc_csv'])
    
    dataset = HRDCInferenceDataset(
        df, CFG['hrdc_imgs'], get_inference_transform(CFG['image_size'])
    )
    loader = DataLoader(
        dataset, batch_size=CFG['batch_size'], shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    # 2. Load Model
    print("Initializing model...")
    model = CoralEfficientNet(CFG['backbone'], CFG['coral_levels'])
    
    print(f"Loading weights from: {CFG['weights_path']}")
    if not os.path.exists(CFG['weights_path']):
        raise FileNotFoundError(f"Weights not found! Expected at {CFG['weights_path']}")
        
    state_dict = torch.load(CFG['weights_path'], map_location='cpu')
    # If weights were saved with DataParallel, remove 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    
    # 3. Predict with TTA
    fnames, cont_preds = tta_predict(model, loader, device)
    
    # 4. We use the optimized thresholds from Phase 1. 
    # Use the thresholds printed at the very end of your Phase 1 Kaggle log!
    # Update these numbers based on the output of your best model.
    # [0.5348, 1.5016, 2.3462, 3.6337] from your v7 log
    thresholds = [0.5348, 1.5016, 2.3462, 3.6337]
    print(f"Applying discrete thresholds: {thresholds}")
    
    disc_preds = np.digitize(cont_preds, thresholds).clip(0, CFG['num_classes'] - 1)
    
    # 5. Build Output DataFrame
    out_df = pd.DataFrame({
        'filename': fnames,
        'dr_pseudo_continuous': cont_preds,
        'dr_pseudo_discrete': disc_preds
    })
    
    # Merge with original HR labels if they exist in the input csv
    if 'grade' in df.columns:
        out_df = pd.merge(out_df, df[['filename', 'grade']], on='filename', how='left')
        out_df.rename(columns={'grade': 'hr_grade'}, inplace=True)
    
    print("\nPseudolabel Distribution:")
    print(out_df['dr_pseudo_discrete'].value_counts().sort_index())
    
    print(f"\nSaving results to {CFG['output_csv']}")
    out_df.to_csv(CFG['output_csv'], index=False)
    print("Done!")

if __name__ == '__main__':
    main()
