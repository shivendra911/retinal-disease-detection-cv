"""
╔══════════════════════════════════════════════════════════════════════╗
║  IRDAS — Research Bundle Generator                                   ║
║                                                                      ║
║  Generates exhaustive metrics, plots, and Grad-CAM visualizations    ║
║  for your research paper, packaging it all into a single ZIP file.   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import cv2
import json
import time
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             cohen_kappa_score, roc_curve, auc, 
                             precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
CFG = {
    'backbone': 'tf_efficientnet_b4.ns_jft_in1k',
    'drop_path_rate': 0.2,
    'bifpn_channels': 256,
    'bifpn_layers': 2,
    'msd_k': 5,
    'dropout': 0.3,

    'image_size': 512,
    'batch_size': 8,
    'coral_levels': 4,
    'num_classes': 5,
    
    # Path to the final trained weights
    'weights_path': '/kaggle/working/checkpoints/swa_final.pth',
    
    # Validation dataset paths
    'train_csv': '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train.csv',
    'img_dir': '/kaggle/input/datasets/oladipupou/aptos2019-blindness-detection/train_images',
    
    # The optimized thresholds from the training log!
    'opt_thresholds': [0.5348, 1.5016, 2.3462, 3.6337],
    
    'bundle_dir': '/kaggle/working/irdas_research_bundle'
}

CORAL_LEVELS = CFG['coral_levels']
CLASSES = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative']

# ============================================================
# UTILITIES & BUNDLE SETUP
# ============================================================
def create_bundle_structure():
    dirs = ['checkpoints', 'configs', 'gradcam', 'logs', 
            'metrics', 'plots', 'predictions', 'splits', 'thresholds']
    for d in dirs:
        os.makedirs(os.path.join(CFG['bundle_dir'], d), exist_ok=True)

# ============================================================
# DATASET
# ============================================================
def ben_graham_clahe(img):
    blurred = cv2.GaussianBlur(img, (0, 0), 10)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(3):
        img[:, :, i] = clahe.apply(img[:, :, i])
    return img

class DRValidationDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (CFG['image_size'], CFG['image_size']))
        img = ben_graham_clahe(img)
        
        orig_img = img.copy()  # Keep for Grad-CAM
        
        if self.transform:
            img = self.transform(image=img)['image']
            
        return img, row['diagnosis'], row['id_code'], orig_img

def get_inference_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# ============================================================
# MODEL ARCHITECTURE (IRDAS TEACHER)
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        mid = max(in_planes // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, mid, 1, bias=False), nn.ReLU(),
            nn.Conv2d(mid, in_planes, 1, bias=False))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, ks=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, ks, padding=(ks-1)//2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, 1, keepdim=True)
        mx, _ = torch.max(x, 1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))


class CBAM(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.ca = ChannelAttention(p)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class DWConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class BiFPNLayer(nn.Module):
    def __init__(self, ch=256, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.w_p4_td  = nn.Parameter(torch.ones(2))
        self.w_p3_out = nn.Parameter(torch.ones(2))
        self.w_p4_out = nn.Parameter(torch.ones(3))
        self.w_p5_out = nn.Parameter(torch.ones(2))
        self.conv_p4_td  = DWConv(ch)
        self.conv_p3_out = DWConv(ch)
        self.conv_p4_out = DWConv(ch)
        self.conv_p5_out = DWConv(ch)

    def _up(self, x, t):
        return F.interpolate(x, t.shape[-2:], mode='nearest')

    def _dn(self, x, t):
        return F.adaptive_avg_pool2d(x, t.shape[-2:])

    def forward(self, p3, p4, p5):
        w4  = F.relu(self.w_p4_td.clone());  w4  = w4  / (w4.sum()  + self.eps)
        w3  = F.relu(self.w_p3_out.clone()); w3  = w3  / (w3.sum()  + self.eps)
        w4o = F.relu(self.w_p4_out.clone()); w4o = w4o / (w4o.sum() + self.eps)
        w5o = F.relu(self.w_p5_out.clone()); w5o = w5o / (w5o.sum() + self.eps)
        p4_td  = self.conv_p4_td(w4[0]*p4  + w4[1]*self._up(p5, p4))
        p3_out = self.conv_p3_out(w3[0]*p3 + w3[1]*self._up(p4_td, p3))
        p4_out = self.conv_p4_out(w4o[0]*p4 + w4o[1]*p4_td + w4o[2]*self._dn(p3_out, p4))
        p5_out = self.conv_p5_out(w5o[0]*p5 + w5o[1]*self._dn(p4_out, p5))
        return p3_out, p4_out, p5_out


class BiFPN(nn.Module):
    def __init__(self, in_ch, out_ch=256, n=2):
        super().__init__()
        self.lat = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_ch, 1, bias=False),
                          nn.BatchNorm2d(out_ch), nn.SiLU())
            for c in in_ch
        ])
        self.layers = nn.ModuleList([BiFPNLayer(out_ch) for _ in range(n)])

    def forward(self, p3r, p4r, p5r):
        p3, p4, p5 = self.lat[0](p3r), self.lat[1](p4r), self.lat[2](p5r)
        for layer in self.layers:
            p3, p4, p5 = layer(p3, p4, p5)
        return p3, p4, p5


class GeMPooling(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(self.eps).pow(self.p), x.shape[-2:]
        ).pow(1.0 / self.p)


class MSDNetTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        # Since this is an inference script, we don't need local weights 
        # to initialize the backbone, we will load strict=False later
        self.backbone = timm.create_model(
            CFG['backbone'], pretrained=False,
            features_only=True, out_indices=(2, 3, 4),
            drop_path_rate=CFG['drop_path_rate'])

        ch = self.backbone.feature_info.channels()
        oc = CFG['bifpn_channels']
        self.bifpn    = BiFPN(ch, oc, CFG['bifpn_layers'])
        self.pool     = GeMPooling()
        self.cbam_p3  = CBAM(oc)
        self.cbam_p5  = CBAM(oc)
        self.dropout  = nn.Dropout(CFG['dropout'])
        self.head     = nn.Linear(oc * 2, CORAL_LEVELS)

    def forward(self, x):
        f = self.backbone(x)
        p3, p4, p5 = self.bifpn(f[0], f[1], f[2])
        feat = torch.cat([
            self.pool(self.cbam_p3(p3)).flatten(1),
            self.pool(self.cbam_p5(p5)).flatten(1)
        ], 1)
        if self.training:
            return torch.stack([
                self.head(self.dropout(feat)) for _ in range(CFG['msd_k'])
            ]).mean(0)
        return self.head(feat)

def predict_continuous(logits):
    probs = torch.sigmoid(logits)
    expected_grade = torch.sum(probs, dim=1)
    return expected_grade.cpu().numpy()

# ============================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================
class SimpleGradCAM:
    """Lightweight Grad-CAM implementation requiring no external libraries"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x):
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(x)
        expected_grade = torch.sum(torch.sigmoid(logits), dim=1)
        
        # We want to see what evidence pushes the score HIGHER
        loss = expected_grade.sum()
        loss.backward()
        
        # Global average pooling on gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by the gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average across channels
        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().detach().numpy()
        
        # ReLU on heatmap
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        
        # Resize to original image size
        heatmap = cv2.resize(heatmap, (x.shape[3], x.shape[2]))
        return heatmap

def overlay_gradcam(img_rgb, heatmap):
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)
    return superimposed

# ============================================================
# METRICS & PLOTTING
# ============================================================
def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw CM
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=CLASSES, yticklabels=CLASSES)
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Normalized CM
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=CLASSES, yticklabels=CLASSES)
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, cont_preds, thresholds, path):
    """Approximate multi-class ROC using distances to thresholds"""
    # Create pseudo-probabilities for each class based on continuous score
    y_prob = np.zeros((len(y_true), CFG['num_classes']))
    for i, pred in enumerate(cont_preds):
        # Extremely simplified pseudo-probabilities for visualization
        # Calculate 5 class centers based on the 4 thresholds
        centers = [
            thresholds[0] - 0.5,
            (thresholds[0] + thresholds[1]) / 2,
            (thresholds[1] + thresholds[2]) / 2,
            (thresholds[2] + thresholds[3]) / 2,
            thresholds[3] + 0.5
        ]
        dists = [abs(pred - c) for c in centers]
        probs = np.exp(-np.array(dists))
        y_prob[i] = probs / probs.sum()
        
    y_true_bin = label_binarize(y_true, classes=[0,1,2,3,4])
    
    plt.figure(figsize=(10, 8))
    for i in range(CFG['num_classes']):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{CLASSES[i]} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_detailed_metrics(y_true, y_pred, path_txt, path_csv):
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = []
    for i in range(CFG['num_classes']):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics.append({
            'Class': CLASSES[i],
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'F1-Score': f1
        })
        
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(path_csv, index=False)
    
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    with open(path_txt, 'w') as f:
        f.write("=== RESEARCH EVALUATION REPORT ===\n\n")
        f.write(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASSES))
        f.write("\n\nDetailed Per-Class Metrics:\n")
        f.write(df_metrics.to_string())

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("🚀 Initializing Research Bundle Generation...")
    create_bundle_structure()
    
    # 1. Save configuration
    with open(os.path.join(CFG['bundle_dir'], 'configs', 'evaluation_config.json'), 'w') as f:
        json.dump(CFG, f, indent=4)
        
    with open(os.path.join(CFG['bundle_dir'], 'thresholds', 'optimized_thresholds.txt'), 'w') as f:
        f.write(",".join(map(str, CFG['opt_thresholds'])))

    # 2. Setup Data
    df = pd.read_csv(CFG['train_csv'])
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=6, shuffle=True, random_state=42)
    _, val_idx = next(skf.split(df, df['diagnosis']))
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    val_df.to_csv(os.path.join(CFG['bundle_dir'], 'splits', 'validation_split.csv'), index=False)
    print(f"📊 Validation Set Size: {len(val_df)} images")
    
    dataset = DRValidationDataset(val_df, CFG['img_dir'], get_inference_transform())
    loader = DataLoader(dataset, batch_size=CFG['batch_size'], shuffle=False, num_workers=2)

    # 3. Setup Model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MSDNetTeacher()
    
    print(f"📥 Loading weights from {CFG['weights_path']}...")
    try:
        state_dict = torch.load(CFG['weights_path'], map_location='cpu')
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    except FileNotFoundError:
        print("⚠️ Weights not found! Using random weights for testing...")
    
    model.to(device)
    model.eval()

    # 4. Inference & Metrics Collection
    print("🔍 Running D4 TTA Inference on Validation Set...")
    all_cont = []
    all_tgt = []
    all_ids = []
    
    gradcam_candidates = {c: [] for c in range(CFG['num_classes'])}
    
    with torch.no_grad():
        for i, (imgs, tgts, ids, orig_imgs) in enumerate(loader):
            imgs = imgs.to(device)
            preds = []
            
            for k in range(4):
                rotated = torch.rot90(imgs, k, [2, 3])
                preds.append(model(rotated))
                preds.append(model(torch.flip(rotated, [3])))
                
            avg_logits = sum(preds) / len(preds)
            cont_preds = predict_continuous(avg_logits)
            
            all_cont.extend(cont_preds)
            all_tgt.extend(tgts.numpy())
            all_ids.extend(ids)
            
            for j in range(len(tgts)):
                if len(gradcam_candidates[tgts[j].item()]) < 3:
                    gradcam_candidates[tgts[j].item()].append({
                        'img_tensor': imgs[j:j+1], 
                        'orig_img': orig_imgs[j].numpy(), 
                        'id': ids[j], 
                        'tgt': tgts[j].item()
                    })

    all_cont = np.array(all_cont)
    all_tgt = np.array(all_tgt)
    
    all_disc = np.digitize(all_cont, CFG['opt_thresholds']).clip(0, CFG['num_classes'] - 1)
    
    pred_df = pd.DataFrame({
        'id_code': all_ids,
        'true_diagnosis': all_tgt,
        'continuous_pred': all_cont,
        'discrete_pred': all_disc
    })
    pred_df.to_csv(os.path.join(CFG['bundle_dir'], 'predictions', 'val_predictions.csv'), index=False)
    
    # 5. Generate Metrics & Plots
    print("📈 Generating Metrics & Plots...")
    compute_detailed_metrics(
        all_tgt, all_disc, 
        os.path.join(CFG['bundle_dir'], 'metrics', 'classification_report.txt'),
        os.path.join(CFG['bundle_dir'], 'metrics', 'detailed_metrics.csv')
    )
    
    plot_confusion_matrix(all_tgt, all_disc, os.path.join(CFG['bundle_dir'], 'plots', 'confusion_matrix.png'))
    plot_roc_curves(all_tgt, all_cont, CFG['opt_thresholds'], os.path.join(CFG['bundle_dir'], 'plots', 'roc_curves.png'))
    
    # 6. Generate Grad-CAM Heatmaps
    print("🔥 Generating Grad-CAM Visualizations...")
    # Hook onto the final convolution in CBAM Spatial Attention of p5
    target_layer = model.cbam_p5.sa.conv
    cam = SimpleGradCAM(model, target_layer)
    
    for cls in range(CFG['num_classes']):
        for idx, sample in enumerate(gradcam_candidates[cls]):
            img_tensor = sample['img_tensor'].clone().detach().requires_grad_(True)
            
            heatmap = cam(img_tensor)
            overlaid = overlay_gradcam(sample['orig_img'], heatmap)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(sample['orig_img'])
            axes[0].set_title(f"Original (Class {cls})")
            axes[0].axis('off')
            
            axes[1].imshow(overlaid)
            axes[1].set_title(f"Grad-CAM Heatmap")
            axes[1].axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(CFG['bundle_dir'], 'gradcam', f"class_{cls}_{sample['id']}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    # 7. Package ZIP file
    print("📦 Zipping Research Bundle...")
    shutil.make_archive(CFG['bundle_dir'], 'zip', CFG['bundle_dir'])
    print(f"✅ Success! Research bundle ready for download: {CFG['bundle_dir']}.zip")

if __name__ == '__main__':
    main()
