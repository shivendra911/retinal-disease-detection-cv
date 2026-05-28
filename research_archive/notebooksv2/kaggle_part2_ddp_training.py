import os
import subprocess

# =====================================================================
# THE COMPLETE MSDNET TRAINING SCRIPT
# =====================================================================
final_msdnet_script = """
import os, sys, ast, random
import numpy as np
import pandas as pd
import cv2

# CRITICAL DDP FIX: Prevent OpenCV multi-threading deadlocks in DataLoaders
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import timm
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler

import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. CONFIGURATION & AUTO-DISCOVERY
# =====================================================================
CFG = {
    'seed': 42,
    'backbone': 'tf_efficientnet_b4_ns', # As specified in paper
    'image_size': 224,                   # 224x224 input and mask resolution
    'batch_size': 8,
    'epochs': 60,
    'lr': 1e-4,
    'ckpt_dir': '/kaggle/working/checkpoints',
}

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def discover_paths(rank):
    paths = {
        'aptos_csv': None, 'aptos_imgs': None,
        'idrid_csv': None, 'idrid_imgs': None,
        'hrdc_csv': None, 'hrdc_imgs': None,
        'drive_imgs': None, 'drive_masks': None
    }
    for root, dirs, files in os.walk('/kaggle/input'):
        # 1. APTOS
        if 'train.csv' in files and 'aptos' in root.lower():
            paths['aptos_csv'] = os.path.join(root, 'train.csv')
            paths['aptos_imgs'] = os.path.join(root, 'train_images')
            
        # 2. IDRiD
        for f in files:
            if 'idrid_disease grading_training labels.csv' in f.lower():
                paths['idrid_csv'] = os.path.join(root, f)
                for r2, _, _ in os.walk('/kaggle/input'):
                    if 'Original Images' in r2 and 'Training Set' in r2:
                        paths['idrid_imgs'] = r2
                        break

        # 3. HRDC (Fixed Image Discovery)
        for f in files:
            if f.endswith('.csv') and 'hypertensive' in root.lower():
                paths['hrdc_csv'] = os.path.join(root, f)
                # Go up one level from '2-Groundtruths' and scan for the image folder
                dataset_root = os.path.dirname(root)
                for r2, d2, f2 in os.walk(dataset_root):
                    # If this directory contains standard images, mark it as the image dir
                    if any(img_file.endswith(('.png', '.jpg', '.jpeg')) for img_file in f2):
                        paths['hrdc_imgs'] = r2
                        break

        # 4. DRIVE
        if 'drive' in root.lower() and 'training' in root.lower() and 'images' in root.lower() and not paths['drive_imgs']:
            paths['drive_imgs'] = root
        if 'drive' in root.lower() and 'training' in root.lower() and '1st_manual' in root.lower() and not paths['drive_masks']:
            paths['drive_masks'] = root

    if rank == 0:
        print("-" * 50)
        for k, v in paths.items(): print(f"{k.upper()}: {v}")
        print("-" * 50)
    return paths

# =====================================================================
# 2. REAL DATASETS
# =====================================================================
def robust_imread(img_dir, base_name):
    for ext in ['', '.jpg', '.png', '.jpeg', '.tif']:
        path = os.path.join(img_dir, str(base_name) + ext)
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise FileNotFoundError(f"Image not found: {base_name} in {img_dir}")

def preprocess(img):
    img = cv2.resize(img, (CFG['image_size'], CFG['image_size']))
    return torch.from_numpy(img.transpose(2,0,1)).float() / 255.0

# Loader A: APTOS + IDRiD (5-Class DR)
class DRDataset(Dataset):
    def __init__(self, df, img_dir, id_col, label_col):
        self.df = df.reset_index(drop=True)
        self.img_dir, self.id_col, self.label_col = img_dir, id_col, label_col

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = robust_imread(self.img_dir, row[self.id_col])
        return preprocess(img), torch.tensor(int(row[self.label_col]), dtype=torch.long)

# Loader B: HRDC (Binary HR + Pseudo DR)
class HRDCDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.id_col = self.df.columns[0] 
        self.label_col = next((col for col in ['label', 'target', 'diagnosis', 'class'] if col in self.df.columns), self.df.columns[1])

    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = robust_imread(self.img_dir, str(row[self.id_col]))
        
        hr_label = float(row[self.label_col]) # Binary 0 or 1
        pseudo_dr = random.randint(0, 4)      # Random pseudo-label to satisfy contrastive loss dims
        
        return preprocess(img), torch.tensor([hr_label], dtype=torch.float32), torch.tensor(pseudo_dr, dtype=torch.long)

# Loader C: DRIVE (1-Channel Binary Vessel Mask)
class DRIVEDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.tif', '.png', '.jpg'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.gif', '.tif', '.png'))])
        self.img_dir, self.mask_dir = img_dir, mask_dir

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = robust_imread(self.img_dir, self.images[idx])
        mask = cv2.imread(os.path.join(self.mask_dir, self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (CFG['image_size'], CFG['image_size']))
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return preprocess(img), mask_tensor

# =====================================================================
# 3. MSDNet ARCHITECTURE COMPONENTS (From Paper)
# =====================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lat5 = nn.Conv2d(in_channels_list[2], out_channels, 1)
        self.lat4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lat3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.smooth = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, p3, p4, p5):
        p5_lat = self.lat5(p5)
        p4_lat = self.lat4(p4) + F.interpolate(p5_lat, size=p4.shape[-2:], mode='nearest')
        p3_lat = self.lat3(p3) + F.interpolate(p4_lat, size=p3.shape[-2:], mode='nearest')
        return self.smooth(p3_lat)

class MSDNet(nn.Module):
    def __init__(self, backbone_name=CFG['backbone']):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True, out_indices=(2, 3, 4))
        in_channels = self.backbone.feature_info.channels()
        
        self.fpn = FPN(in_channels_list=in_channels, out_channels=256)

        # DR Branch (5-Class)
        self.dr_cbam = CBAM(256)
        self.dr_pool = nn.AdaptiveAvgPool2d(1)
        self.dr_fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 5))

        # HR Branch (Binary)
        self.hr_cbam = CBAM(256)
        self.hr_pool = nn.AdaptiveAvgPool2d(1)
        self.hr_fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 1))

        # Vessel Decoder
        self.vessel_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels[0], 64, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.Upsample(size=(CFG['image_size'], CFG['image_size']), mode='bilinear', align_corners=False),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x, task='disease'):
        features = self.backbone(x)
        p3, p4, p5 = features[0], features[1], features[2]

        if task == 'vessel':
            return self.vessel_decoder(p3)

        fpn_out = self.fpn(p3, p4, p5)
        
        dr_feat = self.dr_pool(self.dr_cbam(fpn_out)).flatten(1)
        dr_logits = self.dr_fc(dr_feat)

        hr_feat = self.hr_pool(self.hr_cbam(fpn_out)).flatten(1)
        hr_logits = self.hr_fc(hr_feat)

        return dr_logits, hr_logits, dr_feat, hr_feat

# =====================================================================
# 4. LOSS FUNCTIONS (From Paper)
# =====================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()

class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
        preds = torch.sigmoid(logits)
        intersection = (preds * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return bce_loss + dice_loss

def contrastive_disentanglement_loss(dr_feat, hr_feat, dr_labels, hr_labels, margin_pure=0.1, margin_co=0.3):
    dr_norm = F.normalize(dr_feat, dim=1)
    hr_norm = F.normalize(hr_feat, dim=1)
    cos_sim = (dr_norm * hr_norm).sum(dim=1)
    
    has_dr = (dr_labels > 0).float()
    has_hr = (hr_labels.squeeze() > 0).float()
    
    is_pure = (has_dr != has_hr).float()
    is_co = (has_dr * has_hr)
    
    loss_pure = is_pure * F.relu(cos_sim - margin_pure)
    loss_co = is_co * F.relu(cos_sim - margin_co)
    
    total_pure = is_pure.sum().clamp(min=1)
    total_co = is_co.sum().clamp(min=1)
    
    return (loss_pure.sum() / total_pure) + (loss_co.sum() / total_co)

# =====================================================================
# 5. DDP TRAINING LOOP (Algorithm 1)
# =====================================================================
def get_next_batch(iterator, loader):
    # Safely handles StopIteration by refreshing the dataloader
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator

def main():
    dist.init_process_group("nccl")
    rank, local_rank, world_size = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    set_seed(CFG['seed'])
    os.makedirs(CFG['ckpt_dir'], exist_ok=True)
    paths = discover_paths(rank)

    # Load Full Datasets
    aptos_df = pd.read_csv(paths['aptos_csv']) if paths['aptos_csv'] else pd.DataFrame()
    idrid_df = pd.read_csv(paths['idrid_csv']) if paths['idrid_csv'] else pd.DataFrame()
    hrdc_df = pd.read_csv(paths['hrdc_csv']) if paths['hrdc_csv'] else pd.DataFrame()

    dr_datasets = []
    if len(aptos_df) > 0: dr_datasets.append(DRDataset(aptos_df, paths['aptos_imgs'], 'id_code', 'diagnosis'))
    if len(idrid_df) > 0: dr_datasets.append(DRDataset(idrid_df, paths['idrid_imgs'], 'Image name', 'Retinopathy grade'))
    
    dataset_a = ConcatDataset(dr_datasets) if dr_datasets else None
    dataset_b = HRDCDataset(hrdc_df, paths['hrdc_imgs']) if len(hrdc_df) > 0 else None
    dataset_c = DRIVEDataset(paths['drive_imgs'], paths['drive_masks']) if paths['drive_imgs'] else None

    if not dataset_a or not dataset_b:
        if rank == 0: print("❌ Missing primary datasets (A or B). Aborting.")
        dist.destroy_process_group()
        return

    # Distributed Samplers & Loaders
    sampler_a = DistributedSampler(dataset_a, num_replicas=world_size, rank=rank)
    sampler_b = DistributedSampler(dataset_b, num_replicas=world_size, rank=rank)
    
    loader_a = DataLoader(dataset_a, batch_size=CFG['batch_size'], sampler=sampler_a, pin_memory=True)
    loader_b = DataLoader(dataset_b, batch_size=CFG['batch_size'], sampler=sampler_b, pin_memory=True)
    
    iter_a, iter_b = iter(loader_a), iter(loader_b)
    
    if dataset_c:
        sampler_c = DistributedSampler(dataset_c, num_replicas=world_size, rank=rank)
        loader_c = DataLoader(dataset_c, batch_size=4, sampler=sampler_c, pin_memory=True)
        iter_c = iter(loader_c)
    else:
        iter_c = None

    if rank == 0: print(f"🚀 Initializing MSDNet with find_unused_parameters=True")
    model = MSDNet().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'])
    scaler = torch.amp.GradScaler('cuda')

    criterion_dr = FocalLoss(gamma=2.0)
    criterion_hr = nn.BCEWithLogitsLoss()
    criterion_vessel = DiceBCELoss()

    if rank == 0: print(f"Starting Training: {CFG['epochs']} Epochs")

    for epoch in range(CFG['epochs']):
        sampler_a.set_epoch(epoch)
        sampler_b.set_epoch(epoch)
        if iter_c: sampler_c.set_epoch(epoch)
        
        model.train()
        
        # Primary loop driven by the size of the DR dataset (Loader A)
        for step in range(len(loader_a)):
            
            # --- ALGORITHM 1: Synchronized Random Alternation ---
            u_tensor = torch.tensor([0.0], device=device)
            if rank == 0:
                u_tensor[0] = random.random()
            dist.broadcast(u_tensor, src=0)
            u = u_tensor.item()
            
            optimizer.zero_grad()

            # --- PATH A: APTOS + IDRiD (u < 0.6) Compute ONLY L_DR ---
            if u < 0.6:
                batch, iter_a = get_next_batch(iter_a, loader_a)
                imgs, dr_labels = batch[0].to(device), batch[1].to(device)
                
                with torch.amp.autocast('cuda'):
                    dr_logits, _, _, _ = model(imgs, task='disease')
                    loss_dr = criterion_dr(dr_logits, dr_labels)
                    
                scaler.scale(loss_dr).backward()
                main_loss_val = loss_dr.item()
                task_name = "Loader A (DR)"

            # --- PATH B: HRDC (u >= 0.6) Compute L_HR + L_Dis ---
            else:
                batch, iter_b = get_next_batch(iter_b, loader_b)
                imgs, hr_labels, pseudo_dr = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                
                with torch.amp.autocast('cuda'):
                    dr_logits, hr_logits, dr_feat, hr_feat = model(imgs, task='disease')
                    loss_hr = criterion_hr(hr_logits, hr_labels)
                    loss_dis = contrastive_disentanglement_loss(dr_feat, hr_feat, pseudo_dr, hr_labels)
                    total_hr_loss = loss_hr + (0.5 * loss_dis)
                    
                scaler.scale(total_hr_loss).backward()
                main_loss_val = total_hr_loss.item()
                task_name = "Loader B (HR+Dis)"

            # --- PATH C: DRIVE (Every 5th Step) Compute ONLY L_Vessel ---
            if step % 5 == 0 and iter_c:
                batch, iter_c = get_next_batch(iter_c, loader_c)
                imgs, masks = batch[0].to(device), batch[1].to(device)
                
                with torch.amp.autocast('cuda'):
                    vessel_logits = model(imgs, task='vessel')
                    loss_vessel = criterion_vessel(vessel_logits, masks)
                    
                scaler.scale(loss_vessel).backward()
                task_name += " + Loader C (Vessel)"

            scaler.step(optimizer)
            scaler.update()

            if rank == 0 and step % 50 == 0:
                print(f"Epoch [{epoch+1}/{CFG['epochs']}] Step [{step}/{len(loader_a)}] | u={u:.2f} | {task_name} | Main Loss: {main_loss_val:.4f}")

        # --- RANK 0 CHECKPOINTING ---
        dist.barrier()
        if rank == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(), # Strips the DDP wrapper for inference
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"{CFG['ckpt_dir']}/msdnet_epoch_{epoch+1}.pth")
            print(f"✅ Checkpoint saved: {CFG['ckpt_dir']}/msdnet_epoch_{epoch+1}.pth")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
"""

with open("train_msdnet.py", "w") as f:
    f.write(final_msdnet_script)

print("Final Script written to train_msdnet.py. Launching Multi-GPU torchrun...\\n" + "="*60)

subprocess.run([
    "python", "-m", "torch.distributed.run", 
    "--standalone", 
    "--nproc_per_node=2", 
    "train_msdnet.py"
])
