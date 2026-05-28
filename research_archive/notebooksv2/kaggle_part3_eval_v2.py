import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import google.generativeai as genai
import timm

# %%
# =====================================================================
# CELL 1: Isolate the Architecture Definition
# =====================================================================
# When PyTorch loads a saved .pth file, it needs the exact blueprint 
# of the model in its memory namespace.
CFG = {
    'backbone': 'tf_efficientnet_b4_ns',
    'image_size': 224,
}

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
        # Ensure pretrained=False for evaluation
        self.backbone = timm.create_model(backbone_name, pretrained=False, features_only=True, out_indices=(2, 3, 4))
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


# %%
# =====================================================================
# CELL 2: Build the Generalization Testing Loop
# =====================================================================
def test_generalization(model, idrid_loader, device):
    model.eval()
    all_dr_preds = []
    all_dr_targets = []
    
    print("Evaluating DR Generalization on IDRiD Dataset...")
    with torch.no_grad():
        for imgs, labels in idrid_loader:
            imgs = imgs.to(device)
            dr_logits, hr_logits, _, _ = model(imgs, task='disease')
            
            # Convert logits to class predictions
            preds = torch.argmax(dr_logits, dim=1).cpu().numpy()
            all_dr_preds.extend(preds)
            all_dr_targets.extend(labels.numpy())
            
    # Calculate Quadratic Weighted Kappa for DR Generalization
    qwk = cohen_kappa_score(all_dr_targets, all_dr_preds, weights='quadratic')
    print(f"✅ IDRiD Generalization DR QWK: {qwk:.4f}")
    return qwk


# %%
# =====================================================================
# CELL 3: Implement the MC Dropout Uncertainty Engine
# =====================================================================
def predict_with_uncertainty(model, image_tensor, num_passes=30):
    model.train() # CRITICAL: Keeps dropout layers active
    
    dr_preds = []
    hr_preds = []
    
    with torch.no_grad():
        for _ in range(num_passes):
            dr_logits, hr_logits, _, _ = model(image_tensor, task='disease')
            dr_preds.append(torch.softmax(dr_logits, dim=1).cpu().numpy())
            hr_preds.append(torch.sigmoid(hr_logits).cpu().numpy())
            
    dr_preds = np.array(dr_preds) # Shape: (30, 1, 5)
    
    # Mean probability across 30 passes
    dr_mean = np.mean(dr_preds, axis=0) 
    # Variance (Uncertainty Score)
    dr_variance = np.var(dr_preds, axis=0).mean() 
    
    hr_preds = np.array(hr_preds)
    hr_mean = np.mean(hr_preds, axis=0)
    
    # 1. Define your Nelder-Mead optimized thresholds from your training logs
    thresholds = [0.875, 1.469, 2.401, 2.890]
    
    # 2. Calculate the Expected Value (Continuous DR Score)
    classes = np.array([0, 1, 2, 3, 4])
    dr_continuous_score = np.sum(dr_mean[0] * classes)
    
    # 3. Apply the threshold brackets to get the final ordinal integer grade
    dr_grade = 0
    for threshold in thresholds:
        if dr_continuous_score > threshold:
            dr_grade += 1
    hr_bool = bool(hr_mean[0][0] > 0.5)
    
    return dr_grade, hr_bool, float(dr_variance)


# %%
# =====================================================================
# CELL 4: Hook the Dual-Branch Grad-CAM++
# =====================================================================
def generate_heatmaps(model, img_tensor, original_rgb_img):
    model.eval() # Grad-CAM should be run in eval mode
    
    # Hook into the DR Branch (Spatial Attention Conv)
    dr_cam = GradCAMPlusPlus(model=model, target_layers=[model.dr_cbam.sa.conv])
    
    # Hook into the HR Branch (Spatial Attention Conv)
    hr_cam = GradCAMPlusPlus(model=model, target_layers=[model.hr_cbam.sa.conv])

    # Generate Heatmaps
    dr_grayscale_cam = dr_cam(input_tensor=img_tensor)[0]
    hr_grayscale_cam = hr_cam(input_tensor=img_tensor)[0]

    # Overlay on original image (needs to be normalized to 0-1)
    original_rgb_img_norm = original_rgb_img / 255.0
    dr_visualization = show_cam_on_image(original_rgb_img_norm, dr_grayscale_cam, use_rgb=True)
    hr_visualization = show_cam_on_image(original_rgb_img_norm, hr_grayscale_cam, use_rgb=True)
    
    return dr_visualization, hr_visualization


# %%
# =====================================================================
# CELL 5: Attach the Gemini Pro Multilingual Reporter
# =====================================================================
def generate_clinical_report(dr_grade, hr_bool, variance, api_key):
    genai.configure(api_key=api_key)
    # Using gemini-pro (or gemini-1.5-pro) for text generation
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    You are an ophthalmic reporting assistant. The MSDNet AI has analyzed a patient's fundus image. 
    DR Grade: {dr_grade} (Scale 0-4)
    HR Present: {hr_bool}
    AI Uncertainty Score: {variance:.4f}
    
    Generate a plain-language screening report in English, Hindi, and Tamil. 
    If Uncertainty > 0.05, add a strict warning that manual specialist review is highly recommended due to complex imaging artifacts.
    Format clearly using markdown.
    """
    
    response = model.generate_content(prompt)
    return response.text


# %%
# =====================================================================
# EXAMPLE EXECUTION STUB (How to use this notebook)
# =====================================================================
if __name__ == "__main__":
    print("✅ Inference Pipeline Ready!")
    print("How to run this in your Notebook/Backend:")
    print("1. model = MSDNet().to(device)")
    print("2. model.load_state_dict(torch.load('checkpoints/msdnet_best.pth'))")
    print("3. qwk = test_generalization(model, idrid_loader, device)")
    print("4. dr_grade, hr_bool, variance = predict_with_uncertainty(model, img_tensor)")
    print("5. dr_vis, hr_vis = generate_heatmaps(model, img_tensor, rgb_img)")
    print("6. report = generate_clinical_report(dr_grade, hr_bool, variance, 'YOUR_GEMINI_KEY')")
