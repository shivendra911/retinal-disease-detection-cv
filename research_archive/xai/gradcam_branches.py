"""
IRDAS XAI — Per-Branch Grad-CAM++ Heatmaps
=============================================

Generates SEPARATE Grad-CAM++ heatmaps for DR and HR branches.

Each heatmap shows WHERE in the image the model found evidence
for that specific disease:
- DR heatmap → highlights: microaneurysms, hemorrhage dots, exudates
- HR heatmap → highlights: arteriovenous nicking, vessel caliber changes

These heatmaps serve two purposes:
1. Paper Figure 2 — qualitative XAI results (proves disentanglement works)
2. Stage 3 input — tells the LLM WHERE the disease was found

Why Grad-CAM++ (not Grad-CAM)?
Grad-CAM++ handles multiple instances better. Retinal images often have
multiple scattered lesions — Grad-CAM++ captures all of them.

Reference: Chattopadhyay et al., "Grad-CAM++", WACV 2018
"""

import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image


def get_per_branch_heatmaps(model, input_tensor, original_img_np):
    """
    Generate separate Grad-CAM++ heatmaps for DR and HR branches.
    
    Args:
        model: Trained MSDNet instance
        input_tensor: (1, 3, 224, 224) preprocessed image tensor (GPU)
        original_img_np: (224, 224, 3) float [0,1] for overlay
    
    Returns:
        overlay_dr: RGB numpy array — DR heatmap overlaid on original
        overlay_hr: RGB numpy array — HR heatmap overlaid on original
        grayscale_dr: (224, 224) — raw DR heatmap values
        grayscale_hr: (224, 224) — raw HR heatmap values
    """
    model.eval()
    model.training_mode = False
    
    # Target: spatial attention output of each branch's CBAM
    dr_target_layer = [model.dr_branch.cbam.spatial.conv]
    hr_target_layer = [model.hr_branch.cbam.spatial.conv]
    
    cam_dr = GradCAMPlusPlus(model=model, target_layers=dr_target_layer)
    cam_hr = GradCAMPlusPlus(model=model, target_layers=hr_target_layer)
    
    # Generate heatmaps
    grayscale_dr = cam_dr(input_tensor=input_tensor)
    grayscale_hr = cam_hr(input_tensor=input_tensor)
    
    # Overlay on original image
    overlay_dr = show_cam_on_image(original_img_np, grayscale_dr[0], use_rgb=True)
    overlay_hr = show_cam_on_image(original_img_np, grayscale_hr[0], use_rgb=True)
    
    return overlay_dr, overlay_hr, grayscale_dr[0], grayscale_hr[0]


def describe_heatmap_regions(grayscale_cam, threshold=0.5):
    """
    Converts heatmap to text description for LangChain prompt.
    
    Identifies which quadrant of the retina has the highest activation.
    Used in Stage 3 to tell the LLM WHERE the disease was found.
    
    Retinal quadrant naming follows ophthalmological convention:
    - Superior nasal / Superior temporal (top half)
    - Inferior nasal / Inferior temporal (bottom half)
    - Central (macula) — most critical area for vision
    
    Args:
        grayscale_cam: (H, W) float array, values 0-1
        threshold: Activation threshold to consider a region "active"
    
    Returns:
        String describing the active retinal regions
    """
    h, w = grayscale_cam.shape
    quadrants = {
        'superior nasal'  : grayscale_cam[:h//2, :w//2].mean(),
        'superior temporal': grayscale_cam[:h//2, w//2:].mean(),
        'inferior nasal'  : grayscale_cam[h//2:, :w//2].mean(),
        'inferior temporal': grayscale_cam[h//2:, w//2:].mean(),
        'central (macula)': grayscale_cam[h//3:2*h//3, w//3:2*w//3].mean(),
    }
    active = [q for q, v in quadrants.items() if v > threshold]
    if not active:
        return "no specific region highlighted"
    return "primarily in the " + " and ".join(active) + " region"


def save_heatmap_pair(overlay_dr, overlay_hr, original_img_np, save_path, image_id):
    """
    Save a side-by-side comparison of DR and HR heatmaps.
    
    Creates a figure with 3 panels:
    [Original | DR Heatmap | HR Heatmap]
    
    Args:
        overlay_dr: DR heatmap overlay (RGB numpy)
        overlay_hr: HR heatmap overlay (RGB numpy)
        original_img_np: Original image (RGB, float 0-1)
        save_path: Directory to save the figure
        image_id: Identifier string for the filename
    """
    import matplotlib.pyplot as plt
    import os
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img_np)
    axes[0].set_title('Original Fundus', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(overlay_dr)
    axes[1].set_title('DR Attention (Grad-CAM++)', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(overlay_hr)
    axes[2].set_title('HR Attention (Grad-CAM++)', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'xai_{image_id}.png'), dpi=150, bbox_inches='tight')
    plt.close()
