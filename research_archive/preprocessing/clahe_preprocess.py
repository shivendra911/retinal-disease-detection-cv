"""
IRDAS Preprocessing — Fundus Image Preprocessing Pipeline
==========================================================

Applies a standardized 5-step preprocessing pipeline to every retinal fundus image:
1. Ben Graham illumination normalization (2015 Kaggle DR winner technique)
2. CLAHE local contrast enhancement on L channel
3. Optic disc suppression via inpainting
4. Circular field-of-view cropping (removes black borders)
5. Resize + ImageNet normalization

Order matters — do NOT change the sequence.

References:
- Ben Graham: https://kaggle.com/competitions/diabetic-retinopathy-detection/discussion/15801
- CLAHE: Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization.
"""

import cv2
import numpy as np


def ben_graham_preprocess(img, sigmaX=10):
    """
    Ben Graham's preprocessing — winner of 2015 Kaggle DR competition.
    
    Removes uneven illumination across the fundus image by subtracting
    a heavily blurred version of itself, then re-centering pixel values.
    Dramatically sharpens vessel edges and lesion boundaries.
    
    Formula: result = 4 × img - 4 × GaussianBlur(img) + 128
    
    Args:
        img: RGB image as numpy array (uint8 or float)
        sigmaX: Gaussian blur sigma — controls how much illumination is removed.
                Higher = removes more low-frequency illumination variation.
    
    Returns:
        Illumination-normalized image
    """
    return cv2.addWeighted(
        img, 4,
        cv2.GaussianBlur(img, (0, 0), sigmaX), -4,
        128
    )


def apply_clahe(img, clip_limit=2.0, tile_size=8):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel.
    
    Retinal pathological features have highest contrast in the green channel,
    but applying CLAHE in LAB color space on the L (lightness) channel
    preserves color information while enhancing local contrast.
    
    Why CLAHE and not regular histogram equalization?
    - Regular HE amplifies noise uniformly
    - CLAHE divides the image into tiles and equalizes each separately
    - clip_limit prevents over-amplification in uniform regions (e.g., sclera)
    
    Args:
        img: RGB image as numpy array
        clip_limit: Contrast amplification limit (2.0 is standard for retinal)
        tile_size: Grid size for adaptive equalization (8×8 is standard)
    
    Returns:
        Contrast-enhanced RGB image
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def green_channel_clahe(img, clip_limit=2.0, tile_size=8):
    """
    Enhanced CLAHE with green channel emphasis.
    
    The green channel of retinal fundus images carries ~80% of the
    diagnostically relevant information (microaneurysms, hemorrhages,
    hard exudates). This function applies CLAHE separately to the
    green channel and blends the result with standard LAB-space CLAHE.
    
    Args:
        img: RGB image as numpy array (uint8)
        clip_limit: CLAHE clip limit
        tile_size: CLAHE tile grid size
    Returns:
        Enhanced RGB image with green channel emphasis
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    
    # Standard LAB CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    standard = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    
    # Green channel CLAHE
    green = img[:, :, 1]
    green_enhanced = clahe.apply(green)
    green_img = img.copy()
    green_img[:, :, 1] = green_enhanced
    
    # Blend: 60% green-emphasized + 40% standard
    result = cv2.addWeighted(green_img, 0.6, standard, 0.4, 0)
    return result


def suppress_optic_disc(img, threshold=220, kernel_size=25, inpaint_radius=5):
    """
    Locates and suppresses the optic disc (bright central region).
    
    Critical preprocessing step: the optic disc is the brightest structure
    in a fundus image. Without suppression, models frequently confuse it
    with hard exudates (bright yellow pathological deposits), leading to
    false positives for DR grade 2+.
    
    Method:
    1. Convert to grayscale
    2. Threshold to find brightest region (OD is always the brightest)
    3. Dilate the mask to ensure complete coverage
    4. Inpaint (fill) the region with surrounding tissue color
    
    Args:
        img: RGB image as numpy array
        threshold: Brightness threshold for OD detection (220/255)
        kernel_size: Morphological dilation kernel size
        inpaint_radius: Pixel radius for inpainting neighborhood
    
    Returns:
        Image with optic disc suppressed
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Dilate mask slightly to ensure complete OD coverage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bright_mask = cv2.dilate(bright_mask, kernel, iterations=2)
    # Inpaint: fill with local neighborhood mean using Telea method
    result = cv2.inpaint(img, bright_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return result


def crop_circle(img):
    """
    Remove black border from circular fundus field of view.
    
    Fundus cameras capture a circular view of the retina, surrounded by
    a black border. This border wastes computation and can confuse models
    (dark pixels at edges look like hemorrhages to an untrained model).
    
    Method: threshold → find largest contour → crop to bounding rectangle
    
    Args:
        img: RGB image with potential black borders
    
    Returns:
        Cropped image containing only the circular fundus region
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return img[y:y+h, x:x+w]


def preprocess_fundus(img, size=384, apply_ben_graham=True, apply_clahe_step=True,
                      apply_od_suppress=True, apply_crop=True,
                      green_channel_emphasis=True):
    """
    Master preprocessing function. Apply to every image from every dataset.
    
    Processing order (do NOT change sequence):
    1. Resize to 512×512 (work at high resolution first for quality preprocessing)
    2. Ben Graham illumination normalization
    3. CLAHE local contrast enhancement (with optional green channel emphasis)
    4. Optic disc suppression
    5. Circle crop (remove black border)
    6. Resize to target size (384×384 default for higher-resolution models)
    7. Float32 conversion + ImageNet normalization
    
    Args:
        img: RGB image as numpy array (uint8)
        size: Final output size (384 default — higher resolution captures
              fine-grained lesions like microaneurysms better than 224)
        apply_ben_graham: Whether to apply Ben Graham preprocessing
        apply_clahe_step: Whether to apply CLAHE
        apply_od_suppress: Whether to suppress optic disc
        apply_crop: Whether to crop circular field of view
        green_channel_emphasis: Whether to use green-channel-enhanced CLAHE.
            The green channel carries ~80% of DR-relevant information.
            When True, applies CLAHE to the green channel separately and
            blends at 60/40 ratio with standard LAB CLAHE.
    
    Returns:
        Preprocessed image as float32 numpy array, ImageNet-normalized,
        shape (size, size, 3)
    """
    # Step 1: Work at high resolution for quality preprocessing
    img = cv2.resize(img, (512, 512))
    
    # Step 2: Illumination normalization
    if apply_ben_graham:
        img = ben_graham_preprocess(img)
    
    # Step 3: Local contrast enhancement
    if apply_clahe_step:
        if green_channel_emphasis:
            img = green_channel_clahe(img)
        else:
            img = apply_clahe(img)
    
    # Step 4: Remove optic disc (prevents false positive exudates)
    if apply_od_suppress:
        img = suppress_optic_disc(img)
    
    # Step 5: Remove black border
    if apply_crop:
        img = crop_circle(img)
    
    # Step 6: Final resize to model input size
    img = cv2.resize(img, (size, size))
    
    # Step 7: Float conversion + ImageNet normalization
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    
    return img
