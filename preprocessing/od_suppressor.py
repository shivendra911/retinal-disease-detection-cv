"""
IRDAS Preprocessing — Optic Disc Suppressor (Advanced)
======================================================

Advanced optic disc detection and suppression using morphological operations.
This module provides an enhanced OD suppressor that handles edge cases like:
- Very bright cataracts (entire image is bright)
- Multiple bright spots (OD + hard exudates)
- Dark/low-quality images where OD is not clearly visible

The basic version in clahe_preprocess.py handles most cases.
Use this module when the basic version produces artifacts.
"""

import cv2
import numpy as np


def detect_optic_disc_center(img, min_radius=30, max_radius=100):
    """
    Detect the optic disc center using Hough Circle Transform.
    
    More robust than simple thresholding for challenging images.
    The OD is roughly circular with known radius range relative to image size.
    
    Args:
        img: RGB image (512×512 recommended)
        min_radius: Minimum OD radius in pixels
        max_radius: Maximum OD radius in pixels
    
    Returns:
        (center_x, center_y, radius) or None if not detected
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=1, minDist=200,
        param1=100, param2=30,
        minRadius=min_radius, maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Take the brightest circle (OD is always brightest)
        best_circle = None
        best_brightness = 0
        for c in circles[0]:
            cx, cy, r = c
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (cx, cy), r, 255, -1)
            brightness = gray[mask > 0].mean()
            if brightness > best_brightness:
                best_brightness = brightness
                best_circle = (int(cx), int(cy), int(r))
        return best_circle
    
    return None


def suppress_optic_disc_advanced(img, expansion_factor=1.3):
    """
    Advanced OD suppression using circle detection + inpainting.
    
    1. Detect OD as a circle using Hough Transform
    2. Create circular mask with slight expansion
    3. Inpaint using Navier-Stokes method (smoother than Telea for large regions)
    
    Args:
        img: RGB image as numpy array
        expansion_factor: How much to expand the detected OD circle
    
    Returns:
        Image with OD suppressed, or original if OD not detected
    """
    od = detect_optic_disc_center(img)
    
    if od is None:
        # Fallback to basic thresholding method
        from preprocessing.clahe_preprocess import suppress_optic_disc
        return suppress_optic_disc(img)
    
    cx, cy, r = od
    expanded_r = int(r * expansion_factor)
    
    # Create circular mask
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), expanded_r, 255, -1)
    
    # Inpaint using Navier-Stokes (smoother for large regions)
    result = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS)
    
    return result
