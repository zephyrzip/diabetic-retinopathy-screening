"""
Fundus Image Quality Assessment and Adaptive Enhancement
Implements:
1. Quality checks (Illumination, Laplacian focus metric, Field-of-View mask)
2. Green-channel Contrast-Limited Adaptive Histogram Equalization (CLAHE)
3. Ben Graham illumination subtraction (local color constancy normalization)
"""

import cv2
import numpy as np


def assess_and_enhance_fundus(img_rgb, target_size=512):
    """
    Takes an RGB uint8 fundus image (HxWx3).
    Returns:
        enhanced_img: HxWx3 uint8 processed fundus image
        quality_info: dict containing illumination, focus score, and quality pass flag
    """
    if img_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        
    img_resized = cv2.resize(img_rgb, (target_size, target_size))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    
    # 1. Quality Assessment & Retinal Chromaticity Gate
    mean_int = float(gray.mean())
    illum_ok = 15.0 < mean_int < 240.0
    
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    focus_score = float(lap.var())
    focus_ok = focus_score > 6.0
    
    mask = gray > 15
    fov_fraction = float(mask.sum() / mask.size)
    circular_fov = (0.20 <= fov_fraction <= 1.00)
    
    # Human retina fundus reflectance is dominated by vascular choroid: R > G > B (B strongly absorbed)
    r_mean = float(img_resized[:, :, 0][mask].mean()) if mask.sum() > 0 else 0.0
    g_mean = float(img_resized[:, :, 1][mask].mean()) if mask.sum() > 0 else 0.0
    b_mean = float(img_resized[:, :, 2][mask].mean()) if mask.sum() > 0 else 0.0
    
    retinal_hue = (r_mean > g_mean * 1.10) and (r_mean > b_mean * 1.35) and (b_mean < r_mean * 0.65) and (r_mean > 35)
    is_real_retina = illum_ok and focus_ok and circular_fov and retinal_hue
    quality_ok = is_real_retina
    
    quality_info = {
        "mean_intensity": round(mean_int, 2),
        "focus_score": round(focus_score, 2),
        "fov_fraction": round(fov_fraction, 2),
        "quality_pass": quality_ok,
        "is_retina": is_real_retina,
        "rgb_means": (round(r_mean, 1), round(g_mean, 1), round(b_mean, 1))
    }
    
    # 2. Green Channel CLAHE
    # The green channel in retinal imaging provides optimal optical absorption
    # for hemoglobin (microaneurysms and hemorrhages) and lipid reflectance (exudates)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    green = img_resized[:, :, 1]
    green_enhanced = clahe.apply(green)
    green_enhanced = cv2.GaussianBlur(green_enhanced, (0, 0), sigmaX=0.5)
    
    enhanced = img_resized.copy()
    enhanced[:, :, 1] = green_enhanced
    
    # 3. Ben Graham Illumination Normalization
    # Subtracts local blurred mean to correct non-uniform camera flash illumination
    sigma = target_size / 30.0
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=sigma)
    normalized = enhanced.astype(np.float32) - blurred.astype(np.float32) + 128.0
    normalized = np.clip(normalized, 0, 255)
    enhanced_final = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return enhanced_final, quality_info
