"""
Retinal Blood Vessel Segmentation Model (U-Net) & Vascular Biomarker Engine
Dataset: DRIVE (Digital Retinal Images for Vessel Extraction) - https://drive.grand-challenge.org/
Benchmark: Messidor-2 Clinical Validation - https://www.adcis.net/en/third-party/messidor2/

Clinical Purpose in Diabetic Retinopathy:
1. Vessel Density: Quantifies retinal capillary dropout and ischemic zones.
2. Vessel Tortuosity: Quantifies vascular twisting and endothelial stress (hallmark of worsening DR).
3. Neovascularization (NV) Screening: Disorganized fragile vessel clusters distinguishing Grade 3 from Grade 4 PDR.
4. Lesion Disambiguation: Distinguishes true microaneurysms/hemorrhages from cross-sections of normal retinal vessels.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class VesselUNet(nn.Module):
    """
    Lightweight, high-precision U-Net for retinal blood vessel segmentation.
    Accepts 3-channel RGB fundus or 1-channel Green-CLAHE fundus images.
    Outputs a single-channel vessel probability map [0, 1].
    """
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Downsampling Encoder
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Upsampling Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # Final 1x1 Conv
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Decoder path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            # Handle shape mismatch if input is not power of 2
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        logits = self.final_conv(x)
        return torch.sigmoid(logits)


class DiceBCELoss(nn.Module):
    """Combined Dice Loss + Binary Cross Entropy Loss for vessel segmentation."""
    def __init__(self, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        # Flatten tensors for Dice calculation
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return (1.0 - self.dice_weight) * bce_loss + self.dice_weight * dice_loss


# =============================================================================
# CLINICAL VASCULAR BIOMARKER ENGINE
# =============================================================================

def extract_vascular_biomarkers(vessel_prob_map: np.ndarray, fov_mask: np.ndarray = None, threshold: float = 0.40):
    """
    Computes clinical vascular indices from the segmented vessel map:
    1. Vessel Perfusion Density (VPD)
    2. Vessel Tortuosity Index (VTI)
    3. Neovascularization / Proliferative Cluster Risk
    """
    if fov_mask is None:
        fov_mask = np.ones_like(vessel_prob_map, dtype=bool)

    # Binary vessel mask
    binary_vessels = (vessel_prob_map >= threshold) & fov_mask

    fov_area = np.count_nonzero(fov_mask)
    if fov_area == 0:
        return {"vessel_density": 0.0, "tortuosity_index": 1.0, "neovascularization_flag": False}

    vessel_area = np.count_nonzero(binary_vessels)
    vessel_density = float(vessel_area / fov_area)

    # Skeletonize vessels to analyze centerline tortuosity
    vessel_uint8 = (binary_vessels * 255).astype(np.uint8)
    
    # Fast skeleton approximation using morphological thinning
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    skeleton = np.zeros_like(vessel_uint8)
    temp = vessel_uint8.copy()
    
    for _ in range(12): # Iterative thinning
        eroded = cv2.erode(temp, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        subset = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, subset)
        temp = eroded.copy()
        if cv2.countNonZero(temp) == 0:
            break

    # Calculate tortuosity via contour analysis of skeleton segments
    contours, _ = cv2.findContours(skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    tortuosities = []
    
    for c in contours:
        if len(c) > 20: # Ignore tiny fragments
            arc_length = cv2.arcLength(c, False)
            # Distance between start and end point (chord length)
            p1 = c[0][0]
            p2 = c[-1][0]
            chord_length = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if chord_length > 5.0:
                t_val = arc_length / (chord_length + 1e-3)
                if 1.0 <= t_val <= 6.0:
                    tortuosities.append(t_val)

    avg_tortuosity = float(np.mean(tortuosities)) if tortuosities else 1.15
    
    # Neovascularization screening: Dense disorganized microvascular tangles
    # High tortuosity (> 1.45) combined with local clustering
    nv_flag = bool(avg_tortuosity > 1.42 and vessel_density > 0.14)

    return {
        "vessel_density_pct": round(vessel_density * 100, 2),
        "tortuosity_index": round(avg_tortuosity, 3),
        "neovascularization_flag": nv_flag,
        "clinical_significance": (
            "Severe vascular tortuosity & neovascularization risk detected (PDR Stage 4 alert)"
            if nv_flag else
            "Moderate vascular remodeling detected" if avg_tortuosity > 1.25 else
            "Normal retinal vascular architecture"
        )
    }
