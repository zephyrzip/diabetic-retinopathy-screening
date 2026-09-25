"""
=============================================================================
ONNX MODEL EXPORTER & VALIDATOR FOR MATLAB DEEP LEARNING TOOLBOX
=============================================================================
This script exports the 3 trained PyTorch deep learning models to ONNX:
1. APTOS 2019 DR Model (ResNet-50 Dual-Head: DR Grade 0-4 + 5 Biomarkers)
   -> outputs/onnx/aptos_dr_resnet50.onnx
2. IDRiD Multi-Task Model (ResNet-50 Triple-Head: DR 0-4 + DME 0-2 + 5 Biomarkers)
   -> outputs/onnx/idrid_multitask_resnet50.onnx
3. DRIVE Retinal Vessel U-Net Model (Vessel Segmentation Map)
   -> outputs/onnx/drive_vessel_unet.onnx

Enables direct one-line loading inside MATLAB via:
  net = importONNXNetwork('outputs/onnx/idrid_multitask_resnet50.onnx', 'OutputDataFormats', 'BC');
=============================================================================
"""

import os
import sys
import time
import torch
import torch.nn as nn
import onnx

os.environ["TORCH_HOME"] = r"D:\torch_cache"

from aptos_explainable_model import ExplainableDRModel
from idrid_explainable_model import ExplainableIDRiDModel
from drive_vessel_model import VesselUNet

BASE_DIR = r"D:\New folder (2)"
ONNX_DIR = os.path.join(BASE_DIR, "outputs", "onnx")
os.makedirs(ONNX_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def export_aptos_dr_model():
    print("\n-----------------------------------------------------------------")
    print(" [1/3] EXPORTING APTOS DR RESNET-50 MODEL TO ONNX")
    print("-----------------------------------------------------------------")
    weights_path = os.path.join(BASE_DIR, "outputs", "explainable_dr_resnet50.pt")
    onnx_path = os.path.join(ONNX_DIR, "aptos_dr_resnet50.onnx")

    model = ExplainableDRModel(num_classes=5, num_biomarkers=5, pretrained=False).to(DEVICE)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f" Loaded trained weights from: {weights_path}")
    else:
        print(f" Warning: Checkpoint not found at {weights_path}, exporting base architecture.")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["grade_logits", "biomarker_logits"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "grade_logits": {0: "batch_size"},
            "biomarker_logits": {0: "batch_size"}
        }
    )

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f" Successfully Exported: {onnx_path}")
    print(f" ONNX Model Size: {file_size_mb:.2f} MB")
    print(f" ONNX Checker Verification: PASSED (Opset 14)")
    return onnx_path


def export_idrid_multitask_model():
    print("\n-----------------------------------------------------------------")
    print(" [2/3] EXPORTING IDRID MULTI-TASK RESNET-50 MODEL TO ONNX")
    print("-----------------------------------------------------------------")
    weights_path = os.path.join(BASE_DIR, "outputs", "idrid", "explainable_idrid_multitask_resnet50.pt")
    onnx_path = os.path.join(ONNX_DIR, "idrid_multitask_resnet50.onnx")

    model = ExplainableIDRiDModel(pretrained=False).to(DEVICE)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f" Loaded trained weights from: {weights_path}")
    else:
        print(f" Warning: Checkpoint not found at {weights_path}, exporting base architecture.")

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["dr_logits", "dme_logits", "biomarker_logits"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "dr_logits": {0: "batch_size"},
            "dme_logits": {0: "batch_size"},
            "biomarker_logits": {0: "batch_size"}
        }
    )

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f" Successfully Exported: {onnx_path}")
    print(f" ONNX Model Size: {file_size_mb:.2f} MB")
    print(f" ONNX Checker Verification: PASSED (Opset 14)")
    return onnx_path


def export_drive_vessel_model():
    print("\n-----------------------------------------------------------------")
    print(" [3/3] EXPORTING DRIVE RETINAL VESSEL U-NET MODEL TO ONNX")
    print("-----------------------------------------------------------------")
    weights_path = os.path.join(BASE_DIR, "outputs", "drive", "vessel_unet_drive.pt")
    onnx_path = os.path.join(ONNX_DIR, "drive_vessel_unet.onnx")

    model = VesselUNet(in_channels=3, out_channels=1, features=[32, 64, 128, 256]).to(DEVICE)
    if os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f" Loaded trained weights from: {weights_path}")
    else:
        print(f" Warning: Checkpoint not found at {weights_path}, exporting base architecture.")

    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256, device=DEVICE)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["vessel_mask"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "vessel_mask": {0: "batch_size"}
        }
    )

    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f" Successfully Exported: {onnx_path}")
    print(f" ONNX Model Size: {file_size_mb:.2f} MB")
    print(f" ONNX Checker Verification: PASSED (Opset 14)")
    return onnx_path


def main():
    print("=" * 70)
    print(" EXPORTING ALL TRAINED MODELS TO ONNX FOR MATLAB PIPELINE")
    print(f" Target Directory: {ONNX_DIR}")
    print("=" * 70)

    p1 = export_aptos_dr_model()
    p2 = export_idrid_multitask_model()
    p3 = export_drive_vessel_model()

    print("\n" + "=" * 70)
    print(" [EXPORT SUCCESS] ALL 3 MODELS EXPORTED TO ONNX FORMAT:")
    print(f"  1. APTOS Model:  {p1}")
    print(f"  2. IDRiD Model:  {p2}")
    print(f"  3. DRIVE Model:  {p3}")
    print("\n MATLAB Command to Import:")
    print("  >> net = importONNXNetwork('outputs/onnx/idrid_multitask_resnet50.onnx', 'OutputDataFormats', 'BC');")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
