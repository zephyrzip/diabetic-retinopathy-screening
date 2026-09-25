"""
Inspect Any Retina Image: Step-by-Step AI Checking Pipeline
Usage:
    python inspect_retina_image.py --image "path/to/retina.jpg" [--open-browser]

Processes any given retinal image through the complete 6-stage diagnostic pipeline:
1. Optical Quality Control (IQA: Illumination, Blur Variance, FOV)
2. Green-Channel CLAHE & Ben Graham Illumination Normalization
3. Anatomical & Vascular Arborization Mapping (Optic Disc, Macula, Vessels)
4. Hallmark Lesion Localization (Microaneurysms, Hemorrhages, Exudates, Ischemia)
5. Multi-Task ResNet-50 Inference & Grad-CAM Attention Heatmap
6. Ophthalmology Etiology Diagnostic Report Generation

Saves visual outputs to outputs/steps/ and can automatically open the interactive inspector in the browser.
"""

import os
import sys
import argparse
import webbrowser
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

# Drive D torch cache
os.environ["TORCH_HOME"] = r"D:\torch_cache"

from aptos_explainable_model import (
    ExplainableDRModel,
    GradCAM,
    ClinicalEtiologyExplainer,
    BIOMARKER_NAMES,
    GRADE_NAMES
)
from show_retina_checking_steps import (
    extract_vessel_tree,
    detect_anatomical_landmarks,
    detect_hallmark_lesions,
    run_pipeline_for_image
)
from build_interactive_presentation import build_presentation

from keras_dr_model import FriendKerasDRModel, DualModelEnsemble, resolve_model_path

BASE_DIR = r"d:\New folder (2)"
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "explainable_dr_resnet50.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def inspect_image(image_path, open_browser=True, model_engine="ensemble"):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)
        
    engine_desc = {
        "ensemble": "Dual-Architecture Ensemble (EfficientNet-B0 + ResNet-50)",
        "keras": "Keras EfficientNet-B0 (Collaborator Trained)",
        "pytorch": "PyTorch ResNet-50"
    }.get(model_engine.lower(), model_engine)

    print(f"\n=======================================================")
    print(f" RETINASCAN AI: STEP-BY-STEP RETINA INSPECTION PIPELINE")
    print(f" Image: {image_path}")
    print(f" Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Engine: {engine_desc}")
    print(f"=======================================================\n")
    
    # 1. Load trained model
    print("[1/4] Loading deep learning diagnostic models...")
    model = ExplainableDRModel(num_classes=5, num_biomarkers=5, pretrained=False).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    gradcam = GradCAM(model)
    
    # 2. Estimate initial grade from quick forward pass
    raw_bgr = cv2.imread(image_path)
    if raw_bgr is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    
    from data_enhancement import assess_and_enhance_fundus
    enhanced_512, q_info = assess_and_enhance_fundus(raw_rgb, target_size=512)
    
    # Pre-flight OOD / Quality Gate
    if not q_info.get("quality_pass", True):
        print("\n=======================================================")
        print(" [SAFETY GUARD TRIGGERED] NON-RETINAL / OOD INPUT DETECTED")
        print(f" R/G/B Means: {q_info.get('rgb_means')} | FOV: {int(q_info.get('fov_fraction', 0)*100)}%")
        print(" Reason: Optical absorption spectrum does not match human retinal fundus.")
        print(" Action: Deep neural network inference aborted to prevent misdiagnosis.")
        print("=======================================================\n")
        return {
            "pred_name": "REJECTED (Non-Retinal Input)",
            "confidence": 0.0,
            "is_referable": False,
            "quality_info": q_info,
            "biomarker_probs": {b: 0.0 for b in BIOMARKER_NAMES},
            "pathology_etiology": "Input image does not exhibit retinal fundus chromaticity or circular aperture.",
            "clinical_recommendation": "SCAN REJECTED. Please acquire and upload a standardized 45°/50° color retinal fundus image."
        }
    
    # Forward pass
    pred_grade = 0
    if model_engine.lower() == "ensemble":
        try:
            ensemble_engine = DualModelEnsemble(device=str(DEVICE))
            ens_res = ensemble_engine.predict(enhanced_512)
            pred_grade = ens_res["pred_grade"]
            conf = ens_res.get("confidence", round(float(np.max(ens_res.get("ensemble_probabilities", [0.5]))) * 100, 1))
            print(f"[2/4] Dual-Architecture Ensemble Inference -> Predicted Severity: Grade {pred_grade} ({GRADE_NAMES[pred_grade]}) [Confidence: {conf}%]")
        except Exception as e:
            print(f"[WARN] Ensemble inference error ({e}), falling back to Keras/ResNet-50.")
            model_engine = "keras"

    if model_engine.lower() == "keras":
        try:
            keras_engine = FriendKerasDRModel(device=str(DEVICE))
            k_res = keras_engine.predict(enhanced_512)
            pred_grade = k_res["pred_grade"]
            print(f"[2/4] Collaborator Keras EfficientNet-B0 Inference -> Predicted Severity: Grade {pred_grade} ({GRADE_NAMES[pred_grade]}) [Confidence: {k_res['confidence']}%]")
        except Exception as e:
            print(f"[WARN] Keras model inference error ({e}), falling back to ResNet-50.")
            model_engine = "pytorch"

    if model_engine.lower() == "pytorch":
        resized_224 = cv2.resize(enhanced_512, (224, 224))
        t = torch.from_numpy(resized_224.transpose(2, 0, 1)).float() / 255.0
        import torchvision.transforms as T
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_tensor = norm(t).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            g_logits, _ = model(input_tensor)
            pred_grade = int(g_logits.argmax(1).item())
        print(f"[2/4] ResNet-50 Model Inference -> Predicted Severity: Grade {pred_grade} ({GRADE_NAMES[pred_grade]})")
    
    # 3. Run full 6-step pipeline
    print("[3/4] Executing 6-stage checking pipeline & Grad-CAM extraction...")
    pipeline_result = run_pipeline_for_image(image_path, pred_grade, model, gradcam)
    gradcam.remove()
    
    # Output summary
    print("\n----------------- CLINICAL DIAGNOSTIC REPORT -----------------")
    print(f" Diagnosis: {pipeline_result['pred_name']}")
    print(f" Confidence: {pipeline_result['confidence']}%")
    print(f" Referable DR: {'YES [URGENT OPHTHALMOLOGY REFERRAL]' if pipeline_result['is_referable'] else 'NO [ROUTINE FOLLOW-UP]'}")
    print(f" Quality Gate: {'PASSED' if pipeline_result['quality_info']['quality_pass'] else 'FAILED'}")
    print(f" Illumination: {pipeline_result['quality_info']['mean_intensity']} | Focus Variance: {pipeline_result['quality_info']['focus_score']}")
    print(" Hallmark Biomarkers:")
    for bio, prob in pipeline_result['biomarker_probs'].items():
        print(f"   - {bio}: {int(prob*100)}% ({'Detected' if prob>=0.5 else 'Absent'})")
    print(f" Etiology: {pipeline_result['pathology_etiology']}")
    print(f" Recommendation: {pipeline_result['clinical_recommendation']}")
    print("--------------------------------------------------------------\n")
    
    # 4. Save dedicated output board
    out_dir = os.path.join(BASE_DIR, "outputs", "steps")
    custom_board = os.path.join(out_dir, "custom_retina_diagnostic_board.png")
    # Copy generated board to custom board
    src_board = os.path.join(out_dir, f"grade_{pred_grade}_diagnostic_board.png")
    if os.path.exists(src_board):
        import shutil
        shutil.copyfile(src_board, custom_board)
        print(f"6-Step Diagnostic Board saved to: {custom_board}")
        
    # Rebuild presentation HTML so it contains latest data
    build_presentation()
    
    html_path = os.path.join(BASE_DIR, "retina_inspection_steps.html")
    print(f"[4/4] Interactive visual presentation updated: {html_path}")
    
    if open_browser:
        print(f"Opening presentation in default web browser...")
        webbrowser.open(f"file:///{html_path.replace(os.sep, '/')}")
        
    return pipeline_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect any retinal image through the 6-step AI pipeline.")
    parser.add_argument("--image", "-i", type=str, default=r"d:\New folder (2)\demo_dataset\train_images\demo_grade_3_00.png",
                        help="Path to the retinal fundus image file (.png, .jpg, .jpeg)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser.")
    parser.add_argument("--model", type=str, default="ensemble", choices=["ensemble", "keras", "pytorch"],
                        help="Deep learning engine: 'ensemble' (Dual EfficientNet-B0 + ResNet-50, default), 'keras', or 'pytorch'")
    args = parser.parse_args()
    
    inspect_image(args.image, open_browser=not args.no_browser, model_engine=args.model)
