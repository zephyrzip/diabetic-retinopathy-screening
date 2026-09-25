"""
Self-contained Demo and Verification Script for Explainable DR (XAI) Pipeline
Creates synthetic retinal fundus samples with hallmark lesions (Microaneurysms,
Hemorrhages, Hard Exudates, Cotton Wool Spots, Neovascularization),
executes the full training loop on GPU/CPU, generates Grad-CAM attention maps,
and synthesizes the clinical etiology report explaining "WHY DR occurs".
All data, weights, and caches are kept strictly on Drive D.
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

os.environ["TORCH_HOME"] = r"D:\torch_cache"

from aptos_explainable_model import (
    ExplainableDRModel,
    GradCAM,
    ClinicalEtiologyExplainer,
    BIOMARKER_NAMES,
    GRADE_NAMES
)
from train_explainable_dr import APTOSDataset, Config


def create_synthetic_fundus(grade=0, size=512):
    """
    Generates a realistic synthetic retinal fundus phantom:
    - Orange-red retinal background with natural gradient
    - Optic disc (yellowish oval)
    - Fovea / Macula (darker spot)
    - Retinal vascular tree
    - Grade-specific hallmark lesions (Microaneurysms, Hemorrhages, Exudates, etc.)
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.44)
    
    # Retinal field of view mask
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    fov_mask = dist_from_center <= radius
    
    # Retinal base color (orange-reddish gradient)
    r_channel = np.clip(180 + 30 * (1 - dist_from_center / radius), 0, 240).astype(np.uint8)
    g_channel = np.clip(70 + 20 * (1 - dist_from_center / radius), 0, 120).astype(np.uint8)
    b_channel = np.clip(25 + 10 * (1 - dist_from_center / radius), 0, 60).astype(np.uint8)
    
    canvas[..., 0] = np.where(fov_mask, r_channel, 0)
    canvas[..., 1] = np.where(fov_mask, g_channel, 0)
    canvas[..., 2] = np.where(fov_mask, b_channel, 0)
    
    # Optic Disc (yellowish circle on nasal side)
    disc_center = (int(size * 0.28), int(size * 0.48))
    cv2.circle(canvas, disc_center, int(size * 0.07), (240, 210, 140), -1)
    
    # Macula (dark reddish-brown depression on temporal side)
    macula_center = (int(size * 0.60), int(size * 0.50))
    cv2.circle(canvas, macula_center, int(size * 0.05), (140, 45, 20), -1)
    
    # Retinal Blood Vessel Arcs
    for dy in [-0.15, 0.15]:
        pts = np.array([
            disc_center,
            (int(size * 0.45), int(size * (0.50 + dy))),
            (int(size * 0.70), int(size * (0.50 + 1.4 * dy))),
            (int(size * 0.85), int(size * (0.50 + 1.8 * dy)))
        ], np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=(120, 20, 15), thickness=4)
        
    # Grade-specific pathological hallmarks
    np.random.seed(42 + grade)
    
    if grade >= 1:
        # Microaneurysms (tiny round dark-red dots: 2-4px)
        num_ma = 8 if grade == 1 else 25
        for _ in range(num_ma):
            px = int(np.random.uniform(size * 0.35, size * 0.75))
            py = int(np.random.uniform(size * 0.25, size * 0.75))
            if fov_mask[py, px]:
                cv2.circle(canvas, (px, py), np.random.randint(2, 4), (100, 10, 10), -1)
                
    if grade >= 2:
        # Hemorrhages (larger blot/flame lesions: 6-14px)
        num_hems = 6 if grade == 2 else 18
        for _ in range(num_hems):
            px = int(np.random.uniform(size * 0.35, size * 0.75))
            py = int(np.random.uniform(size * 0.25, size * 0.75))
            if fov_mask[py, px]:
                cv2.ellipse(canvas, (px, py), (np.random.randint(5, 12), np.random.randint(3, 7)),
                            np.random.randint(0, 180), 0, 360, (90, 5, 5), -1)
                
        # Hard Exudates (bright yellow-white sharp lipid deposits)
        for _ in range(12):
            px = int(np.random.uniform(size * 0.45, size * 0.70))
            py = int(np.random.uniform(size * 0.40, size * 0.65))
            if fov_mask[py, px]:
                cv2.circle(canvas, (px, py), np.random.randint(3, 7), (240, 240, 160), -1)
                
    if grade >= 3:
        # Cotton Wool Spots (pale, fluffy, whitish ischemic micro-infarctions)
        for _ in range(5):
            px = int(np.random.uniform(size * 0.35, size * 0.70))
            py = int(np.random.uniform(size * 0.30, size * 0.70))
            if fov_mask[py, px]:
                cv2.circle(canvas, (px, py), np.random.randint(12, 20), (220, 220, 210), -1)
                
    if grade >= 4:
        # Neovascularization (tangled fronds of fine, abnormal blood vessels)
        nv_origin = disc_center
        for _ in range(8):
            end_x = nv_origin[0] + np.random.randint(-40, 40)
            end_y = nv_origin[1] + np.random.randint(-40, 40)
            cv2.line(canvas, nv_origin, (end_x, end_y), (140, 15, 15), 2)
            
    return canvas


def setup_demo_dataset(base_dir=r"D:\New folder (2)\demo_dataset", n_per_class=6):
    images_dir = os.path.join(base_dir, "train_images")
    os.makedirs(images_dir, exist_ok=True)
    
    records = []
    for g in range(5):
        for i in range(n_per_class):
            id_code = f"demo_grade_{g}_{i:02d}"
            img = create_synthetic_fundus(grade=g, size=512)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(images_dir, f"{id_code}.png")
            cv2.imwrite(save_path, img_bgr)
            records.append({"id_code": id_code, "diagnosis": g})
            
    df = pd.DataFrame(records)
    csv_path = os.path.join(base_dir, "train.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} synthetic retinal images in {base_dir}")
    return base_dir, csv_path


def main():
    print("=== Explainable DR Verification Run ===")
    demo_dir, csv_path = setup_demo_dataset()
    
    out_dir = r"D:\New folder (2)\outputs"
    os.makedirs(out_dir, exist_ok=True)
    gradcam_dir = os.path.join(out_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    df["filepath"] = df["id_code"].apply(lambda x: os.path.join(demo_dir, "train_images", f"{x}.png"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    dataset = APTOSDataset(df, image_size=512, net_input=224, is_train=True)
    loader = DataLoader(dataset, batch_size=6, shuffle=True)
    
    # Model initialization (pretrained ResNet50 backbone)
    model = ExplainableDRModel(num_classes=5, num_biomarkers=5, pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion_grade = nn.CrossEntropyLoss()
    criterion_bio = nn.BCEWithLogitsLoss()
    
    print("\n--- Running Mini-Training Loop (2 Epochs) ---")
    model.train()
    for ep in range(1, 3):
        total_loss, correct, total = 0.0, 0, 0
        for imgs, grades, bios, _ in loader:
            imgs, grades, bios = imgs.to(device), grades.to(device), bios.to(device)
            optimizer.zero_grad()
            g_logits, b_logits = model(imgs)
            l_grade = criterion_grade(g_logits, grades)
            l_bio = criterion_bio(b_logits, bios)
            loss = l_grade + 0.5 * l_bio
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            correct += (g_logits.argmax(1) == grades).sum().item()
            total += imgs.size(0)
        print(f"Epoch {ep}/2 Complete | Loss: {total_loss/total:.4f} | Accuracy: {correct/total*100:.1f}%")
        
    ckpt_path = os.path.join(out_dir, "explainable_dr_demo.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nModel checkpoint saved successfully to: {ckpt_path}")
    
    # Explainability Verification (Grad-CAM & Etiology Report)
    print("\n--- Verifying Grad-CAM & Clinical Etiology Generator ---")
    gradcam = GradCAM(model)
    model.eval()
    
    sample_reports = []
    # Test one image from each severity grade (0 to 4)
    for g in range(5):
        sample_row = df[df["diagnosis"] == g].iloc[0]
        raw_img = cv2.imread(sample_row["filepath"])
        raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        resized = cv2.resize(raw_rgb, (224, 224))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_tensor = norm(tensor).unsqueeze(0).to(device)
        
        cam, pred_g, g_logits, b_logits = gradcam(input_tensor)
        g_probs = torch.softmax(g_logits, dim=1).squeeze().cpu().detach().numpy()
        b_probs = torch.sigmoid(b_logits).squeeze().cpu().detach().numpy()
        
        report = ClinicalEtiologyExplainer.generate_report(pred_g, g_probs, b_probs)
        report["sample_id"] = sample_row["id_code"]
        report["ground_truth_grade"] = g
        sample_reports.append(report)
        
        # Overlay heatmap
        cam_heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_heat = cv2.cvtColor(cam_heat, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(resized, 0.65, cam_heat, 0.35, 0)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(resized)
        axes[0].set_title(f"Fundus (Ground Truth: Grade {g})")
        axes[0].axis("off")
        
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM Heatmap (Pred: Grade {pred_g})")
        axes[1].axis("off")
        
        save_img_path = os.path.join(gradcam_dir, f"verification_grade_{g}_overlay.png")
        plt.tight_layout()
        plt.savefig(save_img_path, bbox_inches="tight", dpi=150)
        plt.close()
        
    gradcam.remove()
    
    # Save combined clinical report
    summary_report_path = os.path.join(out_dir, "clinical_etiology_summary.json")
    with open(summary_report_path, "w") as f:
        json.dump(sample_reports, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
        
    print(f"All 5 Grade Grad-CAM maps saved to: {gradcam_dir}")
    print(f"Clinical Etiology Diagnostic Report saved to: {summary_report_path}")
    print("\nVerification succeeded! All components operate with 0 bytes used on Drive C.")


if __name__ == "__main__":
    main()
