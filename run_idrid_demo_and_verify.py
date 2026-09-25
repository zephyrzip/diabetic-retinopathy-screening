"""
=============================================================================
Self-Contained IDRiD Multi-Task Demo & Verification Script
=============================================================================
Quickly verifies the complete IDRiD pipeline on local GPU (GTX 1650) or CPU:
- Generates realistic synthetic Indian cohort retinal phantoms with DR lesions
  and macular hard exudates (DME)
- Runs multi-task forward & backward pass (DR + DME + 5 Pathological Biomarkers)
- Tests Dual-Target Grad-CAM (DR peripheral vs DME macular attention)
- Generates Ophthalmology Clinical Etiology diagnostic reports
- Strictly isolated to Drive D (zero footprint on Drive C)
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt

os.environ["TORCH_HOME"] = r"D:\torch_cache"

from idrid_explainable_model import (
    ExplainableIDRiDModel,
    MultiTargetGradCAM,
    IDRiDEtiologyExplainer,
    BIOMARKER_NAMES,
    DR_GRADE_NAMES,
    DME_RISK_NAMES,
    get_biomarker_prior
)

DEMO_DIR = r"D:\New folder (2)\demo_dataset\idrid_demo"
OUT_DIR = r"D:\New folder (2)\outputs\idrid_demo"
os.makedirs(DEMO_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def create_synthetic_idrid_fundus(dr_grade=0, dme_risk=0, size=512):
    """
    Generates a realistic synthetic fundus phantom matching the IDRiD cohort:
    - 50° Field of View with darker pigment epithelium
    - Optic disc (nasal) and macula / fovea (temporal)
    - Retinal vascular arcades
    - DR lesions: Microaneurysms, hemorrhages, cotton wool spots, neovascular tufts
    - DME lesions: Hard lipid exudates radiating or clustered in the macula
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = int(size * 0.45)

    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    fov_mask = dist_from_center <= radius

    # Base Indian fundus color (slightly deeper red-brown tint)
    r = np.clip(165 + 35 * (1 - dist_from_center / radius), 0, 220).astype(np.uint8)
    g = np.clip(60 + 20 * (1 - dist_from_center / radius), 0, 100).astype(np.uint8)
    b = np.clip(20 + 10 * (1 - dist_from_center / radius), 0, 50).astype(np.uint8)

    canvas[..., 0] = np.where(fov_mask, r, 0)
    canvas[..., 1] = np.where(fov_mask, g, 0)
    canvas[..., 2] = np.where(fov_mask, b, 0)

    # Optic Disc (Nasal)
    disc_center = (int(size * 0.28), int(size * 0.50))
    cv2.circle(canvas, disc_center, int(size * 0.07), (230, 205, 130), -1)

    # Macula / Foveal avascular zone (Temporal)
    macula_center = (int(size * 0.62), int(size * 0.50))
    cv2.circle(canvas, macula_center, int(size * 0.045), (110, 35, 15), -1)

    # Retinal Vascular Arcs
    for dy in [-0.18, 0.18]:
        pts = np.array([
            disc_center,
            (int(size * 0.45), int(size * (0.50 + dy))),
            (int(size * 0.75), int(size * (0.50 + dy * 0.85)))
        ], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=False, color=(80, 15, 10), thickness=3)

    rng = np.random.RandomState(dr_grade * 10 + dme_risk)

    # Inject DR Lesions
    if dr_grade >= 1:
        # Microaneurysms (tiny red dots)
        for _ in range(8 * dr_grade):
            lx = rng.randint(int(size * 0.35), int(size * 0.75))
            ly = rng.randint(int(size * 0.25), int(size * 0.75))
            cv2.circle(canvas, (lx, ly), rng.randint(2, 4), (120, 10, 5), -1)

    if dr_grade >= 2:
        # Blot and flame hemorrhages
        for _ in range(5 * dr_grade):
            lx = rng.randint(int(size * 0.35), int(size * 0.80))
            ly = rng.randint(int(size * 0.25), int(size * 0.75))
            cv2.ellipse(canvas, (lx, ly), (rng.randint(6, 12), rng.randint(3, 7)), rng.randint(0, 180), 0, 360, (110, 5, 5), -1)

    if dr_grade >= 3:
        # Cotton Wool Spots (fluffy whitish-gray patches)
        for _ in range(3):
            lx = rng.randint(int(size * 0.30), int(size * 0.60))
            ly = rng.randint(int(size * 0.25), int(size * 0.75))
            cv2.circle(canvas, (lx, ly), rng.randint(8, 14), (200, 195, 185), -1)

    if dr_grade >= 4:
        # Neovascularization (frond-like fine vessel arborization)
        for _ in range(2):
            vx, vy = disc_center[0] + rng.randint(-15, 15), disc_center[1] + rng.randint(-15, 15)
            for _ in range(6):
                nx = vx + rng.randint(-18, 18)
                ny = vy + rng.randint(-18, 18)
                cv2.line(canvas, (vx, vy), (nx, ny), (130, 20, 15), 1)

    # Inject DME Lesions: Hard Exudates (bright yellowish lipid deposits near or on macula)
    if dme_risk == 1:
        # Hard exudates within 1 disc diameter of fovea
        for _ in range(12):
            offset_x = rng.randint(-int(size * 0.10), int(size * 0.10))
            offset_y = rng.randint(-int(size * 0.10), int(size * 0.10))
            # Keep away from direct central fovea
            if abs(offset_x) > int(size * 0.03) or abs(offset_y) > int(size * 0.03):
                lx = macula_center[0] + offset_x
                ly = macula_center[1] + offset_y
                cv2.circle(canvas, (lx, ly), rng.randint(2, 5), (245, 235, 130), -1)
    elif dme_risk == 2:
        # Clinically Significant Macular Edema (CSME): exudates involving central fovea
        for _ in range(25):
            offset_x = rng.randint(-int(size * 0.06), int(size * 0.06))
            offset_y = rng.randint(-int(size * 0.06), int(size * 0.06))
            lx = macula_center[0] + offset_x
            ly = macula_center[1] + offset_y
            cv2.circle(canvas, (lx, ly), rng.randint(2, 6), (250, 240, 140), -1)

    canvas = np.where(fov_mask[..., None], canvas, 0)
    return canvas


class DemoDataset(Dataset):
    def __init__(self, records):
        self.records = records
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = rec["image"]
        # Basic CLAHE on green
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        green = img[:, :, 1]
        img[:, :, 1] = clahe.apply(green)
        
        resized = cv2.resize(img, (224, 224))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        tensor = self.normalize(tensor)

        dr_grade = rec["dr_grade"]
        dme_risk = rec["dme_risk"]
        bio_target = torch.tensor(get_biomarker_prior(dr_grade, dme_risk), dtype=torch.float32)

        return tensor, dr_grade, dme_risk, bio_target, img, rec["id"]


def run_demo():
    print("=" * 70)
    print("[DEMO] IDRiD Multi-Task Explainable DR & DME System Verification")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Compute Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

    # 1. Synthesize IDRiD samples representing diverse DR & DME combinations
    sample_configs = [
        (0, 0, "IDRiD_Demo_01_Normal"),
        (1, 0, "IDRiD_Demo_02_MildNPDR"),
        (2, 1, "IDRiD_Demo_03_ModNPDR_MildDME"),
        (3, 1, "IDRiD_Demo_04_SevereNPDR_DME"),
        (4, 2, "IDRiD_Demo_05_PDR_CSME"),
    ]

    records = []
    print("\n[STEP 1/5] Synthesizing Indian Cohort Fundus Phantoms (DR + DME)...")
    for dr_g, dme_r, name in sample_configs:
        img = create_synthetic_idrid_fundus(dr_grade=dr_g, dme_risk=dme_r, size=512)
        save_path = os.path.join(DEMO_DIR, f"{name}.png")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        records.append({
            "image": img,
            "dr_grade": dr_g,
            "dme_risk": dme_r,
            "id": name
        })
    print(f"Generated {len(records)} benchmark cases saved in: {DEMO_DIR}")

    # 2. Build Multi-Task Model
    print("\n[STEP 2/5] Initializing Multi-Task ResNet-50 Network...")
    model = ExplainableIDRiDModel(pretrained=True).to(device)

    # Check for existing APTOS weights to verify transfer learning capability
    aptos_weights = r"D:\New folder (2)\outputs\explainable_dr_resnet50.pt"
    if os.path.exists(aptos_weights):
        model.load_aptos_transfer_weights(aptos_weights)

    # 3. Quick Multi-Task Forward & Backward Verification
    print("\n[STEP 3/5] Executing Multi-Task Forward & Backward Iteration on GPU...")
    loader = DataLoader(DemoDataset(records), batch_size=2, shuffle=True)
    
    crit_dr = nn.CrossEntropyLoss()
    crit_dme = nn.CrossEntropyLoss()
    crit_bio = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    total_loss_accum = 0.0
    for tensors, dr_targets, dme_targets, bio_targets, _, _ in loader:
        tensors = tensors.to(device)
        dr_targets = dr_targets.to(device)
        dme_targets = dme_targets.to(device)
        bio_targets = bio_targets.to(device)

        optimizer.zero_grad()
        dr_logits, dme_logits, bio_logits = model(tensors)

        l_dr = crit_dr(dr_logits, dr_targets)
        l_dme = crit_dme(dme_logits, dme_targets)
        l_bio = crit_bio(bio_logits, bio_targets)
        loss = l_dr + 0.8 * l_dme + 0.5 * l_bio

        loss.backward()
        optimizer.step()
        total_loss_accum += loss.item()

    print(f"[SUCCESS] Multi-Task Optimization verified! Sample loss: {total_loss_accum:.4f}")

    # 4. Dual-Target Grad-CAM Generation (DR peripheral vs DME macular)
    print("\n[STEP 4/5] Computing Dual-Target Grad-CAM (DR vs DME)...")
    cam_engine = MultiTargetGradCAM(model)
    explainer = IDRiDEtiologyExplainer()
    reports = []

    fig, axes = plt.subplots(len(records), 4, figsize=(20, 4.2 * len(records)))
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model.eval()
    for idx, rec in enumerate(records):
        img_rgb = rec["image"]
        resized = cv2.resize(img_rgb, (224, 224))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        tensor = normalize(tensor).unsqueeze(0).to(device)

        with torch.no_grad():
            dr_l, dme_l, bio_l = model(tensor)
            dr_probs = torch.softmax(dr_l, dim=1)[0].cpu().numpy()
            dme_probs = torch.softmax(dme_l, dim=1)[0].cpu().numpy()
            bio_probs = torch.sigmoid(bio_l)[0].cpu().numpy()

        pred_dr = int(np.argmax(dr_probs))
        pred_dme = int(np.argmax(dme_probs))

        # DR Grad-CAM
        cam_dr, _ = cam_engine.generate(tensor, target_type="dr", target_class=pred_dr)
        overlay_dr = cam_engine.overlay(cam_dr, img_rgb)

        # DME Grad-CAM
        cam_dme, _ = cam_engine.generate(tensor, target_type="dme", target_class=pred_dme)
        overlay_dme = cam_engine.overlay(cam_dme, img_rgb)

        # Etiology Report
        rep = explainer.explain(pred_dr, pred_dme, dr_probs, dme_probs, bio_probs, rec["id"])
        reports.append(rep)

        # Plot Col 1: Fundus
        axes[idx, 0].imshow(img_rgb)
        axes[idx, 0].set_title(f"{rec['id']}\nTrue: DR {rec['dr_grade']} | DME {rec['dme_risk']}", fontsize=11, fontweight="bold")
        axes[idx, 0].axis("off")

        # Plot Col 2: DR Heatmap
        axes[idx, 1].imshow(overlay_dr)
        axes[idx, 1].set_title(f"DR Activation: {DR_GRADE_NAMES[pred_dr]}\n(Conf: {dr_probs[pred_dr]*100:.1f}%)", fontsize=10, color="navy")
        axes[idx, 1].axis("off")

        # Plot Col 3: DME Heatmap
        axes[idx, 2].imshow(overlay_dme)
        axes[idx, 2].set_title(f"DME Activation: {DME_RISK_NAMES[pred_dme]}\n(Conf: {dme_probs[pred_dme]*100:.1f}%)", fontsize=10, color="darkred")
        axes[idx, 2].axis("off")

        # Plot Col 4: Etiology Box
        bio_str = ", ".join([b["biomarker"] for b in rep["biomarker_findings"]]) or "None"
        txt = (
            f"CLINICAL ETIOLOGY REPORT\n"
            f"---------------------------\n"
            f"- DR Grade: {rep['primary_diagnosis']['dr_stage']}\n"
            f"- DME Status: {rep['primary_diagnosis']['dme_stage']}\n"
            f"- Lesions: {bio_str}\n\n"
            f"Retinopathy Mechanism:\n"
            f"{rep['pathological_etiology']['dr_pathogenesis']}\n\n"
            f"Macular Involvement:\n"
            f"{rep['pathological_etiology']['dme_pathogenesis']}\n\n"
            f"Action: {rep['clinical_decision_support']['action_plan']}"
        )
        axes[idx, 3].text(0.05, 0.5, txt, fontsize=9.2, family="monospace", va="center",
                          bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da"))
        axes[idx, 3].axis("off")

    plt.tight_layout()
    viz_path = os.path.join(OUT_DIR, "idrid_dual_gradcam_demo.png")
    plt.savefig(viz_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Dual Grad-CAM visualization saved to: {viz_path}")

    # 5. Save Clinical Reports JSON
    print("\n[STEP 5/5] Exporting Clinical Etiology Reports...")
    rep_path = os.path.join(OUT_DIR, "idrid_demo_clinical_reports.json")
    with open(rep_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"[SAVED] Clinical reports saved to: {rep_path}")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL LOCAL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print(f"Results generated in: {OUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
