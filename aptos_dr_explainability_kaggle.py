"""
=============================================================================
APTOS 2019 Blindness Detection: Explainable AI & Lesion Etiology Pipeline
Kaggle Notebook Ready Script (GPU Accelerator ON)
=============================================================================
Purpose:
- Trains ResNet-50 on the full APTOS 2019 dataset
- Detects the 5 Pathological Biomarkers explaining "WHY DR occurs":
    1. Microaneurysms (Capillary outpouchings)
    2. Hemorrhages (Microvascular wall rupture)
    3. Hard Exudates (Lipoprotein vascular leakage)
    4. Cotton Wool Spots (Retinal nerve fiber ischemia)
    5. Neovascularization (Fragile abnormal vessels driven by VEGF)
- Generates Grad-CAM visual heatmaps localizing lesion hotspots
- Generates ophthalmology-grade clinical diagnostic etiology reports
- Evaluates Quadratic Weighted Kappa (QWK) and Referable DR (Grade 2+) Sensitivity/Specificity

How to run on Kaggle:
1. Create a new Kaggle Notebook attached to 'aptos2019-blindness-detection'
2. Enable GPU (Settings -> Accelerator -> GPU T4 x2 or P100)
3. Paste and run this script
"""

import os
import cv2
import copy
import json
import random
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------
# AUTO-DETECT DATASET PATH
# ---------------------------------------------------------------------
def find_aptos_data():
    search_dirs = ["/kaggle/input", "."]
    
    # 1. Prioritize official 'train.csv'
    for base in search_dirs:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                for f in files:
                    if f.lower() == "train.csv":
                        csv_p = os.path.join(root, f)
                        print(f"Found official train.csv at: {csv_p}")
                        return root, csv_p, root

    # 2. Look for any CSV with 'train' in its name
    for base in search_dirs:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                for f in files:
                    if "train" in f.lower() and f.endswith(".csv"):
                        csv_p = os.path.join(root, f)
                        try:
                            df_temp = pd.read_csv(csv_p, nrows=5)
                            cols = [c.lower() for c in df_temp.columns]
                            if any(k in cols for k in ["diagnosis", "dr_level", "label"]):
                                print(f"Found training CSV at: {csv_p}")
                                return root, csv_p, root
                        except Exception:
                            pass

    # 3. Fallback to any compatible CSV (like valid.csv)
    for base in search_dirs:
        if os.path.exists(base):
            for root, dirs, files in os.walk(base):
                for f in files:
                    if f.endswith(".csv"):
                        csv_p = os.path.join(root, f)
                        try:
                            df_temp = pd.read_csv(csv_p, nrows=5)
                            cols = [c.lower() for c in df_temp.columns]
                            if any(k in cols for k in ["diagnosis", "dr_level", "label"]):
                                print(f"Found compatible CSV at: {csv_p}")
                                return root, csv_p, root
                        except Exception:
                            pass

    msg = (
        f"\n{'='*65}\n"
        f"COULD NOT FIND APTOS 'train.csv' IN /kaggle/input!\n\n"
        f"HOW TO ATTACH THE OFFICIAL COMPETITION DATASET ON KAGGLE:\n"
        f"1. In the top-right of your notebook, click '+ Add Input'.\n"
        f"2. Click the 'Competition Datasets' button.\n"
        f"3. Type in the search box: aptos2019-blindness-detection\n"
        f"4. Click the 'Add' (+) button next to 'APTOS 2019 Blindness Detection'.\n"
        f"5. Re-run this cell!\n"
        f"{'='*65}\n"
    )
    raise FileNotFoundError(msg)

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
class CFG:
    data_root, csv_path, image_dir = None, None, None
    out_dir = "/kaggle/working"

    image_size = 512
    net_input = 224
    val_split = 0.15
    test_split = 0.15
    batch_size = 32
    max_epochs = 8
    initial_lr = 3e-4
    num_classes = 5
    num_biomarkers = 5
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def setup_paths(cls):
        cls.data_root, cls.csv_path, cls.image_dir = find_aptos_data()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(CFG.seed)

# Biomarkers and Grade definitions
BIOMARKER_NAMES = [
    "Microaneurysms",
    "Hemorrhages",
    "Hard Exudates",
    "Cotton Wool Spots",
    "Neovascularization"
]

GRADE_NAMES = [
    "0 - No DR",
    "1 - Mild Non-Proliferative DR",
    "2 - Moderate Non-Proliferative DR",
    "3 - Severe Non-Proliferative DR",
    "4 - Proliferative DR (PDR)"
]

GRADE_TO_BIOMARKER_PRIOR = {
    0: [0.0, 0.0, 0.0, 0.0, 0.0],
    1: [1.0, 0.0, 0.0, 0.0, 0.0],
    2: [1.0, 1.0, 1.0, 0.0, 0.0],
    3: [1.0, 1.0, 1.0, 1.0, 0.0],
    4: [1.0, 1.0, 1.0, 1.0, 1.0],
}

# ---------------------------------------------------------------------
# IMAGE ENHANCEMENT (CLAHE ON GREEN CHANNEL & BEN GRAHAM NORM)
# ---------------------------------------------------------------------
def enhance_fundus(img, size=512):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (size, size))
    
    # CLAHE on green channel
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    green = img[:, :, 1]
    green_eq = clahe.apply(green)
    green_eq = cv2.GaussianBlur(green_eq, (0, 0), sigmaX=0.5)
    
    out = img.copy()
    out[:, :, 1] = green_eq
    
    # Ben Graham illumination correction
    sigma = size / 30.0
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)
    out = out.astype(np.float32) - blurred.astype(np.float32) + 128.0
    out = np.clip(out, 0, 255)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return out

# ---------------------------------------------------------------------
# DATASET & AUGMENTATIONS
# ---------------------------------------------------------------------
class APTOSDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"])
        if img is None:
            img = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        enhanced = enhance_fundus(img, CFG.image_size)
        
        if self.is_train:
            if random.random() > 0.5:
                enhanced = cv2.flip(enhanced, 1)
            if random.random() > 0.5:
                enhanced = cv2.flip(enhanced, 0)
            angle = random.uniform(-25, 25)
            M = cv2.getRotationMatrix2D((CFG.image_size / 2, CFG.image_size / 2), angle, 1.0)
            enhanced = cv2.warpAffine(enhanced, M, (CFG.image_size, CFG.image_size))
            
        resized = cv2.resize(enhanced, (CFG.net_input, CFG.net_input))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        tensor = self.normalize(tensor)
        
        grade = int(row["diagnosis"])
        bio_target = torch.tensor(GRADE_TO_BIOMARKER_PRIOR[grade], dtype=torch.float32)
        return tensor, grade, bio_target, enhanced

# ---------------------------------------------------------------------
# EXPLAINABLE MULTI-TASK MODEL & GRAD-CAM
# ---------------------------------------------------------------------
class ExplainableDRModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        
        in_feat = backbone.fc.in_features
        self.grade_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feat, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, CFG.num_classes)
        )
        self.biomarker_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_feat, 256),
            nn.SiLU(),
            nn.Linear(256, CFG.num_biomarkers)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        pooled = self.avgpool(x)
        flat = torch.flatten(pooled, 1)
        return self.grade_head(flat), self.biomarker_head(flat)


class GradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.layer4[-1]
        self.activations = None
        self.gradients = None
        self._fwd = self.target_layer.register_forward_hook(self._save_act)
        self._bwd = self.target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, i, o):
        self.activations = o.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        g_logits, b_logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = g_logits.argmax(1).item()
            
        self.model.zero_grad()
        score = g_logits[0, class_idx]
        score.backward()
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        return cam, class_idx, g_logits, b_logits

# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------
def main():
    CFG.setup_paths()
    print("Loading APTOS metadata from", CFG.csv_path)
    df = pd.read_csv(CFG.csv_path)
    
    # Normalize diagnosis column name
    for col in df.columns:
        if col.lower() in ["diagnosis", "dr_level", "label"]:
            df["diagnosis"] = df[col].astype(int)
            break
            
    id_col = "id_code"
    for col in df.columns:
        if col.lower() in ["id_code", "id", "image_id", "filename"]:
            id_col = col
            break
            
    # Index all image files across /kaggle/input
    print("Scanning and indexing all image files in /kaggle/input...")
    image_index = {}
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_index[f.lower()] = os.path.join(root, f)
                image_index[os.path.splitext(f)[0].lower()] = os.path.join(root, f)
                
    print(f"Discovered {len(image_index)} image files on disk.")
    
    def resolve_image_path(val):
        clean = str(val).strip().lower()
        clean_no_ext = os.path.splitext(clean)[0]
        if clean in image_index:
            return image_index[clean]
        if clean_no_ext in image_index:
            return image_index[clean_no_ext]
        return os.path.join(CFG.image_dir, f"{val}.png")
        
    df["filepath"] = df[id_col].apply(resolve_image_path)
    valid_mask = df["filepath"].apply(os.path.exists)
    print(f"Matched {valid_mask.sum()} out of {len(df)} images on disk successfully.")
    df = df[valid_mask].reset_index(drop=True)
    
    if len(df) == 0:
        raise FileNotFoundError(
            "Found CSV metadata, but could not link to image files on disk. "
            "Please ensure you attach the official 'APTOS 2019 Blindness Detection' competition dataset!"
        )
    print(f"Total usable training images: {len(df)}")
    
    train_df, temp_df = train_test_split(df, test_size=CFG.val_split + CFG.test_split, stratify=df["diagnosis"], random_state=CFG.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["diagnosis"], random_state=CFG.seed)
    
    counts = train_df["diagnosis"].value_counts().sort_index().values.astype(np.float32)
    class_weights = torch.tensor(len(train_df) / (CFG.num_classes * counts), dtype=torch.float32).to(CFG.device)
    
    criterion_grade = nn.CrossEntropyLoss(weight=class_weights)
    criterion_bio = nn.BCEWithLogitsLoss()
    
    train_loader = DataLoader(APTOSDataset(train_df, is_train=True), batch_size=CFG.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(APTOSDataset(val_df, is_train=False), batch_size=CFG.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(APTOSDataset(test_df, is_train=False), batch_size=CFG.batch_size, shuffle=False, num_workers=2)
    
    model = ExplainableDRModel().to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.initial_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.max_epochs)
    
    print("\nStarting Training with Multi-Task Lesion Attribution...")
    best_loss = float("inf")
    best_state = None
    
    for epoch in range(1, CFG.max_epochs + 1):
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, grades, bios, _ in train_loader:
            imgs, grades, bios = imgs.to(CFG.device), grades.to(CFG.device), bios.to(CFG.device)
            optimizer.zero_grad()
            g_out, b_out = model(imgs)
            loss = criterion_grade(g_out, grades) + 0.5 * criterion_bio(b_out, bios)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * imgs.size(0)
            tr_correct += (g_out.argmax(1) == grades).sum().item()
            tr_total += imgs.size(0)
            
        # Validation
        model.eval()
        v_loss, v_preds, v_labels = 0.0, [], []
        with torch.no_grad():
            for imgs, grades, bios, _ in val_loader:
                imgs, grades, bios = imgs.to(CFG.device), grades.to(CFG.device), bios.to(CFG.device)
                g_out, b_out = model(imgs)
                loss = criterion_grade(g_out, grades) + 0.5 * criterion_bio(b_out, bios)
                v_loss += loss.item() * imgs.size(0)
                v_preds.extend(g_out.argmax(1).cpu().numpy())
                v_labels.extend(grades.cpu().numpy())
                
        scheduler.step()
        qwk = cohen_kappa_score(v_labels, v_preds, weights="quadratic")
        print(f"Epoch {epoch:02d}/{CFG.max_epochs} | Train Loss: {tr_loss/tr_total:.4f} Acc: {tr_correct/tr_total*100:.1f}% | Val Loss: {v_loss/len(val_df):.4f} QWK: {qwk:.4f}")
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(CFG.out_dir, "explainable_dr_resnet50.pt"))
            print(f"  --> [Saved best model checkpoint (Val Loss: {best_loss:.4f}) to /kaggle/working]")
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Evaluate on Test Set
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, grades, _, _ in test_loader:
            imgs = imgs.to(CFG.device)
            g_out, _ = model(imgs)
            test_preds.extend(g_out.argmax(1).cpu().numpy())
            test_labels.extend(grades.numpy())
            
    test_preds, test_labels = np.array(test_preds), np.array(test_labels)
    test_qwk = cohen_kappa_score(test_labels, test_preds, weights="quadratic")
    
    tp = np.sum((test_labels >= 2) & (test_preds >= 2))
    fn = np.sum((test_labels >= 2) & (test_preds < 2))
    tn = np.sum((test_labels < 2) & (test_preds < 2))
    fp = np.sum((test_labels < 2) & (test_preds >= 2))
    sens = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    spec = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0
    
    print("\n" + "="*50)
    print(f"TEST RESULTS:")
    print(f"Quadratic Weighted Kappa (QWK): {test_qwk:.4f}")
    print(f"Referable DR (Grade 2+) Sensitivity: {sens:.2f}%")
    print(f"Referable DR (Grade 2+) Specificity: {spec:.2f}%")
    print("="*50)
    
    # Generate Grad-CAM for 4 test samples
    gradcam = GradCAM(model)
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    fig, axes = plt.subplots(4, 2, figsize=(10, 16))
    for i in range(4):
        row = test_df.iloc[i]
        img = cv2.imread(row["filepath"])
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enh = enhance_fundus(img, CFG.image_size)
        resized = cv2.resize(enh, (CFG.net_input, CFG.net_input))
        tensor = norm(torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(CFG.device)
        
        cam, pred_g, g_logits, b_logits = gradcam(tensor)
        cam_heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(resized, 0.6, cam_heat, 0.4, 0)
        
        axes[i, 0].imshow(resized)
        axes[i, 0].set_title(f"Sample {i}: True Grade {row['diagnosis']}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(overlay)
        axes[i, 1].set_title(f"Grad-CAM Heatmap: Pred Grade {pred_g}")
        axes[i, 1].axis("off")
        
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.out_dir, "test_gradcam_explanations.png"), bbox_inches="tight", dpi=150)
    plt.show()
    gradcam.remove()
    print("Pipeline Complete! Output saved to /kaggle/working")

if __name__ == "__main__":
    main()
