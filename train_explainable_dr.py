"""
Training Pipeline for Explainable Diabetic Retinopathy (XAI) & Biomarker Etiology
Configured strictly for D: Drive storage and GTX 1650 GPU optimization.
"""

import os
import sys
import copy
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Strict D: Drive cache configuration
os.environ["TORCH_HOME"] = r"D:\torch_cache"
os.environ["HF_HOME"] = r"D:\torch_cache"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import cv2

from data_enhancement import assess_and_enhance_fundus
from aptos_explainable_model import (
    ExplainableDRModel,
    GradCAM,
    ClinicalEtiologyExplainer,
    BIOMARKER_NAMES,
    GRADE_NAMES
)

# Clinical mapping priors: mapping DR grade to expected pathological biomarker profile
GRADE_TO_BIOMARKER_PRIOR = {
    0: [0.0, 0.0, 0.0, 0.0, 0.0],  # No lesions
    1: [1.0, 0.0, 0.0, 0.0, 0.0],  # Microaneurysms only
    2: [1.0, 1.0, 1.0, 0.0, 0.0],  # Microaneurysms + Hemorrhages + Hard Exudates
    3: [1.0, 1.0, 1.0, 1.0, 0.0],  # Above + Cotton Wool Spots / Severe Ischemia
    4: [1.0, 1.0, 1.0, 1.0, 1.0],  # Above + Neovascularization (Proliferative)
}


class Config:
    def __init__(self, data_root="D:\\aptos_dataset", out_dir="D:\\New folder (2)\\outputs"):
        self.data_root = data_root
        self.image_dir = os.path.join(data_root, "train_images")
        self.csv_path = os.path.join(data_root, "train.csv")
        self.out_dir = out_dir
        
        self.image_size = 512
        self.net_input = 224
        self.batch_size = 16  # Optimized for 4GB GTX 1650 VRAM
        self.num_epochs = 15
        self.lr = 2e-4
        self.num_classes = 5
        self.num_biomarkers = 5
        self.val_split = 0.15
        self.test_split = 0.15
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class APTOSDataset(Dataset):
    def __init__(self, dataframe, image_size=512, net_input=224, is_train=True):
        self.df = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.net_input = net_input
        self.is_train = is_train
        
        # Standard normalization for ImageNet backbones
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        
        # Load and verify fundus image
        img = cv2.imread(img_path)
        if img is None:
            # Fallback black canvas if corrupt
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        enhanced_img, _ = assess_and_enhance_fundus(img, self.image_size)
        
        # Data Augmentations
        if self.is_train:
            if random.random() > 0.5:
                enhanced_img = cv2.flip(enhanced_img, 1)  # Horizontal flip
            if random.random() > 0.5:
                enhanced_img = cv2.flip(enhanced_img, 0)  # Vertical flip
            angle = random.uniform(-25, 25)
            M = cv2.getRotationMatrix2D((self.image_size / 2, self.image_size / 2), angle, 1.0)
            enhanced_img = cv2.warpAffine(enhanced_img, M, (self.image_size, self.image_size))
            
        resized = cv2.resize(enhanced_img, (self.net_input, self.net_input))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        tensor = self.normalize(tensor)
        
        grade_label = int(row["diagnosis"])
        biomarker_target = torch.tensor(GRADE_TO_BIOMARKER_PRIOR[grade_label], dtype=torch.float32)
        
        return tensor, grade_label, biomarker_target, enhanced_img


def train_epoch(model, loader, optimizer, criterion_grade, criterion_biomarker, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    
    for imgs, grade_labels, biomarker_targets, _ in loader:
        imgs = imgs.to(device)
        grade_labels = grade_labels.to(device)
        biomarker_targets = biomarker_targets.to(device)
        
        optimizer.zero_grad()
        grade_logits, biomarker_logits = model(imgs)
        
        loss_grade = criterion_grade(grade_logits, grade_labels)
        loss_bio = criterion_biomarker(biomarker_logits, biomarker_targets)
        loss = loss_grade + 0.5 * loss_bio
        
        loss.backward()
        optimizer.step()
        
        preds = grade_logits.argmax(dim=1)
        total_correct += (preds == grade_labels).sum().item()
        total_loss += loss.item() * imgs.size(0)
        total_count += imgs.size(0)
        
    return total_loss / total_count, total_correct / total_count


def evaluate(model, loader, criterion_grade, criterion_biomarker, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for imgs, grade_labels, biomarker_targets, _ in loader:
            imgs = imgs.to(device)
            grade_labels = grade_labels.to(device)
            biomarker_targets = biomarker_targets.to(device)
            
            grade_logits, biomarker_logits = model(imgs)
            loss_grade = criterion_grade(grade_logits, grade_labels)
            loss_bio = criterion_biomarker(biomarker_logits, biomarker_targets)
            loss = loss_grade + 0.5 * loss_bio
            
            preds = grade_logits.argmax(dim=1)
            total_correct += (preds == grade_labels).sum().item()
            total_loss += loss.item() * imgs.size(0)
            total_count += imgs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(grade_labels.cpu().numpy())
            
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    accuracy = total_correct / total_count if total_count > 0 else 0.0
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def run_pipeline(cfg):
    seed_everything(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.out_dir, "gradcam"), exist_ok=True)
    
    print(f"=== Starting Explainable DR Training Pipeline ===")
    print(f"Device: {cfg.device}")
    print(f"Outputs will be saved strictly to: {cfg.out_dir}")
    
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"APTOS train.csv not found at {cfg.csv_path}")
        
    df = pd.read_csv(cfg.csv_path)
    df["filepath"] = df["id_code"].apply(lambda x: os.path.join(cfg.image_dir, f"{x}.png"))
    print(f"Dataset loaded: {len(df)} samples.")
    print("Class distribution:\n", df["diagnosis"].value_counts().sort_index())
    
    # Train / Val / Test split
    train_df, temp_df = train_test_split(
        df, test_size=(cfg.val_split + cfg.test_split),
        stratify=df["diagnosis"], random_state=cfg.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=(cfg.test_split / (cfg.val_split + cfg.test_split)),
        stratify=temp_df["diagnosis"], random_state=cfg.seed
    )
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Class weights for CrossEntropyLoss
    label_counts = train_df["diagnosis"].value_counts().sort_index().values.astype(np.float32)
    weights = len(train_df) / (cfg.num_classes * label_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(cfg.device)
    
    criterion_grade = nn.CrossEntropyLoss(weight=class_weights)
    criterion_biomarker = nn.BCEWithLogitsLoss()
    
    train_ds = APTOSDataset(train_df, cfg.image_size, cfg.net_input, is_train=True)
    val_ds = APTOSDataset(val_df, cfg.image_size, cfg.net_input, is_train=False)
    test_ds = APTOSDataset(test_df, cfg.image_size, cfg.net_input, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    model = ExplainableDRModel(
        num_classes=cfg.num_classes,
        num_biomarkers=cfg.num_biomarkers,
        pretrained=True
    ).to(cfg.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)
    
    best_val_loss = float("inf")
    best_model_state = None
    
    for epoch in range(1, cfg.num_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion_grade, criterion_biomarker, cfg.device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion_grade, criterion_biomarker, cfg.device)
        scheduler.step()
        
        qwk = cohen_kappa_score(val_labels, val_preds, weights="quadratic")
        print(f"Epoch [{epoch:02d}/{cfg.num_epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} QWK: {qwk:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            
    # Save best checkpoint
    model.load_state_dict(best_model_state)
    ckpt_path = os.path.join(cfg.out_dir, "explainable_dr_resnet50.pt")
    torch.save({
        "model_state": model.state_dict(),
        "cfg": vars(cfg),
        "best_val_loss": best_val_loss
    }, ckpt_path)
    print(f"Saved best model checkpoint to {ckpt_path}")
    
    # Test evaluation & Clinical KPI
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion_grade, criterion_biomarker, cfg.device)
    test_qwk = cohen_kappa_score(test_labels, test_preds, weights="quadratic")
    
    true_ref = test_labels >= 2
    pred_ref = test_preds >= 2
    tp = np.sum(true_ref & pred_ref)
    tn = np.sum(~true_ref & ~pred_ref)
    fp = np.sum(~true_ref & pred_ref)
    fn = np.sum(true_ref & ~pred_ref)
    
    sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0
    
    print("\n--- TEST SET EVALUATION ---")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Quadratic Weighted Kappa (QWK): {test_qwk:.4f}")
    print(f"Referable DR (Grade 2+) Sensitivity: {sensitivity:.2f}%")
    print(f"Referable DR (Grade 2+) Specificity: {specificity:.2f}%")
    
    # Save Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1", "2", "3", "4"])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", values_format=".2f")
    plt.title("Explainable DR - Confusion Matrix (Normalized)")
    cm_path = os.path.join(cfg.out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")
    
    # Generate Grad-CAM Explainability Samples
    print("\nGenerating Grad-CAM Explainability Reports on test samples...")
    gradcam = GradCAM(model)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    for i in range(min(5, len(test_df))):
        sample_row = test_df.iloc[i]
        img_raw = cv2.imread(sample_row["filepath"])
        if img_raw is None:
            continue
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        enhanced_img, _ = assess_and_enhance_fundus(img_rgb, cfg.image_size)
        
        resized = cv2.resize(enhanced_img, (cfg.net_input, cfg.net_input))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
        tensor = normalize(tensor).unsqueeze(0).to(cfg.device)
        
        cam, pred_grade, grade_logits, bio_logits = gradcam(tensor)
        grade_probs = torch.softmax(grade_logits, dim=1).squeeze().cpu().detach().numpy()
        bio_probs = torch.sigmoid(bio_logits).squeeze().cpu().detach().numpy()
        
        # Clinical Etiology Report
        report = ClinicalEtiologyExplainer.generate_report(pred_grade, grade_probs, bio_probs)
        report_path = os.path.join(cfg.out_dir, "gradcam", f"sample_{i}_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        # Visual overlay
        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(resized, 0.6, cam_heatmap, 0.4, 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cv2.resize(img_rgb, (cfg.net_input, cfg.net_input)))
        axes[0].set_title(f"Original Fundus (True: {sample_row['diagnosis']})")
        axes[0].axis("off")
        
        axes[1].imshow(resized)
        axes[1].set_title("Enhanced (CLAHE Green Channel)")
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Grad-CAM Attribution (Pred: {pred_grade})")
        axes[2].axis("off")
        
        fig_path = os.path.join(cfg.out_dir, "gradcam", f"sample_{i}_xai_overlay.png")
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close()
        
    gradcam.remove()
    print("Pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explainable DR Training Pipeline")
    parser.add_argument("--data_root", type=str, default="D:\\aptos_dataset")
    parser.add_argument("--out_dir", type=str, default="D:\\New folder (2)\\outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    cfg = Config(data_root=args.data_root, out_dir=args.out_dir)
    cfg.num_epochs = args.epochs
    cfg.batch_size = args.batch_size
    run_pipeline(cfg)
