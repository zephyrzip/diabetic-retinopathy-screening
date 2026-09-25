"""
APTOS 2019 Blindness Detection - DR Severity Grading Pipeline (Python/PyTorch port)
Kaggle-ready version of the original MATLAB hackathon pipeline.

Run this as a Kaggle Notebook (GPU on). Paste into cells or run as a script.

Expected data (Kaggle "APTOS 2019 Blindness Detection" dataset, added to the notebook):
    /kaggle/input/aptos2019-blindness-detection/train_images/*.png
    /kaggle/input/aptos2019-blindness-detection/train.csv   (id_code, diagnosis)

diagnosis labels: 0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR
"Referable DR" = diagnosis >= 2 (clinical decision threshold)

pip installs needed on Kaggle (usually already present):
    pip install -q albumentations opencv-python-headless
"""

import os
import cv2
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------
class CFG:
    data_root   = "/kaggle/input/aptos2019-blindness-detection"
    image_dir   = os.path.join(data_root, "train_images")
    csv_path    = os.path.join(data_root, "train.csv")
    out_dir     = "/kaggle/working"

    image_size  = 512          # working resolution before network input resize
    net_input   = 224          # ResNet-50 input size
    val_split   = 0.15
    test_split  = 0.15
    batch_size  = 16
    max_epochs  = 20
    initial_lr  = 3e-4
    lr_drop_factor = 0.5
    lr_drop_period = 5
    val_patience   = 6
    num_classes = 5
    seed        = 42
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG.seed)

# ---------------------------------------------------------------------
# 1. LOAD LABELS
# ---------------------------------------------------------------------
df = pd.read_csv(CFG.csv_path)
df["filepath"] = df["id_code"].apply(lambda x: os.path.join(CFG.image_dir, f"{x}.png"))
print(f"Loaded {len(df)} images. Class distribution:")
print(df["diagnosis"].value_counts().sort_index())

# ---------------------------------------------------------------------
# 2. STRATIFIED TRAIN / VAL / TEST SPLIT
# ---------------------------------------------------------------------
train_df, temp_df = train_test_split(
    df, test_size=(CFG.val_split + CFG.test_split),
    stratify=df["diagnosis"], random_state=CFG.seed,
)
val_df, test_df = train_test_split(
    temp_df, test_size=CFG.test_split / (CFG.val_split + CFG.test_split),
    stratify=temp_df["diagnosis"], random_state=CFG.seed,
)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---------------------------------------------------------------------
# 3. CLASS IMBALANCE HANDLING
# ---------------------------------------------------------------------
label_counts = train_df["diagnosis"].value_counts().sort_index().values.astype(np.float64)
class_weights = len(train_df) / (CFG.num_classes * label_counts)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(CFG.device)
print("Class weights (0-4):", class_weights)

# ---------------------------------------------------------------------
# 4. IMAGE QUALITY ASSESSMENT + ENHANCEMENT (requirement 1)
# ---------------------------------------------------------------------
def assess_and_enhance_fundus(img, size):
    """img: HxWx3 uint8 RGB. Returns (enhanced_img, quality_ok)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, (size, size))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # --- Quality checks ---
    mean_int = gray.mean()
    illum_ok = 15 < mean_int < 240

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    focus_score = lap.var()
    focus_ok = focus_score > 8  # tune empirically

    mask = gray > 10
    fov_fraction = mask.sum() / mask.size
    fov_ok = fov_fraction > 0.35

    quality_ok = illum_ok and focus_ok and fov_ok

    # --- Adaptive enhancement (always applied) ---
    # CLAHE on the green channel (best vessel/lesion contrast in fundus imaging)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    green = img[:, :, 1]
    green_eq = clahe.apply(green)
    green_eq = cv2.GaussianBlur(green_eq, (0, 0), sigmaX=0.5)

    out = img.copy()
    out[:, :, 1] = green_eq

    # Illumination normalization (subtract local mean, Ben Graham style)
    sigma = size / 30
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=sigma)
    out = out.astype(np.float32) - blurred.astype(np.float32) + 128.0
    out = np.clip(out, 0, 255)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return out, quality_ok

# ---------------------------------------------------------------------
# 5. DATASET + AUGMENTATION
# ---------------------------------------------------------------------
train_aug = A.Compose([
    A.Rotate(limit=25, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Affine(translate_percent=(0.0, 0.04), scale=(0.9, 1.1), p=1.0),
    A.Resize(CFG.net_input, CFG.net_input),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

eval_aug = A.Compose([
    A.Resize(CFG.net_input, CFG.net_input),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

class APTOSDataset(Dataset):
    def __init__(self, dataframe, transform, image_size):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row["filepath"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _ = assess_and_enhance_fundus(img, self.image_size)
        img = self.transform(image=img)["image"]
        label = int(row["diagnosis"])
        return img, label

train_ds = APTOSDataset(train_df, train_aug, CFG.image_size)
val_ds   = APTOSDataset(val_df, eval_aug, CFG.image_size)
test_ds  = APTOSDataset(test_df, eval_aug, CFG.image_size)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=2, pin_memory=True)

# ---------------------------------------------------------------------
# 6. MODEL: TRANSFER LEARNING (ResNet-50 backbone)
# ---------------------------------------------------------------------
model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
model.fc = nn.Linear(model.fc.in_features, CFG.num_classes)
model = model.to(CFG.device)

# ---------------------------------------------------------------------
# 7. LOSS / OPTIMIZER / SCHEDULER
# ---------------------------------------------------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.initial_lr)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=CFG.lr_drop_period, gamma=CFG.lr_drop_factor
)

# ---------------------------------------------------------------------
# 8. TRAIN (with early stopping on val loss)
# ---------------------------------------------------------------------
def run_epoch(loader, train_mode):
    model.train() if train_mode else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    torch.set_grad_enabled(train_mode)
    for imgs, labels in loader:
        imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
        if train_mode:
            optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        if train_mode:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_correct += (outputs.argmax(1) == labels).sum().item()
        total_n += imgs.size(0)
    return total_loss / total_n, total_correct / total_n

best_val_loss = float("inf")
best_state = None
patience_counter = 0

print("Starting training...")
for epoch in range(CFG.max_epochs):
    train_loss, train_acc = run_epoch(train_loader, train_mode=True)
    val_loss, val_acc = run_epoch(val_loader, train_mode=False)
    scheduler.step()

    print(f"Epoch {epoch+1}/{CFG.max_epochs} | "
          f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
          f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= CFG.val_patience:
            print(f"Early stopping at epoch {epoch+1} (no val improvement for {CFG.val_patience} epochs).")
            break

model.load_state_dict(best_state)
ckpt_path = os.path.join(CFG.out_dir, "aptos_dr_resnet50.pt")
torch.save({"model_state": model.state_dict(), "cfg": vars(CFG)}, ckpt_path)
print(f"Saved best model to {ckpt_path}")

# ---------------------------------------------------------------------
# 9. EVALUATE ON HELD-OUT TEST SET
# ---------------------------------------------------------------------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(CFG.device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# --- Quadratic Weighted Kappa (the clinically standard metric here) ---
qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
print(f"Quadratic Weighted Kappa: {qwk:.4f}")

# --- Referable DR (>=2) sensitivity / specificity - the clinical KPI ---
true_referable = all_labels >= 2
pred_referable = all_preds >= 2

TP = np.sum(true_referable & pred_referable)
TN = np.sum(~true_referable & ~pred_referable)
FP = np.sum(~true_referable & pred_referable)
FN = np.sum(true_referable & ~pred_referable)

sensitivity = TP / (TP + FN) if (TP + FN) else float("nan")
specificity = TN / (TN + FP) if (TN + FP) else float("nan")

print(f"Referable DR (Level 2+) - Sensitivity: {sensitivity*100:.2f}% (target >90%)")
print(f"Referable DR (Level 2+) - Specificity: {specificity*100:.2f}% (target >85%)")

cm = confusion_matrix(all_labels, all_preds, normalize="true")
disp = ConfusionMatrixDisplay(cm, display_labels=["0", "1", "2", "3", "4"])
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", values_format=".2f")
plt.title("DR Severity Grading - Confusion Matrix (row-normalized)")
plt.savefig(os.path.join(CFG.out_dir, "confusion_matrix.png"), bbox_inches="tight")
plt.show()

# ---------------------------------------------------------------------
# 10. GRAD-CAM EXPLAINABILITY (real implementation, not the hackathon stub)
# ---------------------------------------------------------------------
class GradCAM:
    """
    Real Grad-CAM via forward/backward hooks on a target conv layer.
    For torchvision resnet50, a good target layer is model.layer4[-1]
    (the last residual block before global average pooling).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        """
        input_tensor: 1x3xHxW, already normalized, requires no grad set by caller.
        Returns (cam [H,W] in 0..1, predicted_class_idx).
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)  # 1 x num_classes

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # activations: 1xCxH'xW', gradients: 1xCxH'xW'
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # 1xCx1x1 (GAP of gradients)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # 1x1xH'xW'
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        return cam, class_idx

def show_gradcam(model, image_rgb_uint8, cfg=CFG, class_idx=None):
    """
    image_rgb_uint8: HxWx3 uint8 image, already enhanced (see step 4),
    at cfg.image_size resolution.
    """
    aug = eval_aug(image=image_rgb_uint8)
    input_tensor = aug["image"].unsqueeze(0).to(cfg.device)

    gradcam = GradCAM(model, model.layer4[-1])
    cam, predicted_class = gradcam(input_tensor, class_idx=class_idx)
    gradcam.remove()

    display_img = cv2.resize(image_rgb_uint8, (cfg.net_input, cfg.net_input))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_img, 0.6, heatmap, 0.4, 0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(display_img); axes[0].set_title("Enhanced fundus image"); axes[0].axis("off")
    axes[1].imshow(overlay); axes[1].set_title(f"Grad-CAM (pred class {predicted_class})"); axes[1].axis("off")
    plt.tight_layout()
    plt.show()
    return cam, predicted_class

# Example usage on a single test image:
# sample_row = test_df.iloc[0]
# sample_img = cv2.cvtColor(cv2.imread(sample_row["filepath"]), cv2.COLOR_BGR2RGB)
# sample_img, _ = assess_and_enhance_fundus(sample_img, CFG.image_size)
# show_gradcam(model, sample_img)

print("\nPhase 1 (APTOS) complete. Model saved to", ckpt_path)
print("Next: use this trained backbone as the classification head while")
print("training IDRiD-based U-Net segmentation models for lesion-level")
print("evidence, and validate cross-dataset generalization on Messidor-2.")
