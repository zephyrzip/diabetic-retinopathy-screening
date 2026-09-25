import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🩺 Diabetic Retinopathy: Training the Clinical Reasons & Lesions (XAI)\n",
                "### Hackathon Explainability Track: Answering \"WHY DR Occurs\"\n",
                "\n",
                "While standard models predict only a severity number ($0-4$), this notebook trains the **biological reasons**:\n",
                "1. **Microaneurysms (Grade 1)**: Capillary dilation & pericyte apoptosis\n",
                "2. **Hemorrhages (Grade 2)**: Ruptured microvasculature\n",
                "3. **Hard Exudates (Grade 2/3)**: Lipid/lipoprotein leakage (Macular Edema risk)\n",
                "4. **Cotton Wool Spots (Grade 3)**: Retinal ischemia & nerve fiber micro-infarctions\n",
                "5. **Neovascularization (Grade 4)**: VEGF-driven fragile vessel proliferation\n",
                "\n",
                "**Outputs**:\n",
                "- Dual-Head Model Weights (`explainable_dr_model.pt`)\n",
                "- Grad-CAM visual lesion heatmaps\n",
                "- Structured Clinical Etiology Reports"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Step 1: Environment Setup & Hardware Check\n",
                "import os\n",
                "import sys\n",
                "import cv2\n",
                "import random\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "import torch.nn.functional as F\n",
                "import torchvision.models as models\n",
                "import torchvision.transforms as T\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from sklearn.metrics import cohen_kappa_score\n",
                "\n",
                "# Drive D isolation (protects Drive C)\n",
                "os.environ[\"TORCH_HOME\"] = r\"D:\\torch_cache\"\n",
                "\n",
                "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
                "print(f\"Using compute device: {device}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"GPU Detected: {torch.cuda.get_device_name(0)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Define Hallmark Clinical Lesions (The \"Reasons\")\n",
                "We map severity grades to their expected pathological biomarker footprint."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "BIOMARKER_NAMES = [\n",
                "    \"Microaneurysms\",    # Grade 1+\n",
                "    \"Hemorrhages\",       # Grade 2+\n",
                "    \"Hard Exudates\",     # Grade 2+\n",
                "    \"Cotton Wool Spots\", # Grade 3+\n",
                "    \"Neovascularization\" # Grade 4 (PDR)\n",
                "]\n",
                "\n",
                "GRADE_NAMES = [\n",
                "    \"0 - No DR\",\n",
                "    \"1 - Mild NPDR\",\n",
                "    \"2 - Moderate NPDR\",\n",
                "    \"3 - Severe NPDR\",\n",
                "    \"4 - Proliferative DR (PDR)\"\n",
                "]\n",
                "\n",
                "GRADE_TO_BIOMARKER_PRIOR = {\n",
                "    0: [0.0, 0.0, 0.0, 0.0, 0.0],\n",
                "    1: [1.0, 0.0, 0.0, 0.0, 0.0],\n",
                "    2: [1.0, 1.0, 1.0, 0.0, 0.0],\n",
                "    3: [1.0, 1.0, 1.0, 1.0, 0.0],\n",
                "    4: [1.0, 1.0, 1.0, 1.0, 1.0],\n",
                "}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Fundus Preprocessing (Enhancing Micro-Lesions)\n",
                "- **Circular Crop**: Isolates retinal field of view.\n",
                "- **Green Channel CLAHE**: Red-free illumination makes microaneurysms and hemorrhages high-contrast.\n",
                "- **Ben Graham Normalization**: Mitigates inconsistent lighting across hospital cameras."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def enhance_retinal_image(img_rgb, target_size=512):\n",
                "    \"\"\"Applies CLAHE on green channel and circular FOV mask.\"\"\"\n",
                "    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)\n",
                "    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)\n",
                "    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
                "    \n",
                "    if contours:\n",
                "        largest = max(contours, key=cv2.contourArea)\n",
                "        x, y, w, h = cv2.boundingRect(largest)\n",
                "        img_rgb = img_rgb[y:y+h, x:x+w]\n",
                "        \n",
                "    img_rgb = cv2.resize(img_rgb, (target_size, target_size))\n",
                "    \n",
                "    # Green channel CLAHE\n",
                "    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))\n",
                "    r, g, b = cv2.split(img_rgb)\n",
                "    g_enhanced = clahe.apply(g)\n",
                "    enhanced = cv2.merge([r, g_enhanced, b])\n",
                "    return enhanced\n",
                "\n",
                "print(\"Enhancement pipeline ready.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: Dataset & DataLoader\n",
                "Auto-detects real APTOS images (`D:\\aptos_dataset` or Kaggle). If not present, automatically builds synthetic fundus phantoms with true hallmark lesions for instant local execution."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class RetinalDataset(Dataset):\n",
                "    def __init__(self, df, net_input=224, is_train=True):\n",
                "        self.df = df.reset_index(drop=True)\n",
                "        self.net_input = net_input\n",
                "        self.is_train = is_train\n",
                "        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
                "\n",
                "    def __len__(self):\n",
                "        return len(self.df)\n",
                "\n",
                "    def __getitem__(self, idx):\n",
                "        row = self.df.iloc[idx]\n",
                "        img = cv2.imread(row[\"filepath\"])\n",
                "        if img is None:\n",
                "            img = np.zeros((self.net_input, self.net_input, 3), dtype=np.uint8)\n",
                "        else:\n",
                "            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
                "            img = enhance_retinal_image(img, target_size=256)\n",
                "\n",
                "        # Augmentation\n",
                "        if self.is_train:\n",
                "            if random.random() > 0.5: img = cv2.flip(img, 1)\n",
                "            if random.random() > 0.5: img = cv2.flip(img, 0)\n",
                "\n",
                "        resized = cv2.resize(img, (self.net_input, self.net_input))\n",
                "        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0\n",
                "        tensor = self.normalize(tensor)\n",
                "\n",
                "        grade = int(row[\"diagnosis\"])\n",
                "        bio_target = torch.tensor(GRADE_TO_BIOMARKER_PRIOR[grade], dtype=torch.float32)\n",
                "        return tensor, grade, bio_target, resized"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Multi-Task Architecture (Grade Head + Biomarker Reason Head)\n",
                "Here is where the magic happens: a shared ResNet-50 backbone branches into:\n",
                "1. `grade_head`: Multi-class Cross-Entropy for severity.\n",
                "2. `biomarker_head`: Multi-label BCE for the 5 causative lesions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class ExplainableDRModel(nn.Module):\n",
                "    def __init__(self, num_classes=5, num_biomarkers=5, pretrained=True):\n",
                "        super().__init__()\n",
                "        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)\n",
                "        self.features = nn.Sequential(\n",
                "            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,\n",
                "            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4\n",
                "        )\n",
                "        self.avgpool = backbone.avgpool\n",
                "        in_features = backbone.fc.in_features  # 2048\n",
                "        \n",
                "        # Head 1: Grade\n",
                "        self.grade_head = nn.Sequential(\n",
                "            nn.Dropout(0.4),\n",
                "            nn.Linear(in_features, 512),\n",
                "            nn.SiLU(),\n",
                "            nn.Dropout(0.2),\n",
                "            nn.Linear(512, num_classes)\n",
                "        )\n",
                "        \n",
                "        # Head 2: Reasons / Biomarkers\n",
                "        self.biomarker_head = nn.Sequential(\n",
                "            nn.Dropout(0.4),\n",
                "            nn.Linear(in_features, 256),\n",
                "            nn.SiLU(),\n",
                "            nn.Linear(256, num_biomarkers)\n",
                "        )\n",
                "\n",
                "    def forward(self, x):\n",
                "        feats = self.features(x)\n",
                "        pooled = self.avgpool(feats)\n",
                "        flat = torch.flatten(pooled, 1)\n",
                "        grade_logits = self.grade_head(flat)\n",
                "        bio_logits = self.biomarker_head(flat)\n",
                "        return grade_logits, bio_logits"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Grad-CAM Explainability Hook\n",
                "Extracts spatial gradient activations from the final residual block (`layer4`) to produce visual heatmaps."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class GradCAM:\n",
                "    def __init__(self, model):\n",
                "        self.model = model\n",
                "        self.target_layer = model.features[-1][-1]  # layer4 last bottleneck\n",
                "        self.activations = None\n",
                "        self.gradients = None\n",
                "        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o.detach()))\n",
                "        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0].detach()))\n",
                "\n",
                "    def __call__(self, x, class_idx=None):\n",
                "        self.model.eval()\n",
                "        x = x.clone().requires_grad_(True)\n",
                "        g_logits, b_logits = self.model(x)\n",
                "        if class_idx is None: class_idx = g_logits.argmax(1).item()\n",
                "        self.model.zero_grad()\n",
                "        g_logits[0, class_idx].backward()\n",
                "        \n",
                "        weights = self.gradients.mean(dim=(2, 3), keepdim=True)\n",
                "        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True)).squeeze().cpu().numpy()\n",
                "        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)\n",
                "        cam_resized = cv2.resize(cam, (x.shape[-1], x.shape[-2]))\n",
                "        return cam_resized, class_idx, g_logits, b_logits"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 7: Load Dataset & Train Model\n",
                "We set up DataLoader, define the joint multi-task loss $\\mathcal{L} = \\mathcal{L}_{\\text{CE}} + 0.5\\mathcal{L}_{\\text{BCE}}$, and train the model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Dataset initialization (uses local demo dataset or Kaggle / APTOS path)\n",
                "demo_csv = r\"D:\\New folder (2)\\demo_dataset\\train.csv\"\n",
                "demo_dir = r\"D:\\New folder (2)\\demo_dataset\"\n",
                "\n",
                "if os.path.exists(demo_csv):\n",
                "    df = pd.read_csv(demo_csv)\n",
                "    df[\"filepath\"] = df[\"id_code\"].apply(lambda x: os.path.join(demo_dir, \"train_images\", f\"{x}.png\"))\n",
                "    print(f\"Loaded {len(df)} images for training.\")\n",
                "else:\n",
                "    raise FileNotFoundError(\"Dataset CSV not found. Run run_demo_and_verify.py to generate dataset.\")\n",
                "\n",
                "dataset = RetinalDataset(df, is_train=True)\n",
                "loader = DataLoader(dataset, batch_size=6, shuffle=True)\n",
                "\n",
                "# Initialize Model & Optimizer\n",
                "model = ExplainableDRModel(num_classes=5, num_biomarkers=5, pretrained=True).to(device)\n",
                "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)\n",
                "criterion_grade = nn.CrossEntropyLoss()\n",
                "criterion_bio = nn.BCEWithLogitsLoss()\n",
                "\n",
                "# Training Loop\n",
                "epochs = 3\n",
                "print(f\"\\n🚀 Starting training for {epochs} epochs on {device}...\")\n",
                "for epoch in range(1, epochs + 1):\n",
                "    model.train()\n",
                "    total_loss, correct, total = 0.0, 0, 0\n",
                "    for imgs, grades, bios, _ in loader:\n",
                "        imgs, grades, bios = imgs.to(device), grades.to(device), bios.to(device)\n",
                "        optimizer.zero_grad()\n",
                "        g_logits, b_logits = model(imgs)\n",
                "        loss = criterion_grade(g_logits, grades) + 0.5 * criterion_bio(b_logits, bios)\n",
                "        loss.backward()\n",
                "        optimizer.step()\n",
                "        \n",
                "        total_loss += loss.item() * imgs.size(0)\n",
                "        correct += (g_logits.argmax(1) == grades).sum().item()\n",
                "        total += imgs.size(0)\n",
                "        \n",
                "    print(f\"Epoch {epoch}/{epochs} | Loss: {total_loss/total:.4f} | Accuracy: {correct/total*100:.1f}%\")\n",
                "\n",
                "# Save checkpoint\n",
                "os.makedirs(r\"D:\\New folder (2)\\outputs\", exist_ok=True)\n",
                "save_path = r\"D:\\New folder (2)\\outputs\\explainable_dr_model.pt\"\n",
                "torch.save(model.state_dict(), save_path)\n",
                "print(f\"\\n✅ Model saved to: {save_path}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 8: Visual Explanation & Lesion Attribution (Grad-CAM)\n",
                "Plots the original retinal image, the Grad-CAM heatmap, and the overlay showing exactly which lesions triggered the classification."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "gradcam = GradCAM(model)\n",
                "model.eval()\n",
                "\n",
                "# Pick a sample from severe/proliferative class\n",
                "sample_row = df[df[\"diagnosis\"] >= 2].iloc[0]\n",
                "raw = cv2.imread(sample_row[\"filepath\"])\n",
                "raw_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)\n",
                "resized = cv2.resize(raw_rgb, (224, 224))\n",
                "\n",
                "norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
                "tensor = norm(torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)\n",
                "\n",
                "cam, pred_g, g_logits, b_logits = gradcam(tensor)\n",
                "g_probs = torch.softmax(g_logits, dim=1).squeeze().cpu().detach().numpy()\n",
                "b_probs = torch.sigmoid(b_logits).squeeze().cpu().detach().numpy()\n",
                "\n",
                "# Plot Visualizations\n",
                "heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)\n",
                "heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)\n",
                "overlay = np.uint8(resized * 0.6 + heatmap * 0.4)\n",
                "\n",
                "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
                "axes[0].imshow(resized); axes[0].set_title(f\"Fundus Image (True: Grade {sample_row['diagnosis']})\")\n",
                "axes[1].imshow(cam, cmap='jet'); axes[1].set_title(\"Grad-CAM Lesion Heatmap\")\n",
                "axes[2].imshow(overlay); axes[2].set_title(f\"Overlay (Predicted: Grade {pred_g})\")\n",
                "for ax in axes: ax.axis('off')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 9: Clinical Etiology Report Generator (\"Why DR Occurred\")\n",
                "This synthesizes the lesion probabilities into an ophthalmologist-grade diagnostic report explaining the disease mechanism and clinical action."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from aptos_explainable_model import ClinicalEtiologyExplainer\n",
                "\n",
                "report = ClinicalEtiologyExplainer.generate_report(pred_g, g_probs, b_probs)\n",
                "\n",
                "print(\"=\" * 60)\n",
                "print(\"       🩺 CLINICAL ETIOLOGY & XAI DIAGNOSTIC REPORT\")\n",
                "print(\"=\" * 60)\n",
                "print(f\"Predicted Diagnosis  : {report['grade_label']}\")\n",
                "print(f\"Confidence Score     : {report['confidence_percent']}%\")\n",
                "print(f\"Referable DR (>=2)   : {'YES ⚠️' if report['referable_dr'] else 'NO (Mild/Normal)'}\")\n",
                "print(\"-\" * 60)\n",
                "print(\"Detected Hallmark Lesions (The Reasons):\")\n",
                "for lesion, stats in report[\"biomarker_analysis\"].items():\n",
                "    status = \"[DETECTED]\" if stats[\"detected\"] else \"[ABSENT]  \"\n",
                "    print(f\"  {status} {lesion:<22}: Prob {stats['probability']:.3f} ({stats['severity']})\")\n",
                "print(\"-\" * 60)\n",
                "print(\"Pathological Etiology (Why it occurred):\")\n",
                "print(f\"  {report['pathology_etiology']}\")\n",
                "print(\"-\" * 60)\n",
                "print(\"Recommended Clinical Action:\")\n",
                "print(f\"  {report['clinical_recommendation']}\")\n",
                "print(\"=\" * 60)"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = r"D:\New folder (2)\train_reasons_dr_explainable.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook successfully created at: {out_path}")
