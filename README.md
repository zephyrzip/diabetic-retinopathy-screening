# APTOS 2019 Blindness Detection: Explainable AI (XAI) & Lesion Etiology

This repository trains an **Explainable Deep Learning Pipeline** on retinal fundus imaging from the **APTOS 2019 Blindness Detection** dataset.

Unlike standard "black-box" models that simply output a severity number ($0$ to $4$), this system specifically trains and explains **WHY Diabetic Retinopathy (DR) occurs and WHY a specific diagnosis was made**.

---

## 1. Understanding "Why DR Occurs" (Pathology & AI Architecture)

### The Biological Mechanism
1. **Prolonged Hyperglycemia**: Elevated blood glucose induces biochemical stress on retinal microvasculature.
2. **Pericyte Loss & Capillary Dilation**: Capillary wall weakens, forming balloon-like outpouchings called **Microaneurysms** (**Grade 1 - Mild NPDR**).
3. **Vascular Permeability & Rupture**: Compromised blood-retinal barrier causes:
   - **Hemorrhages**: Blood leaking into retinal layers (**Grade 2 - Moderate NPDR**).
   - **Hard Exudates**: Lipid and lipoprotein deposits leaking from permeable capillaries, risking **Diabetic Macular Edema (DME)**.
4. **Capillary Non-Perfusion & Ischemia**: Precapillary occlusions cause nerve fiber layer micro-infarctions visible as **Cotton Wool Spots (Soft Exudates)** (**Grade 3 - Severe NPDR**).
5. **VEGF Hyper-Secretion & Neovascularization**: Hypoxic retina secretes massive Vascular Endothelial Growth Factor (VEGF), stimulating fragile, abnormal new vessels (**Grade 4 - Proliferative DR / PDR**). These vessels risk vitreous hemorrhage and tractional retinal detachment.

### How Our Model Learns "Why"
- **Dual-Head Architecture**:
  - `grade_head`: Predicts severity class ($0$ to $4$).
  - `biomarker_head`: Multi-label branch detecting presence of the 5 hallmark lesions (Microaneurysms, Hemorrhages, Hard Exudates, Cotton Wool Spots, Neovascularization).
- **Grad-CAM Visual Heatmaps**: Shows the exact pixel regions and lesion clusters that activated the neural network.
- **Clinical Etiology Explainer**: Generates structured clinical diagnostic reports translating image findings into medical etiology.

---

## 2. Storage & Drive Isolation Policy

> [!IMPORTANT]
> **Strict Drive D Storage Policy**:
> To protect Drive C (which has limited space), all components are strictly isolated to Drive D:
> - Virtual Environment: `D:\New folder (2)\venv`
> - Pip Wheel Cache: `D:\pip_cache`
> - PyTorch Model Cache: `D:\torch_cache`
> - Dataset & Outputs: `D:\New folder (2)\` and `D:\aptos_dataset`

---

## 3. Project Structure

```
D:\New folder (2)\
├── aptos_explainable_model.py          # ResNet-50 dual-head architecture, Grad-CAM, & Etiology explainer
├── data_enhancement.py                 # CLAHE on green channel & Ben Graham illumination subtraction
├── train_explainable_dr.py             # Full local training pipeline with QWK & referable DR metrics
├── run_demo_and_verify.py              # Instant end-to-end demo & verification script
├── aptos_dr_explainability_kaggle.py   # Self-contained script ready for Kaggle GPU execution
├── outputs/                            # Model checkpoints, Grad-CAM overlays, confusion matrices
└── venv/                               # Isolated Python environment on Drive D
```

---

## 4. How to Run

### A. Quick Verification & Demo (Runs Instantly)
Validates training, backpropagation, Grad-CAM visualization, and clinical reports on your GPU:
```powershell
& "d:\New folder (2)\venv\Scripts\python.exe" "d:\New folder (2)\run_demo_and_verify.py"
```

### B. Train on Full APTOS Dataset Locally (Drive D)
If you have downloaded the APTOS dataset into `D:\aptos_dataset`:
```powershell
& "d:\New folder (2)\venv\Scripts\python.exe" "d:\New folder (2)\train_explainable_dr.py" --data_root "D:\aptos_dataset" --epochs 15 --batch_size 16
```

### C. Train on Kaggle (Free Cloud GPUs)
1. Open [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection).
2. Create a new notebook with the `aptos2019-blindness-detection` competition data attached.
3. Turn on GPU accelerator (Settings $\to$ Accelerator $\to$ GPU P100 or T4).
4. Copy and paste `aptos_dr_explainability_kaggle.py` into a code cell and run!

---

## 5. Clinical Metrics Tracked

1. **Quadratic Weighted Kappa (QWK)**: The official competition metric measuring agreement with clinical consensus.
2. **Referable DR ($\ge 2$) Sensitivity & Specificity**:
   - Sensitivity target: $>90\%$ (ensuring patients requiring clinical intervention are not missed).
   - Specificity target: $>85\%$ (preventing unnecessary specialist referrals).
3. **Attribution Accuracy**: Verifying that Grad-CAM heatmaps focus on genuine retinal lesions rather than camera artifacts.
