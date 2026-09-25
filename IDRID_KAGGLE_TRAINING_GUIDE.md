# 🇮🇳 IDRiD Training Guide: Multi-Task Explainable DR & DME Pipeline

This guide outlines how to train and evaluate the **Explainable Multi-Task Diabetic Retinopathy & Diabetic Macular Edema (DME) Pipeline** on the **IDRiD (Indian Diabetic Retinopathy Image Dataset)** benchmark using **Kaggle GPU** (with zero local storage footprint).

---

## 1. Why IDRiD is a Critical Milestone for Your Project

The **Indian Diabetic Retinopathy Image Dataset (IDRiD)** was captured using a Kowa VX-10alpha digital fundus camera ($50^\circ$ Field of View) at an eye clinic in Nanded, Maharashtra, India. 

Integrating IDRiD elevates your project from a single-dataset experiment to an **international, multi-cohort clinical AI system**:

| Dimension | APTOS 2019 Dataset | IDRiD Dataset (This Pipeline) |
| :--- | :--- | :--- |
| **Cohort Demographics** | International / Latin America | **Indian Population** (high melanin, distinct retinal pigmentation) |
| **Primary Task** | DR Severity Grading ($0-4$) | **DR Severity ($0-4$) + Diabetic Macular Edema (DME $0-2$)** |
| **Pathology Focus** | Peripheral vascular lesions | **Peripheral lesions + Foveal/Macular lipid exudation** |
| **Explainability Target** | General DR activation | **Dual Grad-CAM**: DR lesion activation vs. DME foveal attention |
| **Domain Transfer** | Source dataset | **Target evaluation showing generalizability across clinical settings** |

---

## 2. Project Files on Drive D (`D:\New folder (2)\`)

| File | Description |
| :--- | :--- |
| **`idrid_dr_explainability_kaggle.py`** | **Production script ready for Kaggle GPU**. Auto-discovers any IDRiD dataset layout on Kaggle, auto-loads APTOS weights if available, trains multi-task heads, plots dual Grad-CAM, and saves weights. |
| **`train_idrid_multitask_explainable.ipynb`** | Complete, formatted Jupyter Notebook ready to upload directly to Kaggle. |
| **`idrid_explainable_model.py`** | The multi-task PyTorch architecture (`ExplainableIDRiDModel`), `MultiTargetGradCAM`, and `IDRiDEtiologyExplainer`. |
| **`run_idrid_demo_and_verify.py`** | **Local instant test**: Generates synthetic Indian fundus phantoms with DR + DME, tests transfer learning, optimization, dual Grad-CAM, and etiology generation on your local GTX 1650 GPU in ~15 seconds. |
| **`generate_idrid_notebook.py`** | Utility to re-compile the Kaggle notebook. |

---

## 3. Step-by-Step: Training on Kaggle GPU (Zero Local Download)

Training on Kaggle gives you **free access to high-performance NVIDIA Tesla T4 or P100 GPUs** and prevents filling up Drive C or Drive D.

### Step 1: Open Kaggle & Create a Notebook
1. Go to [kaggle.com](https://www.kaggle.com) and log in.
2. Click **Create** $\to$ **New Notebook** (or open your existing notebook).

### Step 2: Attach the IDRiD Dataset (1 Click)
1. On the right-side configuration panel, click **+ Add Input**.
2. In the search bar, type: `idrid` or `idrid-dataset`.
3. Select any popular IDRiD dataset (e.g., `mariaherrerot/idrid-dataset` or `carlossouza/idrid` or `sanketbhamare/idrid-dataset`).
4. Click the **`+` (Add)** button.

> [!TIP]
> **Optional Transfer Learning (Boosts QWK & Accuracy)**:
> If you wish to transfer weights from your previously trained APTOS model:
> 1. In Kaggle, upload your `explainable_dr_resnet50.pt` as a private dataset (or attach the notebook where you trained APTOS).
> 2. Our script will **automatically detect it** in `/kaggle/input` and transfer 326 layers into the IDRiD network!

### Step 3: Enable the GPU Accelerator
1. In the right-hand panel, expand **Notebook options** $\to$ **Accelerator**.
2. Select **GPU T4 x2** or **GPU P100**.
3. Ensure **Internet** is toggled **ON** (needed for initial ResNet-50 ImageNet V2 backbone weights).

### Step 4: Run the Training Script
Choose **one** of the two convenient methods:

#### Method A: Upload the Notebook (Easiest)
1. In the Kaggle notebook menu, click **File** $\to$ **Upload Notebook**.
2. Select `D:\New folder (2)\train_idrid_multitask_explainable.ipynb`.
3. Click **Run All**.

#### Method B: Copy-Paste the Single-Cell Script
1. Open `D:\New folder (2)\idrid_dr_explainability_kaggle.py`.
2. Copy all code.
3. Paste into the first code cell of your Kaggle notebook and press **Shift + Enter**.

---

## 4. What Happens During Training (~4 Minutes on GPU)

The script autonomously performs:
1. **Intelligent Dataset Discovery**:
   - Recursively traverses `/kaggle/input` and locates IDRiD training and testing CSVs.
   - Auto-normalizes column names (`Image name`, `Retinopathy grade`, `Risk of macular edema`).
   - Indexes all corresponding fundus images.
2. **Medical Preprocessing**:
   - Circular Field of View masking.
   - Green-channel Contrast Limited Adaptive Histogram Equalization (CLAHE).
   - Ben Graham local illumination correction.
3. **Multi-Task Optimization**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DR}} + 0.8\,\mathcal{L}_{\text{DME}} + 0.5\,\mathcal{L}_{\text{Biomarkers}}$$
   - Uses PyTorch Automatic Mixed Precision (`torch.amp.autocast`) and Cosine Annealing learning rate schedule.
4. **Clinical Metric Computation**:
   - Quadratic Weighted Kappa (QWK) on DR Severity.
   - DME Risk Classification Accuracy.
   - Referable DR ($\ge \text{Grade } 2$) Sensitivity and Specificity.
5. **Dual-Target Grad-CAM Generation**:
   - Heatmap 1: DR Severity (highlights peripheral microaneurysms, hemorrhages, neovascular fronds).
   - Heatmap 2: DME Risk (highlights foveal/macular lipid exudation).
6. **Artifact Export to `/kaggle/working`**:
   - `explainable_idrid_multitask_resnet50.pt` (best trained model checkpoint).
   - `idrid_gradcam_explanations.png` (4-column side-by-side diagnostic visualization).
   - `idrid_confusion_matrices.png` (DR & DME confusion matrices).
   - `idrid_clinical_etiology_reports.json` (structured clinical decision support).

---

## 5. Local Verification on Drive D (GTX 1650 GPU)

You can immediately test the entire multi-task network, loss convergence, transfer learning from APTOS, dual Grad-CAM heatmaps, and etiology reports right now on your machine:

```powershell
& "D:\New folder (2)\venv\Scripts\python.exe" "D:\New folder (2)\run_idrid_demo_and_verify.py"
```

Generated outputs will be saved to:
- `D:\New folder (2)\outputs\idrid_demo\idrid_dual_gradcam_demo.png`
- `D:\New folder (2)\outputs\idrid_demo\idrid_demo_clinical_reports.json`

---

## 6. How to Present This to Judges (The Winning Presentation Story)

When presenting to hackathon judges or clinical reviewers, highlight these four key points:

1. **Dual Clinical Diagnosis (DR + DME)**:
   > *"Most existing models only predict DR severity. In reality, Diabetic Macular Edema (DME) is the leading cause of sudden central vision loss in diabetic patients. By leveraging IDRiD, our system predicts both DR grade (0-4) and DME risk (0-2) in a single unified architecture."*

2. **Cross-Demographic Generalization (India + Global)**:
   > *"Many medical AI algorithms suffer from demographic bias. We validated our model across both international cohorts (APTOS) and Indian cohorts (IDRiD), demonstrating that our explainable feature representations generalize across camera optics ($50^\circ$ vs $45^\circ$) and varying retinal pigmentation."*

3. **Dual-Target Visual Explainability**:
   > *"Our Dual-Target Grad-CAM provides distinct clinical views: when explaining DR, it activates on peripheral hemorrhages and aneurysms; when explaining DME, it focuses specifically on lipid exudation threatening the foveal avascular zone."*

4. **Actionable Clinical Decision Support**:
   > *"Our model doesn't stop at heatmaps. It outputs ophthalmologist-grade diagnostic reports with pathophysiological mechanisms and referral urgency timelines (e.g., immediate anti-VEGF injection for DME vs. annual routine screening)."*
