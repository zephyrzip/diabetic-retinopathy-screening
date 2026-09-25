"""
Generates the Jupyter Notebook for IDRiD Multi-Task Training on Kaggle.
"""

import json
import os

def create_idrid_notebook():
    nb_path = "train_idrid_multitask_explainable.ipynb"
    
    with open("idrid_dr_explainability_kaggle.py", "r", encoding="utf-8") as f:
        script_code = f.read()

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🇮🇳 IDRiD: Multi-Task Explainable Diabetic Retinopathy & DME Pipeline\n",
                "### Indian Diabetic Retinopathy Image Dataset (IEEE Dataport / ISBI 2018 Benchmark)\n",
                "\n",
                "This notebook trains a **Multi-Task Clinical Explainable Deep Learning System** on the Indian Diabetic Retinopathy Image Dataset (IDRiD).\n",
                "\n",
                "---\n",
                "\n",
                "## 🌟 Clinical Innovation & Multi-Task Scope\n",
                "1. **Diabetic Retinopathy (DR) Severity Grading ($0-4$)**: International Clinical Diabetic Retinopathy scale.\n",
                "2. **Diabetic Macular Edema (DME) Risk Grading ($0-2$)**: Assessing exudation proximity to the foveal avascular zone.\n",
                "3. **5 Hallmark Pathological Lesions**: Microaneurysms, Hemorrhages, Hard Exudates, Cotton Wool Spots, Neovascularization.\n",
                "4. **Dual-Target Grad-CAM Visual Heatmaps**: Contrastive attention maps comparing peripheral vascular lesions (DR) vs central macular lipid deposits (DME).\n",
                "5. **Ophthalmology Clinical Etiology Engine**: Translates neural activations into diagnostic reports explaining *WHY* both DR and DME occur."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚙️ Hardware Check & Setup\n",
                "Ensure your Kaggle notebook has GPU turned on: `Settings -> Accelerator -> GPU T4 x2 or P100`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "print(f\"PyTorch Version: {torch.__version__}\")\n",
                "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"GPU Name: {torch.cuda.get_device_name(0)}\")\n",
                "else:\n",
                "    print(\"WARNING: GPU is not enabled! Go to Settings -> Accelerator -> GPU\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🚀 Multi-Task Training, Dual Grad-CAM & Clinical Diagnostics\n",
                "The code cell below executes the full pipeline with automatic dataset detection, transfer learning (if APTOS weights exist), mixed-precision optimization, and artifact export."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [script_code]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 🖼️ Display Dual Grad-CAM Visual Heatmaps\n",
                "Let's visualize the generated side-by-side explanations showing DR and DME lesions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from IPython.display import Image, display\n",
                "\n",
                "gradcam_file = \"/kaggle/working/idrid_gradcam_explanations.png\"\n",
                "if os.path.exists(gradcam_file):\n",
                "    display(Image(filename=gradcam_file))\n",
                "else:\n",
                "    print(\"Grad-CAM file not found yet. Run the training cell above first.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 📊 Display Confusion Matrices\n",
                "Review classification performance for both DR Severity and DME Risk."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "cm_file = \"/kaggle/working/idrid_confusion_matrices.png\"\n",
                "if os.path.exists(cm_file):\n",
                "    display(Image(filename=cm_file))\n",
                "else:\n",
                "    print(\"Confusion matrices file not found yet.\")"
            ]
        }
    ]

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"[SUCCESS] Generated notebook at: {os.path.abspath(nb_path)}")

if __name__ == "__main__":
    create_idrid_notebook()
