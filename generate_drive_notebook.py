"""
Generates the Jupyter Notebook for DRIVE Vessel Segmentation Training on Kaggle.
"""

import json
import os

def create_drive_notebook():
    nb_path = "train_drive_vessel_unet.ipynb"
    
    with open("drive_vessel_segmentation_kaggle.py", "r", encoding="utf-8") as f:
        script_code = f.read()

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 👁️ DRIVE: Retinal Blood Vessel Segmentation & Vascular Biomarker Engine\n",
                "### Digital Retinal Images for Vessel Extraction (Grand Challenge / IEEE TMI)\n",
                "\n",
                "This notebook trains a **High-Precision U-Net Architecture** for automated retinal blood vessel extraction and vascular caliber quantification.\n",
                "\n",
                "---\n",
                "\n",
                "## 🌟 Clinical Purpose in Diabetic Retinopathy\n",
                "1. **Retinal Vessel Perfusion Density (VPD)**: Quantifies capillary non-perfusion and ischemic zones.\n",
                "2. **Vessel Tortuosity Index (VTI)**: Tracks vascular endothelial strain and vascular remodeling.\n",
                "3. **Neovascularization (NV) Alert**: Automatically flags chaotic vascular loops and fronds to detect Grade 4 Proliferative DR (PDR).\n",
                "4. **Export to ONNX for MATLAB**: Integrates into the district-level telemedicine screening pipeline."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## ⚙️ Hardware Check\n",
                "Select `GPU T4 x2 or P100` under Notebook Settings."
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
                "## 🚀 Complete Training, Evaluation & Vascular Biomarker Extraction Code"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                script_code
            ]
        }
    ]

    nb_data = {
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
        "nbformat_minor": 5
    }

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=2)

    print(f"[OK] Generated {nb_path} successfully!")


if __name__ == "__main__":
    create_drive_notebook()
