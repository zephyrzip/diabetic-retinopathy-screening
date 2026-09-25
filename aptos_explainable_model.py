"""
Explainable Diabetic Retinopathy (XAI) & Biomarker Etiology Model
Architecture:
- Backbone: ResNet-50 / EfficientNet feature extractor
- Dual Output Heads:
    1. Grade Head: Severity Classification (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative DR)
    2. Biomarker Head: Hallmark Lesion Detection (Microaneurysms, Hemorrhages, Hard Exudates, Cotton Wool Spots, Neovascularization)
- Grad-CAM Explainability: Generates spatial heatmaps showing exactly where lesions trigger the diagnosis
- Clinical Etiology Engine: Generates human-readable clinical reports explaining "why DR occurs" for each fundus image
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Hallmark lesion biomarkers that explain "Why DR occurs"
BIOMARKER_NAMES = [
    "Microaneurysms",         # Capillary wall outpouching due to pericyte loss (earliest sign, Grade 1)
    "Hemorrhages",            # Microvascular rupture into retinal layers (Grade 2/3)
    "Hard Exudates",          # Lipid/lipoprotein leakage from hyperpermeable capillaries (Grade 2/3)
    "Cotton Wool Spots",      # Micro-infarcts / nerve fiber ischemia from capillary occlusion (Grade 3)
    "Neovascularization"      # Proliferation of fragile new vessels driven by VEGF (Grade 4 PDR)
]

GRADE_NAMES = [
    "0 - No DR",
    "1 - Mild Non-Proliferative DR",
    "2 - Moderate Non-Proliferative DR",
    "3 - Severe Non-Proliferative DR",
    "4 - Proliferative Diabetic Retinopathy (PDR)"
]


class ExplainableDRModel(nn.Module):
    """
    Multi-task deep neural network for DR grading and pathological biomarker attribution.
    """
    def __init__(self, num_classes=5, num_biomarkers=5, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        
        # Feature extractor layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.avgpool = backbone.avgpool
        in_features = backbone.fc.in_features  # 2048
        
        # Dual Heads:
        # Head 1: Grade classification (0 to 4)
        self.grade_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
        
        # Head 2: Biomarker presence / attribution logits
        self.biomarker_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Linear(256, num_biomarkers)
        )

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        feat_map = self.forward_features(x)
        pooled = self.avgpool(feat_map)
        flat = torch.flatten(pooled, 1)
        
        grade_logits = self.grade_head(flat)
        biomarker_logits = self.biomarker_head(flat)
        return grade_logits, biomarker_logits


class GradCAM:
    """
    Grad-CAM implementation to inspect the spatial decision rationale.
    Hooks into the deepest conv block (model.layer4[-1]).
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.layer4[-1]
        self.activations = None
        self.gradients = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        grade_logits, biomarker_logits = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = grade_logits.argmax(dim=1).item()
            
        self.model.zero_grad()
        score = grade_logits[0, class_idx]
        score.backward()
        
        # Channel-wise mean of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        cam_resized = cv2.resize(cam, (w, h))
        return cam_resized, class_idx, grade_logits, biomarker_logits


class ClinicalEtiologyExplainer:
    """
    Translates model predictions, biomarker evidence, and visual heatmaps into an
    ophthalmology-grade explanation of WHY Diabetic Retinopathy occurred in the eye.
    """
    @staticmethod
    def generate_report(grade_idx, grade_probs, biomarker_probs, attention_stats=None):
        grade_idx = int(grade_idx)
        grade_name = GRADE_NAMES[grade_idx]
        conf = float(grade_probs[grade_idx]) * 100.0
        
        report = {
            "predicted_grade": int(grade_idx),
            "grade_label": str(grade_name),
            "confidence_percent": round(float(conf), 2),
            "referable_dr": bool(grade_idx >= 2),
            "biomarker_analysis": {},
            "pathology_etiology": "",
            "clinical_recommendation": ""
        }
        
        # Biomarker breakdown
        for name, prob in zip(BIOMARKER_NAMES, biomarker_probs):
            detected = bool(prob >= 0.50)
            report["biomarker_analysis"][name] = {
                "detected": detected,
                "probability": round(float(prob), 3),
                "severity": "High" if prob >= 0.75 else ("Moderate" if prob >= 0.50 else "Negligible")
            }
            
        # Clinical Etiology (Answering "Why DR Occurred")
        if grade_idx == 0:
            report["pathology_etiology"] = (
                "Normal retinal microvasculature. No detectable capillary wall dilation (microaneurysms) "
                "or blood-retinal barrier breakdown. Capillary pericytes and basement membranes remain intact."
            )
            report["clinical_recommendation"] = "Routine annual diabetic retinal screening."
            
        elif grade_idx == 1:
            report["pathology_etiology"] = (
                "Early microvascular injury triggered by chronic hyperglycemia. Hyperglycemia-induced apoptosis "
                "of capillary pericytes weakens vessel walls, forming focal outpouchings (Microaneurysms). "
                "These are the earliest visible morphological hallmarks of diabetic retinopathy."
            )
            report["clinical_recommendation"] = "Glycemic and blood pressure control. Re-evaluation in 6-12 months."
            
        elif grade_idx == 2:
            report["pathology_etiology"] = (
                "Moderate non-proliferative disease with active blood-retinal barrier compromise. "
                "Vascular wall fragility has progressed to intraretinal dot-and-blot hemorrhages. "
                "Increased capillary permeability permits serum lipoproteins and lipid deposits to accumulate "
                "as Hard Exudates, representing potential risk for Diabetic Macular Edema (DME)."
            )
            report["clinical_recommendation"] = (
                "REFERABLE DR. Prompt referral to ophthalmologist/retinal specialist. "
                "Assess for macular edema via OCT."
            )
            
        elif grade_idx == 3:
            report["pathology_etiology"] = (
                "Severe microvascular ischemia and capillary non-perfusion. Arteriolar precapillary occlusions "
                "cause micro-infarctions of the retinal nerve fiber layer (Cotton Wool Spots / Soft Exudates). "
                "Extensive intraretinal microvascular abnormalities (IRMA) and widespread hemorrhages signify "
                "profound retinal hypoxia approaching the threshold for neovascularization."
            )
            report["clinical_recommendation"] = (
                "URGENT REFERABLE DR. High risk of conversion to proliferative retinopathy within 1 year. "
                "Close ophthalmological monitoring (every 2-4 months) and consider preventive laser therapy."
            )
            
        else: # Grade 4
            report["pathology_etiology"] = (
                "Proliferative Diabetic Retinopathy (PDR). Severe, sustained retinal hypoxia has triggered massive "
                "upregulation and secretion of Vascular Endothelial Growth Factor (VEGF). This induces active "
                "Neovascularization—the formation of highly fragile, aberrant new blood vessels on the optic disc (NVD) "
                "or retina (NVE). These vessels are prone to catastrophic vitreous hemorrhage and fibrovascular contraction, "
                "which can lead to tractional retinal detachment and severe, irreversible vision loss."
            )
            report["clinical_recommendation"] = (
                "IMMEDIATE SURGICAL / RETINAL REFERRAL. Urgent intervention required: Anti-VEGF intravitreal injections, "
                "Panretinal Photocoagulation (PRP), or vitrectomy to prevent permanent blindness."
            )
            
        return report
