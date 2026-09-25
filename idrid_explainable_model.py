"""
=============================================================================
IDRiD (Indian Diabetic Retinopathy Image Dataset) Multi-Task XAI Architecture
=============================================================================
Architecture:
- Shared Backbone: ResNet-50 Feature Extractor
- Multi-Task Output Heads:
    1. DR Grade Head: Retinopathy Severity (0=No DR, 1=Mild, 2=Moderate, 3=Severe, 4=PDR)
    2. DME Risk Head: Diabetic Macular Edema (0=No DME, 1=Mild/Moderate, 2=Severe CSME)
    3. Biomarker Head: Hallmark Lesion Attribution (Microaneurysms, Hemorrhages,
                       Hard Exudates, Cotton Wool Spots, Neovascularization)
- Dual-Target Grad-CAM:
    - DR Severity Grad-CAM: highlights peripheral microvascular lesions & neovascular tufts
    - DME Risk Grad-CAM: highlights macular/foveal lipid exudates and central thickening
- Clinical Etiology Diagnostic Generator:
    - Synthesizes comprehensive ophthalmology reports explaining WHY both DR and DME occur
"""

import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Hallmark lesion biomarkers
BIOMARKER_NAMES = [
    "Microaneurysms",         # Capillary dilation due to pericyte apoptosis (Earliest sign: Grade 1)
    "Hemorrhages",            # Microvascular rupture into intraretinal layers (Grade 2/3)
    "Hard Exudates",          # Lipid & lipoprotein leakage, central to DME pathogenesis (Grade 2/3 + DME)
    "Cotton Wool Spots",      # Retinal nerve fiber layer micro-infarcts / ischemia (Grade 3)
    "Neovascularization"      # Proliferation of fragile abnormal vessels driven by VEGF (Grade 4 PDR)
]

# Clinical Diabetic Retinopathy International Severity Scale (0 to 4)
DR_GRADE_NAMES = [
    "0 - No Apparent DR",
    "1 - Mild Non-Proliferative DR",
    "2 - Moderate Non-Proliferative DR",
    "3 - Severe Non-Proliferative DR",
    "4 - Proliferative Diabetic Retinopathy (PDR)"
]

# Diabetic Macular Edema (DME) International Clinical Classification (0 to 2)
DME_RISK_NAMES = [
    "0 - No Apparent DME (No hard exudates near fovea)",
    "1 - Mild / Moderate DME (Hard exudates present within 1 disc diameter of fovea)",
    "2 - Severe DME / Clinically Significant (Exudates/thickening involving center of fovea)"
]

# Prior mapping: Given DR grade and DME grade, expected biomarker profiles
def get_biomarker_prior(dr_grade: int, dme_risk: int = 0):
    """
    Returns clinical biomarker likelihood prior [MA, HE, EX, SE, NV].
    Incorporates both DR severity and DME status.
    """
    priors = {
        0: [0.0, 0.0, 0.0, 0.0, 0.0],
        1: [1.0, 0.0, 0.0, 0.0, 0.0],
        2: [1.0, 1.0, 1.0, 0.0, 0.0],
        3: [1.0, 1.0, 1.0, 1.0, 0.0],
        4: [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    vec = list(priors.get(dr_grade, [0.0]*5))
    # If DME is present (risk 1 or 2), Hard Exudates (index 2) are guaranteed present
    if dme_risk >= 1:
        vec[2] = 1.0
    return vec


class ExplainableIDRiDModel(nn.Module):
    """
    Multi-task deep neural network for joint DR grading, DME risk assessment,
    and pathological biomarker attribution.
    """
    def __init__(self, num_dr_classes=5, num_dme_classes=3, num_biomarkers=5, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        
        # Shared Feature Extractor
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
        
        # Head 1: DR Grade Classification (0 to 4)
        self.dr_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_dr_classes)
        )
        
        # Head 2: DME Risk Classification (0 to 2) - unique to IDRiD
        self.dme_head = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_dme_classes)
        )
        
        # Head 3: Hallmark Biomarkers Multi-label Attribution
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
        
        dr_logits = self.dr_head(flat)
        dme_logits = self.dme_head(flat)
        biomarker_logits = self.biomarker_head(flat)
        return dr_logits, dme_logits, biomarker_logits

    def load_aptos_transfer_weights(self, aptos_weights_path: str):
        """
        Transfers learned representations from an APTOS model checkpoint.
        Loads backbone, layer1-4, and DR grade head.
        """
        if not os.path.exists(aptos_weights_path):
            print(f"[Transfer Learning] Checkpoint not found at {aptos_weights_path}. Training from ImageNet weights.")
            return False
        
        print(f"[Transfer Learning] Loading pre-trained APTOS weights from {aptos_weights_path}...")
        checkpoint = torch.load(aptos_weights_path, map_location="cpu")
        state_dict = checkpoint if isinstance(checkpoint, dict) and "state_dict" not in checkpoint else checkpoint.get("state_dict", checkpoint)
        
        model_dict = self.state_dict()
        transferred = 0
        for k, v in state_dict.items():
            # Clean module. prefix if trained under DataParallel
            clean_k = k.replace("module.", "")
            if clean_k.startswith("grade_head."):
                # Map old APTOS grade_head to new dr_head
                new_k = clean_k.replace("grade_head.", "dr_head.")
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    model_dict[new_k] = v
                    transferred += 1
            elif clean_k in model_dict and model_dict[clean_k].shape == v.shape:
                model_dict[clean_k] = v
                transferred += 1
                
        self.load_state_dict(model_dict)
        print(f"[Transfer Learning] Successfully loaded {transferred} layers from APTOS checkpoint!")
        return True


class MultiTargetGradCAM:
    """
    Grad-CAM implementation supporting both DR severity targets and DME risk targets.
    Enables visual localization of peripheral lesions (DR) vs central macular exudates (DME).
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.layer4[-1]
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_type="dr", target_class=None):
        """
        Generates visual Grad-CAM heatmap for either 'dr' or 'dme'.
        target_type: 'dr' or 'dme'
        target_class: index of class. If None, uses model's predicted class.
        """
        self.model.eval()
        self.model.zero_grad()
        
        dr_logits, dme_logits, _ = self.model(input_tensor)
        logits = dr_logits if target_type == "dr" else dme_logits
        
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()
            
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        # Pooled gradients across channels
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class

    def overlay(self, cam, base_img_rgb, alpha=0.45, colormap=cv2.COLORMAP_JET):
        """Overlays heatmap on RGB fundus image."""
        cam_uint8 = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        if base_img_rgb.shape[:2] != cam.shape[:2]:
            base_img_rgb = cv2.resize(base_img_rgb, (cam.shape[1], cam.shape[0]))
            
        overlay_img = cv2.addWeighted(heatmap_rgb, alpha, base_img_rgb, 1 - alpha, 0)
        return overlay_img


class IDRiDEtiologyExplainer:
    """
    Ophthalmology Clinical Reasoning Engine for the Indian Diabetic Retinopathy Cohort.
    Synthesizes clinical reports explaining why DR and DME occur and recommends clinical interventions.
    """
    def __init__(self, biomarker_names=BIOMARKER_NAMES):
        self.biomarker_names = biomarker_names

    def explain(self, dr_grade, dme_risk, dr_probs, dme_probs, biomarker_probs, image_id="Patient_Fundus"):
        dr_name = DR_GRADE_NAMES[dr_grade]
        dme_name = DME_RISK_NAMES[dme_risk]
        
        active_biomarkers = []
        for name, prob in zip(self.biomarker_names, biomarker_probs):
            if prob >= 0.40:
                active_biomarkers.append({"biomarker": name, "confidence": float(prob)})

        # Biological rationale for DR
        dr_mechanisms = {
            0: "Normal retinal vasculature without microvascular lesions or capillary dropout.",
            1: "Pericyte loss and localized capillary basement membrane degradation leading to microaneurysm outpouching.",
            2: "Capillary breakdown with intraretinal hemorrhages and lipid exudation. Significant microvascular leakage.",
            3: "Extensive capillary non-perfusion causing nerve fiber layer ischemia (cotton wool spots) and venous beading.",
            4: "Severe retinal hypoxia triggering high VEGF secretion, driving fragile neovascular vessel proliferation with high risk of vitreous hemorrhage."
        }

        # Biological rationale for DME
        dme_mechanisms = {
            0: "Intact inner blood-retinal barrier at the foveal avascular zone; no significant macular lipid leakage.",
            1: "Perifoveal capillary hyperpermeability depositing hard exudates within 1 disc diameter of the foveal center.",
            2: "Clinically Significant Macular Edema (CSME): direct foveal lipid deposition or central thickening threatening central visual acuity."
        }

        # Actionable clinical management recommendations
        if dr_grade == 4 or dme_risk == 2:
            referral = "URGENT (Within 1-2 weeks)"
            management = "Immediate vitreoretinal specialist referral. Candidate for Anti-VEGF intravitreal injections (Aflibercept/Ranibizumab) and/or Panretinal Photocoagulation (PRP)."
        elif dr_grade == 3 or dme_risk == 1:
            referral = "PRIORITY (Within 4-6 weeks)"
            management = "Ophthalmology referral for Optical Coherence Tomography (OCT) macular thickness evaluation and close surveillance."
        elif dr_grade == 2:
            referral = "ROUTINE (Within 3-6 months)"
            management = "Strict glycemic and blood pressure optimization (HbA1c target < 7.0%). Repeat dilated fundus exam in 6 months."
        elif dr_grade == 1:
            referral = "MONITORING (Annual)"
            management = "Annual dilated fundus screening. Counsel patient on diabetic lifestyle and glycemic control."
        else:
            referral = "NORMAL (Annual Screen)"
            management = "Annual routine diabetic retinopathy screening."

        report = {
            "image_id": image_id,
            "cohort": "IDRiD (Indian Diabetic Retinopathy Cohort - 50° FOV)",
            "primary_diagnosis": {
                "dr_grade": int(dr_grade),
                "dr_stage": dr_name,
                "dr_confidence": float(dr_probs[dr_grade]),
                "is_referable_dr": bool(dr_grade >= 2),
                "dme_risk": int(dme_risk),
                "dme_stage": dme_name,
                "dme_confidence": float(dme_probs[dme_risk]),
                "is_high_risk_dme": bool(dme_risk >= 1)
            },
            "biomarker_findings": active_biomarkers,
            "pathological_etiology": {
                "dr_pathogenesis": dr_mechanisms[dr_grade],
                "dme_pathogenesis": dme_mechanisms[dme_risk]
            },
            "clinical_decision_support": {
                "urgency": referral,
                "action_plan": management
            }
        }
        return report
