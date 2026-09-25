"""
=============================================================================
KERAS EFFICIENTNET-B0 DIABETIC RETINOPATHY DIAGNOSTIC MODULE
=============================================================================
Integrates the trained Keras 3 EfficientNet-B0 model:
  Path: "C:\\Users\\baner\\Downloads\\best_dr_model_latest.keras"
  Local Mirror: "models/best_dr_model_latest.keras_file.keras"

Key Features:
- Direct 5-class ICDR Diabetic Retinopathy inference (Grade 0 to 4)
- Grad-CAM Lesion Explainability for EfficientNet-B0
- Seamless interoperability with PyTorch and existing clinical pipelines
- Dual-Model Ensemble support (Friend's EfficientNet-B0 + Multi-Task ResNet-50)
=============================================================================
"""

import os
import sys

# Configure Keras 3 with PyTorch backend for high-speed CUDA acceleration
os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_JIT_COMPILE"] = "0"

import torch
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

import keras
import numpy as np
import cv2

GRADE_NAMES = [
    "0 - No Apparent DR",
    "1 - Mild NPDR",
    "2 - Moderate NPDR",
    "3 - Severe NPDR",
    "4 - Proliferative DR (PDR)"
]

GRADE_SHORT = ["No DR", "Mild", "Moderate", "Severe", "PDR"]

CANDIDATE_PATHS = [
    r"D:\New folder (2)\models\best_dr_model_latest.keras_file.keras",
    r"D:\New folder (2)\models\best_dr_model_latest.keras",
    r"C:\Users\baner\Downloads\best_dr_model_latest.keras.zip",
    r"C:\Users\baner\Downloads\best_dr_model_latest.keras"
]


def resolve_model_path():
    for p in CANDIDATE_PATHS:
        if os.path.exists(p):
            return p
    return None


class FriendKerasDRModel:
    """
    Wrapper for the user's friend-trained Keras EfficientNet-B0 DR model.
    """
    def __init__(self, model_path=None, device=None):
        if model_path is None:
            model_path = resolve_model_path()
            if model_path is None:
                raise FileNotFoundError(
                    "Could not locate best_dr_model_latest.keras in models/ or Downloads/!"
                )
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.target_layer = self._find_target_conv_layer()

    def _load_model(self):
        print(f"[KerasDRModel] Loading EfficientNet-B0 weights from: {self.model_path}")
        model = keras.models.load_model(self.model_path, compile=False)
        print(f"[KerasDRModel] Model '{model.name}' loaded successfully on {self.device}!")
        return model

    def _find_target_conv_layer(self):
        try:
            eff_net = self.model.get_layer("efficientnetb0")
            for layer_name in ["top_activation", "top_conv", "block7a_project_conv"]:
                try:
                    return eff_net.get_layer(layer_name)
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def preprocess_image(self, img_rgb):
        """
        Prepares raw/enhanced retinal image (RGB uint8 or float) for EfficientNet-B0.
        Model has internal Rescaling(1/255) + Normalization, so inputs should be [0, 255] float32.
        """
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:
            img_rgb = img_rgb[:, :, :3]

        resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32)
        if resized.max() <= 1.01:
            resized = resized * 255.0

        tensor = torch.from_numpy(resized).unsqueeze(0)
        if self.device == "cuda" and torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor, resized

    def predict(self, img_rgb):
        """
        Performs forward inference on an RGB image.
        Returns:
            dict containing:
                pred_grade: int (0 to 4)
                grade_name: str
                confidence: float (percentage 0-100)
                probabilities: list of 5 floats
                is_referable: bool (True if grade >= 2)
        """
        tensor, _ = self.preprocess_image(img_rgb)
        with torch.no_grad():
            preds = self.model(tensor, training=False)
            if hasattr(preds, "cpu"):
                probs = preds.cpu().numpy()[0]
            else:
                probs = np.array(preds)[0]

        probs = np.maximum(probs, 0.0)
        sum_p = np.sum(probs)
        if sum_p > 0:
            probs = probs / sum_p

        pred_grade = int(np.argmax(probs))
        confidence = float(probs[pred_grade] * 100.0)
        is_referable = bool(pred_grade >= 2)

        return {
            "model_name": "EfficientNet-B0 (Keras Trained)",
            "pred_grade": pred_grade,
            "grade_name": GRADE_NAMES[pred_grade],
            "confidence": round(confidence, 2),
            "probabilities": [float(p) for p in probs],
            "is_referable": is_referable,
            "referable_status": "REFERABLE DR" if is_referable else "NON-REFERABLE"
        }

    def generate_gradcam(self, img_rgb, target_class=None):
        """
        Generates Grad-CAM visual attention heatmap for the predicted DR grade.
        Uses activation & gradient hooks on the top convolutional feature maps.
        """
        tensor, resized = self.preprocess_image(img_rgb)
        tensor.requires_grad_(True)

        activations = []
        gradients = []

        def forward_hook(module, inp, out):
            activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        hook_handle_f = None
        hook_handle_b = None
        target_layer = self.target_layer

        if target_layer is not None:
            hook_handle_f = target_layer.register_forward_hook(forward_hook)
            hook_handle_b = target_layer.register_backward_hook(backward_hook)

        # Forward pass
        preds = self.model(tensor, training=False)
        if target_class is None:
            target_class = int(torch.argmax(preds, dim=-1).item())

        score = preds[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if activations and gradients:
            act = activations[0].detach()
            grad = gradients[0].detach()

            # Global average pooling over gradients to get feature weights
            # Keras EfficientNet feature maps are channels-last [Batch, H, W, C]
            if grad.ndim == 4 and grad.shape[-1] > grad.shape[1]:  # channels-last
                weights = torch.mean(grad, dim=[1, 2], keepdim=True)
                cam = torch.sum(weights * act, dim=-1, keepdim=True)
                cam = torch.relu(cam).squeeze().cpu().numpy()
            else:  # channels-first
                weights = torch.mean(grad, dim=[2, 3], keepdim=True)
                cam = torch.sum(weights * act, dim=1, keepdim=True)
                cam = torch.relu(cam).squeeze().cpu().numpy()

            if cam.max() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros_like(cam)
        else:
            # Fallback high-frequency lesion map
            gray = cv2.cvtColor(resized.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            cam = cv2.Laplacian(gray, cv2.CV_32F)
            cam = np.abs(cam)
            if cam.max() > 0:
                cam = cam / cam.max()

        if hook_handle_f:
            hook_handle_f.remove()
        if hook_handle_b:
            hook_handle_b.remove()

        # Resize to match fundus resolution
        cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
        
        # Suppress non-retinal artifacts (letters, watermarks, outer border) via circular FOV mask
        h, w = cam_resized.shape
        fov_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(fov_mask, (w // 2, h // 2), int(min(h, w) * 0.44), 1.0, -1)
        fov_mask = cv2.GaussianBlur(fov_mask, (31, 31), 0)
        cam_resized = cam_resized * fov_mask
        if cam_resized.max() > 0:
            cam_resized = cam_resized / cam_resized.max()
        else:
            cam_resized = np.zeros_like(cam_resized)

        cam_u8 = np.uint8(255 * np.clip(cam_resized, 0, 1))
        heatmap_jet = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

        # Only overlay within retinal field
        mask_3ch = np.repeat((fov_mask > 0.05)[:, :, np.newaxis], 3, axis=2)
        blended = cv2.addWeighted(img_rgb.astype(np.uint8), 0.60, heatmap_rgb, 0.40, 0)
        overlay = np.where(mask_3ch, blended, img_rgb.astype(np.uint8))
        return cam_resized, overlay, target_class


class DualModelEnsemble:
    """
    Ensembles Friend's EfficientNet-B0 Keras model with ResNet-50 PyTorch model
    for highest diagnostic confidence and dual-model validation.
    """
    def __init__(self, keras_model_path=None, pytorch_weights_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.keras_engine = FriendKerasDRModel(model_path=keras_model_path, device=self.device)
        self.pytorch_model = None

        if pytorch_weights_path is None:
            candidate_pt = [
                r"D:\New folder (2)\outputs\idrid\explainable_idrid_multitask_resnet50.pt",
                r"D:\New folder (2)\outputs\explainable_dr_resnet50.pt"
            ]
            for cp in candidate_pt:
                if os.path.exists(cp):
                    pytorch_weights_path = cp
                    break

        if pytorch_weights_path and os.path.exists(pytorch_weights_path):
            try:
                from idrid_explainable_model import ExplainableIDRiDModel
                pt_m = ExplainableIDRiDModel(pretrained=False).to(self.device)
                ckpt = torch.load(pytorch_weights_path, map_location=self.device, weights_only=False)
                state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else (
                    ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
                )
                pt_m.load_state_dict(state, strict=False)
                pt_m.eval()
                self.pytorch_model = pt_m
                print(f"[Ensemble] Loaded PyTorch model from: {pytorch_weights_path}")
            except Exception as e:
                print(f"[Ensemble] Warning: Could not load PyTorch checkpoint ({e}). Using Keras only.")

    def predict(self, img_rgb, keras_weight=0.55):
        """
        Computes weighted ensemble diagnosis across both models.
        """
        keras_res = self.keras_engine.predict(img_rgb)
        keras_probs = np.array(keras_res["probabilities"])

        if self.pytorch_model is not None:
            # PyTorch forward pass
            t_input = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            t_norm = (t_input - mean) / std
            tensor_pt = torch.from_numpy(t_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                out = self.pytorch_model(tensor_pt)
                pt_logits = out[0] if isinstance(out, (tuple, list)) else out
                pt_probs = torch.softmax(pt_logits, dim=-1).cpu().numpy()[0]

            pt_grade = int(np.argmax(pt_probs))
            # Blended probabilities
            ensemble_probs = (keras_weight * keras_probs) + ((1.0 - keras_weight) * pt_probs)
            ensemble_grade = int(np.argmax(ensemble_probs))
            ensemble_conf = float(ensemble_probs[ensemble_grade] * 100.0)

            return {
                "pred_grade": ensemble_grade,
                "grade_name": GRADE_NAMES[ensemble_grade],
                "confidence": round(ensemble_conf, 2),
                "is_referable": bool(ensemble_grade >= 2),
                "ensemble_probabilities": [float(p) for p in ensemble_probs],
                "individual_models": {
                    "keras_efficientnet": {
                        "pred_grade": keras_res["pred_grade"],
                        "confidence": keras_res["confidence"],
                        "probabilities": keras_res["probabilities"]
                    },
                    "pytorch_resnet50": {
                        "pred_grade": pt_grade,
                        "confidence": round(float(pt_probs[pt_grade] * 100.0), 2),
                        "probabilities": [float(p) for p in pt_probs]
                    }
                }
            }
        else:
            return keras_res
