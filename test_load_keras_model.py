import os
import sys

os.environ["KERAS_BACKEND"] = "torch"
os.environ["KERAS_JIT_COMPILE"] = "0"

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

import numpy as np
import keras

print(f"Keras Version: {keras.__version__}")
print(f"Backend: {keras.backend.backend()}")
print(f"PyTorch Version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}")

model_file = r"d:\New folder (2)\models\best_dr_model_latest.keras_file.keras"

print("\n--- Loading model from zip file ---")
model = keras.models.load_model(model_file, compile=False)
print("Successfully loaded model!")
print("Model Input Shape:", model.input_shape)
print("Model Output Shape:", model.output_shape)

# Test forward pass with dummy fundus input
dummy_input = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)

# Test calling directly via torch tensor
print("\nTesting forward pass via PyTorch tensor...")
tensor_input = torch.from_numpy(dummy_input)
if torch.cuda.is_available():
    tensor_input = tensor_input.cuda()

with torch.no_grad():
    preds = model(tensor_input, training=False)
    if hasattr(preds, "cpu"):
        preds_np = preds.cpu().numpy()
    else:
        preds_np = np.array(preds)

print("Forward pass successful!")
print("Raw output tensor shape:", preds.shape)
print("Predicted probabilities:", preds_np)
print("Predicted DR Grade:", np.argmax(preds_np, axis=-1)[0])
print("\n[SUCCESS] FRIEND'S KERAS DR MODEL IS FULLY WORKING!")
