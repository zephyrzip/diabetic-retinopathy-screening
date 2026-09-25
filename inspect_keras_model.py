import os
import sys

model_path = r"C:\Users\baner\Downloads\best_dr_model_latest.keras"

print(f"Checking model at: {model_path}")
if not os.path.exists(model_path):
    print("File does not exist!")
    sys.exit(1)

size_mb = os.path.getsize(model_path) / (1024 * 1024)
print(f"File size: {size_mb:.2f} MB")

try:
    import tensorflow as tf
    import keras
    print("TensorFlow:", tf.__version__)
    print("Keras:", keras.__version__)
    
    # Load model
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print("keras.models.load_model failed, trying tf.keras.models.load_model:", e)
        model = tf.keras.models.load_model(model_path, compile=False)
        
    print("\n--- MODEL DETAILS ---")
    print("Model type:", type(model))
    print("Input shape:", getattr(model, 'input_shape', None))
    print("Inputs:", model.inputs)
    print("Output shape:", getattr(model, 'output_shape', None))
    print("Outputs:", model.outputs)
    print("\nLayer names:")
    for i, layer in enumerate(model.layers[:15]):
        print(f"  [{i}] {layer.name}: {layer.__class__.__name__}")
    if len(model.layers) > 15:
        print(f"  ... ({len(model.layers)} total layers) ...")
        for i, layer in enumerate(model.layers[-10:], len(model.layers)-10):
            print(f"  [{i}] {layer.name}: {layer.__class__.__name__}")
            
except Exception as err:
    print("Error inspecting model:", err)
    import traceback
    traceback.print_exc()
