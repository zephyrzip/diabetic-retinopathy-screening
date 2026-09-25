import sys
import json
import os
from io import BytesIO
from urllib.request import urlopen
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = os.environ.get("DR_MODEL_PATH", os.path.join(os.path.dirname(__file__), "dr_model.onnx"))

def predict_retinopathy(image_input_path_or_url):
    if image_input_path_or_url.startswith(("http://", "https://")):
        with urlopen(image_input_path_or_url, timeout=30) as response:
            image_data = response.read(10 * 1024 * 1024 + 1)
        if len(image_data) > 10 * 1024 * 1024:
            raise ValueError("Retinal image exceeds the 10 MB processing limit.")
        img = Image.open(BytesIO(image_data)).convert("RGB")
    else:
        img = Image.open(image_input_path_or_url).convert("RGB")

    img = img.resize((224, 224))
    img_data = np.array(img, dtype=np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    input_tensor = np.expand_dims(img_data, axis=0)

    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"ONNX model not found at {MODEL_PATH}")

    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: input_tensor})[0]
    grade = int(np.argmax(raw_output))
    confidence = float(np.max(raw_output))

    return {"grade": grade, "confidence": round(confidence, 2)}

if __name__ == "__main__":
    if len(sys.argv) > 2:
        image_target = sys.argv[2]
        try:
            res = predict_retinopathy(image_target)
            print(json.dumps(res))
        except Exception as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            sys.exit(1)