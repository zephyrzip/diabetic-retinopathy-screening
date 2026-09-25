"""
Quick test client for RetinaScan AI FastAPI server.
Usage:
    python test_api_client.py --image "path/to/retina.png"
"""

import sys
import argparse
import urllib.request
import json
import mimetypes
import os

API_URL = "http://localhost:8000/api/v1/predict"


def test_api(image_path: str):
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return

    print(f"\n[Client] Sending '{image_path}' to {API_URL}...")

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    filename = os.path.basename(image_path)
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/png"

    with open(image_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("\n================== API RESPONSE ==================")
            print("Status:      ", data.get("status"))
            print("Latency:     ", data.get("latency_sec"), "sec")
            print("Device:      ", data.get("device"))
            diag = data.get("diagnosis", {})
            print("Diagnosis:   ", diag.get("dr_label"), f"(Confidence: {diag.get('dr_confidence')*100:.1f}%)")
            print("DME Risk:    ", diag.get("dme_label"))
            print("Referable:   ", diag.get("is_referable"))
            print("Urgency:     ", diag.get("referral_urgency"))
            print("Action Plan: ", diag.get("action_plan"))
            bio = data.get("biomarkers", {})
            print("Vessel Dens: ", bio.get("vessel_density_pct"), "%")
            print("Lesions:     ", bio.get("lesions"))
            heatmap_str = data.get("visualizations", {}).get("gradcam_base64", "")
            print("Heatmap Data:", heatmap_str[:40], f"... [Total Length: {len(heatmap_str)} chars]")
            print("==================================================\n")
    except Exception as e:
        print(f"[Error] Failed to connect to API ({e}). Make sure 'uvicorn app:app' is running.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=r"demo_dataset/idrid_demo/IDRiD_Demo_01_Normal.png")
    args = parser.parse_args()
    test_api(args.image)
