"""
=============================================================================
RETINASCAN AI: PRODUCTION REST API SERVER (FASTAPI)
=============================================================================
Endpoints:
  GET  /                     Health check & service overview
  GET  /api/v1/health        Detailed engine status & model device info
  POST /api/v1/predict       Upload fundus image -> Diagnosis, Lesions & Grad-CAM
=============================================================================
"""

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import torch

from pipeline_service import run_pipeline, DEVICE

app = FastAPI(
    title="RetinaScan AI Medical Engine",
    description="Multi-Stage Deep Learning & Explainable AI Engine for Retinopathy & DME Screening",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for any frontend (React, Next.js, Vue, mobile apps, local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/inspect", response_class=FileResponse, tags=["Inspection UI"])
def get_inspection_ui():
    """Serves the interactive Step-by-Step Retinal Checking UI directly in browser."""
    html_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retina_inspection_steps.html")
    if os.path.exists(html_file):
        return FileResponse(html_file, media_type="text/html")
    raise HTTPException(status_code=404, detail="retina_inspection_steps.html not found.")


@app.get("/", tags=["General"])
def index():
    return {
        "service": "RetinaScan AI Clinical Diagnostic API",
        "status": "online",
        "device": str(DEVICE),
        "interactive_ui": "/inspect",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "endpoints": {
            "predict": "POST /api/v1/predict",
            "inspect_ui": "GET /inspect",
            "health": "GET /api/v1/health"
        }
    }


@app.get("/api/v1/health", tags=["General"])
def health_check():
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }


@app.post("/api/v1/predict", tags=["Diagnostic Inference"])
async def predict_retina(file: UploadFile = File(...)):
    """
    Direct Retinal Fundus Photo Clinical Audit.
    Accepts: .png, .jpg, .jpeg, .tif
    Returns: Complete diagnostic grade, DME risk, lesion probabilities & Grad-CAM heatmap base64.
    """
    # Verify content type
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/tiff", "image/bmp", "application/octet-stream"]
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload a JPEG or PNG image."
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Execute full multi-stage AI pipeline
        result = run_pipeline(image_bytes)

        if result.get("status") == "error":
            return JSONResponse(status_code=422, content=result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Default local dev port
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
