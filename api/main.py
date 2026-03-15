"""
NeuroAffect-TumorNet  FastAPI Backend
======================================
Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.inference import predict

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuroAffect-TumorNet",
    description=(
        "Multimodal brain tumour detection guided by affective biomarkers.\n\n"
        "Upload an MRI image (required) and optionally a face image or EEG file "
        "to receive a tumour classification with uncertainty estimate."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ─────────────────────────────────────────────────────────────────────────────
#  Schemas
# ─────────────────────────────────────────────────────────────────────────────

from typing import Any, Dict, Optional

class PredictionResponse(BaseModel):
    tumor_class:          str
    tumor_probs:          dict
    emotion_class:        str
    emotion_probs:        dict
    uncertainty:          float
    confidence:           float
    note:                 str
    available_modalities: list
    inference_method:     Optional[str] = None
    gradcam:              Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    version: str


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "NeuroAffect-TumorNet API — visit /docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    mri:  Optional[UploadFile] = File(None,  description="MRI image (.jpg/.png) — primary modality"),
    face: Optional[UploadFile] = File(None,  description="Face photo (.jpg/.png) — optional"),
    eeg:  Optional[UploadFile] = File(None,  description="EEG data (.npy/.csv, shape C×T) — optional"),
):
    """
    Run multimodal prediction.

    - **mri**  : Brain MRI slice (grayscale JPEG/PNG, 224×224 recommended)
    - **face** : Patient face photo (for emotion/distress assessment)
    - **eeg**  : EEG recording as numpy array or CSV (channels × timepoints)

    At least one modality must be provided. MRI alone gives the most reliable
    tumour prediction.
    """
    mri_bytes  = await mri.read()  if mri  else None
    face_bytes = await face.read() if face else None
    eeg_bytes  = await eeg.read()  if eeg  else None

    if not any([mri_bytes, face_bytes, eeg_bytes]):
        raise HTTPException(
            status_code=422,
            detail="At least one file (mri, face, or eeg) must be provided.")

    result = predict(
        mri_bytes=mri_bytes,
        face_bytes=face_bytes,
        eeg_bytes=eeg_bytes,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/labels")
async def get_labels():
    """Returns the class label names used by the model."""
    return {
        "tumor_classes":   ["glioma", "meningioma", "pituitary", "normal"],
        "emotion_classes": ["distress", "fear", "sadness", "neutral"],
    }
