"""
NeuroAffect-TumorNet  FastAPI Backend
======================================
Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.inference import predict

app = FastAPI(
    title="NeuroAffect-TumorNet",
    description=(
        "Multimodal brain tumour detection guided by affective biomarkers.\n\n"
        "Upload an MRI image (required) and optionally a face image, "
        "motor EEG, or affective EEG (DEAP format) to receive a tumour "
        "classification with emotion state and uncertainty estimate."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ── Schemas ───────────────────────────────────────────────────────────────────

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
    status:  str
    version: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def serve_frontend():
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "NeuroAffect-TumorNet API — visit /docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    mri:         Optional[UploadFile] = File(None, description="MRI image (.jpg/.png) — required for tumour classification"),
    face:        Optional[UploadFile] = File(None, description="Face photo (.jpg/.png) — emotion/distress assessment"),
    eeg:         Optional[UploadFile] = File(None, description="Motor EEG (.npy/.csv, 64ch × T) — PhysioNet EEGBCI format"),
    eeg_emotion: Optional[UploadFile] = File(None, description="Affective EEG (.npy/.csv, 32ch × T) — DEAP format, real emotion signal"),
):
    """
    Run multimodal prediction.

    - **mri**         : Brain MRI slice — primary tumour modality
    - **face**        : Patient face photo — facial expression emotion signal
    - **eeg**         : Motor imagery EEG (64 channels, PhysioNet EEGBCI format)
    - **eeg_emotion** : Affective EEG (32 channels, DEAP format) — drives the
                        negative emotion classification [neutral/sadness/fear/distress]

    At least one modality must be provided. For best results provide MRI + eeg_emotion.
    """
    mri_bytes         = await mri.read()         if mri         else None
    face_bytes        = await face.read()        if face        else None
    eeg_bytes         = await eeg.read()         if eeg         else None
    eeg_emotion_bytes = await eeg_emotion.read() if eeg_emotion else None

    if not any([mri_bytes, face_bytes, eeg_bytes, eeg_emotion_bytes]):
        raise HTTPException(
            status_code=422,
            detail="At least one file (mri, face, eeg, or eeg_emotion) must be provided.")

    result = predict(
        mri_bytes=mri_bytes,
        face_bytes=face_bytes,
        eeg_bytes=eeg_bytes,
        eeg_emotion_bytes=eeg_emotion_bytes,
    )

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@app.get("/labels")
async def get_labels():
    return {
        "tumor_classes":   ["glioma", "meningioma", "pituitary", "normal"],
        "emotion_classes": ["neutral", "sadness", "fear", "distress"],
    }