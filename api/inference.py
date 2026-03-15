"""
inference.py  —  Prediction pipeline with Grad-CAM++

Accuracy improvements (no retraining required for MRI):
  1. CLAHE + edge enhancement  — highlights glioma borders
  2. Multi-scale TTA x12       — 3 crop scales × 4 spatial transforms
  3. Temperature scaling T=0.7 — sharpens underconfident predictions
  4. Prior correction          — compensates for glioma underrepresentation
  5. Ensemble MRI + fusion     — averages both heads' calibrated probs
  6. MC-Dropout x20            — stable uncertainty

New in this version:
  7. EEG emotion encoder       — DEAP-trained affective EEG → real emotion signal
     Pass an EEG .npy/.csv file recorded during emotional state assessment.
     The encoder maps 32-channel EEG → [neutral, sadness, fear, distress].
"""

import io
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as TF

from src.utils import load_config, get_device, load_checkpoint, get_logger
from src.models.eeg_encoder import build_eeg_encoder
from src.models.eeg_emotion_encoder import build_eeg_emotion_encoder
from src.models.encoders import build_fmri_encoder, build_face_encoder, build_mri_encoder
from src.models.fusion import build_fusion_model, TUMOR_CLASSES, EMOTION_CLASSES
from api.gradcam import GradCAMPP, heatmap_to_rgb, overlay_heatmap, arr_to_b64, mri_to_b64

log = get_logger("inference")
_MODELS: Dict[str, Any] = {}

# ── Inference-time hyperparameters ────────────────────────────────────────────
_MC_SAMPLES  = 20
_TEMPERATURE = 0.7
_PRIOR_LOGIT_CORRECTION = torch.tensor([1.25, 0.95, 0.95, 1.00])
_MRI_WEIGHT    = 0.55
_FUSION_WEIGHT = 0.45


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_all(cfg, device):
    global _MODELS
    if _MODELS:
        return _MODELS
    ckpt = Path(cfg["paths"]["checkpoint_dir"])

    def _load(model, stage):
        p = ckpt / stage / "best.pt"
        if p.exists():
            load_checkpoint(p, model, device=device)
            log.info(f"  Loaded {stage} checkpoint.")
        else:
            log.warning(f"  {stage} checkpoint missing — random weights.")
        model.eval()
        return model

    mri_enc = _load(build_mri_encoder(cfg).to(device), "mri")
    _MODELS["eeg"]           = _load(build_eeg_encoder(cfg).to(device),       "eeg")
    _MODELS["eeg_emotion"]   = _load(build_eeg_emotion_encoder(cfg).to(device), "eeg_emotion")
    _MODELS["fmri"]          = _load(build_fmri_encoder(cfg).to(device),      "fmri")
    _MODELS["face"]          = _load(build_face_encoder(cfg).to(device),      "face")
    _MODELS["mri"]           = mri_enc
    _MODELS["fusion"]        = _load(build_fusion_model(cfg).to(device),      "fusion")
    _MODELS["gradcam"]       = GradCAMPP(mri_enc)
    _MODELS["device"]        = device
    _MODELS["cfg"]           = cfg
    return _MODELS


# ── CLAHE + edge enhancement ──────────────────────────────────────────────────

def _apply_clahe(pil_img: Image.Image) -> Image.Image:
    try:
        import cv2
        gray  = np.array(pil_img.convert("L"), dtype=np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return Image.fromarray(clahe.apply(gray), mode="L")
    except ImportError:
        return ImageOps.equalize(pil_img.convert("L"))


def _enhance_edges(pil_img: Image.Image, strength: float = 0.4) -> Image.Image:
    gray     = pil_img.convert("L")
    orig_np  = np.array(gray, dtype=np.float32)
    blur_np  = np.array(gray.filter(ImageFilter.GaussianBlur(radius=1)),
                        dtype=np.float32)
    sharp_np = np.clip(orig_np + strength * (orig_np - blur_np), 0, 255).astype(np.uint8)
    return Image.fromarray(sharp_np, mode="L")


def _preprocess_mri(pil_img: Image.Image) -> Image.Image:
    return _enhance_edges(_apply_clahe(pil_img))


# ── Temperature scaling + prior correction ────────────────────────────────────

def _calibrated_probs(logits: torch.Tensor, device: torch.device) -> torch.Tensor:
    logits   = logits.squeeze().float().cpu()
    scaled   = logits / _TEMPERATURE
    adjusted = scaled + torch.log(_PRIOR_LOGIT_CORRECTION)
    return F.softmax(adjusted, dim=-1)


# ── Multi-scale TTA ───────────────────────────────────────────────────────────

def _tta_views(pil_img: Image.Image, target_size: int = 64) -> List[torch.Tensor]:
    """12 views = 3 crop scales × 4 spatial transforms, all with CLAHE+edge."""
    preprocessed = _preprocess_mri(pil_img)
    W, H         = preprocessed.size

    def _centre_crop(img, scale):
        if scale >= 1.0:
            return img
        nw, nh = int(W * scale), int(H * scale)
        l, t   = (W - nw) // 2, (H - nh) // 2
        return img.crop((l, t, l + nw, t + nh))

    def _to_tensor(img):
        img = img.resize((target_size, target_size), Image.BILINEAR)
        return TF.normalize(TF.to_tensor(img), [0.5], [0.5])

    views = []
    for scale in [1.0, 0.90, 0.80]:
        base = _centre_crop(preprocessed, scale)
        views.append(_to_tensor(base))
        views.append(_to_tensor(ImageOps.mirror(base)))
        views.append(_to_tensor(base.rotate( 10, resample=Image.BILINEAR)))
        views.append(_to_tensor(base.rotate(-10, resample=Image.BILINEAR)))
    return views  # 12 × (1, H, W)


def _tta_mri_probs(pil_img, mri_model, device) -> torch.Tensor:
    """TTA-averaged calibrated probs from MRI encoder. Returns (n_classes,)."""
    views = _tta_views(pil_img)
    probs_list = []
    mri_model.eval()
    with torch.no_grad():
        for v in views:
            _, logits = mri_model(v.unsqueeze(0).to(device))
            probs_list.append(_calibrated_probs(logits, device))
    return torch.stack(probs_list).mean(0)


def _tta_mri_emb(pil_img, mri_model, device) -> torch.Tensor:
    """TTA-averaged MRI embedding for fusion head. Returns (1, embed_dim)."""
    views = _tta_views(pil_img)
    embs  = []
    mri_model.eval()
    with torch.no_grad():
        for v in views:
            emb, _ = mri_model(v.unsqueeze(0).to(device))
            embs.append(emb)
    return torch.stack(embs).mean(0)


def _gradcam_tensor(pil_img: Image.Image, size: int = 64) -> torch.Tensor:
    """Clean single tensor for Grad-CAM (no augmentation)."""
    img = _preprocess_mri(pil_img).resize((size, size), Image.BILINEAR)
    return TF.normalize(TF.to_tensor(img), [0.5], [0.5]).unsqueeze(0)


# ── Byte helpers ──────────────────────────────────────────────────────────────

_face_tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def _img_bytes(b, tf):
    return tf(Image.open(io.BytesIO(b)).convert("RGB")).unsqueeze(0)


def _eeg_bytes(b, n_ch, n_t):
    """Generic EEG byte loader — handles .npy and .csv."""
    import csv as csv_mod
    try:
        arr = np.load(io.BytesIO(b)).astype(np.float32)
    except Exception:
        text = b.decode()
        arr  = np.array([list(map(float, r))
                         for r in csv_mod.reader(io.StringIO(text))],
                        dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis]
    C, T = arr.shape
    if C < n_ch: arr = np.pad(arr, ((0, n_ch - C), (0, 0)))
    else:        arr = arr[:n_ch]
    if T < n_t:  arr = np.pad(arr, ((0, 0), (0, n_t - T)))
    else:        arr = arr[:, :n_t]
    arr = (arr - arr.mean(-1, keepdims=True)) / (arr.std(-1, keepdims=True) + 1e-8)
    return torch.from_numpy(arr).unsqueeze(0)


# ── Main predict ──────────────────────────────────────────────────────────────

def predict(
    mri_bytes=None,
    face_bytes=None,
    eeg_bytes=None,           # motor imagery EEG (PhysioNet EEGBCI)
    eeg_emotion_bytes=None,   # affective EEG (DEAP) ← NEW
    cfg_path="config.yaml",
    run_gradcam=True,
):
    cfg    = load_config(cfg_path)
    device = get_device(cfg)
    models = _load_all(cfg, device)

    available       = []
    eeg_emb         = fmri_emb = face_emb = mri_emb = eeg_emotion_emb = None
    mri_t_orig      = None
    mri_pil         = None

    # ── Encode modalities ─────────────────────────────────────────────────
    with torch.no_grad():
        if mri_bytes:
            mri_pil    = Image.open(io.BytesIO(mri_bytes)).convert("RGB")
            mri_emb    = _tta_mri_emb(mri_pil, models["mri"], device)
            mri_t_orig = _gradcam_tensor(mri_pil)
            available.append("mri")

        if face_bytes:
            face_t      = _img_bytes(face_bytes, _face_tf).to(device)
            face_emb, _ = models["face"](face_t)
            available.append("face")

        if eeg_bytes:
            n_ch   = cfg["models"]["eeg_encoder"]["n_channels"]
            n_t    = cfg["models"]["eeg_encoder"]["n_times"]
            eeg_t  = _eeg_bytes(eeg_bytes, n_ch, n_t).to(device)
            eeg_emb, _ = models["eeg"](eeg_t)
            available.append("eeg")

        if eeg_emotion_bytes:
            n_ch_emo = cfg["models"]["eeg_emotion_encoder"]["n_channels"]  # 32
            n_t_emo  = cfg["models"]["eeg_emotion_encoder"]["n_times"]     # 512
            eeg_emo_t = _eeg_bytes(eeg_emotion_bytes, n_ch_emo, n_t_emo).to(device)
            eeg_emotion_emb, eeg_emotion_logits = models["eeg_emotion"](eeg_emo_t)
            available.append("eeg_emotion")

    if not available:
        return {"error": "No input provided.", "available_modalities": []}

    # ── Fusion with MC-Dropout ────────────────────────────────────────────
    original_mc = models["fusion"].mc_dropout_samples
    models["fusion"].mc_dropout_samples = _MC_SAMPLES

    res = models["fusion"].predict_with_uncertainty(
        eeg_emb=eeg_emb, fmri_emb=fmri_emb,
        face_emb=face_emb, mri_emb=mri_emb,
        eeg_emotion_emb=eeg_emotion_emb,
    )

    models["fusion"].mc_dropout_samples = original_mc

    raw_fusion_logits = torch.log(res["tumor_probs"][0].cpu() + 1e-8)
    fusion_t_probs    = _calibrated_probs(raw_fusion_logits, device)

    unc = float(res["tumor_uncertainty"][0])

    # ── Emotion output ────────────────────────────────────────────────────
    # If real affective EEG present, use the encoder's direct prediction
    # (it was trained specifically on DEAP emotion labels).
    # Otherwise fall back to the fusion emotion head.
    if eeg_emotion_bytes is not None:
        emo_probs_direct = F.softmax(eeg_emotion_logits, dim=-1)[0].cpu()
        # Blend with fusion emotion head (which also receives face signal)
        fusion_e_probs   = res["emotion_probs"][0].cpu()
        e_probs_tensor   = 0.65 * emo_probs_direct + 0.35 * fusion_e_probs
    else:
        e_probs_tensor = res["emotion_probs"][0].cpu()

    e_probs = e_probs_tensor.numpy().tolist()
    e_idx   = int(e_probs_tensor.argmax())

    # ── Ensemble tumour probs ─────────────────────────────────────────────
    if mri_pil is not None:
        mri_direct_probs = _tta_mri_probs(mri_pil, models["mri"], device)
        ensemble_probs   = _MRI_WEIGHT * mri_direct_probs + _FUSION_WEIGHT * fusion_t_probs
    else:
        ensemble_probs = fusion_t_probs

    ensemble_probs = ensemble_probs / ensemble_probs.sum()
    t_probs = ensemble_probs.numpy().tolist()
    t_idx   = int(ensemble_probs.argmax())
    conf    = float(ensemble_probs.max())

    if unc > 0.3 or conf < 0.5:
        note = "Low confidence — refer to radiologist for review."
    elif unc > 0.15:
        note = "Moderate confidence — consider additional imaging."
    else:
        note = "High confidence prediction."

    method = (f"Ensemble (TTA×12 + T={_TEMPERATURE} + prior correction "
              f"+ fusion MC×{_MC_SAMPLES})")

    # ── Grad-CAM++ ────────────────────────────────────────────────────────
    gradcam = {}
    if run_gradcam and mri_t_orig is not None:
        try:
            cam = models["gradcam"].generate(
                mri_t_orig.to(device), class_idx=t_idx, out_size=224)
            gradcam = {
                "original_b64": mri_to_b64(mri_t_orig, size=224),
                "heatmap_b64":  arr_to_b64(heatmap_to_rgb(cam)),
                "overlay_b64":  arr_to_b64(
                    overlay_heatmap(mri_t_orig, cam, alpha=0.55, out_size=224)),
                "class_name":   TUMOR_CLASSES[t_idx],
            }
            log.info("Grad-CAM++ generated successfully.")
        except Exception as e:
            log.warning(f"Grad-CAM++ failed: {e}")
            import traceback
            log.warning(traceback.format_exc())

    return {
        "tumor_class":          TUMOR_CLASSES[t_idx],
        "tumor_probs":          dict(zip(TUMOR_CLASSES,
                                         [round(p, 4) for p in t_probs])),
        "emotion_class":        EMOTION_CLASSES[e_idx],
        "emotion_probs":        dict(zip(EMOTION_CLASSES,
                                         [round(p, 4) for p in e_probs])),
        "uncertainty":          round(unc, 5),
        "confidence":           round(conf, 4),
        "note":                 note,
        "inference_method":     method,
        "available_modalities": available,
        "gradcam":              gradcam,
    }