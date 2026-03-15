# NeuroAffect-TumorNet

**Multimodal brain tumour detection guided by affective biomarkers.**

Trains separate encoders on four public datasets, fuses them via cross-modal attention, and serves predictions through a FastAPI + web UI.

---

## Architecture

```
EEG  (PhysioNet EEGBCI)   →  Conv1D + Transformer  ──┐
fMRI (Nilearn Haxby)       →  3D ResNet CNN         ──┤
Face (FER2013)             →  ResNet-18              ──┼──► Cross-modal Fusion ──► Tumour class
MRI  (Figshare 4-class)   →  EfficientNet-B0        ──┘                           Emotion state
                                                                                   Uncertainty
```

Each encoder is trained independently on its own labelled dataset. The fusion head is trained with the MRI encoder frozen, and randomly masks other modalities so it learns to handle any combination of inputs at inference time.

---

## Datasets (all free, auto-downloaded)

| Modality | Dataset | Size | Classes | Download |
|---|---|---|---|---|
| EEG | PhysioNet EEGBCI Motor Imagery | ~30 MB | 2 (left/right hand) | Auto via MNE |
| fMRI | Nilearn Haxby Visual Categories | ~350 MB | 8 (face, cat, house…) | Auto via Nilearn |
| Face | FER2013 Facial Expressions | ~60 MB | 7 emotions | Auto via HuggingFace |
| MRI | Figshare Brain Tumour 4-class | ~150 MB | 4 tumour types | Direct URL |

**Total: ~590 MB**

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-org/neuroaffect-tumornet
cd neuroaffect-tumornet
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download datasets

```bash
python data/download.py
```

This downloads all four datasets into `./data/`. Takes 5–15 minutes depending on connection speed.

### 3. Train encoders

Train each encoder independently (in any order):

```bash
python src/training/run.py --stage mri     # ~50 epochs, fastest to converge
python src/training/run.py --stage face    # ~40 epochs
python src/training/run.py --stage eeg     # ~30 epochs
python src/training/run.py --stage fmri    # ~50 epochs (needs GPU for speed)
```

Or train everything sequentially:

```bash
python src/training/run.py --stage all
```

Checkpoints are saved to `./checkpoints/<stage>/best.pt` whenever validation accuracy improves.

### 4. Train fusion head

```bash
python src/training/run.py --stage fusion
```

This requires the MRI encoder checkpoint to exist. Face encoder checkpoint is used if available.

### 5. Run tests

```bash
pytest tests/ -v
```

All 7 tests should pass on CPU without any downloaded data.

### 6. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

- Web UI:   http://localhost:8000
- API docs: http://localhost:8000/docs

---

## Project Structure

```
neuroaffect-tumornet/
│
├── config.yaml                  ← All hyperparameters in one place
├── requirements.txt
│
├── data/
│   └── download.py              ← Downloads all 4 datasets
│
├── src/
│   ├── datasets/
│   │   ├── eeg_dataset.py       ← PhysioNet EEGBCI loader
│   │   ├── fmri_dataset.py      ← Nilearn Haxby loader
│   │   ├── face_dataset.py      ← FER2013 via HuggingFace
│   │   └── mri_dataset.py       ← Figshare Brain Tumour loader
│   │
│   ├── models/
│   │   ├── eeg_encoder.py       ← Conv1D + Transformer
│   │   ├── encoders.py          ← fMRI (3D CNN) + Face (ResNet-18) + MRI (EfficientNet)
│   │   └── fusion.py            ← Cross-modal attention fusion + MC-Dropout
│   │
│   ├── training/
│   │   ├── trainer.py           ← BaseTrainer (MLflow + checkpointing)
│   │   └── run.py               ← Entry point for all training stages
│   │
│   └── utils/
│       └── __init__.py          ← Config, seeding, metrics, checkpointing
│
├── api/
│   ├── main.py                  ← FastAPI routes
│   └── inference.py             ← Preprocessing + model loading + prediction
│
├── frontend/
│   └── index.html               ← Single-page web UI
│
├── tests/
│   └── test_models.py           ← Shape + forward-pass tests
│
├── checkpoints/                 ← Saved model weights (git-ignored)
└── logs/                        ← MLflow runs (git-ignored)
```

---

## API Reference

### `POST /predict`

Upload files as `multipart/form-data`:

| Field | Type | Required | Description |
|---|---|---|---|
| `mri` | image file | ✅ Yes | Brain MRI slice (.jpg/.png) |
| `face` | image file | ❌ Optional | Patient face photo |
| `eeg` | .npy or .csv | ❌ Optional | EEG array (channels × timepoints) |

**Response:**

```json
{
  "tumor_class": "glioma",
  "tumor_probs": {
    "glioma": 0.812,
    "meningioma": 0.091,
    "pituitary": 0.064,
    "normal": 0.033
  },
  "emotion_class": "distress",
  "emotion_probs": {
    "distress": 0.541,
    "fear": 0.213,
    "sadness": 0.181,
    "neutral": 0.065
  },
  "uncertainty": 0.00312,
  "confidence": 0.812,
  "note": "High confidence prediction.",
  "available_modalities": ["mri", "face"]
}
```

**Example with curl:**

```bash
# MRI only
curl -X POST http://localhost:8000/predict \
  -F "mri=@/path/to/brain_scan.jpg"

# MRI + face
curl -X POST http://localhost:8000/predict \
  -F "mri=@/path/to/brain_scan.jpg" \
  -F "face=@/path/to/patient_face.jpg"

# MRI + EEG (numpy array)
curl -X POST http://localhost:8000/predict \
  -F "mri=@/path/to/brain_scan.jpg" \
  -F "eeg=@/path/to/eeg_recording.npy"
```

---

## Experiment Tracking

Training metrics are logged to MLflow automatically. To view:

```bash
mlflow ui --backend-store-uri ./logs/mlruns --port 5000
```

Then open http://localhost:5000

---

## Expected Performance (after full training)

| Encoder | Dataset | Expected Val Accuracy |
|---|---|---|
| MRI | Figshare 4-class | ~92–96% |
| Face | FER2013 | ~68–72% |
| EEG | EEGBCI Motor Imagery | ~75–85% |
| fMRI | Haxby (8-class) | ~60–70% |
| Fusion | Figshare (primary) | ~93–97% |

MRI accuracy is highest because Figshare is the cleanest, highest-resolution dataset. EEG and fMRI numbers reflect the difficulty of the classification task.

---

## Extending to More Modalities

When you gain access to MEG data or a joint dataset:

1. Add a new encoder in `src/models/` following the same interface:
   ```python
   def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
       # returns (embedding, logits)
   ```
2. Add the modality name to `MODALITIES` in `src/models/fusion.py`
3. Add a dataset loader in `src/datasets/`
4. Add a training stage in `src/training/run.py`
5. Update `config.yaml` with the new hyperparameters

The fusion head handles missing modalities automatically via the mask token mechanism, so existing checkpoints remain valid.

---

## Hardware Requirements

| Setup | Time estimate (full training) |
|---|---|
| CPU only | ~4–8 hours |
| Single GPU (RTX 3060) | ~45–90 minutes |
| Single GPU (A100) | ~15–25 minutes |

The MRI and Face encoders benefit most from GPU. EEG and fMRI encoders are small enough to train on CPU in reasonable time.

---

## Citation

If you use this project in research:

```bibtex
@software{neuroaffect2025,
  title  = {NeuroAffect-TumorNet: Affective-Guided Multimodal Brain Tumour Detection},
  year   = {2025},
  note   = {Trains on PhysioNet EEGBCI, Nilearn Haxby, FER2013, Figshare Brain Tumour MRI}
}
```

**Baseline datasets:**
- PhysioNet EEGBCI: Goldberger et al., PhysioBank (2000)
- Nilearn Haxby: Haxby et al., Science (2001)
- FER2013: Goodfellow et al., ICML Workshop (2013)
- Figshare Brain Tumour MRI: Cheng et al. (2017)
