"""
EEGEmotionDataset  —  DEAP Affective EEG Dataset
=================================================
DEAP: A Database for Emotion Analysis using Physiological Signals
Koelstra et al., IEEE T-AFFC 2012

32 participants × 40 music video trials × 63 s @ 128 Hz
32 EEG channels + 8 peripheral (we use EEG only)
Labels: valence, arousal, dominance, liking  (continuous 1–9)

Negative emotion mapping (valence/arousal threshold → 4 classes):
  0  neutral   — valence ≥ 5  (positive/neutral affect)
  1  sadness   — valence < 5, arousal < 5  (low energy negative)
  2  fear      — valence < 5, arousal ≥ 5, dominance < 5  (uncontrolled high arousal)
  3  distress  — valence < 5, arousal ≥ 5, dominance ≥ 5  (controlled high arousal)

DEAP requires manual download (licence agreement):
  1. Register at https://www.eecs.qmul.ac.uk/mmv/datasets/deap/
  2. Download the preprocessed Python data: data_preprocessed_python.zip
  3. Unzip to:  ./data/eeg_emotion/
  Files should be: s01.dat, s02.dat, ... s32.dat

Output tensor shape: (B, 32, n_times)  — 32 EEG channels
"""

import pickle
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ── Emotion label names ───────────────────────────────────────────────────────
EMOTION_LABEL_NAMES = ["neutral", "sadness", "fear", "distress"]
N_EEG_CHANNELS = 32    # DEAP has 32 EEG channels (channels 0–31)


def _valence_arousal_to_label(valence: float,
                               arousal: float,
                               dominance: float) -> int:
    """
    Maps continuous DEAP valence/arousal/dominance scores to a 4-class
    negative emotion label.

    Thresholds are set at the scale midpoint (5.0 on a 1–9 scale).
    This is the standard binary threshold used in DEAP literature.
    """
    if valence >= 5.0:
        return 0  # neutral
    if arousal < 5.0:
        return 1  # sadness
    if dominance < 5.0:
        return 2  # fear
    return 3       # distress


class DEAPEmotionDataset(Dataset):
    """
    Loads DEAP preprocessed .dat files.

    Each .dat file is a dict with keys:
        'data'   : (40, 40, 8064)  trials × channels × samples
                   channels 0–31 = EEG, 32–39 = peripheral (ignored)
        'labels' : (40, 4)         valence, arousal, dominance, liking per trial

    We:
      1. Slice EEG channels only (0–31)
      2. Crop/pad to n_times samples
      3. z-score normalise each channel per trial
      4. Map valence/arousal/dominance → 4-class emotion label
    """

    def __init__(
        self,
        data_dir: str,
        subjects: List[int] = None,   # e.g. [1, 2, 3] — None = all 32
        n_times: int = 512,           # samples to keep (4s @ 128Hz)
        n_channels: int = 32,
    ):
        super().__init__()
        data_dir = Path(data_dir)
        subjects = subjects or list(range(1, 33))

        self.samples: List[Tuple[np.ndarray, int]] = []

        for subj in subjects:
            dat_file = data_dir / f"s{subj:02d}.dat"
            if not dat_file.exists():
                warnings.warn(
                    f"DEAP file not found: {dat_file}\n"
                    "Download from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/\n"
                    "and unzip to ./data/eeg_emotion/"
                )
                continue

            try:
                with open(dat_file, "rb") as f:
                    subject_data = pickle.load(f, encoding="latin1")

                eeg    = subject_data["data"][:, :n_channels, :]  # (40, 32, 8064)
                labels = subject_data["labels"]                    # (40, 4)

                for trial_idx in range(eeg.shape[0]):
                    trial  = eeg[trial_idx].astype(np.float32)    # (32, 8064)

                    # Crop or pad to n_times
                    T = trial.shape[1]
                    if T >= n_times:
                        # Take the middle segment (avoids baseline at start)
                        start = (T - n_times) // 2
                        trial = trial[:, start:start + n_times]
                    else:
                        trial = np.pad(trial, ((0, 0), (0, n_times - T)))

                    # z-score per channel
                    mu  = trial.mean(-1, keepdims=True)
                    sig = trial.std(-1,  keepdims=True) + 1e-8
                    trial = (trial - mu) / sig

                    valence    = float(labels[trial_idx, 0])
                    arousal    = float(labels[trial_idx, 1])
                    dominance  = float(labels[trial_idx, 2])
                    label      = _valence_arousal_to_label(valence, arousal, dominance)

                    self.samples.append((trial, label))

            except Exception as e:
                warnings.warn(f"Subject {subj} failed to load: {e}")

        if not self.samples:
            raise RuntimeError(
                "No DEAP samples loaded.\n"
                "Download preprocessed Python data from:\n"
                "  https://www.eecs.qmul.ac.uk/mmv/datasets/deap/\n"
                "Unzip s01.dat … s32.dat to: ./data/eeg_emotion/"
            )

        counts = np.bincount([y for _, y in self.samples], minlength=4)
        print(
            f"  DEAPEmotionDataset: {len(self.samples)} trials  "
            f"labels={counts.tolist()}  "
            f"({', '.join(f'{n}={c}' for n, c in zip(EMOTION_LABEL_NAMES, counts))})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def get_eeg_emotion_loaders(cfg: dict, seed: int = 42):
    ecfg = cfg["datasets"]["eeg_emotion"]
    tcfg = cfg["training"]["eeg_emotion"]

    ds = DEAPEmotionDataset(
        data_dir   = ecfg["download_dir"],
        subjects   = ecfg.get("subjects"),
        n_times    = ecfg["n_times"],
        n_channels = ecfg["n_channels"],
    )

    n_val   = max(1, int(len(ds) * tcfg["val_split"]))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    kw = dict(num_workers=0, pin_memory=False)
    return (
        DataLoader(train_ds, tcfg["batch_size"], shuffle=True,  **kw),
        DataLoader(val_ds,   tcfg["batch_size"], shuffle=False, **kw),
    )