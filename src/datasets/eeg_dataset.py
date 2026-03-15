"""
EEGDataset  —  PhysioNet EEGBCI Motor Imagery
Labels: 0 = left-hand, 1 = right-hand
Output: (B, 64, n_times)
"""

import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class EEGDataset(Dataset):
    def __init__(self, data_dir, subjects, runs,
                 tmin=0.0, tmax=4.0, sfreq=160, n_ch=64):
        super().__init__()
        import mne
        from mne.datasets import eegbci
        mne.set_log_level("WARNING")

        self.samples: List[Tuple[np.ndarray, int]] = []

        for subj in subjects:
            try:
                fnames = eegbci.load_data(subj, runs, path=data_dir,
                                          verbose=False, update_path=False)
                raws = [mne.io.read_raw_edf(f, preload=True, verbose=False)
                        for f in fnames]
                raw = mne.concatenate_raws(raws)
                eegbci.standardize(raw)
                raw.resample(sfreq, npad="auto")
                raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)

                events, event_id = mne.events_from_annotations(
                    raw, verbose=False)

                # Keep only T1 / T2 events
                ev_map = {k: v for k, v in event_id.items()
                          if k in ("T1", "T2")}
                if not ev_map:
                    continue

                epochs = mne.Epochs(
                    raw, events, event_id=ev_map,
                    tmin=tmin, tmax=tmax,
                    baseline=None, preload=True, verbose=False)

                data = epochs.get_data().astype(np.float32)  # (N, C, T)

                # ── FIX: get labels from epochs.events, not events_id_to_str ──
                # epochs.events[:, 2] contains the event code
                # ev_map maps name → code, so we invert it
                code_to_label = {}
                for name, code in ev_map.items():
                    code_to_label[code] = 0 if name == "T1" else 1

                labels = [code_to_label[ev[2]] for ev in epochs.events]

                # Normalise each trial
                data = (data - data.mean(-1, keepdims=True)) / (
                    data.std(-1, keepdims=True) + 1e-8)

                # Pad / trim channels to n_ch
                C = data.shape[1]
                if C < n_ch:
                    pad = np.zeros((data.shape[0], n_ch - C, data.shape[2]),
                                   dtype=np.float32)
                    data = np.concatenate([data, pad], axis=1)
                else:
                    data = data[:, :n_ch, :]

                for x, y in zip(data, labels):
                    self.samples.append((x, y))

            except Exception as e:
                warnings.warn(f"Subject {subj} failed: {e}")

        if not self.samples:
            raise RuntimeError(
                "No EEG samples loaded. Run data/download.py first.")

        print(f"  EEGDataset: {len(self.samples)} epochs loaded  "
              f"labels={np.bincount([y for _,y in self.samples]).tolist()}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def get_eeg_loaders(cfg, seed=42):
    ecfg = cfg["datasets"]["eeg"]
    tcfg = cfg["training"]["eeg"]

    ds = EEGDataset(
        data_dir = ecfg["download_dir"],
        subjects = ecfg["subjects"],
        runs     = ecfg["runs"],
        tmin     = ecfg["tmin"],
        tmax     = ecfg["tmax"],
        sfreq    = ecfg["sfreq"],
        n_ch     = ecfg["n_channels"],
    )

    n_val   = max(1, int(len(ds) * tcfg["val_split"]))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed))

    kw = dict(num_workers=0, pin_memory=False)
    return (DataLoader(train_ds, tcfg["batch_size"], shuffle=True,  **kw),
            DataLoader(val_ds,   tcfg["batch_size"], shuffle=False, **kw))