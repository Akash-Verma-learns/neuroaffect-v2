"""
FMRIDataset  —  Nilearn Haxby (visual object categories)
==========================================================
8 visual categories: face, cat, chair, house, scissors, shoe, bottle, scrambled
Each sample = one 3D fMRI volume (40×64×64)
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F


HAXBY_LABELS = {
    "face": 0, "cat": 1, "chair": 2, "house": 3,
    "scissors": 4, "shoe": 5, "bottle": 6, "scrambledpix": 7,
}


class FMRIDataset(Dataset):
    """
    Loads the Nilearn Haxby dataset.

    Each item: (volume_tensor, label)
    volume_tensor: (1, D, H, W) z-scored BOLD volume, resized to vol_shape.
    """

    def __init__(
        self,
        data_dir: str,
        subject: int = 1,
        vol_shape: Tuple[int, int, int] = (40, 64, 64),
    ):
        super().__init__()
        try:
            import nibabel as nib
            from nilearn import datasets as nl_ds
        except ImportError:
            raise ImportError("Install: pip install nilearn nibabel")

        haxby = nl_ds.fetch_haxby(subjects=[subject], data_dir=data_dir, verbose=0)

        # Load labels
        import pandas as pd
        labels_df = pd.read_csv(haxby.session_target[0], sep=" ")
        cond      = labels_df["labels"].values

        # Load 4D functional image
        img  = nib.load(haxby.func[0])
        data = img.get_fdata(dtype=np.float32)     # (X, Y, Z, T)

        # z-score per voxel
        mu  = data.mean(-1, keepdims=True)
        sig = data.std(-1, keepdims=True) + 1e-8
        data = (data - mu) / sig

        self.samples: list = []
        for t, label in enumerate(cond):
            if label not in HAXBY_LABELS:
                continue
            vol = data[:, :, :, t]                 # (X, Y, Z)
            vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # (1,1,X,Y,Z)
            vol_t = F.interpolate(
                vol_t, size=vol_shape, mode="trilinear", align_corners=False
            ).squeeze(0)                           # (1, D, H, W)
            self.samples.append((vol_t, HAXBY_LABELS[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.samples[idx]
        return x, torch.tensor(y, dtype=torch.long)


def get_fmri_loaders(cfg: dict, seed: int = 42):
    fcfg = cfg["datasets"]["fmri"]
    tcfg = cfg["training"]["fmri"]

    ds = FMRIDataset(
        data_dir  = fcfg["download_dir"],
        subject   = fcfg["subject"],
        vol_shape = tuple(fcfg["vol_shape"]),
    )

    n_val   = max(1, int(len(ds) * tcfg["val_split"]))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed))

    loader_kw = dict(num_workers=2, pin_memory=True)
    return (DataLoader(train_ds, tcfg["batch_size"], shuffle=True,  **loader_kw),
            DataLoader(val_ds,   tcfg["batch_size"], shuffle=False, **loader_kw))
