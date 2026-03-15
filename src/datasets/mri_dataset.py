"""
MRIDataset  —  Figshare Brain Tumour 4-class
Uses 64x64 images instead of 224x224 for 10x faster training.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

LABEL_NAMES = ["glioma", "meningioma", "pituitary", "normal"]

FOLDER_TO_CLASS = {
    "glioma_tumor": 0, "meningioma_tumor": 1,
    "pituitary_tumor": 2, "no_tumor": 3,
    "glioma": 0, "meningioma": 1,
    "pituitary": 2, "notumor": 3,
}


class BrainMRIDataset(Dataset):
    def __init__(self, root, split="train", img_size=64, augment=False):
        super().__init__()
        root = Path(root)
        self.samples: List[Tuple[Path, int]] = []

        split_name = "Training" if split == "train" else "Testing"
        split_dir  = self._find_dir(root, split_name)
        search     = split_dir if split_dir else root

        for fname, cls in FOLDER_TO_CLASS.items():
            d = self._find_dir(search, fname)
            if d is None: continue
            for p in sorted(d.glob("*")):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((p, cls))

        if not self.samples:
            # scan whole tree
            for p in root.rglob("*"):
                if p.suffix.lower() not in (".jpg",".jpeg",".png"): continue
                for fname, cls in FOLDER_TO_CLASS.items():
                    if fname in p.parent.name.lower():
                        self.samples.append((p, cls)); break

        if not self.samples:
            print(f"\n  Contents of {root}:")
            for p in sorted(root.rglob("*"))[:20]: print(f"    {p}")
            raise RuntimeError(f"No images found under {root}")

        counts = np.bincount([y for _,y in self.samples], minlength=4)
        self.class_weights = torch.tensor(1.0/(counts+1e-6), dtype=torch.float32)
        self.class_weights /= self.class_weights.sum()

        base = [transforms.Grayscale(), transforms.Resize((img_size, img_size)),
                transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]
        aug  = [transforms.Grayscale(), transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]
        self.tf = transforms.Compose(aug if augment else base)

        print(f"  BrainMRIDataset [{split}] {img_size}x{img_size}: "
              f"{len(self.samples)} imgs  classes={counts.tolist()}")

    @staticmethod
    def _find_dir(root, name):
        for p in root.rglob("*"):
            if p.is_dir() and p.name.lower() == name.lower(): return p
        return None

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        return self.tf(img), torch.tensor(label, dtype=torch.long)


def get_mri_loaders(cfg, seed=42):
    mcfg = cfg["datasets"]["mri"]
    tcfg = cfg["training"]["mri"]
    root = Path(mcfg["download_dir"]) / "extracted"
    size = mcfg.get("img_size_fast", 64)   # use fast size

    print(f"\n  MRI data root: {root.resolve()}")

    try:
        train_ds = BrainMRIDataset(root, "train", size, augment=tcfg.get("augment",True))
        val_ds   = BrainMRIDataset(root, "test",  size, augment=False)
    except RuntimeError:
        full = BrainMRIDataset(root, "train", size, augment=False)
        n_val = max(1, int(len(full) * tcfg["val_split"]))
        train_ds, val_ds = random_split(full, [len(full)-n_val, n_val],
            generator=torch.Generator().manual_seed(seed))

    kw = dict(num_workers=0, pin_memory=False)
    return (DataLoader(train_ds, tcfg["batch_size"], shuffle=True,  **kw),
            DataLoader(val_ds,   tcfg["batch_size"], shuffle=False, **kw))