"""
FaceDataset  —  FER2013
Tries multiple HuggingFace repos, then falls back to
local images built from CSV by data/download.py.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

LABEL_NAMES = ["angry","disgust","fear","happy","sad","surprise","neutral"]

# All known working FER2013 HuggingFace repos to try in order
FER_HF_REPOS = [
    "EduardoPacheco/FER2013",
    "jonatasgrosman/fer2013",
    "trpakov/vit-face-expression",
]


def _try_hf(repo, split, cache_dir):
    """Try loading a HuggingFace dataset, return None on failure."""
    try:
        import datasets as hf_ds
        ds = hf_ds.load_dataset(repo, split=split, cache_dir=cache_dir)
        return ds
    except Exception:
        return None


class FER2013Dataset(Dataset):
    """
    Loads FER2013 from:
      1. Any working HuggingFace repo (tries list above)
      2. Local image folders built by download.py CSV conversion
         (data/face/images/train/angry/*.png  etc.)
      3. Flat CSV at data/face/fer2013.csv (converts on the fly)
    """

    def __init__(self, split="train", img_size=48, augment=False,
                 hf_repo=None, cache_dir=None, download_dir=None):
        super().__init__()
        self.samples: List[Tuple] = []
        self.use_hf  = False
        self.hf_ds   = None

        # ── Try HuggingFace ───────────────────────────────────────────────
        repos = ([hf_repo] if hf_repo else []) + FER_HF_REPOS
        for repo in repos:
            ds = _try_hf(repo, split, cache_dir)
            if ds is not None:
                self.hf_ds   = ds
                self.use_hf  = True
                self.hf_repo = repo
                print(f"  FER2013 [{split}]: loaded from HuggingFace '{repo}' "
                      f"({len(ds)} samples)")
                break

        # ── Try local image folders ───────────────────────────────────────
        if not self.use_hf and download_dir:
            img_root = Path(download_dir) / "images" / split
            if img_root.exists():
                for cls_idx, cls_name in enumerate(LABEL_NAMES):
                    cls_dir = img_root / cls_name
                    if not cls_dir.exists():
                        continue
                    for p in sorted(cls_dir.glob("*.png")) + \
                             sorted(cls_dir.glob("*.jpg")):
                        self.samples.append((p, cls_idx))
                if self.samples:
                    print(f"  FER2013 [{split}]: loaded {len(self.samples)} "
                          f"local images from {img_root}")

        # ── Try CSV conversion on the fly ─────────────────────────────────
        if not self.use_hf and not self.samples and download_dir:
            csv_path = Path(download_dir) / "fer2013.csv"
            if csv_path.exists():
                self.samples = self._load_csv(csv_path, split)
                print(f"  FER2013 [{split}]: loaded {len(self.samples)} "
                      f"samples from CSV")

        if not self.use_hf and not self.samples:
            raise RuntimeError(
                "FER2013 not found.\n"
                "Run: python data/download.py\n"
                "Or manually download fer2013.csv from Kaggle and place at:\n"
                f"  {Path(download_dir or './data/face') / 'fer2013.csv'}"
            )

        # ── Transforms ───────────────────────────────────────────────────
        if augment:
            self.tf = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

    @staticmethod
    def _load_csv(csv_path, split):
        """Load samples directly from FER2013 CSV."""
        import csv as csv_mod
        split_map = {"train": "Training", "test": "PublicTest"}
        target    = split_map.get(split, "Training")
        samples   = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                if row.get("Usage", "Training") != target:
                    continue
                pixels = list(map(int, row["pixels"].split()))
                label  = int(row["emotion"])
                arr    = np.array(pixels, dtype=np.uint8).reshape(48, 48)
                samples.append((arr, label))
        return samples

    def __len__(self):
        if self.use_hf:
            return len(self.hf_ds)
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_hf:
            sample = self.hf_ds[idx]
            img    = sample.get("image") or sample.get("img")
            label  = sample.get("label") or sample.get("labels", 0)
            if not hasattr(img, "convert"):
                img = Image.fromarray(np.array(img, dtype=np.uint8))
            img = img.convert("RGB")
        else:
            item, label = self.samples[idx]
            if isinstance(item, np.ndarray):
                img = Image.fromarray(item)
            else:
                img = Image.open(item).convert("L")

        img = self.tf(img)
        return img, torch.tensor(int(label), dtype=torch.long)


def get_face_loaders(cfg, seed=42):
    fcfg = cfg["datasets"]["face"]
    tcfg = cfg["training"]["face"]

    kw = dict(
        hf_repo      = fcfg.get("hf_repo"),
        cache_dir    = fcfg.get("cache_dir"),
        download_dir = fcfg.get("download_dir"),
        img_size     = fcfg["img_size"],
    )

    train_ds = FER2013Dataset(split="train", augment=tcfg.get("augment", True), **kw)

    # Some HF repos only have "train" split — do manual split
    try:
        val_ds = FER2013Dataset(split="test", augment=False, **kw)
    except Exception:
        n_val    = max(1, int(len(train_ds) * tcfg.get("val_split", 0.1)))
        n_train  = len(train_ds) - n_val
        train_ds, val_ds = random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed))

    kw2 = dict(num_workers=0, pin_memory=False)
    return (DataLoader(train_ds, tcfg["batch_size"], shuffle=True,  **kw2),
            DataLoader(val_ds,   tcfg["batch_size"], shuffle=False, **kw2))