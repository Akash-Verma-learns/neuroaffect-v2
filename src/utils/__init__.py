"""Shared utilities: config loading, seeding, logging, metrics."""
import os, random, logging, yaml
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_device(cfg: Dict) -> torch.device:
    mode = cfg["project"].get("device", "auto")
    if mode == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(mode)

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_checkpoint(model, optimizer, epoch, metric, path: Path, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "metric": metric,
        **(extra or {}),
    }, path)

def load_checkpoint(path: Path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0), ckpt.get("metric", 0.0)

from sklearn.metrics import accuracy_score, f1_score, classification_report

def compute_metrics(y_true, y_pred, label_names=None) -> Dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report": classification_report(y_true, y_pred, target_names=label_names, zero_division=0),
    }
