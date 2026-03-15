"""
Fast encoders — optimised for CPU training.
Trades some accuracy for 10x faster training.

MRI   : SimpleCNN  (64x64 input, 3 conv blocks)   ~2 min/epoch → 5 epochs
Face  : TinyCNN    (48x48 input, 4 conv blocks)    ~1 min/epoch → 5 epochs
fMRI  : Small3DCNN (32x32x32, 2 conv blocks)       ~3 min/epoch → 5 epochs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# ── Shared building block ─────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=k//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)


# ══════════════════════════════════════════════════════════════════════════════
#  MRI ENCODER  —  SimpleCNN  (64×64 greyscale)
# ══════════════════════════════════════════════════════════════════════════════
class MRIEncoder(nn.Module):
    """
    3-block CNN for 64x64 greyscale MRI.
    ~10x faster than EfficientNet-B0 on CPU.
    Expected accuracy: ~85-90% on Figshare 4-class.
    """
    def __init__(self, embed_dim=256, n_classes=4, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  32, pool=True),    # 32
            ConvBlock(32, 64, pool=True),    # 16
            ConvBlock(64, 128, pool=True),   # 8
            ConvBlock(128, 256, pool=True),  # 4
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )
        self.cls_head = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim

    def forward(self, x):
        emb = self.proj(self.features(x))
        return emb, self.cls_head(emb)


def build_mri_encoder(cfg):
    return MRIEncoder(
        embed_dim=cfg["models"]["embed_dim"],
        n_classes=cfg["datasets"]["mri"]["n_classes"],
        dropout=cfg["models"]["mri_encoder"]["dropout"],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  FACE ENCODER  —  TinyCNN  (48×48 greyscale, FER2013 native size)
# ══════════════════════════════════════════════════════════════════════════════
class FaceEncoder(nn.Module):
    """
    4-block CNN for 48x48 greyscale face images.
    Expected accuracy: ~65-70% on FER2013 7-class.
    """
    def __init__(self, embed_dim=256, n_classes=7, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32,  pool=True),   # 24
            ConvBlock(32, 64, pool=True),   # 12
            ConvBlock(64, 128, pool=True),  # 6
            ConvBlock(128, 256, pool=False),# 6
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )
        self.cls_head = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim

    def forward(self, x):
        emb = self.proj(self.features(x))
        return emb, self.cls_head(emb)


def build_face_encoder(cfg):
    return FaceEncoder(
        embed_dim=cfg["models"]["embed_dim"],
        n_classes=cfg["datasets"]["face"]["n_classes"],
        dropout=cfg["models"]["face_encoder"]["dropout"],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  fMRI ENCODER  —  Small3DCNN  (32×32×32 volume)
# ══════════════════════════════════════════════════════════════════════════════
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )
    def forward(self, x): return self.block(x)


class FMRIEncoder(nn.Module):
    """
    2-block 3D CNN for fMRI volumes.
    Uses smaller 32x32x32 input for speed.
    Expected accuracy: ~55-65% on Haxby 8-class.
    """
    def __init__(self, embed_dim=256, n_classes=8, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock3D(1, 32),   # 16
            ConvBlock3D(32, 64),  # 8
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )
        self.cls_head = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim

    def forward(self, x):
        emb = self.proj(self.features(x))
        return emb, self.cls_head(emb)


def build_fmri_encoder(cfg):
    return FMRIEncoder(
        embed_dim=cfg["models"]["embed_dim"],
        n_classes=cfg["datasets"]["fmri"]["n_classes"],
        dropout=cfg["models"]["fmri_encoder"]["dropout"],
    )