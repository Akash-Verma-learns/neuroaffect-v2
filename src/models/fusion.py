"""
FusionModel  —  Cross-modal Attention Fusion
=============================================
5 modalities: EEG (motor), fMRI, Face, MRI, EEG-Emotion (DEAP affective)

Key changes from v1:
  - eeg_emotion added as a 5th modality slot
  - Emotion head now receives a real affective EEG signal (DEAP-trained encoder)
    rather than being unsupervised noise
  - emotion_head attends to [face, eeg_emotion] tokens specifically
    (tumour-irrelevant modalities carry the affective signal)
  - Tumour head unchanged — still MRI-primary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


MODALITIES = ["eeg", "fmri", "face", "mri", "eeg_emotion"]

TUMOR_CLASSES   = ["glioma", "meningioma", "pituitary", "normal"]
EMOTION_CLASSES = ["neutral", "sadness", "fear", "distress"]


# ─────────────────────────────────────────────────────────────────────────────
class CrossAttentionLayer(nn.Module):
    """Bidirectional cross-modal attention (unchanged from v1)."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross = nn.MultiheadAttention(
            dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.gate  = nn.Parameter(torch.tensor(0.1))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross(q, kv, kv)
        x = self.norm1(q + self.gate.clamp(0, 1) * attended)
        x = self.norm2(x + self.ffn(x))
        return x


class FusionBlock(nn.Module):
    """One fusion layer: each modality attends to all others."""

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleDict({
            m: CrossAttentionLayer(dim, n_heads, dropout)
            for m in MODALITIES
        })

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for m in MODALITIES:
            q  = tokens[m]
            kv = torch.cat([tokens[k] for k in MODALITIES if k != m], dim=1)
            out[m] = self.layers[m](q, kv)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class FusionModel(nn.Module):
    """
    Full multimodal fusion model with 5 modalities.

    Emotion head is now informed by real affective EEG (eeg_emotion slot)
    and face expression, making it a meaningful output rather than a proxy.

    Tumour head uses the full joint representation (all 5 modalities).
    The eeg_emotion signal contributes to tumour classification too — there
    is clinical evidence that patient emotional state correlates with tumour
    type and grade.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_fusion_layers: int = 2,
        tumor_classes: int = 4,
        emotion_classes: int = 4,
        mc_dropout_samples: int = 20,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim          = embed_dim
        self.mc_dropout_samples = mc_dropout_samples
        n_mod = len(MODALITIES)   # 5

        # Learned mask tokens for absent modalities
        self.mask_tokens = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            for m in MODALITIES
        })

        # Modality-type positional embeddings
        self.mod_type = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            for m in MODALITIES
        })

        # Fusion layers
        self.fusion = nn.ModuleList([
            FusionBlock(embed_dim, n_heads, dropout)
            for _ in range(n_fusion_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        jd = embed_dim * n_mod   # 128 × 5 = 640 (full joint)

        # Affective joint: face + eeg_emotion tokens only (2 × embed_dim = 256)
        # These carry the genuine emotion signal; keeping them separate
        # prevents the tumour-driven MRI token from dominating the emotion head.
        affective_dim = embed_dim * 2

        # Primary: tumour classification
        self.tumor_head = nn.Sequential(
            nn.LayerNorm(jd),
            nn.Linear(jd, 512), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, tumor_classes),
        )

        # Secondary: emotion — attended specifically to face + eeg_emotion
        self.emotion_head = nn.Sequential(
            nn.LayerNorm(affective_dim),
            nn.Linear(affective_dim, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, emotion_classes),
        )

    def _prepare_tokens(
        self,
        eeg_emb:        Optional[torch.Tensor],
        fmri_emb:       Optional[torch.Tensor],
        face_emb:       Optional[torch.Tensor],
        mri_emb:        Optional[torch.Tensor],
        eeg_emotion_emb: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        inputs = dict(
            eeg=eeg_emb, fmri=fmri_emb, face=face_emb,
            mri=mri_emb, eeg_emotion=eeg_emotion_emb,
        )
        B = next(v.size(0) for v in inputs.values() if v is not None)
        tokens = {}
        for m, emb in inputs.items():
            tok = emb.unsqueeze(1) if emb is not None \
                  else self.mask_tokens[m].expand(B, -1, -1)
            tokens[m] = tok + self.mod_type[m]
        return tokens

    def forward(
        self,
        eeg_emb:         Optional[torch.Tensor] = None,
        fmri_emb:        Optional[torch.Tensor] = None,
        face_emb:        Optional[torch.Tensor] = None,
        mri_emb:         Optional[torch.Tensor] = None,
        eeg_emotion_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        tokens = self._prepare_tokens(
            eeg_emb, fmri_emb, face_emb, mri_emb, eeg_emotion_emb)

        for layer in self.fusion:
            tokens = layer(tokens)

        # Full joint representation for tumour head
        joint = torch.cat(
            [self.norm(tokens[m].squeeze(1)) for m in MODALITIES], dim=-1
        )  # (B, 5 × embed_dim)

        # Affective representation: face + eeg_emotion only
        affective = torch.cat([
            self.norm(tokens["face"].squeeze(1)),
            self.norm(tokens["eeg_emotion"].squeeze(1)),
        ], dim=-1)  # (B, 2 × embed_dim)

        tumor_logits   = self.tumor_head(joint)
        emotion_logits = self.emotion_head(affective)

        return {
            "tumor_logits":   tumor_logits,
            "emotion_logits": emotion_logits,
            "tumor_pred":     tumor_logits.argmax(-1),
            "emotion_pred":   emotion_logits.argmax(-1),
        }

    def predict_with_uncertainty(
        self,
        eeg_emb=None, fmri_emb=None, face_emb=None,
        mri_emb=None, eeg_emotion_emb=None,
    ) -> Dict[str, torch.Tensor]:
        """MC-Dropout: T stochastic forward passes → mean probs + variance."""
        self.train()
        tumor_samples, emotion_samples = [], []

        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                out = self.forward(
                    eeg_emb, fmri_emb, face_emb, mri_emb, eeg_emotion_emb)
                tumor_samples.append(out["tumor_logits"].softmax(-1))
                emotion_samples.append(out["emotion_logits"].softmax(-1))

        self.eval()
        t_stack = torch.stack(tumor_samples)
        e_stack = torch.stack(emotion_samples)

        return {
            "tumor_probs":       t_stack.mean(0),
            "tumor_uncertainty": t_stack.var(0).sum(-1),
            "tumor_pred":        t_stack.mean(0).argmax(-1),
            "emotion_probs":     e_stack.mean(0),
            "emotion_pred":      e_stack.mean(0).argmax(-1),
        }


def build_fusion_model(cfg: dict) -> FusionModel:
    mc = cfg["models"]["fusion"]
    return FusionModel(
        embed_dim          = cfg["models"]["embed_dim"],
        n_heads            = mc["n_heads"],
        n_fusion_layers    = mc["n_fusion_layers"],
        tumor_classes      = mc["tumor_classes"],
        emotion_classes    = mc["emotion_classes"],
        mc_dropout_samples = mc["mc_dropout_samples"],
        dropout            = mc["dropout"],
    )