"""
FusionModel  —  Cross-modal Attention Fusion
=============================================
Combines up to 4 modality embeddings (EEG, fMRI, Face, MRI) via
bidirectional cross-attention.

Key features:
  - Missing modality robustness: absent inputs are zero-masked, with a
    learned mask token so the network can distinguish zero-feature from
    genuine absent modality.
  - Separate classification heads for tumor (primary) and emotion (secondary).
  - MC-Dropout for calibrated uncertainty at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


MODALITIES = ["eeg", "fmri", "face", "mri"]

TUMOR_CLASSES   = ["glioma", "meningioma", "pituitary", "normal"]
EMOTION_CLASSES = ["distress", "fear", "sadness", "neutral"]


# ─────────────────────────────────────────────────────────────────────────────
class CrossAttentionLayer(nn.Module):
    """
    One round of bidirectional cross-modal attention.
    Each modality token attends to ALL other modality tokens.
    """

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
        """q: (B,1,D)  kv: (B,K,D)  → (B,1,D)"""
        attended, _ = self.cross(q, kv, kv)
        x = self.norm1(q + self.gate.clamp(0, 1) * attended)
        x = self.norm2(x + self.ffn(x))
        return x


class FusionBlock(nn.Module):
    """
    One fusion layer: each modality attends to all others simultaneously.
    """

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleDict({
            m: CrossAttentionLayer(dim, n_heads, dropout)
            for m in MODALITIES
        })

    def forward(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """tokens: each (B, 1, D)"""
        out = {}
        for m in MODALITIES:
            q  = tokens[m]
            kv = torch.cat([tokens[k] for k in MODALITIES if k != m], dim=1)
            out[m] = self.layers[m](q, kv)
        return out


# ─────────────────────────────────────────────────────────────────────────────
class FusionModel(nn.Module):
    """
    Full multimodal fusion model.

    Usage
    -----
    model = FusionModel(embed_dim=256, ...)

    # all modalities present
    out = model(eeg_emb=e, fmri_emb=f, face_emb=fa, mri_emb=m)

    # only MRI available (others are None → zero-masked)
    out = model(mri_emb=m)

    Returns dict with:
        tumor_logits  : (B, 4)
        emotion_logits: (B, 4)
        tumor_pred    : (B,) int
        emotion_pred  : (B,) int
    """

    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_fusion_layers: int = 3,
        tumor_classes: int = 4,
        emotion_classes: int = 4,
        mc_dropout_samples: int = 20,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embed_dim         = embed_dim
        self.mc_dropout_samples = mc_dropout_samples

        # Learned mask token: replaces absent modality embedding
        self.mask_tokens = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            for m in MODALITIES
        })

        # Modality-type embeddings (so model knows which slot each token occupies)
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

        jd = embed_dim * len(MODALITIES)   # 256 × 4 = 1024

        # Primary: tumor classification (MRI-driven)
        self.tumor_head = nn.Sequential(
            nn.LayerNorm(jd),
            nn.Linear(jd, 512), nn.GELU(),
            nn.Dropout(dropout),           # MC-Dropout
            nn.Linear(512, tumor_classes),
        )

        # Secondary: emotion state (EEG/fMRI/face-driven)
        self.emotion_head = nn.Sequential(
            nn.LayerNorm(jd),
            nn.Linear(jd, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, emotion_classes),
        )

    def _prepare_tokens(
        self,
        eeg_emb:  Optional[torch.Tensor],
        fmri_emb: Optional[torch.Tensor],
        face_emb: Optional[torch.Tensor],
        mri_emb:  Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Replace None embeddings with learned mask token."""
        inputs = dict(eeg=eeg_emb, fmri=fmri_emb, face=face_emb, mri=mri_emb)
        tokens = {}
        # Determine batch size from first available embedding
        B = next(v.size(0) for v in inputs.values() if v is not None)

        for m, emb in inputs.items():
            if emb is not None:
                tok = emb.unsqueeze(1)                             # (B,1,D)
            else:
                tok = self.mask_tokens[m].expand(B, -1, -1)       # (B,1,D)
            tokens[m] = tok + self.mod_type[m]                    # add type embed
        return tokens

    def forward(
        self,
        eeg_emb:  Optional[torch.Tensor] = None,
        fmri_emb: Optional[torch.Tensor] = None,
        face_emb: Optional[torch.Tensor] = None,
        mri_emb:  Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        tokens = self._prepare_tokens(eeg_emb, fmri_emb, face_emb, mri_emb)

        for layer in self.fusion:
            tokens = layer(tokens)

        # Concatenate all fused tokens
        joint = torch.cat(
            [self.norm(tokens[m].squeeze(1)) for m in MODALITIES], dim=-1
        )                                                          # (B, 4D)

        tumor_logits   = self.tumor_head(joint)
        emotion_logits = self.emotion_head(joint)

        return {
            "tumor_logits":   tumor_logits,
            "emotion_logits": emotion_logits,
            "tumor_pred":     tumor_logits.argmax(-1),
            "emotion_pred":   emotion_logits.argmax(-1),
        }

    # ── MC-Dropout inference ──────────────────────────────────────────────
    def predict_with_uncertainty(
        self,
        eeg_emb=None, fmri_emb=None, face_emb=None, mri_emb=None,
    ) -> Dict[str, torch.Tensor]:
        """Runs T stochastic forward passes, returns mean probs + variance."""
        self.train()                   # keep dropout active
        tumor_samples, emotion_samples = [], []

        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                out = self.forward(eeg_emb, fmri_emb, face_emb, mri_emb)
                tumor_samples.append(out["tumor_logits"].softmax(-1))
                emotion_samples.append(out["emotion_logits"].softmax(-1))

        self.eval()

        t_stack = torch.stack(tumor_samples)        # (T, B, 4)
        e_stack = torch.stack(emotion_samples)

        return {
            "tumor_probs":      t_stack.mean(0),
            "tumor_uncertainty": t_stack.var(0).sum(-1),
            "tumor_pred":       t_stack.mean(0).argmax(-1),
            "emotion_probs":    e_stack.mean(0),
            "emotion_pred":     e_stack.mean(0).argmax(-1),
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
