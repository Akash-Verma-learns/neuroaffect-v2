"""
EEGEncoder  —  Conv1D + Transformer
====================================
Input : (B, n_channels, n_times)
Output: embedding (B, embed_dim) + logits (B, n_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConvBlock(nn.Module):
    """Depthwise separable temporal conv block."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3,
                 stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, k, stride=stride,
                      padding=k // 2, groups=in_ch, bias=False),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class EEGEncoder(nn.Module):
    """
    Temporal conv stack → patch-wise tokens → Transformer → CLS embedding.

    Architecture:
        Conv1D stack (3 blocks, stride=2 each): n_ch → 256, T → T/8
        Positional encoding
        Transformer Encoder (4 layers, 8 heads)
        CLS token → Linear → embed_dim
        Classification head
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 641,
        conv_channels: list = None,
        embed_dim: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 4,
        n_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        conv_channels = conv_channels or [64, 128, 256]

        # ── Temporal feature extraction ────────────────────────────────────
        layers = []
        in_ch  = n_channels
        for out_ch in conv_channels:
            layers.append(DepthwiseConvBlock(in_ch, out_ch, k=7,
                                             stride=2, dropout=dropout))
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)

        seq_len  = n_times
        for _ in conv_channels:
            seq_len = (seq_len + 1) // 2    # each stride=2 halves T

        # Project conv output to transformer d_model
        d_model  = conv_channels[-1]
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1,
                                          dropout=dropout)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=transformer_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, transformer_layers)

        self.norm = nn.LayerNorm(d_model)

        # ── Projection & heads ─────────────────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(d_model, embed_dim), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(embed_dim),
        )
        self.cls_head = nn.Linear(embed_dim, n_classes)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor):
        """
        x : (B, C, T)
        Returns
        -------
        emb    : (B, embed_dim)
        logits : (B, n_classes)
        """
        x = self.conv_stack(x)                     # (B, D, T')
        x = x.transpose(1, 2)                      # (B, T', D)

        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)           # (B, 1+T', D)
        x   = self.pos_enc(x)
        x   = self.transformer(x)
        x   = self.norm(x[:, 0])                   # CLS → (B, D)

        emb    = self.proj(x)
        logits = self.cls_head(emb)
        return emb, logits


def build_eeg_encoder(cfg: dict) -> EEGEncoder:
    mc = cfg["models"]["eeg_encoder"]
    return EEGEncoder(
        n_channels        = mc["n_channels"],
        n_times           = mc["n_times"],
        conv_channels     = mc["conv_channels"],
        embed_dim         = cfg["models"]["embed_dim"],
        transformer_heads = mc["transformer_heads"],
        transformer_layers= mc["transformer_layers"],
        n_classes         = cfg["datasets"]["eeg"]["n_classes"],
        dropout           = mc["dropout"],
    )
