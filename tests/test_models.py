"""
Shape + forward-pass tests for all models including EEGEmotionEncoder.
Run: pytest tests/ -v
"""

import pytest
import torch

CFG = {
    "models": {
        "embed_dim": 64,
        "eeg_encoder": {
            "n_channels": 64, "n_times": 641,
            "conv_channels": [32, 64, 64],
            "transformer_heads": 2, "transformer_layers": 1, "dropout": 0.1,
        },
        "eeg_emotion_encoder": {
            "n_channels": 32, "n_times": 512,
            "conv_channels": [32, 64, 64],
            "transformer_heads": 2, "transformer_layers": 1, "dropout": 0.1,
        },
        "fmri_encoder":  {"dropout": 0.1},
        "face_encoder":  {"in_channels": 1, "img_size": 48, "dropout": 0.1},
        "mri_encoder":   {"in_channels": 1, "img_size": 224, "dropout": 0.1},
        "fusion": {
            "n_heads": 2, "n_fusion_layers": 1,
            "tumor_classes": 4, "emotion_classes": 4,
            "mc_dropout_samples": 3, "dropout": 0.1,
        },
    },
    "datasets": {
        "eeg":         {"n_classes": 2},
        "eeg_emotion": {"n_classes": 4},
        "fmri":        {"n_classes": 8},
        "face":        {"n_classes": 7},
        "mri":         {"n_classes": 4},
    },
}

B = 2
D = CFG["models"]["embed_dim"]


def test_eeg_encoder():
    from src.models.eeg_encoder import build_eeg_encoder
    model = build_eeg_encoder(CFG)
    x = torch.randn(B, 64, 641)
    emb, logits = model(x)
    assert emb.shape    == (B, D), f"EEG emb: {emb.shape}"
    assert logits.shape == (B, 2), f"EEG logits: {logits.shape}"


def test_eeg_emotion_encoder():
    from src.models.eeg_emotion_encoder import build_eeg_emotion_encoder
    model = build_eeg_emotion_encoder(CFG)
    x = torch.randn(B, 32, 512)      # 32 DEAP channels, 512 timepoints
    emb, logits = model(x)
    assert emb.shape    == (B, D), f"EEGEmotion emb: {emb.shape}"
    assert logits.shape == (B, 4), f"EEGEmotion logits: {logits.shape}"


def test_fmri_encoder():
    from src.models.encoders import build_fmri_encoder
    model = build_fmri_encoder(CFG)
    x = torch.randn(B, 1, 20, 32, 32)
    emb, logits = model(x)
    assert emb.shape    == (B, D), f"fMRI emb: {emb.shape}"
    assert logits.shape == (B, 8), f"fMRI logits: {logits.shape}"


def test_face_encoder():
    from src.models.encoders import build_face_encoder
    model = build_face_encoder(CFG)
    x = torch.randn(B, 1, 48, 48)
    emb, logits = model(x)
    assert emb.shape    == (B, D), f"Face emb: {emb.shape}"
    assert logits.shape == (B, 7), f"Face logits: {logits.shape}"


def test_mri_encoder():
    from src.models.encoders import build_mri_encoder
    model = build_mri_encoder(CFG)
    x = torch.randn(B, 1, 224, 224)
    emb, logits = model(x)
    assert emb.shape    == (B, D), f"MRI emb: {emb.shape}"
    assert logits.shape == (B, 4), f"MRI logits: {logits.shape}"


def test_fusion_all_modalities():
    from src.models.fusion import build_fusion_model
    model = build_fusion_model(CFG)
    emb = torch.randn(B, D)
    out = model(
        eeg_emb=emb, fmri_emb=emb, face_emb=emb,
        mri_emb=emb, eeg_emotion_emb=emb,
    )
    assert out["tumor_logits"].shape   == (B, 4)
    assert out["emotion_logits"].shape == (B, 4)
    assert out["tumor_pred"].shape     == (B,)


def test_fusion_mri_only():
    from src.models.fusion import build_fusion_model
    model = build_fusion_model(CFG)
    out = model(mri_emb=torch.randn(B, D))
    assert out["tumor_logits"].shape == (B, 4)


def test_fusion_mri_plus_eeg_emotion():
    """Most clinically relevant combination: MRI tumour + affective EEG emotion."""
    from src.models.fusion import build_fusion_model
    model = build_fusion_model(CFG)
    out = model(
        mri_emb=torch.randn(B, D),
        eeg_emotion_emb=torch.randn(B, D),
    )
    assert out["tumor_logits"].shape   == (B, 4)
    assert out["emotion_logits"].shape == (B, 4)


def test_fusion_uncertainty():
    from src.models.fusion import build_fusion_model
    model = build_fusion_model(CFG)
    result = model.predict_with_uncertainty(
        mri_emb=torch.randn(B, D),
        eeg_emotion_emb=torch.randn(B, D),
    )
    assert result["tumor_probs"].shape       == (B, 4)
    assert result["tumor_uncertainty"].shape == (B,)
    assert result["emotion_probs"].shape     == (B, 4)
    assert torch.allclose(result["tumor_probs"].sum(-1),
                          torch.ones(B), atol=1e-4)