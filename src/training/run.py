"""
Training scripts for all encoders + fusion head.

Usage:
    python src/training/run.py --stage eeg
    python src/training/run.py --stage eeg_emotion   ← NEW
    python src/training/run.py --stage fmri
    python src/training/run.py --stage face
    python src/training/run.py --stage mri
    python src/training/run.py --stage fusion
    python src/training/run.py --stage all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn

from src.utils import load_config, get_device, seed_everything, get_logger
from src.training.trainer import BaseTrainer

log = get_logger("run")


def get_loss(weights=None, device="cpu"):
    if weights is not None:
        return nn.CrossEntropyLoss(weight=weights.to(device))
    return nn.CrossEntropyLoss()


# ─────────────────────────────────────────────────────────────────────────────
def train_eeg(cfg, device):
    log.info("=" * 50)
    log.info("  Training: EEG Encoder  (motor imagery)")
    log.info("=" * 50)
    from src.datasets.eeg_dataset import get_eeg_loaders
    from src.models.eeg_encoder import build_eeg_encoder

    train_ld, val_ld = get_eeg_loaders(cfg, seed=cfg["project"]["seed"])
    model = build_eeg_encoder(cfg)
    opt   = torch.optim.AdamW(model.parameters(),
                               lr=cfg["training"]["eeg"]["lr"],
                               weight_decay=cfg["training"]["eeg"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["training"]["eeg"]["epochs"])
    BaseTrainer(model, train_ld, val_ld, opt, get_loss(), cfg,
                stage_name="eeg", device=device,
                label_names=["left-hand", "right-hand"],
                scheduler=sched).fit()


# ─────────────────────────────────────────────────────────────────────────────
def train_eeg_emotion(cfg, device):
    """
    Train the affective EEG encoder on DEAP valence/arousal labels.

    Expected accuracy: ~55–70% (4-class emotion from EEG is genuinely hard;
    state-of-the-art methods achieve ~70–80% on DEAP with subject-dependent splits).

    The encoder learns to distinguish:
      neutral  — high valence
      sadness  — low valence, low arousal
      fear     — low valence, high arousal, low dominance
      distress — low valence, high arousal, high dominance
    """
    log.info("=" * 50)
    log.info("  Training: EEG Emotion Encoder  (DEAP affective EEG)")
    log.info("=" * 50)
    from src.datasets.eeg_emotion_dataset import get_eeg_emotion_loaders
    from src.models.eeg_emotion_encoder import build_eeg_emotion_encoder

    train_ld, val_ld = get_eeg_emotion_loaders(cfg, seed=cfg["project"]["seed"])
    model = build_eeg_emotion_encoder(cfg)
    opt   = torch.optim.AdamW(model.parameters(),
                               lr=cfg["training"]["eeg_emotion"]["lr"],
                               weight_decay=cfg["training"]["eeg_emotion"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["training"]["eeg_emotion"]["epochs"])
    BaseTrainer(model, train_ld, val_ld, opt, get_loss(), cfg,
                stage_name="eeg_emotion", device=device,
                label_names=cfg["datasets"]["eeg_emotion"]["label_names"],
                scheduler=sched).fit()


# ─────────────────────────────────────────────────────────────────────────────
def train_fmri(cfg, device):
    log.info("=" * 50)
    log.info("  Training: fMRI Encoder")
    log.info("=" * 50)
    from src.datasets.fmri_dataset import get_fmri_loaders
    from src.models.encoders import build_fmri_encoder

    train_ld, val_ld = get_fmri_loaders(cfg, seed=cfg["project"]["seed"])
    model = build_fmri_encoder(cfg)
    opt   = torch.optim.AdamW(model.parameters(),
                               lr=cfg["training"]["fmri"]["lr"],
                               weight_decay=cfg["training"]["fmri"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["training"]["fmri"]["epochs"])
    BaseTrainer(model, train_ld, val_ld, opt, get_loss(), cfg,
                stage_name="fmri", device=device,
                label_names=cfg["datasets"]["fmri"].get("label_names"),
                scheduler=sched).fit()


# ─────────────────────────────────────────────────────────────────────────────
def train_face(cfg, device):
    log.info("=" * 50)
    log.info("  Training: Face Encoder  (FER2013)")
    log.info("=" * 50)
    from src.datasets.face_dataset import get_face_loaders
    from src.models.encoders import build_face_encoder

    train_ld, val_ld = get_face_loaders(cfg)
    model = build_face_encoder(cfg)
    opt   = torch.optim.AdamW(model.parameters(),
                               lr=cfg["training"]["face"]["lr"],
                               weight_decay=cfg["training"]["face"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["training"]["face"]["epochs"])
    BaseTrainer(model, train_ld, val_ld, opt, get_loss(), cfg,
                stage_name="face", device=device,
                label_names=cfg["datasets"]["face"]["label_names"],
                scheduler=sched).fit()


# ─────────────────────────────────────────────────────────────────────────────
def train_mri(cfg, device):
    log.info("=" * 50)
    log.info("  Training: MRI Encoder  (Figshare Brain Tumour)")
    log.info("=" * 50)
    from src.datasets.mri_dataset import get_mri_loaders
    from src.models.encoders import build_mri_encoder

    train_ld, val_ld = get_mri_loaders(cfg, seed=cfg["project"]["seed"])
    try:
        weights = train_ld.dataset.class_weights
    except AttributeError:
        weights = None

    model = build_mri_encoder(cfg)
    opt   = torch.optim.AdamW(model.parameters(),
                               lr=cfg["training"]["mri"]["lr"],
                               weight_decay=cfg["training"]["mri"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["training"]["mri"]["epochs"])
    BaseTrainer(model, train_ld, val_ld, opt, get_loss(weights, device), cfg,
                stage_name="mri", device=device,
                label_names=cfg["datasets"]["mri"]["label_names"],
                scheduler=sched).fit()


# ─────────────────────────────────────────────────────────────────────────────
def train_fusion(cfg, device):
    """
    Fusion training:
      - MRI encoder frozen (primary tumour signal)
      - EEG emotion encoder frozen if checkpoint exists (real affect signal)
      - Face encoder frozen if checkpoint exists
      - Fusion head trainable with random modality masking
    """
    log.info("=" * 50)
    log.info("  Training: Fusion Head")
    log.info("  Strategy: MRI primary + EEG emotion + random masking")
    log.info("=" * 50)

    import random, time, mlflow
    from src.datasets.mri_dataset import get_mri_loaders
    from src.models.encoders import build_mri_encoder, build_face_encoder
    from src.models.eeg_emotion_encoder import build_eeg_emotion_encoder
    from src.models.fusion import build_fusion_model
    from src.utils import load_checkpoint, save_checkpoint

    ckpt_dir  = Path(cfg["paths"]["checkpoint_dir"])
    mask_prob = cfg["training"]["fusion"]["missing_modality_prob"]

    def _frozen(model, stage):
        p = ckpt_dir / stage / "best.pt"
        if p.exists():
            load_checkpoint(p, model, device=device)
            for param in model.parameters():
                param.requires_grad = False
            log.info(f"  Loaded frozen {stage} encoder.")
        else:
            log.warning(f"  {stage} checkpoint missing — random weights (still frozen).")
        model.eval()
        return model

    mri_enc       = _frozen(build_mri_encoder(cfg).to(device),       "mri")
    face_enc      = _frozen(build_face_encoder(cfg).to(device),      "face")
    eeg_emo_enc   = _frozen(build_eeg_emotion_encoder(cfg).to(device), "eeg_emotion")

    fusion    = build_fusion_model(cfg).to(device)
    opt       = torch.optim.AdamW(fusion.parameters(),
                                   lr=cfg["training"]["fusion"]["lr"],
                                   weight_decay=cfg["training"]["fusion"]["weight_decay"])
    sched     = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["training"]["fusion"]["epochs"])
    loss_fn   = nn.CrossEntropyLoss()

    train_ld, val_ld = get_mri_loaders(cfg, seed=cfg["project"]["seed"])

    best_acc  = 0.0
    ckpt_path = ckpt_dir / "fusion" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    epochs    = cfg["training"]["fusion"]["epochs"]
    embed_dim = cfg["models"]["embed_dim"]

    mlflow.set_experiment("fusion")
    with mlflow.start_run(run_name="fusion_run"):
        for epoch in range(1, epochs + 1):
            fusion.train()
            t0 = time.time()
            total_loss = correct = total = 0

            for mri_imgs, labels in train_ld:
                mri_imgs = mri_imgs.to(device)
                labels   = labels.to(device)

                with torch.no_grad():
                    mri_emb, _ = mri_enc(mri_imgs)

                    # Randomly mask face and eeg_emotion with mask_prob each
                    face_emb = (
                        None if random.random() < mask_prob
                        else torch.zeros(mri_emb.size(0), embed_dim, device=device)
                    )
                    eeg_emo_emb = (
                        None if random.random() < mask_prob
                        else torch.zeros(mri_emb.size(0), embed_dim, device=device)
                    )

                out  = fusion(mri_emb=mri_emb,
                              face_emb=face_emb,
                              eeg_emotion_emb=eeg_emo_emb)
                loss = loss_fn(out["tumor_logits"], labels)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
                opt.step()

                total_loss += loss.item()
                correct    += (out["tumor_pred"] == labels).sum().item()
                total      += labels.size(0)

            sched.step()

            # Validation
            fusion.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for mri_imgs, labels in val_ld:
                    mri_imgs = mri_imgs.to(device)
                    labels   = labels.to(device)
                    mri_emb, _ = mri_enc(mri_imgs)
                    out = fusion(mri_emb=mri_emb)
                    val_correct += (out["tumor_pred"] == labels).sum().item()
                    val_total   += labels.size(0)

            val_acc   = val_correct / max(1, val_total)
            train_acc = correct / max(1, total)
            log.info(f"Fusion Ep {epoch:03d}/{epochs}  "
                     f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                     f"({time.time()-t0:.1f}s)")
            mlflow.log_metrics({"train_acc": train_acc, "val_acc": val_acc},
                               step=epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(fusion, opt, epoch, val_acc, ckpt_path)
                log.info(f"  ✓ Saved fusion checkpoint (val_acc={val_acc:.4f})")

    log.info(f"Fusion training complete. Best val_acc = {best_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["eeg", "eeg_emotion", "fmri", "face",
                                 "mri", "fusion", "all"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)
    log.info(f"Device: {device}")

    stages = (["eeg", "eeg_emotion", "fmri", "face", "mri", "fusion"]
              if args.stage == "all" else [args.stage])

    runners = {
        "eeg":         train_eeg,
        "eeg_emotion": train_eeg_emotion,
        "fmri":        train_fmri,
        "face":        train_face,
        "mri":         train_mri,
        "fusion":      train_fusion,
    }

    for s in stages:
        runners[s](cfg, device)


if __name__ == "__main__":
    main()