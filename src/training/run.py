"""
Training scripts for all 4 encoders + the fusion head.

Usage:
    python src/training/run.py --stage eeg
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


# ─────────────────────────────────────────────────────────────────────────────
#  Weighted cross-entropy (handles class imbalance)
# ─────────────────────────────────────────────────────────────────────────────
def get_loss(weights=None, device="cpu"):
    if weights is not None:
        return nn.CrossEntropyLoss(weight=weights.to(device))
    return nn.CrossEntropyLoss()


# ─────────────────────────────────────────────────────────────────────────────
def train_eeg(cfg: dict, device: torch.device):
    log.info("=" * 50)
    log.info("  Training: EEG Encoder")
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

    trainer = BaseTrainer(
        model, train_ld, val_ld, opt, get_loss(), cfg,
        stage_name="eeg", device=device,
        label_names=["left-hand", "right-hand"],
        scheduler=sched,
    )
    trainer.fit()


# ─────────────────────────────────────────────────────────────────────────────
def train_fmri(cfg: dict, device: torch.device):
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

    trainer = BaseTrainer(
        model, train_ld, val_ld, opt, get_loss(), cfg,
        stage_name="fmri", device=device,
        label_names=cfg["datasets"]["fmri"]["label_names"]
                   if "label_names" in cfg["datasets"]["fmri"]
                   else None,
        scheduler=sched,
    )
    trainer.fit()


# ─────────────────────────────────────────────────────────────────────────────
def train_face(cfg: dict, device: torch.device):
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

    trainer = BaseTrainer(
        model, train_ld, val_ld, opt, get_loss(), cfg,
        stage_name="face", device=device,
        label_names=cfg["datasets"]["face"]["label_names"],
        scheduler=sched,
    )
    trainer.fit()


# ─────────────────────────────────────────────────────────────────────────────
def train_mri(cfg: dict, device: torch.device):
    log.info("=" * 50)
    log.info("  Training: MRI Encoder  (Figshare Brain Tumour)")
    log.info("=" * 50)
    from src.datasets.mri_dataset import get_mri_loaders
    from src.models.encoders import build_mri_encoder

    train_ld, val_ld = get_mri_loaders(cfg, seed=cfg["project"]["seed"])

    # Use class weights from dataset if available
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

    trainer = BaseTrainer(
        model, train_ld, val_ld, opt, get_loss(weights, device), cfg,
        stage_name="mri", device=device,
        label_names=cfg["datasets"]["mri"]["label_names"],
        scheduler=sched,
    )
    trainer.fit()


# ─────────────────────────────────────────────────────────────────────────────
def train_fusion(cfg: dict, device: torch.device):
    """
    Fusion training strategy:
    ─────────────────────────
    We have no dataset where all 4 modalities come from the same subject.
    Instead we train the fusion head using MRI samples exclusively (the
    strongest tumor signal) and randomly zero-mask the other modality slots
    with probability `missing_modality_prob`.

    This teaches the fusion to:
      (a) classify tumors correctly when only MRI is present
      (b) improve predictions when other modalities are added
    """
    log.info("=" * 50)
    log.info("  Training: Fusion Head")
    log.info("  Strategy: MRI primary + random modality masking")
    log.info("=" * 50)

    import random
    from src.datasets.mri_dataset import get_mri_loaders, BrainMRIDataset
    from src.models.encoders import build_mri_encoder, build_face_encoder
    from src.models.fusion import build_fusion_model
    from src.utils import load_checkpoint

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    mask_prob = cfg["training"]["fusion"]["missing_modality_prob"]

    # Load pre-trained MRI encoder (frozen)
    mri_enc = build_mri_encoder(cfg).to(device)
    mri_ckpt = ckpt_dir / "mri" / "best.pt"
    if mri_ckpt.exists():
        load_checkpoint(mri_ckpt, mri_enc, device=device)
        for p in mri_enc.parameters():
            p.requires_grad = False
        log.info("  Loaded frozen MRI encoder.")
    else:
        log.warning("  MRI encoder checkpoint not found. Train MRI first.")

    # Load pre-trained Face encoder (frozen, optional)
    face_enc = build_face_encoder(cfg).to(device)
    face_ckpt = ckpt_dir / "face" / "best.pt"
    if face_ckpt.exists():
        load_checkpoint(face_ckpt, face_enc, device=device)
        for p in face_enc.parameters():
            p.requires_grad = False
        log.info("  Loaded frozen Face encoder.")

    # Fusion model (trainable)
    fusion = build_fusion_model(cfg).to(device)
    opt    = torch.optim.AdamW(fusion.parameters(),
                                lr=cfg["training"]["fusion"]["lr"],
                                weight_decay=cfg["training"]["fusion"]["weight_decay"])
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["training"]["fusion"]["epochs"])
    loss_fn = nn.CrossEntropyLoss()

    train_ld, val_ld = get_mri_loaders(cfg, seed=cfg["project"]["seed"])

    # ── Training loop ──────────────────────────────────────────────────────
    import mlflow, time
    from src.utils import save_checkpoint

    best_acc      = 0.0
    ckpt_path     = ckpt_dir / "fusion" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = cfg["training"]["fusion"]["epochs"]
    mlflow.set_experiment("fusion")

    with mlflow.start_run(run_name="fusion_run"):
        for epoch in range(1, epochs + 1):
            fusion.train(); mri_enc.eval(); face_enc.eval()
            t0 = time.time()
            total_loss, correct, total = 0.0, 0, 0

            for mri_imgs, labels in train_ld:
                mri_imgs = mri_imgs.to(device)
                labels   = labels.to(device)

                with torch.no_grad():
                    mri_emb, _  = mri_enc(mri_imgs)
                    # Optionally include face embedding (zero if mask triggered)
                    face_emb = (
                        None if random.random() < mask_prob
                        else torch.zeros(mri_emb.size(0), cfg["models"]["embed_dim"],
                                         device=device)
                    )

                out  = fusion(mri_emb=mri_emb, face_emb=face_emb)
                loss = loss_fn(out["tumor_logits"], labels)

                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
                opt.step()

                total_loss += loss.item()
                correct    += (out["tumor_pred"] == labels).sum().item()
                total      += labels.size(0)

            sched.step()
            train_acc = correct / max(1, total)

            # Validation
            fusion.eval(); val_correct = val_total = 0
            with torch.no_grad():
                for mri_imgs, labels in val_ld:
                    mri_imgs = mri_imgs.to(device)
                    labels   = labels.to(device)
                    mri_emb, _ = mri_enc(mri_imgs)
                    out = fusion(mri_emb=mri_emb)
                    val_correct += (out["tumor_pred"] == labels).sum().item()
                    val_total   += labels.size(0)

            val_acc = val_correct / max(1, val_total)
            log.info(f"Fusion Ep {epoch:03d}/{epochs}  "
                     f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
                     f"({time.time()-t0:.1f}s)")
            mlflow.log_metrics({"train_acc": train_acc, "val_acc": val_acc}, step=epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(fusion, opt, epoch, val_acc, ckpt_path)
                log.info(f"  ✓ Saved fusion checkpoint (val_acc={val_acc:.4f})")

    log.info(f"Fusion training complete. Best val_acc = {best_acc:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["eeg", "fmri", "face", "mri", "fusion", "all"])
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    device = get_device(cfg)
    log.info(f"Device: {device}")

    stages = ["eeg", "fmri", "face", "mri", "fusion"] if args.stage == "all" \
             else [args.stage]

    runners = {
        "eeg":    train_eeg,
        "fmri":   train_fmri,
        "face":   train_face,
        "mri":    train_mri,
        "fusion": train_fusion,
    }

    for s in stages:
        runners[s](cfg, device)


if __name__ == "__main__":
    main()
