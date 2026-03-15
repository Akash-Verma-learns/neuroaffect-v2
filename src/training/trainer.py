"""
BaseTrainer  —  shared train/eval loop used by all training scripts.
"""

import time
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import mlflow

from src.utils import get_logger, save_checkpoint, compute_metrics

log = get_logger("trainer")


class BaseTrainer:
    """
    Handles the full training lifecycle:
        - epoch loop with tqdm progress bars
        - validation after every epoch
        - best-model checkpointing (by val accuracy)
        - cosine LR scheduling
        - MLflow metric logging
        - early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        cfg: dict,
        stage_name: str,             # e.g. "eeg", "mri"
        device: torch.device,
        label_names: Optional[list] = None,
        scheduler=None,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.loss_fn      = loss_fn
        self.device       = device
        self.label_names  = label_names
        self.scheduler    = scheduler
        self.stage_name   = stage_name

        tcfg                = cfg["training"][stage_name]
        self.epochs         = tcfg["epochs"]
        self.ckpt_dir       = Path(cfg["paths"]["checkpoint_dir"]) / stage_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt      = self.ckpt_dir / "best.pt"

        self.best_val_acc   = 0.0
        self.patience       = 10
        self.patience_count = 0

    # ── One epoch ─────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> Dict:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader,
                    desc=f"[{self.stage_name}] Epoch {epoch} train",
                    leave=False)

        for batch in pbar:
            loss, n_correct, n = self._step(batch, train=True)
            total_loss += loss
            correct    += n_correct
            total      += n
            pbar.set_postfix(loss=f"{loss:.3f}")

        return {"loss": total_loss / len(self.train_loader),
                "acc":  correct / max(1, total)}

    @torch.no_grad()
    def _val_epoch(self) -> Dict:
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        for batch in tqdm(self.val_loader,
                          desc=f"[{self.stage_name}] val", leave=False):
            loss, _, _ = self._step(batch, train=False)
            total_loss += loss

            # Collect predictions for metrics
            inputs, labels = self._unpack_batch(batch)
            with torch.enable_grad():
                pass
            # Re-run without grad for preds
            out    = self._forward(inputs)
            preds  = out.argmax(-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

        metrics = compute_metrics(all_labels, all_preds, self.label_names)
        metrics["loss"] = total_loss / len(self.val_loader)
        return metrics

    # ── Subclass interface ─────────────────────────────────────────────────

    def _unpack_batch(self, batch):
        """Override if batch has a non-(X, y) structure."""
        x, y = batch
        return x.to(self.device), y.to(self.device)

    def _forward(self, inputs) -> torch.Tensor:
        """Returns logits tensor. Override for multi-head models."""
        _, logits = self.model(inputs)
        return logits

    def _step(self, batch, train: bool):
        inputs, labels = self._unpack_batch(batch)
        logits = self._forward(inputs)
        loss   = self.loss_fn(logits, labels)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

        preds     = logits.argmax(-1)
        n_correct = (preds == labels).sum().item()
        return loss.item(), n_correct, labels.size(0)

    # ── Full training loop ─────────────────────────────────────────────────

    def fit(self):
        mlflow.set_experiment(self.stage_name)
        with mlflow.start_run(run_name=f"{self.stage_name}_run"):
            mlflow.log_params({
                "stage": self.stage_name,
                "epochs": self.epochs,
            })

            for epoch in range(1, self.epochs + 1):
                t0 = time.time()

                train_m = self._train_epoch(epoch)
                val_m   = self._val_epoch()

                if self.scheduler:
                    self.scheduler.step()

                elapsed = time.time() - t0
                log.info(
                    f"[{self.stage_name}] Ep {epoch:03d}/{self.epochs}  "
                    f"train_loss={train_m['loss']:.4f}  "
                    f"val_acc={val_m['accuracy']:.4f}  "
                    f"val_f1={val_m['f1_weighted']:.4f}  "
                    f"({elapsed:.1f}s)"
                )

                mlflow.log_metrics({
                    "train_loss": train_m["loss"],
                    "train_acc":  train_m["acc"],
                    "val_loss":   val_m["loss"],
                    "val_acc":    val_m["accuracy"],
                    "val_f1":     val_m["f1_weighted"],
                }, step=epoch)

                # Checkpoint best model
                if val_m["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_m["accuracy"]
                    self.patience_count = 0
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        val_m["accuracy"], self.best_ckpt)
                    log.info(f"  ✓ Saved best model  (val_acc={val_m['accuracy']:.4f})")
                    mlflow.log_artifact(str(self.best_ckpt))
                else:
                    self.patience_count += 1
                    if self.patience_count >= self.patience:
                        log.info(f"Early stopping at epoch {epoch}.")
                        break

            log.info(f"Training complete. Best val_acc = {self.best_val_acc:.4f}")
            log.info(f"Best checkpoint: {self.best_ckpt}")
