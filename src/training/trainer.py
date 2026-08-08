"""
Training loop for IMUSA sentiment classifier.

Supports:
  - Mixed precision (fp16)
  - Gradient accumulation
  - Early stopping
  - Learning rate scheduling
  - Differential learning rates (different LR for encoder vs. classifier)
  - Checkpoint saving (top-k by F1)
  - Wandb logging (optional)
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from src.training.losses import build_loss_fn
from src.evaluation.metrics import compute_metrics
from src.utils.helpers import count_parameters, save_json

logger = logging.getLogger("imusa.trainer")


class Trainer:
    """
    Training orchestrator for IMUSA models.

    Handles the full training loop including validation,
    checkpointing, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Training config
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 10)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.patience = train_cfg.get("early_stopping_patience", 3)
        self.fp16 = train_cfg.get("fp16", True) and torch.cuda.is_available()
        self.grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 1)

        # Device
        self.device = self._get_device()
        self.model = self.model.to(self.device)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        # Loss function
        self.criterion = build_loss_fn(config, class_weights)

        # Optimizer with optional differential learning rates
        self.optimizer = self._build_optimizer()

        # Scheduler
        total_steps = len(train_loader) * self.epochs // self.grad_accum_steps
        self.scheduler = self._build_scheduler(total_steps)

        # Mixed precision
        self.scaler = GradScaler(enabled=self.fp16)

        # Checkpointing
        output_cfg = config.get("output", {})
        self.checkpoint_dir = Path(output_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = output_cfg.get("save_top_k", 3)

        # Wandb
        wandb_cfg = config.get("wandb", {})
        self.use_wandb = wandb_cfg.get("enabled", False)
        if self.use_wandb:
            self._init_wandb(wandb_cfg)

        # State
        self.best_f1 = 0.0
        self.patience_counter = 0
        self.global_step = 0
        self.top_checkpoints = []  # [(f1, path), ...] sorted ascending

        # Log model info
        params = count_parameters(model)
        logger.info(
            f"Model: {params['total_m']} params total, "
            f"{params['trainable_m']} trainable"
        )
        logger.info(f"Device: {self.device}, FP16: {self.fp16}")

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            dict with training history and best metrics
        """
        history = {"train_loss": [], "val_loss": [], "val_metrics": []}

        logger.info(f"Starting training for {self.epochs} epochs")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            # Train
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)

            # Validate
            val_loss, val_metrics = self._validate(epoch)
            history["val_loss"].append(val_loss)
            history["val_metrics"].append(val_metrics)

            # Log
            logger.info(
                f"Epoch {epoch}/{self.epochs} — "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            if self.use_wandb:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))},
                    }
                )

            # Checkpointing
            current_f1 = val_metrics["f1"]
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self._save_checkpoint(epoch, current_f1)
                self.patience_counter = 0
                logger.info(f"  ✓ New best F1: {current_f1:.4f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"  ✗ No improvement ({self.patience_counter}/{self.patience})"
                )

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed / 60:.1f} min. Best F1: {self.best_f1:.4f}")

        # Save history
        run_name = config.get("run", {}).get("name", "unnamed")
        save_json(history, str(self.checkpoint_dir / f"{run_name}_history.json"))

        return history

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )

        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            batch = self._to_device(batch)

            with autocast(enabled=self.fp16):
                logits = self.model(batch)
                loss = self.criterion(logits, batch["label"])
                loss = loss / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

                self.global_step += 1

            total_loss += loss.item() * self.grad_accum_steps
            num_batches += 1
            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, epoch: int) -> tuple[float, dict]:
        """Run validation and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
            batch = self._to_device(batch)

            with autocast(enabled=self.fp16):
                logits = self.model(batch)
                loss = self.criterion(logits, batch["label"])

            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1).cpu()
            labels = batch["label"].cpu()
            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        val_loss = total_loss / max(num_batches, 1)
        metrics = compute_metrics(all_labels, all_preds)

        return val_loss, metrics

    def _save_checkpoint(self, epoch: int, f1: float):
        """Save model checkpoint, maintaining only top-k by F1."""
        run_name = self.config.get("run", {}).get("name", "model")
        filename = f"{run_name}_epoch{epoch}_f1{f1:.4f}.pt"
        filepath = self.checkpoint_dir / filename

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "f1": f1,
                "config": self.config,
            },
            filepath,
        )

        self.top_checkpoints.append((f1, str(filepath)))
        self.top_checkpoints.sort(key=lambda x: x[0])

        # Remove excess checkpoints
        while len(self.top_checkpoints) > self.save_top_k:
            _, old_path = self.top_checkpoints.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
                logger.debug(f"Removed old checkpoint: {old_path}")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer with optional differential learning rates."""
        train_cfg = self.config.get("training", {})
        opt_cfg = self.config.get("optimizer", {})

        base_lr = train_cfg.get("learning_rate", 2e-5)
        weight_decay = train_cfg.get("weight_decay", 0.01)

        # Check for differential learning rates
        text_lr = train_cfg.get("text_lr")
        image_lr = train_cfg.get("image_lr")
        fusion_lr = train_cfg.get("fusion_lr")

        if text_lr or image_lr or fusion_lr:
            # Group parameters by component
            param_groups = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue

                lr = base_lr
                if "text_encoder" in name and text_lr:
                    lr = text_lr
                elif "image_encoder" in name and image_lr:
                    lr = image_lr
                elif "fusion" in name and fusion_lr:
                    lr = fusion_lr

                param_groups.append({"params": [param], "lr": lr})

            logger.info(
                f"Differential LRs: text={text_lr or base_lr}, "
                f"image={image_lr or base_lr}, fusion={fusion_lr or base_lr}"
            )
        else:
            param_groups = [
                {"params": [p for p in self.model.parameters() if p.requires_grad]}
            ]

        return torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
            weight_decay=weight_decay,
        )

    def _build_scheduler(self, total_steps: int):
        """Build learning rate scheduler."""
        sched_cfg = self.config.get("scheduler", {})
        sched_name = sched_cfg.get("name", "cosine")
        warmup_steps = sched_cfg.get("num_warmup_steps", 100)

        from transformers import get_scheduler

        return get_scheduler(
            name=sched_name.replace("_", "_with_") if "restart" in sched_name else sched_name,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _to_device(self, batch: dict) -> dict:
        """Move all tensor values in batch to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _init_wandb(self, wandb_cfg: dict):
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            wandb.init(
                project=wandb_cfg.get("project", "imusa-fire2026"),
                entity=wandb_cfg.get("entity"),
                config=self.config,
                name=self.config.get("run", {}).get("name"),
            )
            logger.info("Wandb initialized")
        except ImportError:
            logger.warning("wandb not installed, disabling")
            self.use_wandb = False

    def load_checkpoint(self, checkpoint_path: str):
        """Load a model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            f"Loaded checkpoint: {checkpoint_path} "
            f"(epoch {checkpoint['epoch']}, F1={checkpoint['f1']:.4f})"
        )
        return checkpoint
