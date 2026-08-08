#!/usr/bin/env python3
"""
Main training script for IMUSA.

Usage:
    python scripts/train.py --config configs/text_only.yaml
    python scripts/train.py --config configs/multimodal.yaml
    python scripts/train.py --config configs/image_only.yaml

Options:
    --config       Path to run-specific config YAML
    --data-dir     Override data directory
    --output-dir   Override output directory
    --seed         Override random seed
    --epochs       Override number of epochs
    --batch-size   Override batch size
    --lr           Override learning rate
    --no-fp16      Disable mixed precision
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.utils.config import load_config, set_seed
from src.utils.helpers import setup_logging
from src.data.dataset import load_data, create_train_val_split, IMUSADataset
from src.models.classifier import build_model
from src.training.trainer import Trainer
from src.evaluation.metrics import print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="IMUSA Training Script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to run-specific config YAML",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load Configuration ───────────────────────────────
    config = load_config(args.config)

    # Apply CLI overrides
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.output_dir:
        config["output"]["output_dir"] = args.output_dir
    if args.seed:
        config["training"]["seed"] = args.seed
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.no_fp16:
        config["training"]["fp16"] = False

    # ── Setup ────────────────────────────────────────────
    logger = setup_logging(config["output"].get("log_dir", "outputs/logs"))
    set_seed(config["training"]["seed"])

    run_name = config.get("run", {}).get("name", "unnamed")
    logger.info(f"=" * 60)
    logger.info(f"IMUSA Training — Run: {run_name}")
    logger.info(f"=" * 60)

    # ── Data ─────────────────────────────────────────────
    data_cfg = config["data"]
    data_dir = data_cfg["data_dir"]
    mode = config["model"]["type"]

    # Check if train/val split already exists
    train_csv = os.path.join(data_cfg["processed_dir"], "train.csv")
    val_csv = os.path.join(data_cfg["processed_dir"], "val.csv")

    if not os.path.exists(train_csv):
        # Look for the raw training data CSV
        raw_csv = os.path.join(data_cfg["raw_dir"], "train", "labels.csv")
        if not os.path.exists(raw_csv):
            logger.error(
                f"Training data not found at {raw_csv}. "
                "Please download the dataset and place it in data/raw/train/"
            )
            sys.exit(1)

        logger.info("Creating train/val split...")
        train_csv, val_csv = create_train_val_split(
            raw_csv,
            data_cfg["processed_dir"],
            val_ratio=data_cfg["val_split"],
            seed=config["training"]["seed"],
        )

    # Build datasets
    tokenizer_name = config["model"].get("text_encoder", {}).get("name", "google/muril-base-cased")

    train_dataset = load_data(
        csv_path=train_csv,
        data_dir=data_dir,
        mode=mode,
        tokenizer_name=tokenizer_name,
        max_seq_length=data_cfg["max_seq_length"],
        image_size=data_cfg["image_size"],
        is_test=False,
    )

    val_dataset = load_data(
        csv_path=val_csv,
        data_dir=data_dir,
        mode=mode,
        tokenizer_name=tokenizer_name,
        max_seq_length=data_cfg["max_seq_length"],
        image_size=data_cfg["image_size"],
        is_test=False,
    )

    # DataLoaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    logger.info(
        f"Data: {len(train_dataset)} train, {len(val_dataset)} val, "
        f"batch_size={batch_size}"
    )
    logger.info(f"Label distribution (train): {train_dataset.get_label_distribution()}")

    # ── Model ────────────────────────────────────────────
    model = build_model(config)
    logger.info(f"Model type: {mode}")

    # ── Class Weights ────────────────────────────────────
    class_weights = train_dataset.get_class_weights()

    # ── Train ────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        class_weights=class_weights,
    )

    history = trainer.train()

    # ── Final Results ────────────────────────────────────
    if history["val_metrics"]:
        best_idx = max(range(len(history["val_metrics"])), key=lambda i: history["val_metrics"][i]["f1"])
        best_metrics = history["val_metrics"][best_idx]
        logger.info(f"\nBest validation results (epoch {best_idx + 1}):")
        print_metrics(best_metrics)

    logger.info(f"\nTraining complete! Checkpoints saved to: {trainer.checkpoint_dir}")


if __name__ == "__main__":
    main()
