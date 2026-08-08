#!/usr/bin/env python3
"""
Ensemble script for IMUSA (Run 3).

Combines predictions from multiple models using:
  - Soft voting (average probabilities)
  - Weighted voting (optimized weights)
  - Stacking (meta-classifier on validation predictions)

Usage:
    python scripts/ensemble.py \
        --checkpoints outputs/checkpoints/run1.pt outputs/checkpoints/run2.pt outputs/checkpoints/image_only.pt \
        --test-csv data/raw/test/labels.csv \
        --val-csv data/processed/val.csv \
        --output submissions/run3_ensemble.csv \
        --method weighted
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from itertools import product

from src.utils.config import set_seed
from src.utils.helpers import setup_logging
from src.data.dataset import load_data, ID2LABEL
from src.models.classifier import build_model
from src.evaluation.metrics import compute_metrics, print_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="IMUSA Ensemble Script")
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model checkpoints",
    )
    parser.add_argument("--test-csv", type=str, required=True)
    parser.add_argument("--val-csv", type=str, default=None, help="Val CSV for weight optimization")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="submissions/run3_ensemble.csv")
    parser.add_argument(
        "--method",
        type=str,
        default="weighted",
        choices=["soft", "weighted", "stacking"],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def get_model_predictions(
    checkpoint_path: str,
    data_loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run inference and return probability matrix (N, C)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Predicting ({os.path.basename(checkpoint_path)})", leave=False):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)

    return np.array(all_probs)


def optimize_weights(
    model_probs: list[np.ndarray],
    true_labels: np.ndarray,
    step: float = 0.1,
) -> tuple[list[float], float]:
    """
    Grid search for optimal ensemble weights on validation set.

    Returns:
        (best_weights, best_f1)
    """
    n_models = len(model_probs)
    best_f1 = 0.0
    best_weights = [1.0 / n_models] * n_models

    # Generate weight combinations that sum to 1
    weight_values = np.arange(0, 1 + step, step)
    weight_combos = [
        combo
        for combo in product(weight_values, repeat=n_models)
        if abs(sum(combo) - 1.0) < 1e-6
    ]

    print(f"Searching {len(weight_combos)} weight combinations...")

    for weights in weight_combos:
        # Weighted average of probabilities
        ensemble_probs = sum(w * p for w, p in zip(weights, model_probs))
        preds = ensemble_probs.argmax(axis=1)
        metrics = compute_metrics(true_labels, preds)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_weights = list(weights)

    return best_weights, best_f1


def main():
    args = parse_args()
    logger = setup_logging()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_models = len(args.checkpoints)
    logger.info(f"Ensemble with {n_models} models, method={args.method}")

    # ── Get predictions from all models ──────────────────
    # We need to create data loaders compatible with each model
    # For simplicity, we'll use the first model's config for data loading
    # (in practice, each model may need its own DataLoader if modes differ)

    test_probs_list = []
    val_probs_list = []

    for ckpt_path in args.checkpoints:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        config = checkpoint["config"]
        mode = config["model"]["type"]
        data_cfg = config.get("data", {})
        tokenizer_name = config["model"].get("text_encoder", {}).get(
            "name", "google/muril-base-cased"
        )

        # Test loader
        test_dataset = load_data(
            csv_path=args.test_csv,
            data_dir=args.data_dir,
            mode=mode,
            tokenizer_name=tokenizer_name,
            max_seq_length=data_cfg.get("max_seq_length", 128),
            image_size=data_cfg.get("image_size", 224),
            is_test=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        test_probs = get_model_predictions(ckpt_path, test_loader, device)
        test_probs_list.append(test_probs)
        logger.info(f"  {os.path.basename(ckpt_path)}: {test_probs.shape}")

        # Val loader (for weight optimization)
        if args.val_csv and args.method == "weighted":
            val_dataset = load_data(
                csv_path=args.val_csv,
                data_dir=args.data_dir,
                mode=mode,
                tokenizer_name=tokenizer_name,
                max_seq_length=data_cfg.get("max_seq_length", 128),
                image_size=data_cfg.get("image_size", 224),
                is_test=False,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
            )
            val_probs = get_model_predictions(ckpt_path, val_loader, device)
            val_probs_list.append(val_probs)

    # ── Ensemble ─────────────────────────────────────────
    if args.method == "soft":
        # Simple average
        weights = [1.0 / n_models] * n_models
        logger.info(f"Soft voting with equal weights: {weights}")

    elif args.method == "weighted":
        if val_probs_list and args.val_csv:
            # Optimize weights on validation set
            val_df = pd.read_csv(args.val_csv)
            from src.data.dataset import LABEL2ID

            val_labels = val_df["label"].map(LABEL2ID).values
            weights, best_f1 = optimize_weights(val_probs_list, val_labels)
            logger.info(f"Optimized weights: {weights} (val F1={best_f1:.4f})")
        else:
            # Default weights: higher for multimodal
            weights = [0.3, 0.5, 0.2][:n_models]
            total = sum(weights)
            weights = [w / total for w in weights]
            logger.info(f"Default weights: {weights}")

    elif args.method == "stacking":
        logger.warning("Stacking not yet implemented, falling back to weighted")
        weights = [1.0 / n_models] * n_models

    # Apply weights
    ensemble_probs = sum(w * p for w, p in zip(weights, test_probs_list))
    final_preds = ensemble_probs.argmax(axis=1)

    # ── Save Results ─────────────────────────────────────
    test_df = pd.read_csv(args.test_csv)
    id_col = "image_id" if "image_id" in test_df.columns else "image_path"

    results = pd.DataFrame(
        {
            id_col: test_df[id_col].values[: len(final_preds)],
            "predicted_label": [ID2LABEL[p] for p in final_preds],
        }
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)

    logger.info(f"Saved {len(results)} ensemble predictions to {args.output}")
    logger.info(f"Distribution:\n{results['predicted_label'].value_counts()}")

    # Evaluate on val if available
    if val_probs_list and args.val_csv:
        val_df = pd.read_csv(args.val_csv)
        from src.data.dataset import LABEL2ID

        val_labels = val_df["label"].map(LABEL2ID).values
        val_ensemble = sum(w * p for w, p in zip(weights, val_probs_list))
        val_preds = val_ensemble.argmax(axis=1)
        metrics = compute_metrics(val_labels, val_preds)
        print_metrics(metrics)


if __name__ == "__main__":
    main()
