#!/usr/bin/env python3
"""
Prediction / submission generation script for IMUSA.

Usage:
    python scripts/predict.py \
        --checkpoint outputs/checkpoints/run2_multimodal_epoch8_f10.7523.pt \
        --test-csv data/raw/test/labels.csv \
        --output submissions/run2_predictions.csv
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

from src.utils.config import set_seed
from src.utils.helpers import setup_logging
from src.data.dataset import load_data, ID2LABEL
from src.models.classifier import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="IMUSA Prediction Script")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--data-dir", type=str, default="data", help="Data root directory")
    parser.add_argument("--output", type=str, default="submissions/predictions.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-probs", action="store_true", help="Save class probabilities")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()
    set_seed(42)

    # ── Load Checkpoint ──────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    logger.info(f"  Epoch: {checkpoint['epoch']}, F1: {checkpoint['f1']:.4f}")

    # ── Build Model ──────────────────────────────────────
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ── Load Test Data ───────────────────────────────────
    mode = config["model"]["type"]
    data_cfg = config.get("data", {})
    tokenizer_name = config["model"].get("text_encoder", {}).get("name", "google/muril-base-cased")

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
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Test set: {len(test_dataset)} samples")

    # ── Inference ────────────────────────────────────────
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with autocast(enabled=torch.cuda.is_available()):
                logits = model(batch)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)

    # ── Save Predictions ─────────────────────────────────
    test_df = pd.read_csv(args.test_csv)

    # Use image_id or image_path as identifier
    id_col = "image_id" if "image_id" in test_df.columns else "image_path"

    results = pd.DataFrame(
        {
            id_col: test_df[id_col].values[: len(all_preds)],
            "predicted_label": [ID2LABEL[p] for p in all_preds],
        }
    )

    if args.save_probs:
        probs_array = np.array(all_probs)
        for i, name in ID2LABEL.items():
            results[f"prob_{name}"] = probs_array[:, i]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results.to_csv(args.output, index=False)

    logger.info(f"Saved {len(results)} predictions to {args.output}")

    # Print prediction distribution
    dist = results["predicted_label"].value_counts()
    logger.info(f"Prediction distribution:\n{dist}")


if __name__ == "__main__":
    main()
