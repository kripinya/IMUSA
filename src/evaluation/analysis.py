"""
Error analysis and visualization for IMUSA.

Generates:
  - Confusion matrix heatmaps
  - Per-class accuracy bar charts
  - Misclassification galleries
  - Modality contribution analysis
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import ID2LABEL

logger = logging.getLogger("imusa.analysis")

CLASS_NAMES = [ID2LABEL[i] for i in range(4)]
CLASS_EMOJIS = {"Sarcasm": "😏", "Neutral": "😐", "Offensive": "⚠️", "Motivational": "💪"}


def plot_confusion_matrix(
    cm: np.ndarray | list,
    save_path: str = "outputs/confusion_matrix.png",
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """Plot and save a confusion matrix heatmap."""
    cm = np.array(cm)

    if normalize:
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None,
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add raw counts as secondary annotations
    if normalize:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j + 0.5,
                    i + 0.72,
                    f"(n={cm[i, j]})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="gray",
                )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_per_class_metrics(
    metrics: dict,
    save_path: str = "outputs/per_class_metrics.png",
):
    """Plot per-class precision, recall, and F1 as grouped bars."""
    per_class = metrics["per_class"]

    classes = list(per_class.keys())
    precision = [per_class[c]["precision"] for c in classes]
    recall = [per_class[c]["recall"] for c in classes]
    f1 = [per_class[c]["f1"] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#3b82f6")
    bars2 = ax.bar(x, recall, width, label="Recall", color="#f97316")
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#22c55e")

    ax.set_xlabel("Sentiment Category", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{CLASS_EMOJIS.get(c, '')} {c}" for c in classes])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved per-class metrics to {save_path}")


def find_misclassified(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_per_pair: int = 5,
) -> pd.DataFrame:
    """
    Find misclassified samples grouped by (true, predicted) pairs.

    Returns a DataFrame with columns:
        image_path, text, true_label, predicted_label
    """
    misclassified = []
    pair_counts = {}

    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            pair = (ID2LABEL[y_true[i]], ID2LABEL[y_pred[i]])
            pair_counts.setdefault(pair, 0)

            if pair_counts[pair] < max_per_pair:
                misclassified.append(
                    {
                        "index": i,
                        "image_path": df.iloc[i].get("image_path", ""),
                        "text": df.iloc[i].get("text", ""),
                        "true_label": pair[0],
                        "predicted_label": pair[1],
                    }
                )
                pair_counts[pair] += 1

    result = pd.DataFrame(misclassified)
    logger.info(f"Found {len(result)} misclassified examples across {len(pair_counts)} confusion pairs")
    return result


def compare_runs(
    run_metrics: dict[str, dict],
    save_path: str = "outputs/run_comparison.png",
):
    """
    Compare multiple runs side by side.

    Args:
        run_metrics: {run_name: metrics_dict}
    """
    run_names = list(run_metrics.keys())
    metric_names = ["accuracy", "precision", "recall", "f1"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metric_names))
    width = 0.8 / len(run_names)
    colors = ["#3b82f6", "#f97316", "#22c55e", "#a855f7"]

    for i, (name, metrics) in enumerate(run_metrics.items()):
        values = [metrics[m] for m in metric_names]
        offset = (i - len(run_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=colors[i % len(colors)])

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Run Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metric_names])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved run comparison to {save_path}")
