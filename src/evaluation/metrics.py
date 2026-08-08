"""
Evaluation metrics for IMUSA sentiment classification.

Computes:
  - Accuracy (overall correctness)
  - Precision (per-class and macro)
  - Recall (per-class and macro)
  - F1-Score (per-class and macro)
  - Confusion matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.data.dataset import ID2LABEL

CLASS_NAMES = [ID2LABEL[i] for i in range(4)]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Ground truth labels (integer-encoded)
        y_pred: Predicted labels (integer-encoded)

    Returns:
        dict with accuracy, precision, recall, f1, per-class metrics,
        classification report, and confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)

    # Macro-averaged metrics (treats all classes equally)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Weighted-averaged metrics (weighted by class support)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(4)), zero_division=0
    )

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision_per[i]),
            "recall": float(recall_per[i]),
            "f1": float(f1_per[i]),
            "support": int(support_per[i]),
        }

    # Classification report (pretty-printed)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))

    return {
        "accuracy": float(accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class,
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


def print_metrics(metrics: dict):
    """Pretty-print evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Macro Precision:    {metrics['precision']:.4f}")
    print(f"  Macro Recall:       {metrics['recall']:.4f}")
    print(f"  Macro F1-Score:     {metrics['f1']:.4f}")
    print(f"  Weighted F1-Score:  {metrics['f1_weighted']:.4f}")
    print("-" * 60)
    print("\nPer-Class Breakdown:")
    print(f"  {'Class':<15} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("  " + "-" * 47)
    for name, vals in metrics["per_class"].items():
        print(
            f"  {name:<15} {vals['precision']:>8.4f} {vals['recall']:>8.4f} "
            f"{vals['f1']:>8.4f} {vals['support']:>8d}"
        )
    print("-" * 60)
    print("\nFull Classification Report:")
    print(metrics["report"])
