"""
Loss functions for IMUSA sentiment classification.

Provides class-imbalance-aware losses:
  - FocalLoss — down-weights well-classified examples
  - LabelSmoothingCE — prevents overconfident predictions
  - WeightedCE — standard cross-entropy with class weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for handling class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma > 0 reduces the loss for well-classified examples,
      focusing training on hard, misclassified samples.
    - alpha provides per-class weighting.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) raw model output
            targets: (batch,) integer class labels
        """
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # p_t = probability of correct class

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing to prevent overconfident predictions.

    Smooths the target distribution:
        y_smooth = (1 - ε) * y_onehot + ε / K
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / num_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / num_classes)

        loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.weight is not None:
            weight_t = self.weight.to(logits.device)[targets]
            loss = loss * weight_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss_fn(config: dict, class_weights: torch.Tensor | None = None) -> nn.Module:
    """Factory function to build loss from config."""
    loss_cfg = config.get("loss", {})
    loss_name = loss_cfg.get("name", "focal")

    if loss_name == "focal":
        gamma = loss_cfg.get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma, alpha=class_weights)

    elif loss_name == "label_smoothing":
        smoothing = loss_cfg.get("label_smoothing", 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)

    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)

    else:
        raise ValueError(f"Unknown loss: {loss_name}")
