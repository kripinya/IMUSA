"""
Full classifier models for IMUSA sentiment classification.

Three model types:
  1. TextOnlyClassifier   — Run 1 baseline
  2. ImageOnlyClassifier  — Ablation / ensemble component
  3. MultimodalClassifier — Run 2 (CLIP + MuRIL + fusion)
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from src.models.text_encoder import TextEncoder
from src.models.image_encoder import CLIPImageEncoder, build_image_encoder
from src.models.fusion import build_fusion_module

logger = logging.getLogger("imusa.classifier")


class ClassifierHead(nn.Module):
    """Shared classifier head with configurable hidden layers."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 4,
        hidden_dims: list[int] = [256],
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hdim),
                    nn.LayerNorm(hdim),
                    act_fn(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class TextOnlyClassifier(nn.Module):
    """
    Run 1: Text-only sentiment classifier.

    Architecture:
        Text → MuRIL → [CLS] → Classifier Head → 4 classes
    """

    def __init__(self, config: dict):
        super().__init__()
        text_cfg = config["model"]["text_encoder"]
        cls_cfg = config["model"]["classifier"]
        num_classes = config["project"]["num_classes"]

        self.text_encoder = TextEncoder(
            model_name=text_cfg["name"],
            pooling=text_cfg.get("pooling", "cls"),
            freeze_layers=text_cfg.get("freeze_layers", 0),
        )

        self.classifier = ClassifierHead(
            input_dim=self.text_encoder.get_output_dim(),
            num_classes=num_classes,
            hidden_dims=cls_cfg.get("hidden_dims", [256]),
            dropout=cls_cfg.get("dropout", 0.3),
            activation=cls_cfg.get("activation", "relu"),
        )

        logger.info(f"TextOnlyClassifier: {text_cfg['name']} → {num_classes} classes")

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys: input_ids, attention_mask, [token_type_ids]

        Returns:
            (batch, num_classes) logits
        """
        text_feat = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        return self.classifier(text_feat)


class ImageOnlyClassifier(nn.Module):
    """
    Image-only sentiment classifier (ablation / ensemble component).

    Architecture:
        Image → CLIP ViT → Image Embedding → Classifier Head → 4 classes
    """

    def __init__(self, config: dict):
        super().__init__()
        image_cfg = config["model"]["image_encoder"]
        cls_cfg = config["model"]["classifier"]
        num_classes = config["project"]["num_classes"]

        self.image_encoder = build_image_encoder(image_cfg)

        self.classifier = ClassifierHead(
            input_dim=self.image_encoder.get_output_dim(),
            num_classes=num_classes,
            hidden_dims=cls_cfg.get("hidden_dims", [256]),
            dropout=cls_cfg.get("dropout", 0.3),
            activation=cls_cfg.get("activation", "relu"),
        )

        logger.info(
            f"ImageOnlyClassifier: {image_cfg['name']} → {num_classes} classes"
        )

    def forward(self, batch: dict) -> torch.Tensor:
        image_feat = self.image_encoder(batch["image"])
        return self.classifier(image_feat)


class MultimodalClassifier(nn.Module):
    """
    Run 2: Multimodal sentiment classifier with cross-modal fusion.

    Architecture:
        Text  → MuRIL    → text_embedding ──┐
                                              ├→ Fusion → Classifier → 4 classes
        Image → CLIP ViT → image_embedding ─┘
    """

    def __init__(self, config: dict):
        super().__init__()
        text_cfg = config["model"]["text_encoder"]
        image_cfg = config["model"]["image_encoder"]
        fusion_cfg = config["model"]["fusion"]
        cls_cfg = config["model"]["classifier"]
        num_classes = config["project"]["num_classes"]

        # Encoders
        self.text_encoder = TextEncoder(
            model_name=text_cfg["name"],
            pooling=text_cfg.get("pooling", "cls"),
            freeze_layers=text_cfg.get("freeze_layers", 0),
        )

        self.image_encoder = build_image_encoder(image_cfg)

        # Fusion module
        self.fusion = build_fusion_module(
            config=fusion_cfg,
            text_dim=self.text_encoder.get_output_dim(),
            image_dim=self.image_encoder.get_output_dim(),
        )

        # Classifier
        fusion_output_dim = fusion_cfg.get("projection_dim", 256)
        self.classifier = ClassifierHead(
            input_dim=fusion_output_dim,
            num_classes=num_classes,
            hidden_dims=cls_cfg.get("hidden_dims", [256, 128]),
            dropout=cls_cfg.get("dropout", 0.3),
            activation=cls_cfg.get("activation", "gelu"),
        )

        logger.info(
            f"MultimodalClassifier: {text_cfg['name']} + {image_cfg['name']} "
            f"→ {fusion_cfg['type']} fusion → {num_classes} classes"
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Args:
            batch: dict with keys:
                - input_ids, attention_mask, [token_type_ids] (text)
                - image (pixel_values)

        Returns:
            (batch, num_classes) logits
        """
        # Encode modalities
        text_feat = self.text_encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        image_feat = self.image_encoder(batch["image"])

        # Fuse
        fused = self.fusion(text_feat, image_feat)

        # Classify
        return self.classifier(fused)


def build_model(config: dict) -> nn.Module:
    """Factory function to build the appropriate model from config."""
    model_type = config["model"]["type"]

    if model_type == "text_only":
        return TextOnlyClassifier(config)
    elif model_type == "image_only":
        return ImageOnlyClassifier(config)
    elif model_type == "multimodal":
        return MultimodalClassifier(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
