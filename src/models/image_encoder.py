"""
Image encoder wrappers for IMUSA.

Supports:
  - CLIP ViT-B/32  (openai/clip-vit-base-patch32) — best for multimodal
  - ResNet-50       (torchvision)                  — lightweight fallback
  - ViT-B/16        (google/vit-base-patch16-224)  — standalone ViT
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("imusa.image_encoder")


class CLIPImageEncoder(nn.Module):
    """
    CLIP Vision Transformer encoder.

    Extracts image features from CLIP's visual backbone,
    already pretrained to align with text in a shared space.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.model_name = model_name

        # Load CLIP model and extract visual component
        clip_model = CLIPModel.from_pretrained(model_name)
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection

        # CLIP ViT-B/32 outputs 512-d after projection
        self.hidden_size = clip_model.config.projection_dim  # typically 512

        logger.info(
            f"Loaded CLIP image encoder: {model_name} "
            f"(output_dim={self.hidden_size})"
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            logger.info("Froze CLIP vision backbone")

        # Optional additional projection
        self.projection = None
        if output_dim and output_dim != self.hidden_size:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
            self.hidden_size = output_dim

        # Clean up the full CLIP model (we only need the vision part)
        del clip_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Encode images through CLIP vision backbone.

        Args:
            pixel_values: (batch, 3, H, W) normalized images

        Returns:
            (batch, hidden_size) image embeddings
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output  # (batch, vision_hidden)
        image_features = self.visual_projection(pooled_output)  # (batch, proj_dim)

        if self.projection is not None:
            image_features = self.projection(image_features)

        return image_features

    def get_output_dim(self) -> int:
        return self.hidden_size


class ResNetImageEncoder(nn.Module):
    """
    ResNet-50 image encoder (lightweight alternative to CLIP).

    Uses torchvision pretrained weights and removes the final FC layer.
    """

    def __init__(
        self,
        freeze_backbone: bool = False,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        from torchvision import models

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Remove final FC layer — keep up to avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.hidden_size = 2048  # ResNet-50 output before FC

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Froze ResNet-50 backbone")

        self.projection = None
        if output_dim and output_dim != self.hidden_size:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
            self.hidden_size = output_dim

        logger.info(f"Loaded ResNet-50 image encoder (output_dim={self.hidden_size})")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (batch, 3, H, W)

        Returns:
            (batch, hidden_size) image features
        """
        features = self.backbone(pixel_values)  # (batch, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (batch, 2048)

        if self.projection is not None:
            features = self.projection(features)

        return features

    def get_output_dim(self) -> int:
        return self.hidden_size


def build_image_encoder(config: dict) -> nn.Module:
    """Factory function to build the image encoder from config."""
    name = config.get("name", "openai/clip-vit-base-patch32")
    freeze = config.get("freeze_backbone", False)
    output_dim = config.get("output_dim", None)

    if "clip" in name.lower():
        return CLIPImageEncoder(
            model_name=name,
            freeze_backbone=freeze,
            output_dim=output_dim,
        )
    elif "resnet" in name.lower():
        return ResNetImageEncoder(
            freeze_backbone=freeze,
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown image encoder: {name}")
