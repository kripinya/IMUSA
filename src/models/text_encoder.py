"""
Text encoder wrappers for IMUSA.

Supports:
  - MuRIL (google/muril-base-cased)    — best for Indian languages
  - PunjabiBERT (l3cube-pune/punjabi-bert) — Punjabi-specific
  - XLM-RoBERTa (xlm-roberta-base)     — multilingual fallback
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

logger = logging.getLogger("imusa.text_encoder")


class TextEncoder(nn.Module):
    """
    Wrapper around HuggingFace transformer models for text encoding.

    Extracts a fixed-size representation from the model output
    using one of: [CLS] token, mean pooling, or max pooling.
    """

    def __init__(
        self,
        model_name: str = "google/muril-base-cased",
        pooling: str = "cls",  # cls, mean, max
        freeze_layers: int = 0,  # 0 = fine-tune all, -1 = freeze all
        output_dim: Optional[int] = None,  # project to this dim if set
    ):
        super().__init__()
        self.model_name = model_name
        self.pooling = pooling

        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size

        logger.info(
            f"Loaded text encoder: {model_name} "
            f"(hidden_size={self.hidden_size}, pooling={pooling})"
        )

        # Freeze layers if requested
        if freeze_layers != 0:
            self._freeze_layers(freeze_layers)

        # Optional projection head
        self.projection = None
        if output_dim and output_dim != self.hidden_size:
            self.projection = nn.Sequential(
                nn.Linear(self.hidden_size, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
            self.hidden_size = output_dim

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode text and return pooled representation.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) — optional

        Returns:
            (batch, hidden_size) pooled text representation
        """
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        # Pool the output
        if self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif self.pooling == "mean":
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand_as(hidden).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        elif self.pooling == "max":
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand_as(hidden)
            hidden[~mask.bool()] = float("-inf")
            pooled = hidden.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Project if needed
        if self.projection is not None:
            pooled = self.projection(pooled)

        return pooled

    def _freeze_layers(self, num_layers: int):
        """Freeze the first `num_layers` encoder layers. -1 = freeze all."""
        # Always freeze embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        if num_layers == -1:
            # Freeze everything
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Froze all text encoder parameters")
            return

        # Freeze the first N encoder layers
        if hasattr(self.encoder, "encoder"):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, "transformer"):
            layers = self.encoder.transformer.layer
        else:
            logger.warning("Could not find encoder layers to freeze")
            return

        for i, layer in enumerate(layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        logger.info(f"Froze embeddings + first {num_layers} encoder layers")

    def get_output_dim(self) -> int:
        """Return the output dimensionality."""
        return self.hidden_size
