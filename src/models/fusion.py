"""
Cross-modal fusion modules for combining text and image representations.

Implements four fusion strategies:
  1. Concatenation — simple but effective baseline
  2. Bilinear Pooling — captures multiplicative interactions
  3. Cross-Attention — transformer-style inter-modal attention
  4. Gated Fusion — learnable gating to control modality contributions
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("imusa.fusion")


class ConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion: [text; image] → Linear → output.

    Fast and effective — often a strong baseline.
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(text_dim + image_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = output_dim

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([text_feat, image_feat], dim=-1)
        return self.projection(combined)


class BilinearFusion(nn.Module):
    """
    Low-rank bilinear pooling for capturing multiplicative cross-modal interactions.

    Instead of full bilinear: z = x^T W y (expensive),
    uses low-rank factorization: z = (U^T x) ⊙ (V^T y)
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        output_dim: int = 256,
        rank: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, rank)
        self.image_proj = nn.Linear(image_dim, rank)
        self.output_proj = nn.Sequential(
            nn.Linear(rank, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = output_dim

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        text_proj = self.text_proj(text_feat)    # (batch, rank)
        image_proj = self.image_proj(image_feat)  # (batch, rank)
        interaction = text_proj * image_proj      # Hadamard product
        return self.output_proj(interaction)


class CrossAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion using multi-head attention.

    Text-to-Image: text queries attend to image keys/values
    Image-to-Text: image queries attend to text keys/values

    Final output is the combination of both directions.
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Project both modalities to the same dimension
        self.text_proj = nn.Linear(text_dim, output_dim)
        self.image_proj = nn.Linear(image_dim, output_dim)

        # Text-to-Image attention: text queries, image keys/values
        self.t2i_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Image-to-Text attention: image queries, text keys/values
        self.i2t_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Combine both attention outputs
        self.fusion_norm = nn.LayerNorm(output_dim)
        self.fusion_ff = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feat:  (batch, text_dim)
            image_feat: (batch, image_dim)

        Returns:
            (batch, output_dim) fused representation
        """
        # Project to shared dimension and add sequence dimension
        text_proj = self.text_proj(text_feat).unsqueeze(1)    # (batch, 1, output_dim)
        image_proj = self.image_proj(image_feat).unsqueeze(1)  # (batch, 1, output_dim)

        # Text attends to Image
        t2i_out, _ = self.t2i_attention(
            query=text_proj, key=image_proj, value=image_proj
        )  # (batch, 1, output_dim)

        # Image attends to Text
        i2t_out, _ = self.i2t_attention(
            query=image_proj, key=text_proj, value=text_proj
        )  # (batch, 1, output_dim)

        # Squeeze and combine
        t2i_out = t2i_out.squeeze(1)  # (batch, output_dim)
        i2t_out = i2t_out.squeeze(1)  # (batch, output_dim)

        # Fuse the two attention directions
        combined = torch.cat([t2i_out, i2t_out], dim=-1)  # (batch, output_dim*2)
        fused = self.fusion_ff(combined)                    # (batch, output_dim)
        fused = self.output_norm(fused + t2i_out)          # Residual connection

        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion with learnable modality weighting.

    gate = σ(W_g · [text; image] + b_g)
    fused = gate ⊙ text_proj + (1 - gate) ⊙ image_proj

    The gate learns to dynamically weight each modality per sample.
    """

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        output_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.output_dim = output_dim

        # Project both to same dimension
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(text_dim + image_dim, output_dim),
            nn.Sigmoid(),
        )

        # Post-fusion FFN
        self.output_ff = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        text_proj = self.text_proj(text_feat)
        image_proj = self.image_proj(image_feat)

        gate_input = torch.cat([text_feat, image_feat], dim=-1)
        gate_value = self.gate(gate_input)

        fused = gate_value * text_proj + (1 - gate_value) * image_proj
        return self.output_ff(fused)


def build_fusion_module(config: dict, text_dim: int, image_dim: int) -> nn.Module:
    """Factory function to build a fusion module from config."""
    fusion_type = config.get("type", "cross_attention")
    output_dim = config.get("projection_dim", 256)
    dropout = config.get("fusion_dropout", 0.2)

    if fusion_type == "concatenation":
        return ConcatenationFusion(text_dim, image_dim, output_dim, dropout)
    elif fusion_type == "bilinear":
        rank = config.get("bilinear_rank", 64)
        return BilinearFusion(text_dim, image_dim, output_dim, rank, dropout)
    elif fusion_type == "cross_attention":
        num_heads = config.get("num_attention_heads", 4)
        return CrossAttentionFusion(text_dim, image_dim, output_dim, num_heads, dropout)
    elif fusion_type == "gated":
        return GatedFusion(text_dim, image_dim, output_dim, dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
