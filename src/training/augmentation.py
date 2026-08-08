"""
Data augmentation pipelines for IMUSA.

Image augmentations via albumentations.
Text augmentations via simple transformations.
"""

import random
import logging

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

logger = logging.getLogger("imusa.augmentation")


def get_train_image_transform(image_size: int = 224, config: dict | None = None) -> A.Compose:
    """
    Training-time image augmentations for memes.

    Carefully chosen to preserve text readability:
    - No rotation (text becomes unreadable)
    - Mild color jitter (meme text contrast matters)
    - Horizontal flip only if meme meaning is preserved
    """
    aug_cfg = (config or {}).get("augmentation", {}).get("image", {})

    transforms = [
        A.Resize(image_size, image_size),
    ]

    # Random crop with resize
    if aug_cfg.get("random_crop", True):
        transforms.append(
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
                p=0.5,
            )
        )

    # Horizontal flip (be careful — may reverse text meaning)
    flip_p = aug_cfg.get("horizontal_flip", 0.3)
    if flip_p > 0:
        transforms.append(A.HorizontalFlip(p=flip_p))

    # Color jitter (mild to preserve text readability)
    cj = aug_cfg.get("color_jitter", {})
    if cj:
        transforms.append(
            A.ColorJitter(
                brightness=cj.get("brightness", 0.2),
                contrast=cj.get("contrast", 0.2),
                saturation=cj.get("saturation", 0.1),
                hue=cj.get("hue", 0.05),
                p=0.5,
            )
        )

    # Gaussian blur
    blur_p = aug_cfg.get("gaussian_blur", 0.1)
    if blur_p > 0:
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=blur_p))

    # Random erasing (CoarseDropout) — simulates occluded meme regions
    erasing_p = aug_cfg.get("random_erasing", 0.1)
    if erasing_p > 0:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(0.05, 0.15),
                hole_width_range=(0.05, 0.15),
                fill="random",
                p=erasing_p,
            )
        )

    # CLIP normalization
    transforms.extend(
        [
            A.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
            ToTensorV2(),
        ]
    )

    pipeline = A.Compose(transforms)
    logger.info(f"Training image augmentations: {len(transforms)} transforms")
    return pipeline


def get_val_image_transform(image_size: int = 224) -> A.Compose:
    """Validation/test image transforms (no augmentation)."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
            ToTensorV2(),
        ]
    )


class TextAugmentor:
    """
    Simple text augmentations for Punjabi meme text.

    Operates at the word level to avoid breaking Gurmukhi characters.
    """

    def __init__(
        self,
        word_dropout_p: float = 0.1,
        word_swap_p: float = 0.05,
        enabled: bool = True,
    ):
        self.word_dropout_p = word_dropout_p
        self.word_swap_p = word_swap_p
        self.enabled = enabled

    def __call__(self, text: str) -> str:
        if not self.enabled or not text:
            return text

        words = text.split()
        if len(words) < 3:
            return text  # Too short to augment

        # Word dropout: randomly remove words
        if random.random() < 0.5:
            words = [w for w in words if random.random() > self.word_dropout_p]

        # Word swap: randomly swap adjacent words
        if random.random() < 0.5 and len(words) > 1:
            for i in range(len(words) - 1):
                if random.random() < self.word_swap_p:
                    words[i], words[i + 1] = words[i + 1], words[i]

        return " ".join(words) if words else text
