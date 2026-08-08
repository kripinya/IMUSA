"""
PyTorch Dataset for IMUSA Punjabi meme sentiment classification.

Supports three modes:
  - text_only:   Returns tokenized text only
  - image_only:  Returns transformed image only
  - multimodal:  Returns both image and text
"""

import os
import logging
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer

from src.data.preprocessing import TextPreprocessor

logger = logging.getLogger("imusa.dataset")

# Label mapping
LABEL2ID = {"Sarcasm": 0, "Neutral": 1, "Offensive": 2, "Motivational": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class IMUSADataset(Dataset):
    """
    IMUSA Punjabi Meme Dataset.

    Expects a CSV/DataFrame with columns:
      - image_path: relative or absolute path to the meme image
      - text: extracted/provided meme text (Punjabi)
      - label: one of {Sarcasm, Neutral, Offensive, Motivational}
              (optional for test set)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        data_dir: str,
        mode: str = "multimodal",  # text_only, image_only, multimodal
        tokenizer_name: str = "google/muril-base-cased",
        max_seq_length: int = 128,
        image_size: int = 224,
        image_transform: Optional[Callable] = None,
        text_preprocessor: Optional[TextPreprocessor] = None,
        is_test: bool = False,
    ):
        self.data = data.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        self.is_test = is_test

        # Text preprocessing
        self.text_preprocessor = text_preprocessor or TextPreprocessor()

        # Tokenizer (lazy-loaded to avoid loading for image_only mode)
        self._tokenizer = None
        self._tokenizer_name = tokenizer_name

        # Image transform
        self.image_transform = image_transform or self._default_image_transform()

        # Validate mode
        assert mode in ("text_only", "image_only", "multimodal"), (
            f"Invalid mode: {mode}. Must be text_only, image_only, or multimodal"
        )

        logger.info(
            f"Initialized IMUSADataset: {len(self.data)} samples, "
            f"mode={mode}, test={is_test}"
        )

    @property
    def tokenizer(self):
        """Lazy-load tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name)
            logger.info(f"Loaded tokenizer: {self._tokenizer_name}")
        return self._tokenizer

    def _default_image_transform(self):
        """Default image transforms (CLIP-compatible normalization)."""
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP mean
                    std=[0.26862954, 0.26130258, 0.27577711],  # CLIP std
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data.iloc[idx]
        sample = {"idx": idx}

        # ── Image ────────────────────────────────────────
        if self.mode in ("image_only", "multimodal"):
            image_path = self._resolve_image_path(row["image_path"])
            try:
                image = Image.open(image_path).convert("RGB")
                image = self.image_transform(image)
            except Exception as e:
                logger.warning(f"Failed to load image {image_path}: {e}. Using blank.")
                image = torch.zeros(3, self.image_size, self.image_size)
            sample["image"] = image

        # ── Text ─────────────────────────────────────────
        if self.mode in ("text_only", "multimodal"):
            raw_text = str(row.get("text", ""))
            clean_text = self.text_preprocessor(raw_text)

            encoding = self.tokenizer(
                clean_text,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            sample["input_ids"] = encoding["input_ids"].squeeze(0)
            sample["attention_mask"] = encoding["attention_mask"].squeeze(0)
            if "token_type_ids" in encoding:
                sample["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        # ── Label ────────────────────────────────────────
        if not self.is_test and "label" in row:
            label_str = row["label"]
            if isinstance(label_str, str):
                sample["label"] = torch.tensor(LABEL2ID[label_str], dtype=torch.long)
            else:
                sample["label"] = torch.tensor(int(label_str), dtype=torch.long)

        return sample

    def _resolve_image_path(self, path: str) -> str:
        """Resolve image path relative to data_dir if not absolute."""
        if os.path.isabs(path):
            return path
        return str(self.data_dir / path)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss weighting."""
        if self.is_test or "label" not in self.data.columns:
            return torch.ones(len(LABEL2ID))

        labels = self.data["label"].map(LABEL2ID).values
        class_counts = pd.Series(labels).value_counts().sort_index()

        # Inverse frequency weighting: w_i = N / (C * n_i)
        n_total = len(labels)
        n_classes = len(LABEL2ID)
        weights = n_total / (n_classes * class_counts.values)
        weights = torch.tensor(weights, dtype=torch.float32)

        logger.info(f"Class weights: {dict(zip(LABEL2ID.keys(), weights.tolist()))}")
        return weights

    def get_label_distribution(self) -> dict:
        """Return label distribution as a dictionary."""
        if "label" not in self.data.columns:
            return {}
        return self.data["label"].value_counts().to_dict()


def load_data(
    csv_path: str,
    data_dir: str,
    mode: str = "multimodal",
    tokenizer_name: str = "google/muril-base-cased",
    max_seq_length: int = 128,
    image_size: int = 224,
    image_transform: Optional[Callable] = None,
    is_test: bool = False,
) -> IMUSADataset:
    """Convenience function to load a dataset from CSV."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")

    # Validate required columns
    required = ["image_path"]
    if not is_test:
        required.append("label")

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {csv_path}")

    # If text column is missing, fill with empty strings (OCR will be needed)
    if "text" not in df.columns:
        logger.warning("No 'text' column found — using empty strings. Run OCR first!")
        df["text"] = ""

    return IMUSADataset(
        data=df,
        data_dir=data_dir,
        mode=mode,
        tokenizer_name=tokenizer_name,
        max_seq_length=max_seq_length,
        image_size=image_size,
        image_transform=image_transform,
        is_test=is_test,
    )


def create_train_val_split(
    csv_path: str,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Perform stratified train/val split and save to CSV.

    Creates:
      - {output_dir}/train.csv
      - {output_dir}/val.csv
    """
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df["label"],
        random_state=seed,
    )

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(
        f"Split: {len(train_df)} train, {len(val_df)} val → saved to {output_dir}"
    )

    return train_path, val_path
