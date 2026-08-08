"""
OCR text extraction pipeline for Punjabi (Gurmukhi) meme images.

Uses a multi-engine approach:
  1. PaddleOCR (primary) — best for scene text in memes
  2. EasyOCR (fallback)  — good Gurmukhi support

Results are cached to avoid re-extraction.
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("imusa.ocr")


class OCRExtractor:
    """Multi-engine OCR extractor for Punjabi meme images."""

    def __init__(
        self,
        cache_dir: str = "data/ocr_cache",
        primary_engine: str = "paddleocr",
        use_fallback: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.primary_engine = primary_engine
        self.use_fallback = use_fallback

        self._paddle_reader = None
        self._easy_reader = None

    @property
    def paddle_reader(self):
        """Lazy-load PaddleOCR."""
        if self._paddle_reader is None:
            try:
                from paddleocr import PaddleOCR

                self._paddle_reader = PaddleOCR(
                    use_angle_cls=True,
                    lang="en",  # PaddleOCR uses 'en' but handles Devanagari-like scripts
                    show_log=False,
                    use_gpu=True,
                )
                logger.info("PaddleOCR initialized successfully")
            except ImportError:
                logger.warning("PaddleOCR not available, will use EasyOCR only")
        return self._paddle_reader

    @property
    def easy_reader(self):
        """Lazy-load EasyOCR."""
        if self._easy_reader is None:
            try:
                import easyocr

                self._easy_reader = easyocr.Reader(
                    ["pa", "en"],  # Punjabi (Gurmukhi) + English
                    gpu=True,
                )
                logger.info("EasyOCR initialized with Punjabi + English")
            except ImportError:
                logger.warning("EasyOCR not available")
        return self._easy_reader

    def extract(self, image_path: str, force: bool = False) -> dict:
        """
        Extract text from a meme image.

        Args:
            image_path: Path to the meme image.
            force: If True, skip cache and re-extract.

        Returns:
            dict with keys:
                - text: Combined extracted text
                - raw_results: Detailed per-region results
                - engine: Which engine produced the result
                - confidence: Average confidence score
        """
        cache_key = self._get_cache_key(image_path)
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache
        if not force and cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Preprocess image for OCR
        processed_img = self._preprocess_image(image_path)

        # Try primary engine
        result = None
        if self.primary_engine == "paddleocr" and self.paddle_reader:
            result = self._extract_paddle(processed_img, image_path)

        # Fallback to EasyOCR if primary fails or returns empty
        if (result is None or not result["text"].strip()) and self.use_fallback:
            easy_result = self._extract_easyocr(image_path)
            if easy_result and easy_result["text"].strip():
                result = easy_result

        # Default empty result
        if result is None:
            result = {
                "text": "",
                "raw_results": [],
                "engine": "none",
                "confidence": 0.0,
            }

        # Cache result
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image to improve OCR accuracy on meme text."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter for noise reduction while preserving edges
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    def _extract_paddle(self, processed_img: np.ndarray, image_path: str) -> Optional[dict]:
        """Extract text using PaddleOCR."""
        try:
            # PaddleOCR works better with the original color image
            results = self.paddle_reader.ocr(image_path, cls=True)

            if not results or not results[0]:
                return None

            texts = []
            confidences = []
            raw_results = []

            for line in results[0]:
                bbox, (text, conf) = line[0], line[1]
                texts.append(text)
                confidences.append(conf)
                raw_results.append(
                    {
                        "text": text,
                        "confidence": float(conf),
                        "bbox": [[float(p) for p in point] for point in bbox],
                    }
                )

            return {
                "text": " ".join(texts),
                "raw_results": raw_results,
                "engine": "paddleocr",
                "confidence": float(np.mean(confidences)) if confidences else 0.0,
            }
        except Exception as e:
            logger.error(f"PaddleOCR failed for {image_path}: {e}")
            return None

    def _extract_easyocr(self, image_path: str) -> Optional[dict]:
        """Extract text using EasyOCR."""
        try:
            if self.easy_reader is None:
                return None

            results = self.easy_reader.readtext(image_path)

            if not results:
                return None

            texts = []
            confidences = []
            raw_results = []

            for bbox, text, conf in results:
                texts.append(text)
                confidences.append(conf)
                raw_results.append(
                    {
                        "text": text,
                        "confidence": float(conf),
                        "bbox": [[float(p) for p in point] for point in bbox],
                    }
                )

            return {
                "text": " ".join(texts),
                "raw_results": raw_results,
                "engine": "easyocr",
                "confidence": float(np.mean(confidences)) if confidences else 0.0,
            }
        except Exception as e:
            logger.error(f"EasyOCR failed for {image_path}: {e}")
            return None

    def _get_cache_key(self, image_path: str) -> str:
        """Generate a unique cache key for an image."""
        path_hash = hashlib.md5(os.path.abspath(image_path).encode()).hexdigest()[:12]
        filename = Path(image_path).stem
        return f"{filename}_{path_hash}"

    def batch_extract(self, image_paths: list[str], force: bool = False) -> list[dict]:
        """Extract text from multiple images with progress bar."""
        from tqdm import tqdm

        results = []
        for path in tqdm(image_paths, desc="Extracting OCR text"):
            try:
                result = self.extract(path, force=force)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract text from {path}: {e}")
                results.append(
                    {"text": "", "raw_results": [], "engine": "error", "confidence": 0.0}
                )
        return results
