"""
VLM (Vision-Language Model) baseline for zero/few-shot classification.

Uses large pre-trained VLMs like Gemini to classify memes
without any fine-tuning — useful for ablation studies in the paper.
"""

import os
import json
import logging
import base64
from pathlib import Path
from typing import Optional

logger = logging.getLogger("imusa.vlm_baseline")

VALID_LABELS = {"Sarcasm", "Neutral", "Offensive", "Motivational"}

ZERO_SHOT_PROMPT = """You are a sentiment classifier for Punjabi memes (images with text in Punjabi/Gurmukhi script).

Classify this meme into exactly ONE of the following categories:

1. Sarcasm — The intended meaning differs from the literal expression. Uses irony, exaggeration, or indirect references.
2. Neutral — No strong emotional tone or opinion. General information or observations.
3. Offensive — Abusive, insulting, or inappropriate content targeting individuals or groups.
4. Motivational — Intended to inspire, encourage positive thinking, or provide emotional support.

Look at the image carefully. Consider both the visual content AND any text in the image.

Respond with ONLY the category name (Sarcasm, Neutral, Offensive, or Motivational). Nothing else."""

FEW_SHOT_PROMPT_TEMPLATE = """You are a sentiment classifier for Punjabi memes.

Here are some examples:
{examples}

Now classify this meme into exactly ONE category:
- Sarcasm
- Neutral
- Offensive
- Motivational

Respond with ONLY the category name."""


class VLMClassifier:
    """
    Vision-Language Model classifier using Gemini or similar APIs.

    This is NOT a trainable model — it uses API calls for inference.
    Useful for comparison with fine-tuned models in the paper.
    """

    def __init__(
        self,
        provider: str = "gemini",  # gemini, openai
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        prompt_type: str = "zero_shot",  # zero_shot, few_shot
        few_shot_examples: Optional[list] = None,
    ):
        self.provider = provider
        self.model_name = model_name
        self.prompt_type = prompt_type
        self.few_shot_examples = few_shot_examples or []

        # Set up API
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No API key found. Set GOOGLE_API_KEY or OPENAI_API_KEY.")

        self._client = None

    def classify_image(self, image_path: str, ocr_text: str = "") -> dict:
        """
        Classify a single meme image.

        Args:
            image_path: Path to the meme image
            ocr_text: Pre-extracted OCR text (optional, helps the model)

        Returns:
            dict with 'label', 'raw_response', 'confidence'
        """
        prompt = self._build_prompt(ocr_text)

        try:
            if self.provider == "gemini":
                response = self._call_gemini(image_path, prompt)
            elif self.provider == "openai":
                response = self._call_openai(image_path, prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # Parse the response
            label = self._parse_response(response)

            return {
                "label": label,
                "raw_response": response,
                "valid": label in VALID_LABELS,
            }
        except Exception as e:
            logger.error(f"VLM classification failed for {image_path}: {e}")
            return {"label": "Neutral", "raw_response": str(e), "valid": False}

    def batch_classify(
        self,
        image_paths: list[str],
        ocr_texts: Optional[list[str]] = None,
    ) -> list[dict]:
        """Classify multiple images with progress tracking."""
        from tqdm import tqdm

        if ocr_texts is None:
            ocr_texts = [""] * len(image_paths)

        results = []
        for path, text in tqdm(
            zip(image_paths, ocr_texts),
            total=len(image_paths),
            desc="VLM Classification",
        ):
            result = self.classify_image(path, text)
            results.append(result)

        # Report accuracy
        valid = sum(1 for r in results if r["valid"])
        logger.info(f"VLM: {valid}/{len(results)} valid responses")

        return results

    def _build_prompt(self, ocr_text: str = "") -> str:
        """Build the classification prompt."""
        if self.prompt_type == "zero_shot":
            prompt = ZERO_SHOT_PROMPT
            if ocr_text:
                prompt += f"\n\nOCR-extracted text from this meme: \"{ocr_text}\""
            return prompt
        elif self.prompt_type == "few_shot":
            examples_str = ""
            for i, ex in enumerate(self.few_shot_examples, 1):
                examples_str += f"\nExample {i}: {ex['label']} — {ex.get('description', '')}"
            prompt = FEW_SHOT_PROMPT_TEMPLATE.format(examples=examples_str)
            if ocr_text:
                prompt += f"\n\nOCR-extracted text: \"{ocr_text}\""
            return prompt
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")

    def _call_gemini(self, image_path: str, prompt: str) -> str:
        """Call Google Gemini API."""
        import google.generativeai as genai

        if self._client is None:
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(self.model_name)

        # Load image
        from PIL import Image

        image = Image.open(image_path)

        response = self._client.generate_content(
            [prompt, image],
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=20,
            ),
        )
        return response.text.strip()

    def _call_openai(self, image_path: str, prompt: str) -> str:
        """Call OpenAI API with vision."""
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)

        # Encode image as base64
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=20,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    def _parse_response(self, response: str) -> str:
        """Parse the VLM response to extract a valid label."""
        response_clean = response.strip().lower()

        label_map = {
            "sarcasm": "Sarcasm",
            "sarcastic": "Sarcasm",
            "neutral": "Neutral",
            "offensive": "Offensive",
            "abusive": "Offensive",
            "motivational": "Motivational",
            "motivating": "Motivational",
            "inspirational": "Motivational",
        }

        for key, label in label_map.items():
            if key in response_clean:
                return label

        logger.warning(f"Could not parse VLM response: '{response}' → defaulting to 'Neutral'")
        return "Neutral"
