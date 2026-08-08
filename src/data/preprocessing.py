"""
Text preprocessing for Punjabi meme text.

Handles:
  - Unicode normalization (Gurmukhi NFC)
  - Code-mixing detection (Gurmukhi / Latin / Devanagari)
  - URL / hashtag / mention removal
  - Emoji to text conversion
  - Script-specific cleaning
"""

import re
import unicodedata
import logging
from typing import Optional

logger = logging.getLogger("imusa.preprocessing")

# ── Unicode Ranges ──────────────────────────────────────
GURMUKHI_RANGE = (0x0A00, 0x0A7F)
DEVANAGARI_RANGE = (0x0900, 0x097F)
LATIN_BASIC_RANGE = (0x0041, 0x007A)

# ── Regex Patterns ──────────────────────────────────────
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_PATTERN = re.compile(r"#\w+")
MENTION_PATTERN = re.compile(r"@\w+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & pictographs
    "\U0001F680-\U0001F6FF"  # Transport & map
    "\U0001F1E0-\U0001F1FF"  # Flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
MULTIPLE_SPACES = re.compile(r"\s+")
REPEATED_CHARS = re.compile(r"(.)\1{3,}")  # Reduce "hahahahaha" → "haha"


class TextPreprocessor:
    """Preprocess Punjabi meme text for NLP models."""

    def __init__(
        self,
        remove_urls: bool = True,
        remove_hashtags: bool = False,  # Hashtags can carry sentiment
        remove_mentions: bool = True,
        remove_emojis: bool = False,  # Emojis may encode sentiment
        normalize_unicode: bool = True,
        lowercase_latin: bool = True,
        reduce_repetition: bool = True,
        max_length: Optional[int] = None,
    ):
        self.remove_urls = remove_urls
        self.remove_hashtags = remove_hashtags
        self.remove_mentions = remove_mentions
        self.remove_emojis = remove_emojis
        self.normalize_unicode = normalize_unicode
        self.lowercase_latin = lowercase_latin
        self.reduce_repetition = reduce_repetition
        self.max_length = max_length

    def __call__(self, text: str) -> str:
        """Apply full preprocessing pipeline."""
        if not text or not isinstance(text, str):
            return ""

        text = text.strip()

        # Step 1: Unicode normalization (NFC for Gurmukhi)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # Step 2: Remove URLs
        if self.remove_urls:
            text = URL_PATTERN.sub("", text)

        # Step 3: Remove mentions
        if self.remove_mentions:
            text = MENTION_PATTERN.sub("", text)

        # Step 4: Handle hashtags (optionally remove # but keep word)
        if self.remove_hashtags:
            text = HASHTAG_PATTERN.sub("", text)
        else:
            text = text.replace("#", " ")

        # Step 5: Handle emojis
        if self.remove_emojis:
            text = EMOJI_PATTERN.sub("", text)

        # Step 6: Lowercase Latin segments only
        if self.lowercase_latin:
            text = self._lowercase_latin_only(text)

        # Step 7: Reduce character repetition
        if self.reduce_repetition:
            text = REPEATED_CHARS.sub(r"\1\1", text)

        # Step 8: Normalize Gurmukhi-specific patterns
        text = self._normalize_gurmukhi(text)

        # Step 9: Clean whitespace
        text = MULTIPLE_SPACES.sub(" ", text).strip()

        # Step 10: Truncate if needed
        if self.max_length and len(text) > self.max_length:
            text = text[: self.max_length]

        return text

    def _lowercase_latin_only(self, text: str) -> str:
        """Lowercase Latin characters while preserving Gurmukhi and Devanagari."""
        result = []
        for char in text:
            code_point = ord(char)
            if LATIN_BASIC_RANGE[0] <= code_point <= LATIN_BASIC_RANGE[1]:
                result.append(char.lower())
            else:
                result.append(char)
        return "".join(result)

    def _normalize_gurmukhi(self, text: str) -> str:
        """Apply Gurmukhi-specific normalization rules."""
        # Normalize common Gurmukhi character variations
        # Bindi (ਂ) normalization
        text = re.sub(r"\u0A02{2,}", "\u0A02", text)

        # Tippi (ੰ) normalization
        text = re.sub(r"\u0A70{2,}", "\u0A70", text)

        # Remove zero-width characters
        text = text.replace("\u200B", "")  # Zero-width space
        text = text.replace("\u200C", "")  # Zero-width non-joiner
        text = text.replace("\u200D", "")  # Zero-width joiner
        text = text.replace("\uFEFF", "")  # BOM

        return text

    def detect_scripts(self, text: str) -> dict:
        """Detect which scripts are present in the text."""
        scripts = {"gurmukhi": 0, "devanagari": 0, "latin": 0, "other": 0}
        total = 0

        for char in text:
            if char.isspace() or not char.isalpha():
                continue
            total += 1
            code_point = ord(char)

            if GURMUKHI_RANGE[0] <= code_point <= GURMUKHI_RANGE[1]:
                scripts["gurmukhi"] += 1
            elif DEVANAGARI_RANGE[0] <= code_point <= DEVANAGARI_RANGE[1]:
                scripts["devanagari"] += 1
            elif LATIN_BASIC_RANGE[0] <= code_point <= LATIN_BASIC_RANGE[1]:
                scripts["latin"] += 1
            else:
                scripts["other"] += 1

        # Convert to percentages
        if total > 0:
            scripts = {k: round(v / total * 100, 1) for k, v in scripts.items()}

        return scripts

    def is_code_mixed(self, text: str, threshold: float = 10.0) -> bool:
        """Check if text contains significant code-mixing (>threshold% non-primary script)."""
        scripts = self.detect_scripts(text)
        primary_pct = max(scripts.values())
        return primary_pct < (100.0 - threshold)
