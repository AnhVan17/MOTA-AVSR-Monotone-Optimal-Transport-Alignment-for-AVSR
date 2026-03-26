import re
import unicodedata
from typing import List
from jiwer import wer, cer
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


def normalize_vietnamese(text: str) -> str:
    """
    Vietnamese text normalization for WER/CER evaluation.

    Vietnamese có đặc điểm:
    - Tone marks: à, ả, ạ, ã, ạ — PHÂN BIỆT nghĩa, giữ nguyên
    - Diacritics: â, ơ, ư — PHÂN BIỆT nghĩa, giữ nguyên
    - Uppercase/lowercase: "Việt Nam" vs "việt nam" — không phân biệt trong ASR
    - Digraphs: ch, gh, ng, nh, ph, qu, th, tr — 2 chars = 1 sound, nhưng WER xử lý đúng

    Steps:
      1. Unicode NFC normalization
      2. Lowercase
      3. Strip extra whitespace
      4. Remove punctuation marks (.,!?:;()[]{}"')
      5. Normalize unicode quotes/dashes
    """
    if not text:
        return ""

    # Unicode NFC — đảm bảo các ký tự được biểu diễn nhất quán
    text = unicodedata.normalize('NFC', text)

    # Lowercase
    text = text.lower()

    # Normalize unicode quotes and dashes
    text = text.replace('\u2018', "'").replace('\u2019', "'")   # curly quotes → straight
    text = text.replace('\u201c', '"').replace('\u201d', '"')   # curly double → straight
    text = text.replace('\u2013', '-').replace('\u2014', '-')   # em-dash → hyphen

    # Remove punctuation
    text = re.sub(r'[.,!?;:()[\]{}"\'\/\\]', ' ', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


class MetricCalculator:
    """
    Calculates ASR metrics: WER and CER with Vietnamese text normalization.
    """

    @staticmethod
    def compute_wer(
        predictions: List[str],
        references: List[str],
        normalize: bool = True,
    ) -> float:
        """
        Returns WER as percentage (0-100).

        Args:
            predictions: List of predicted transcriptions
            references: List of reference transcriptions
            normalize: If True, apply Vietnamese text normalization
        """
        if not predictions or not references:
            return 100.0

        refs = references
        preds = predictions

        if normalize:
            refs = [normalize_vietnamese(r) for r in references]
            preds = [normalize_vietnamese(p) for p in predictions]

        try:
            # normalize_vietnamese đã lowercase + remove punctuation rồi
            error_rate = wer(refs, preds)
            return error_rate * 100
        except Exception as e:
            logger.error(f"WER Calculation Error: {e}")
            return 100.0

    @staticmethod
    def compute_cer(
        predictions: List[str],
        references: List[str],
        normalize: bool = True,
    ) -> float:
        """
        Returns CER as percentage (0-100).
        """
        if not predictions or not references:
            return 100.0

        refs = references
        preds = predictions

        if normalize:
            refs = [normalize_vietnamese(r) for r in references]
            preds = [normalize_vietnamese(p) for p in predictions]

        try:
            error_rate = cer(refs, preds)
            return error_rate * 100
        except Exception as e:
            logger.error(f"CER Calculation Error: {e}")
            return 100.0