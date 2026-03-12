from typing import List
from jiwer import wer, cer
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class MetricCalculator:
    """
    Calculates ASR metrics: WER (Word Error Rate) and CER (Character Error Rate).
    """
    
    @staticmethod
    def compute_wer(predictions: List[str], references: List[str]) -> float:
        """
        Returns WER as percentage (0-100).
        """
        if not predictions or not references:
            return 100.0
            
        try:
            error_rate = wer(references, predictions)
            return error_rate * 100
        except Exception as e:
            logger.error(f"WER Calculation Error: {e}")
            return 100.0

    @staticmethod
    def compute_cer(predictions: List[str], references: List[str]) -> float:
        """
        Returns CER as percentage (0-100).
        """
        if not predictions or not references:
            return 100.0
            
        try:
            error_rate = cer(references, predictions)
            return error_rate * 100
        except Exception as e:
            logger.error(f"CER Calculation Error: {e}")
            return 100.0