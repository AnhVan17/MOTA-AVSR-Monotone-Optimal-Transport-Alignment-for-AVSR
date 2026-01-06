from transformers import WhisperTokenizer as HfWhisperTokenizer
import torch
from typing import List
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

class WhisperTokenizer:
    def __init__(
        self,
        language: str = "vi",  # Vietnamese for ViCocktail
        model: str = "openai/whisper-small",  # Use small for better Vietnamese support
        task: str = "transcribe",
    ):
        self.language = language
        self.model = model
        self.task = task

        self.tokenizer = HfWhisperTokenizer.from_pretrained(
            model,
            language=language,
            task=task
        )

        # expose commonly used attributes (READ-ONLY)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.sot_token_id = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftranscript|>")

        logger.debug(f"WhisperTokenizer initialized")
        logger.debug(f"   Vocab size: {self.vocab_size}")
        logger.debug(f"   Language: {language}")
        logger.debug(f"   Model: {model}")
        
        # Log token range for debugging
        logger.debug(f"   Pad token ID: {self.pad_token_id}")
        logger.debug(f"   Special token range: {self.sot_token_id} - {self.eot_token_id}")

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode text to token IDs"""
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens
        )
    
    def encode_for_ctc(
        self,
        text: str,
    ) -> List[int]:
        """
        Encode text for CTC loss (without special tokens)
        
        This is CRITICAL for CTC training because:
        1. CTC should only see content tokens
        2. Special tokens (<|startoftranscript|>, <|vi|>, etc.) confuse alignment
        3. Cleaner targets = faster convergence
        
        Args:
            text: Input text string
            
        Returns:
            List of token IDs (content only, no special tokens)
        """
        return self.tokenizer.encode(
            text,
            add_special_tokens=False  # KEY: Remove special tokens for CTC
        )

    def decode(
        self,
        ids,
        skip_special_tokens: bool = True
    ) -> str:
        """Decode single sequence of token IDs to text"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        batch_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode batch of token IDs"""
        return self.tokenizer.batch_decode(
            batch_ids,
            skip_special_tokens=skip_special_tokens
        )

    def get_vocab(self) -> dict:
        return self.tokenizer.get_vocab()

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return (
            f"WhisperTokenizer(model={self.model}, "
            f"language={self.language}, vocab_size={self.vocab_size})"
        )