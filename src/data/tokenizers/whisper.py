"""
Whisper Tokenizer with Vietnamese Vocab Pruning
================================================
Reduces vocab from 51,865 -> ~5,000 for faster training

USAGE:
1. Normal mode (full vocab):
   tokenizer = WhisperTokenizer(use_pruned_vocab=False)

2. Pruned mode (Vietnamese-only, 10x faster):
   tokenizer = WhisperTokenizer(use_pruned_vocab=True)
"""

from transformers import WhisperTokenizer as HfWhisperTokenizer
import torch
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class WhisperTokenizer:
    def __init__(
        self,
        language: str = "vi",
        model: str = "openai/whisper-small",
        task: str = "transcribe",
        use_pruned_vocab: bool = False,  # 🔧 NEW: Enable vocab pruning
        pruned_vocab_path: Optional[str] = None,
    ):
        """
        Args:
            language: Target language (default: "vi")
            model: Whisper model name
            task: Task type (default: "transcribe")
            use_pruned_vocab: If True, use Vietnamese-only vocab (~5K tokens)
            pruned_vocab_path: Path to pruned vocab files (auto-detect if None)
        """
        self.language = language
        self.model = model
        self.task = task
        self.use_pruned_vocab = use_pruned_vocab

        # Load base tokenizer
        self.tokenizer = HfWhisperTokenizer.from_pretrained(
            model,
            language=language,
            task=task
        )

        # 🔧 NEW: Load pruned vocab if enabled
        self.id_mapping = None
        self.reverse_mapping = None
        self.pruned_vocab_size = None
        
        if use_pruned_vocab:
            self._load_pruned_vocab(pruned_vocab_path)
        
        # Expose commonly used attributes
        self._setup_special_tokens()
        
        # Log initialization
        self._log_init()

    def _setup_special_tokens(self):
        """Setup special token IDs"""
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        
        # Whisper specific
        self.sot_token_id = self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftranscript|>")
        self.no_timestamps_token_id = self.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        
        # 🔧 NEW: Remap special tokens if using pruned vocab
        if self.use_pruned_vocab and self.id_mapping is not None:
            # Map special tokens if they exist in mapping, else keep original (but usually they should be in mapping)
            # We assume critical special tokens are PRESAVED in the pruning process
            self.pad_token_id = self.id_mapping.get(self.pad_token_id, 0)
            self.bos_token_id = self.id_mapping.get(self.bos_token_id, 1)
            # eos is usually same as eot in whisper
            self.eos_token_id = self.id_mapping.get(self.eos_token_id, 2) 
            self.unk_token_id = self.id_mapping.get(self.unk_token_id, 3)
            self.sot_token_id = self.id_mapping.get(self.sot_token_id, 4)
            self.eot_token_id = self.id_mapping.get(self.eot_token_id, 5)

    def _load_pruned_vocab(self, vocab_path: Optional[str] = None):
        """
        Load pruned vocabulary mapping
        
        Args:
            vocab_path: Path to pruned vocab directory
        """
        # Auto-detect path if not provided
        if vocab_path is None:
            # Try common locations
            possible_paths = [
                "./vocab_pruned",
                "./data/vocab_pruned",
                "../vocab_pruned",
                "src/data/vocab_pruned"
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    vocab_path = path
                    break
            
            if vocab_path is None:
                logger.warning("⚠️ Pruned vocab not found! Need 'id_mapping.pkl'.")
                logger.warning("   Please run: python src/utils/prune_whisper_vocab.py")
                logger.warning("   Falling back to FULL vocab.")
                self.use_pruned_vocab = False
                return
        
        vocab_dir = Path(vocab_path)
        mapping_file = vocab_dir / "id_mapping.pkl"
        
        if not mapping_file.exists():
            logger.warning(f"⚠️ Mapping file not found: {mapping_file}")
            logger.warning("   Falling back to FULL vocab.")
            self.use_pruned_vocab = False
            return
        
        # Load ID mapping (old_id -> new_id)
        with open(mapping_file, 'rb') as f:
            self.id_mapping = pickle.load(f)
        
        # Create reverse mapping (new_id -> old_id)
        self.reverse_mapping = {v: k for k, v in self.id_mapping.items()}
        
        self.pruned_vocab_size = len(self.id_mapping)
        
        logger.info(f"✅ Loaded pruned vocab: {self.pruned_vocab_size} tokens")
        logger.info(f"   Original: {self.tokenizer.vocab_size} -> Pruned: {self.pruned_vocab_size}")
        logger.info(f"   Reduction: {(1 - self.pruned_vocab_size / self.tokenizer.vocab_size) * 100:.1f}%")

    def _log_init(self):
        """Log initialization info"""
        logger.info(f"WhisperTokenizer initialized (Language: {self.language})")
        logger.info(f"   Mode: {'PRUNED' if self.use_pruned_vocab else 'FULL'}")
        logger.info(f"   Vocab Size: {self.vocab_size}")

    @property
    def vocab_size(self):
        """Return vocab size (pruned if enabled, else full)"""
        if self.use_pruned_vocab and self.pruned_vocab_size is not None:
            return self.pruned_vocab_size
        return self.tokenizer.vocab_size

    def _remap_ids(self, token_ids: List[int]) -> List[int]:
        """
        Remap token IDs from full vocab to pruned vocab
        
        Args:
            token_ids: Original token IDs
            
        Returns:
            Remapped token IDs (OOV tokens removed)
        """
        if not self.use_pruned_vocab or self.id_mapping is None:
            return token_ids
        
        remapped = []
        for old_id in token_ids:
            if old_id in self.id_mapping:
                remapped.append(self.id_mapping[old_id])
            else:
                # OOV token - simply skip it! 
                # Ideally we map to UNK, but CTC blank is better for ASR than UNK noise
                pass
        
        return remapped

    def _reverse_remap_ids(self, token_ids: List[int]) -> List[int]:
        """
        Reverse remap: pruned vocab -> full vocab (for decoding)
        
        Args:
            token_ids: Pruned vocab token IDs
            
        Returns:
            Original vocab token IDs
        """
        if not self.use_pruned_vocab or self.reverse_mapping is None:
            return token_ids
        
        original = []
        for new_id in token_ids:
            if new_id in self.reverse_mapping:
                original.append(self.reverse_mapping[new_id])
            else:
                # Invalid ID map to UNK
                original.append(self.unk_token_id)
        
        return original

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Encode text to token IDs
        """
        # Encode with base tokenizer
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens
        )
        
        # Remap if using pruned vocab
        if self.use_pruned_vocab:
            token_ids = self._remap_ids(token_ids)
        
        return token_ids
    
    def encode_for_ctc(
        self,
        text: str,
    ) -> List[int]:
        """
        Encode text for CTC loss (without special tokens)
        """
        # Encode without special tokens
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )
        
        # Remap if using pruned vocab
        if self.use_pruned_vocab:
            token_ids = self._remap_ids(token_ids)
        
        return token_ids

    def decode(
        self,
        ids,
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode single sequence of token IDs to text
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        # Reverse remap if using pruned vocab
        if self.use_pruned_vocab:
            ids = self._reverse_remap_ids(ids)
        
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        batch_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode batch"""
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.tolist()
        
        # Reverse remap if using pruned vocab
        if self.use_pruned_vocab:
            batch_ids = [self._reverse_remap_ids(ids) for ids in batch_ids]
        
        return self.tokenizer.batch_decode(
            batch_ids,
            skip_special_tokens=skip_special_tokens
        )

    def get_vocab(self) -> dict:
        """Get vocab"""
        if self.use_pruned_vocab and self.id_mapping is not None:
             # Return fake pruned vocab for debugging
             return {f"token_{k}": v for k, v in self.id_mapping.items()}
        return self.tokenizer.get_vocab()

    def __len__(self):
        return self.vocab_size


# Factory
def create_tokenizer(use_pruned: bool = True, **kwargs):
    return WhisperTokenizer(use_pruned_vocab=use_pruned, **kwargs)