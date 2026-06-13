"""SentencePiece tokenizer for Vietnamese — compact ~2000 vocab vs Whisper's 51865.

Matches the ``WhisperTokenizer`` interface (encode/decode/vocab_size/eot_token_id/...) so the
loader, ``HybridLoss`` (``create_loss`` derives pad/blank/special_ids FROM the tokenizer) and
``CTCDecoder`` all work unchanged — only the tokenizer object swaps.

Reserved ids (fixed at SP train time): unk=0, bos=1, eos=2, pad=3, ``<blank>``=4 (CTC blank),
real pieces from 5. Convention (mirrors Whisper's eot==pad==blank=50257): ``eot_token_id`` ==
``pad_token_id`` == ``<blank>`` id, so the collate pad value, the CTC blank and the CE pad all
reference one reserved token (masking is positional, so this stays consistent).
"""
from typing import Any, Dict, List, Union

import sentencepiece as spm
import torch

from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

BLANK_PIECE = "<blank>"


class SentencePieceTokenizer:
    def __init__(self, model_path: str, language: str = "vi"):
        self.language = language
        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

        self.unk_token_id = self.sp.unk_id()                    # 0
        self.bos_token_id = self.sp.bos_id()                    # 1
        self.eos_token_id = self.sp.eos_id()                    # 2
        self.sot_token_id = self.bos_token_id                   # decoder start-of-sequence = bos
        # CTC blank doubles as eot + collate pad (mirrors Whisper eot==pad==blank).
        self.blank_token_id = self.sp.piece_to_id(BLANK_PIECE)  # 4
        self.eot_token_id = self.blank_token_id
        self.pad_token_id = self.blank_token_id

        logger.debug(
            f"SentencePieceTokenizer({model_path}) vocab={self.vocab_size} "
            f"blank/eot/pad={self.blank_token_id} bos={self.bos_token_id} eos={self.eos_token_id}"
        )

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    @property
    def model_vocab_size(self) -> int:
        return self.sp.vocab_size()

    @property
    def all_special_ids(self) -> List[int]:
        """Control ids filtered from CTC targets: unk/bos/eos/pad-piece/blank."""
        ids = {self.unk_token_id, self.bos_token_id, self.eos_token_id, self.blank_token_id}
        pad_piece = self.sp.pad_id()
        if pad_piece >= 0:
            ids.add(pad_piece)
        return sorted(i for i in ids if i >= 0)

    @property
    def max_token_id(self) -> int:
        return self.vocab_size - 1

    @property
    def total_token_count(self) -> int:
        return self.vocab_size

    def vocab_info(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "model_vocab_size": self.model_vocab_size,
            "max_token_id": self.max_token_id,
            "eot_token_id": self.eot_token_id,
            "sot_token_id": self.sot_token_id,
            "pad_token_id": self.pad_token_id,
            "blank_token_id": self.blank_token_id,
            "language": self.language,
            "model": self.model_path,
        }

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Text → ids. With specials: [bos] + pieces + [eos] (decoder learns start/stop)."""
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if skip_special_tokens:
            special = set(self.all_special_ids)
            ids = [int(i) for i in ids if int(i) not in special]
        else:
            ids = [int(i) for i in ids]
        return self.sp.decode(ids)

    def batch_decode(
        self,
        batch_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.tolist()
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_ids]

    def get_vocab(self) -> Dict[str, int]:
        return {self.sp.id_to_piece(i): i for i in range(self.vocab_size)}

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self):
        return f"SentencePieceTokenizer(model={self.model_path}, vocab_size={self.vocab_size})"
