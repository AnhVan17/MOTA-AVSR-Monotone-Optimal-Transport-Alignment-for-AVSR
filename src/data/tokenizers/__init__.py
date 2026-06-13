"""Tokenizer factory — pick the tokenizer by ``config['tokenizer']['type']``.

  - sentencepiece : Vietnamese SentencePiece (~2000 vocab) — default for the AVSR pipeline.
  - whisper       : HF Whisper tokenizer (51865) — GRID / legacy.

Both expose the same interface (encode/decode/vocab_size/eot_token_id/all_special_ids/...), so the
loader, HybridLoss and CTCDecoder consume either unchanged.
"""
import os
from pathlib import Path
from typing import Dict


def _resolve(path: str) -> str:
    """Resolve a (possibly repo-relative) path so it works from any CWD and in Modal (/root)."""
    if os.path.exists(path):
        return path
    # repo root = src/data/tokenizers/__init__.py → parents[3]
    root_rel = Path(__file__).resolve().parents[3] / path
    if root_rel.exists():
        return str(root_rel)
    container_rel = Path("/root") / path
    if container_rel.exists():
        return str(container_rel)
    return path  # fall through → tokenizer raises a clear "model not found" error


def build_tokenizer(config: Dict):
    """Build the tokenizer selected in config['tokenizer'] (default: whisper, backward-compatible)."""
    tcfg = (config or {}).get("tokenizer", {}) or {}
    ttype = str(tcfg.get("type", "whisper")).lower()
    language = tcfg.get("language", "vi")

    if ttype == "sentencepiece":
        from .sentencepiece import SentencePieceTokenizer
        return SentencePieceTokenizer(model_path=_resolve(tcfg["model_path"]), language=language)
    if ttype == "whisper":
        from .whisper import WhisperTokenizer
        return WhisperTokenizer(model=tcfg.get("model", "openai/whisper-small"), language=language)
    raise ValueError(f"Unknown tokenizer type: {ttype!r} (expected 'sentencepiece' or 'whisper')")
