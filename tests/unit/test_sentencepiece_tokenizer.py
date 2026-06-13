"""SentencePiece Vietnamese tokenizer — interface + round-trip + CTC-blank wiring.

Skipped until the SP model is downloaded to assets/ (trained by
scripts/modal/data_prep/train_sentencepiece.py). Once present, validates the contract the
loader/HybridLoss/CTCDecoder rely on.
"""
import os

import pytest

MODEL = "assets/tokenizer/vi_sp_2000.model"
pytestmark = pytest.mark.skipif(not os.path.exists(MODEL), reason="SP model not downloaded yet")


def _tok():
    from src.data.tokenizers.sentencepiece import SentencePieceTokenizer
    return SentencePieceTokenizer(MODEL)


def test_round_trip_vietnamese():
    tok = _tok()
    text = "xin chào việt nam hôm nay trời đẹp"
    ids = tok.encode(text)                              # [bos] + pieces + [eos]
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id
    assert tok.decode(ids).strip() == text             # exact reconstruction (char coverage 1.0)


def test_vocab_size_and_special_ids():
    tok = _tok()
    assert 1500 <= tok.vocab_size <= 2500              # ~2000
    # eot == pad == blank (one reserved token; mirrors Whisper convention)
    assert tok.eot_token_id == tok.pad_token_id == tok.blank_token_id
    specials = set(tok.all_special_ids)
    assert {tok.unk_token_id, tok.bos_token_id, tok.eos_token_id, tok.blank_token_id} <= specials


def test_decode_skips_blank_and_specials():
    tok = _tok()
    ids = (
        [tok.bos_token_id]
        + tok.encode("chào", add_special_tokens=False)
        + [tok.blank_token_id, tok.eos_token_id]
    )
    out = tok.decode(ids, skip_special_tokens=True)
    assert "<blank>" not in out and out.strip() != ""
