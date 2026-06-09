"""Unit tests for the batch Collator (src/data/collate.py).

Focus: the defensive None-filtering (corrupt samples) + padding/masking correctness.
"""
import torch

from src.data.collate import Collator


def _sample(a_len, v_len, t_len, text="hi"):
    return {
        "audio": torch.randn(a_len, 768),
        "visual": torch.randn(v_len, 512),
        "target": torch.arange(t_len),
        "text": text,
        "rel_path": "x",
    }


def test_collate_empty_batch_returns_none():
    assert Collator()([]) is None


def test_collate_all_none_returns_none():
    assert Collator()([None, None]) is None


def test_collate_filters_none_and_pads_to_max():
    batch = [_sample(10, 5, 3), None, _sample(7, 8, 4)]
    out = Collator(pad_id=50257)(batch)

    assert out is not None
    # None dropped → 2 valid samples, padded to per-modality max length.
    assert out["audio"].shape == (2, 10, 768)
    assert out["visual"].shape == (2, 8, 512)
    assert out["target"].shape == (2, 4)
    assert len(out["text"]) == 2


def test_collate_masks_mark_padding():
    batch = [_sample(10, 5, 3), _sample(7, 8, 4)]
    out = Collator()(batch)

    # First sample fills full audio length; second is padded after 7 frames.
    assert out["audio_mask"][0].all()
    assert out["audio_mask"][1, :7].all()
    assert out["audio_mask"][1, 7:].sum().item() == 0
