"""F1 (CTC input_lengths real) + F2 (label smoothing) + fallback warnings."""
import logging

import torch

from src.training.losses import HybridLoss
from src.utils.warn_once import reset_warn_once


def _loss(label_smoothing=0.0):
    return HybridLoss(vocab_size=50, ctc_weight=0.5, ce_weight=0.5,
                      pad_id=4, blank_id=4, label_smoothing=label_smoothing)


def _batch():
    B, V, L = 2, 50, 6
    return (torch.randn(B, 40, V), torch.randn(B, L, V),
            torch.randint(5, V, (B, L)), torch.ones(B, L, dtype=torch.bool))


def test_input_lengths_changes_ctc():
    ctc, ar, tgt, tm = _batch()
    loss = _loss()
    full = loss(ctc, ar, tgt, target_mask=tm)["ctc_loss"]                                 # None → full padded
    short = loss(ctc, ar, tgt, target_mask=tm, input_lengths=torch.tensor([10, 12]))["ctc_loss"]
    assert not torch.isclose(full, short), "real input_lengths must change CTC vs full padded"


def test_input_lengths_none_warns_once(caplog):
    reset_warn_once()
    ctc, ar, tgt, tm = _batch()
    with caplog.at_level(logging.WARNING):
        _loss()(ctc, ar, tgt, target_mask=tm)  # input_lengths=None → fallback warn
    assert any("ctc_input_lengths" in r.message for r in caplog.records)


def test_label_smoothing_changes_ce():
    ctc, ar, tgt, tm = _batch()
    il = torch.tensor([20, 20])
    ce0 = _loss(0.0)(ctc, ar, tgt, target_mask=tm, input_lengths=il)["ce_loss"]
    ce1 = _loss(0.1)(ctc, ar, tgt, target_mask=tm, input_lengths=il)["ce_loss"]
    assert not torch.isclose(ce0, ce1)


def test_ctc_empty_flag_when_no_valid_tokens(caplog):
    reset_warn_once()
    # targets = toàn special (blank/pad id=4) → sau lọc còn 0 token → CTC fallback.
    B, V, L = 2, 50, 5
    ctc, ar = torch.randn(B, 40, V), torch.randn(B, L, V)
    tgt = torch.full((B, L), 4)  # all blank/pad
    tm = torch.ones(B, L, dtype=torch.bool)
    with caplog.at_level(logging.WARNING):
        out = _loss()(ctc, ar, tgt, target_mask=tm, input_lengths=torch.tensor([20, 20]))
    assert float(out["ctc_empty"]) == 1.0
    assert float(out["ctc_loss"]) == 0.0
    assert any("ctc_empty_target" in r.message for r in caplog.records)


def test_compute_ctc_only_accepts_visual_lengths():
    loss = _loss()
    B, V, L = 2, 50, 6
    visual_logits = torch.randn(B, 9, V)
    targets = torch.randint(5, V, (B, L))
    target_mask = torch.ones(B, L, dtype=torch.bool)
    out = loss.compute_ctc_only(
        visual_logits,
        targets,
        target_mask=target_mask,
        input_lengths=torch.tensor([9, 7]),
    )
    assert torch.isfinite(out["loss"])
    assert float(out["empty"]) == 0.0
