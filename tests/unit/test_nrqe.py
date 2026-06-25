"""Unit tests for NRQE (Noise-Robust Quality Estimator) — Section B.2.

Acceptance (ENGINEERING_PLAN §B.2): q_a,q_v ∈ [0,1]; shapes preserved; gradient flows;
cross-modal consistency high when streams agree, low when they disagree.
"""
import torch

from src.models.fusion.nrqe import NRQE


def test_nrqe_output_shapes_and_range():
    B, T, D = 2, 50, 256
    nrqe = NRQE(d_model=D, chunk_size=15)
    out = nrqe(torch.randn(B, T, D), torch.randn(B, T, D))
    assert out["q_a"].shape == (B, T)
    assert out["q_v"].shape == (B, T)
    assert out["consistency"].shape == (B, T)
    for k in ("q_a", "q_v", "consistency"):
        assert float(out[k].detach().min()) >= 0.0 and float(out[k].detach().max()) <= 1.0


def test_nrqe_handles_chunk_not_dividing_T():
    # T=37 not a multiple of chunk_size=15 → padding path must not crash / must keep shape.
    nrqe = NRQE(d_model=64, chunk_size=15)
    out = nrqe(torch.randn(1, 37, 64), torch.randn(1, 37, 64))
    assert out["q_a"].shape == (1, 37)
    assert out["q_v"].shape == (1, 37)


def test_nrqe_gradient_flows():
    nrqe = NRQE(d_model=64, chunk_size=8)
    out = nrqe(torch.randn(2, 24, 64), torch.randn(2, 24, 64))
    (out["q_a"].mean() + out["q_v"].mean()).backward()
    assert any(
        p.grad is not None and float(p.grad.abs().sum()) > 0 for p in nrqe.parameters()
    )


def test_nrqe_consistency_high_when_streams_agree():
    # Identical streams → per-chunk cosine ≈ 1 → consistency ≈ 1.
    nrqe = NRQE(d_model=32, chunk_size=4)
    x = torch.randn(1, 16, 32)
    out = nrqe(x, x.clone())
    assert float(out["consistency"].mean()) > 0.9


def test_nrqe_consistency_lower_when_streams_disagree():
    torch.manual_seed(0)
    nrqe = NRQE(d_model=32, chunk_size=4)
    x = torch.randn(1, 16, 32)
    agree = float(nrqe(x, x.clone())["consistency"].mean())
    disagree = float(nrqe(x, torch.randn(1, 16, 32))["consistency"].mean())
    assert disagree < agree
