"""Unit tests for RGF (Router-Gated Fusion) — Section B.3.

Acceptance (ENGINEERING_PLAN §B.3): shape [B,T,d] preserved; router weights sum=1, no NaN;
q_a,q_v ∈ [0,1]; gradient flows to NRQE + router; eval = hard one-hot routing.
(The behavioural "audio=noise → route to visual" property is a *post-training* property,
not a module-init invariant, so it is validated in integration, not here.)
"""
import torch

from src.models.fusion.router_gate import RouterGatedFusion


def _rgf(d=64, K=8):
    return RouterGatedFusion(d_model=d, chunk_size=K)


def test_rgf_output_shape_preserved():
    out = _rgf()(torch.randn(2, 40, 64), torch.randn(2, 40, 64))
    assert out["fused"].shape == (2, 40, 64)


def test_rgf_handles_Ta_ne_Tv():
    # audio/visual different length → QualityGate aligns visual to audio timeline.
    out = _rgf()(torch.randn(2, 40, 64), torch.randn(2, 25, 64))
    assert out["fused"].shape == (2, 40, 64)


def test_rgf_router_weights_sum_to_one_no_nan():
    rgf = _rgf().train()
    out = rgf(torch.randn(2, 40, 64), torch.randn(2, 40, 64))
    w = out["router_weights"]  # [B, Ta, 3]
    assert w.shape == (2, 40, 3)
    assert torch.allclose(w.sum(-1), torch.ones(2, 40), atol=1e-4)
    assert not torch.isnan(w).any()
    assert not torch.isnan(out["fused"]).any()


def test_rgf_router_probs_per_chunk_shape():
    out = _rgf(K=8)(torch.randn(2, 40, 64), torch.randn(2, 40, 64))
    assert out["router_probs"].shape == (2, 5, 3)  # 40 / 8 = 5 chunks


def test_rgf_q_in_range():
    out = _rgf()(torch.randn(2, 40, 64), torch.randn(2, 40, 64))
    for k in ("q_audio", "q_visual"):
        assert float(out[k].detach().min()) >= 0.0 and float(out[k].detach().max()) <= 1.0


def test_rgf_eval_mode_hard_one_hot():
    rgf = _rgf().eval()
    with torch.no_grad():
        out = rgf(torch.randn(1, 32, 64), torch.randn(1, 32, 64))
    w = out["router_weights"]
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)))
    assert ((w == 0) | (w == 1)).all()  # exactly one expert per frame


def test_rgf_gradient_flows_to_router_and_nrqe():
    rgf = _rgf().train()
    rgf(torch.randn(2, 40, 64), torch.randn(2, 40, 64))["fused"].mean().backward()

    def has_grad(m):
        return any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in m.parameters())

    assert has_grad(rgf.router) and has_grad(rgf.nrqe)


def test_mota_use_rgf_end_to_end():
    """MOTA with use_rgf=true builds, runs a forward, and exposes router_probs."""
    from src.models.mota import create_model

    cfg = {
        "audio_dim": 768, "visual_dim": 512, "d_model": 64,
        "num_encoder_layers": 1, "num_decoder_layers": 1, "num_heads": 4,
        "vocab_size": 50, "dropout": 0.1,
        "use_rgf": True, "rgf": {"chunk_size": 4},
    }
    model = create_model(cfg)
    out = model(torch.randn(2, 20, 768), torch.randn(2, 16, 512), torch.randint(0, 50, (2, 8)))
    assert out["ctc_logits"].shape[0] == 2
    assert "router_probs" in out and out["router_probs"].shape[-1] == 3


def test_hybrid_loss_load_balancing():
    """RGF load-balancing term is added iff router_probs is passed (Switch-Transformer L_bal)."""
    from src.training.losses import HybridLoss

    crit = HybridLoss(
        vocab_size=50, ctc_weight=0.5, ce_weight=0.5, quality_loss_weight=0.0,
        lambda_bal=0.01, pad_id=4, blank_id=4, special_ids=[0, 1, 2, 3, 4],
    )
    B, T, V, L = 2, 20, 50, 8
    ctc_logits, ar_logits = torch.randn(B, T, V), torch.randn(B, L, V)
    targets = torch.randint(5, V, (B, L))
    rp = torch.rand(B, 5, 3)
    rp = rp / rp.sum(-1, keepdim=True)

    assert float(crit(ctc_logits, ar_logits, targets, router_probs=rp)["bal_loss"]) > 0.0
    assert float(crit(ctc_logits, ar_logits, targets)["bal_loss"]) == 0.0  # no router_probs → 0

