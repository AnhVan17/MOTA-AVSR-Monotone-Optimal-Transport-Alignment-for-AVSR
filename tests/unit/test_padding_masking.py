"""F3 GOLD test: padding-invariance. Adding padding (with masks) must NOT change outputs at the
real positions. Also a sanity check that WITHOUT masks padding DOES leak (mask is necessary)."""
import torch

from src.models.fusion.quality_gate import QualityGate
from src.models.layers.conformer import ConformerBlock
from src.models.layers.decoders import HybridDecoder
from src.models.mota import create_model


def _pad_time(x, n):
    """Append n random frames along time (dim=1)."""
    shape = list(x.shape); shape[1] = n
    return torch.cat([x, torch.randn(*shape)], dim=1)


def _pad_mask(m, n):
    return torch.cat([m, torch.zeros(m.shape[0], n, dtype=torch.bool)], dim=1)


def test_conformer_padding_invariant():
    torch.manual_seed(0)
    blk = ConformerBlock(d_model=32, num_heads=4, conv_kernel=15, dropout=0.0).eval()
    B, T = 2, 10
    x = torch.randn(B, T, 32)
    m = torch.ones(B, T, dtype=torch.bool)
    out1 = blk(x, pad_mask=~m)
    out2 = blk(_pad_time(x, 5), pad_mask=~_pad_mask(m, 5))
    assert torch.allclose(out1, out2[:, :T], atol=1e-5)


def test_conformer_unmasked_leaks():
    # Sanity: without mask, the appended frames DO change real-position outputs (mask is needed).
    torch.manual_seed(0)
    blk = ConformerBlock(d_model=32, num_heads=4, conv_kernel=15, dropout=0.0).eval()
    B, T = 2, 10
    x = torch.randn(B, T, 32)
    out1 = blk(x)
    out2 = blk(_pad_time(x, 5))
    assert not torch.allclose(out1, out2[:, :T], atol=1e-4)


def test_quality_gate_padding_invariant():
    torch.manual_seed(0)
    qg = QualityGate(d_model=32, num_heads=4, dropout=0.0).eval()
    B, Ta, Tv = 2, 12, 8
    a, v = torch.randn(B, Ta, 32), torch.randn(B, Tv, 32)
    am, vm = torch.ones(B, Ta, dtype=torch.bool), torch.ones(B, Tv, dtype=torch.bool)
    out1 = qg(a, v, am, vm)["fused"]
    out2 = qg(_pad_time(a, 4), _pad_time(v, 3), _pad_mask(am, 4), _pad_mask(vm, 3))["fused"]
    assert torch.allclose(out1, out2[:, :Ta], atol=1e-5)


def test_decoder_memory_padding_invariant():
    torch.manual_seed(0)
    dec = HybridDecoder(d_model=32, num_heads=4, num_layers=2, vocab_size=50, dropout=0.0).eval()
    B, Tm, L = 2, 10, 6
    mem, tgt = torch.randn(B, Tm, 32), torch.randint(0, 50, (B, L))
    mm = torch.ones(B, Tm, dtype=torch.bool)
    ar1 = dec(mem, tgt, memory_key_padding_mask=~mm)["ar_logits"]
    ar2 = dec(_pad_time(mem, 4), tgt, memory_key_padding_mask=~_pad_mask(mm, 4))["ar_logits"]
    assert torch.allclose(ar1, ar2, atol=1e-5)


def test_mota_end_to_end_padding_invariant():
    torch.manual_seed(0)
    cfg = dict(audio_dim=768, visual_dim=512, d_model=64, num_encoder_layers=2,
               num_decoder_layers=1, num_heads=4, vocab_size=50, dropout=0.0,
               use_mqot=False, use_backbones=False, mqot={}, rgf={})
    m = create_model(cfg).eval()
    B, Ta, Tv = 2, 16, 10
    a, v, tgt = torch.randn(B, Ta, 768), torch.randn(B, Tv, 512), torch.randint(0, 50, (B, 5))
    am, vm = torch.ones(B, Ta, dtype=torch.bool), torch.ones(B, Tv, dtype=torch.bool)
    ctc1 = m(a, v, tgt, audio_mask=am, visual_mask=vm)["ctc_logits"]
    ctc2 = m(_pad_time(a, 6), _pad_time(v, 4), tgt,
             audio_mask=_pad_mask(am, 6), visual_mask=_pad_mask(vm, 4))["ctc_logits"]
    assert torch.allclose(ctc1, ctc2[:, :Ta], atol=1e-4)
