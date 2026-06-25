"""Unit tests for modality dropout (noise-robust AV training)."""
import torch

from src.data.augmentations import FeatureAugmenter


def test_modality_dropout_zeros_exactly_one_stream():
    # prob=1.0 → always drop one; audio/visual feature aug off so only dropout acts.
    aug = FeatureAugmenter(
        audio_conf={"prob": 0.0, "modality_dropout_prob": 1.0}, visual_conf={"prob": 0.0}
    )
    a, v = torch.randn(20, 768), torch.randn(20, 512)
    n_audio_zero = n_visual_zero = 0
    for _ in range(60):
        ao, vo = aug(a.clone(), v.clone())
        assert ao.shape == a.shape and vo.shape == v.shape
        a_zero, v_zero = bool((ao == 0).all()), bool((vo == 0).all())
        assert not (a_zero and v_zero)  # never both dropped
        assert a_zero or v_zero  # exactly one dropped (prob=1)
        n_audio_zero += a_zero
        n_visual_zero += v_zero
    assert n_audio_zero > 0 and n_visual_zero > 0  # both directions occur (~50/50)


def test_modality_dropout_off_by_default():
    # no modality_dropout_prob + aug prob 0 → inputs returned unchanged (backward-compatible)
    aug = FeatureAugmenter(audio_conf={"prob": 0.0}, visual_conf={"prob": 0.0})
    a, v = torch.randn(10, 768), torch.randn(10, 512)
    ao, vo = aug(a.clone(), v.clone())
    assert torch.allclose(ao, a) and torch.allclose(vo, v)


def test_modality_dropout_handles_raw_frames():
    # visual as raw frames [T,C,H,W] (4D) must also be droppable
    aug = FeatureAugmenter(
        audio_conf={"prob": 0.0, "modality_dropout_prob": 1.0}, visual_conf={"prob": 0.0}
    )
    a, frames = torch.randn(16, 768), torch.rand(16, 3, 88, 88)
    ao, vo = aug(a.clone(), frames.clone())
    assert ao.shape == a.shape and vo.shape == frames.shape
    assert bool((ao == 0).all()) or bool((vo == 0).all())


# --- Deterministic (fixed noisy dev) seeding ---------------------------------------------------

def _noisy_aug(moddrop: float = 0.0) -> FeatureAugmenter:
    # Full noise recipe (prob=1.0 → always applies) so determinism is actually exercised.
    return FeatureAugmenter(
        audio_conf={"prob": 1.0, "noise_std": 0.1, "time_mask_param": 10,
                    "freq_mask_param": 64, "modality_dropout_prob": moddrop},
        visual_conf={"prob": 1.0, "dropout_prob": 0.1, "frame_mask_param": 5},
    )


def test_same_seed_is_deterministic():
    # Same seed → byte-identical augmentation across calls (the fixed-noisy-dev guarantee).
    aug = _noisy_aug()
    a, v = torch.randn(20, 768), torch.randn(20, 512)
    a1, v1 = aug(a.clone(), v.clone(), seed=123)
    a2, v2 = aug(a.clone(), v.clone(), seed=123)
    assert torch.equal(a1, a2) and torch.equal(v1, v2)


def test_different_seed_differs():
    aug = _noisy_aug()
    a, v = torch.randn(20, 768), torch.randn(20, 512)
    a1, _ = aug(a.clone(), v.clone(), seed=1)
    a2, _ = aug(a.clone(), v.clone(), seed=2)
    assert not torch.equal(a1, a2)


def test_seed_none_is_random():
    # seed=None (train path) → fresh noise each call (NOT frozen).
    aug = _noisy_aug()
    a, v = torch.randn(20, 768), torch.randn(20, 512)
    a1, _ = aug(a.clone(), v.clone())
    a2, _ = aug(a.clone(), v.clone())
    assert not torch.equal(a1, a2)


def test_modality_dropout_is_seed_deterministic():
    # The drop DECISION (which stream is zeroed) must also be frozen by the seed.
    aug = _noisy_aug(moddrop=1.0)
    a, v = torch.randn(20, 768), torch.randn(20, 512)
    a1, v1 = aug(a.clone(), v.clone(), seed=7)
    a2, v2 = aug(a.clone(), v.clone(), seed=7)
    assert torch.equal(a1, a2) and torch.equal(v1, v2)


def test_seeded_call_restores_global_rng():
    # A deterministic (seeded) call must NOT perturb the global RNG stream training relies on.
    aug = _noisy_aug()
    a, v = torch.randn(8, 768), torch.randn(8, 512)
    torch.manual_seed(999)
    expected = torch.rand(3)
    torch.manual_seed(999)
    aug(a.clone(), v.clone(), seed=42)  # seeded call between the two draws
    after = torch.rand(3)
    assert torch.equal(expected, after)
