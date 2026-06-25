"""Model forward tests for the frame-shard path (visual backbone runs at train time).

The frame WebDataset pipeline stores raw mouth-crop frames [T,C,H,W] for visual and
precomputed Whisper features [T_a,768] for audio. So the model must:
  - run the visual ResNet on 5D frame input when ``use_backbones`` is set, and
  - NOT load a Whisper audio backbone (audio is already features) unless
    ``use_audio_backbone`` is explicitly set.

These build the model with ``visual_pretrained=False`` so no ImageNet weights are fetched
(offline / CI-safe) — we only check wiring + shapes + gradient flow on synthetic tensors.
"""
import torch

from src.models.mota import create_model

VOCAB = 100


def _cfg(**over):
    cfg = dict(
        audio_dim=768,
        visual_dim=512,
        d_model=64,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        vocab_size=VOCAB,
        dropout=0.0,
        use_mqot=False,
        use_backbones=True,        # visual ResNet on raw frames
        use_audio_backbone=False,  # audio already = Whisper features → no Whisper load
        visual_pretrained=False,   # offline-safe (no ImageNet download in tests)
    )
    cfg.update(over)
    return cfg


def test_frame_mode_forward_backward_and_no_audio_backbone():
    """use_backbones loads ONLY the visual ResNet; Whisper must NOT be loaded."""
    torch.manual_seed(0)
    model = create_model(_cfg())

    assert hasattr(model, "visual_backbone"), "use_backbones=True must load the visual ResNet"
    assert not hasattr(model, "whisper"), "use_audio_backbone=False must NOT load Whisper"

    B, Ta, Tv, L = 2, 12, 6, 5
    audio = torch.randn(B, Ta, 768)               # precomputed Whisper feats
    visual = torch.rand(B, Tv, 3, 88, 88)         # raw mouth-crop frames [0,1]
    target = torch.randint(0, VOCAB, (B, L))

    out = model(audio, visual, target)
    assert out["ctc_logits"].shape == (B, Ta, VOCAB)
    assert out["ar_logits"].shape == (B, L, VOCAB)

    (out["ctc_logits"].sum() + out["ar_logits"].sum()).backward()
    assert model.audio_proj.weight.grad is not None
    assert model.visual_proj.weight.grad is not None


def test_feature_mode_still_works():
    """Regression: the precomputed-feature path (no backbones) is unchanged."""
    torch.manual_seed(0)
    model = create_model(_cfg(use_backbones=False))

    assert not hasattr(model, "visual_backbone")
    assert not hasattr(model, "whisper")

    B, Ta, Tv = 2, 12, 6
    audio = torch.randn(B, Ta, 768)
    visual = torch.randn(B, Tv, 512)              # precomputed ResNet features
    out = model(audio, visual)
    assert out["ctc_logits"].shape == (B, Ta, VOCAB)
    assert out["ar_logits"] is None               # no target → no AR logits


def test_visual_ctc_aux_uses_visual_timeline():
    """Bootstrap head should emit logits on Tv, independent from the audio encoder length."""
    torch.manual_seed(0)
    model = create_model(_cfg(use_backbones=False, use_visual_ctc_aux=True))

    B, Ta, Tv = 2, 12, 7
    audio = torch.randn(B, Ta, 768)
    visual = torch.randn(B, Tv, 512)
    out = model(audio, visual)

    assert out["ctc_logits"].shape == (B, Ta, VOCAB)
    assert out["visual_ctc_logits"].shape == (B, Tv, VOCAB)
