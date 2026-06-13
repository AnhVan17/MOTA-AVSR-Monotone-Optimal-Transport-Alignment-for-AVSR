"""The yaml `mqot:` block must actually reach MQOTLayer.

Bug: trainer built the model with `create_model(config['model'])`, but the `mqot:` block is a
sibling of `model:` in the YAML, so MQOT silently used hardcoded defaults (tuning the yaml had no
effect). The fix merges `config['mqot']` into the model config; these tests pin that contract.
"""
from src.models.mota import create_model


def _cfg(**over):
    cfg = dict(
        audio_dim=768, visual_dim=512, d_model=64,
        num_encoder_layers=1, num_decoder_layers=1, num_heads=4,
        vocab_size=100, dropout=0.0,
        use_mqot=True, use_backbones=False,
    )
    cfg.update(over)
    return cfg


def test_mqot_block_reaches_layer():
    """Mirror the trainer merge: config['mqot'] hyperparams must land on MQOTLayer."""
    model = create_model({**_cfg(), "mqot": {"n_iters": 7, "kl_penalty": 0.25}})
    assert model.mqot.n_iters == 7
    assert abs(model.mqot.kl_penalty - 0.25) < 1e-9


def test_mqot_defaults_when_block_absent():
    """No `mqot` key → documented defaults (n_iters=20)."""
    model = create_model(_cfg())
    assert model.mqot.n_iters == 20
