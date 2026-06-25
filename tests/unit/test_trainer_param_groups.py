"""Unit tests for Trainer._build_param_groups (gradual-unfreeze optimizer wiring).

Tests the staticmethod directly (no Trainer instantiation → no dataloaders/Modal needed).
"""
from src.models.mota import create_model
from src.training.trainer import Trainer


def _model():
    cfg = dict(
        audio_dim=768, visual_dim=512, d_model=256, num_encoder_layers=2,
        num_decoder_layers=1, num_heads=4, vocab_size=100, dropout=0.1,
        use_mqot=False, use_backbones=True,
        visual_frontend="lipreading_tcn", visual_frontend_relu="relu",
        mqot={}, rgf={},
    )
    return create_model(cfg)


def test_two_groups_when_visual_backbone_lr_set():
    model = _model()
    groups = Trainer._build_param_groups(model, {"learning_rate": 1e-4, "visual_backbone_lr": 1e-5})
    assert len(groups) == 2
    assert {g["name"] for g in groups} == {"head", "visual_backbone"}
    vb = next(g for g in groups if g["name"] == "visual_backbone")
    head = next(g for g in groups if g["name"] == "head")
    assert vb["lr"] == 1e-5 and head["lr"] == 1e-4
    # visual group = EXACTLY the frontend's last block (layer4)
    assert list(vb["params"]) == list(model.visual_backbone.last_block_parameters())
    # head excludes ALL visual_backbone params (incl. the frozen stem/early-trunk → in no group)
    vb_ids = {id(p) for p in model.visual_backbone.parameters()}
    assert all(id(p) not in vb_ids for p in head["params"])


def test_single_group_when_no_visual_backbone_lr():
    model = _model()
    groups = Trainer._build_param_groups(model, {"learning_rate": 1e-4})  # no visual_backbone_lr
    assert len(groups) == 1
    assert groups[0]["name"] == "all"


def test_single_group_when_no_lip_frontend():
    # Legacy 2D backbone has no `last_block_parameters` → single group even with visual_backbone_lr.
    cfg = dict(
        audio_dim=768, visual_dim=512, d_model=256, num_encoder_layers=2,
        num_decoder_layers=1, num_heads=4, vocab_size=100, dropout=0.1,
        use_mqot=False, use_backbones=True, visual_frontend="resnet2d",
        visual_pretrained=False, mqot={}, rgf={},
    )
    model = create_model(cfg)
    groups = Trainer._build_param_groups(model, {"learning_rate": 1e-4, "visual_backbone_lr": 1e-5})
    assert len(groups) == 1
