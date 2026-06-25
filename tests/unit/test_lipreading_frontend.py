"""Unit tests for the pretrained lip-reading visual frontend (Conv3D stem + 2D ResNet trunk).

Covers: output shape, grayscale+normalize, frozen-by-default, gradual unfreeze (only layer4),
BN-eval invariance of frozen parts, and selective checkpoint loading (frontend3D+trunk, drop tcn).
No pretrained weights needed (random init) — these test the wiring, not accuracy.
"""
import torch

from src.models.visual.lipreading_frontend import LipReadingFrontend, threeD_to_2D
from src.models.visual.resnet_lipreading import BasicBlock, ResNet


def _frontend(relu_type: str = "relu") -> LipReadingFrontend:
    # relu (paramless) keeps tests fast/deterministic; prelu/swish paths exercised in load test.
    return LipReadingFrontend(weights=None, relu_type=relu_type)


def test_output_shape_btx512():
    fe = _frontend()
    frames = torch.rand(2, 7, 3, 88, 88)  # [B,T,C,H,W] RGB in [0,1]
    out = fe(frames)
    assert out.shape == (2, 7, 512)


def test_time_dimension_preserved_by_3d_stem():
    # The Conv3D stem (temporal kernel 5, stride 1) must preserve T → one 512-vec per input frame.
    fe = _frontend()
    for t in (3, 11, 25):
        out = fe(torch.rand(1, t, 3, 88, 88))
        assert out.shape == (1, t, 512)


def test_temporal_kernel_mixes_frames():
    # Motion modeling: perturbing ONE frame must change neighbours' features (5-frame 3D kernel),
    # proving the stem is genuinely temporal (unlike the old per-frame 2D backbone).
    fe = _frontend().eval()
    frames = torch.rand(1, 9, 3, 88, 88)
    base = fe(frames)
    perturbed = frames.clone()
    perturbed[:, 4] = torch.rand(1, 3, 88, 88)  # change only frame 4
    out = fe(perturbed)
    # neighbour frame 3 (within the 5-tap window of frame 4) must move; a far frame (0) stays put
    assert not torch.allclose(out[:, 3], base[:, 3], atol=1e-5)
    assert torch.allclose(out[:, 0], base[:, 0], atol=1e-5)


def test_grayscale_single_channel_input():
    # Already-gray input [B,T,1,H,W] must also work (skip luma).
    fe = _frontend()
    out = fe(torch.rand(2, 5, 1, 88, 88))
    assert out.shape == (2, 5, 512)


def test_frozen_by_default():
    fe = _frontend()
    assert all(not p.requires_grad for p in fe.parameters())


def test_unfreeze_only_last_block():
    fe = _frontend()
    fe.unfreeze_last_block()
    # layer4 trainable; everything else still frozen
    assert all(p.requires_grad for p in fe.trunk.layer4.parameters())
    assert all(not p.requires_grad for p in fe.trunk.layer1.parameters())
    assert all(not p.requires_grad for p in fe.frontend3D.parameters())
    # the dedicated param-group helper returns exactly the layer4 params
    assert list(fe.last_block_parameters()) == list(fe.trunk.layer4.parameters())


def test_frozen_parts_stay_eval_in_train_mode():
    # requires_grad=False does NOT stop BN running-stat drift; train() override must keep
    # frozen sub-modules in eval() so the pretrained BN stats are preserved.
    fe = _frontend().train()
    assert not fe.frontend3D.training
    assert not fe.trunk.layer1.training
    assert not fe.trunk.layer4.training  # still frozen → eval
    fe.unfreeze_last_block()
    fe.train()
    assert fe.trunk.layer4.training       # now fine-tuning → train mode (BN adapts)
    assert not fe.trunk.layer1.training   # still frozen


def test_layer4_grad_flows_only_to_layer4_after_unfreeze():
    fe = _frontend()
    fe.unfreeze_last_block()
    out = fe(torch.rand(1, 4, 3, 88, 88))
    out.sum().backward()
    assert any(p.grad is not None for p in fe.trunk.layer4.parameters())
    assert all(p.grad is None for p in fe.trunk.layer1.parameters())  # frozen, no graph


def test_load_pretrained_selects_frontend_and_trunk_only(tmp_path):
    # Synthetic checkpoint that also contains tcn.* (+ a module. prefix) → only frontend3D/trunk load.
    fe = _frontend()
    sd = {f"module.{k}": v.clone() for k, v in fe.state_dict().items()}
    sd["module.tcn.weird.weight"] = torch.randn(3, 3)  # must be ignored
    ckpt_path = tmp_path / "lrw_fake.pth"
    torch.save({"model_state_dict": sd}, ckpt_path)

    fe2 = _frontend()
    fe2.load_pretrained(str(ckpt_path))  # must not raise on tcn.* / module. prefix
    # a frontend3D weight was actually copied over
    k = "frontend3D.0.weight"
    assert torch.allclose(dict(fe2.state_dict())[k], dict(fe.state_dict())[k])


def test_threeD_to_2D_helper():
    x = torch.randn(2, 64, 5, 22, 22)  # [B,C,T,H,W]
    y = threeD_to_2D(x)
    assert y.shape == (10, 64, 22, 22)  # [B*T,C,H,W]


def test_trunk_outputs_512():
    trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type="relu")
    out = trunk(torch.randn(3, 64, 22, 22))  # stem-less: input is the 3D-frontend feature map
    assert out.shape == (3, 512)
