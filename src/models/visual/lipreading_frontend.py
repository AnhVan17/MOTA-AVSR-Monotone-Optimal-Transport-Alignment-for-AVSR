"""Pretrained lip-reading visual frontend (Conv3D stem + ResNet trunk) → 512-D/frame.

Drop-in replacement for the old torchvision-ResNet18-2D visual backbone in MOTA. Unlike the
2D-per-frame ImageNet backbone (which carried no lip motion → visual stream was useless), this
is a motion-aware lip-reading encoder pretrained on LRW
(``mpc001/Lipreading_using_Temporal_Convolutional_Networks``), used frozen (Whisper-Flamingo
style: frozen-audio + strong-frozen-visual). The trunk's last block can be unfrozen for light
fine-tuning (gradual unfreezing).

Pipeline (see plan ``recursive-hatching-haven.md`` Phase A2):
  frames [B,T,C,H,W] in [0,1] (RGB, 88x88, 25fps)
    → grayscale (luma) → [B,1,T,88,88] → normalize (x-0.421)/0.165
    → frontend3D (Conv3d 5x7x7, T preserved) → threeD_to_2D → ResNet trunk → [B,T,512]

Frozen-correctness: the frozen sub-modules are kept in ``eval()`` regardless of the parent's
train/eval mode, so their BatchNorm running stats do NOT drift during training (requires_grad
alone does not stop BN buffer updates).
"""
import logging
import os
from typing import Iterator, Optional

import torch
import torch.nn as nn

from src.models.visual.resnet_lipreading import BasicBlock, ResNet

logger = logging.getLogger(__name__)

# LRW grayscale normalization (frames already scaled to [0,1] upstream in shards.decode).
_GRAY_MEAN = 0.421
_GRAY_STD = 0.165
# ITU-R 601 luma weights for RGB → gray (frames are stored RGB; see base.py:135).
_LUMA = (0.299, 0.587, 0.114)


def threeD_to_2D(x: torch.Tensor) -> torch.Tensor:
    """[B, C, T, H, W] → [B*T, C, H, W] (per-frame for the 2D trunk)."""
    n, c, t, h, w = x.shape
    return x.transpose(1, 2).reshape(n * t, c, h, w)


class LipReadingFrontend(nn.Module):
    """Frozen (by default) pretrained lip-reading frontend; outputs [B, T, 512]."""

    def __init__(self, weights: Optional[str] = None, relu_type: str = "prelu"):
        super().__init__()
        frontend_relu = (
            nn.PReLU(num_parameters=64) if relu_type == "prelu"
            else (nn.SiLU(inplace=True) if relu_type == "swish" else nn.ReLU(inplace=True))
        )
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            frontend_relu,
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        self.train_last_block = False
        # Frozen feature extractor by default (gradient OFF on everything).
        self.requires_grad_(False)

        if weights and os.path.exists(weights):
            self.load_pretrained(weights)
        elif weights:
            logger.warning(
                f"LipReadingFrontend: weights not found at {weights} → random init "
                f"(OK offline/tests; on the training host the Phase-B pre-flight MUST catch this)."
            )

    # ---- (un)freeze control ------------------------------------------------------------------
    def unfreeze_last_block(self) -> None:
        """Enable gradients on the trunk's last block (layer4) for light fine-tuning."""
        self.train_last_block = True
        self.trunk.layer4.requires_grad_(True)
        if self.training:
            self.trunk.layer4.train()
        logger.info("LipReadingFrontend: unfroze trunk.layer4 (gradual unfreeze).")

    def last_block_parameters(self) -> Iterator[nn.Parameter]:
        """Params of the unfreezable block (for a dedicated optimizer param-group)."""
        return self.trunk.layer4.parameters()

    def train(self, mode: bool = True):  # type: ignore[override]
        """Keep frozen sub-modules in eval() so their BN running stats never drift."""
        super().train(mode)
        self.frontend3D.eval()
        self.trunk.layer1.eval()
        self.trunk.layer2.eval()
        self.trunk.layer3.eval()
        if not self.train_last_block:
            self.trunk.layer4.eval()
        return self

    # ---- weight loading ----------------------------------------------------------------------
    def load_pretrained(self, path: str) -> dict:
        """Load only ``frontend3D.*`` + ``trunk.*`` from an LRW checkpoint (drop ``tcn.*``).

        Returns ``{loaded, missing, unexpected}`` so callers (the Phase-B pre-flight) can assert the
        checkpoint actually matched — ``missing`` non-empty usually means a wrong ``relu_type``/arch
        silently left those weights at random init.
        """
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        subset = {
            k.replace("module.", "", 1): v
            for k, v in state.items()
            if k.replace("module.", "", 1).startswith(("frontend3D.", "trunk."))
        }
        if not subset:
            raise ValueError(f"No frontend3D.*/trunk.* keys found in checkpoint: {path}")
        missing, unexpected = self.load_state_dict(subset, strict=False)
        # missing = our params not in the subset (expected: none for frontend3D+trunk if matched).
        real_missing = [m for m in missing if m.startswith(("frontend3D.", "trunk."))]
        logger.info(
            f"LipReadingFrontend: loaded {len(subset)} keys from {path}; "
            f"missing(frontend/trunk)={len(real_missing)}, unexpected={len(unexpected)}."
        )
        if real_missing:
            logger.warning(f"LipReadingFrontend missing keys (check relu_type): {real_missing[:6]}")
        return {"loaded": len(subset), "missing": real_missing, "unexpected": list(unexpected)}

    # ---- forward -----------------------------------------------------------------------------
    def _to_gray_ncthw(self, frames: torch.Tensor) -> torch.Tensor:
        """[B,T,C,H,W] in [0,1] → normalized grayscale [B,1,T,H,W]."""
        c = frames.shape[2]
        if c == 1:
            gray = frames[:, :, 0]
        else:  # RGB → luma
            gray = _LUMA[0] * frames[:, :, 0] + _LUMA[1] * frames[:, :, 1] + _LUMA[2] * frames[:, :, 2]
        x = gray.unsqueeze(1)  # [B,1,T,H,W]
        return (x - _GRAY_MEAN) / _GRAY_STD

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames [B,T,C,H,W] in [0,1] → features [B,T,512]."""
        b, t = frames.shape[0], frames.shape[1]
        x = self._to_gray_ncthw(frames)
        # Frozen stem + early trunk run under no_grad (no graph, no BN drift).
        with torch.no_grad():
            x = self.frontend3D(x)        # [B,64,T,H',W'] (T preserved)
            x = threeD_to_2D(x)           # [B*T,64,H',W']
            x = self.trunk.layer1(x)
            x = self.trunk.layer2(x)
            x = self.trunk.layer3(x)
            if not self.train_last_block:
                x = self.trunk.layer4(x)
                x = self.trunk.avgpool(x).flatten(1)  # [B*T,512]
        if self.train_last_block:         # layer4 with gradient (input detached → only layer4 trains)
            x = self.trunk.layer4(x)
            x = self.trunk.avgpool(x).flatten(1)
        return x.view(b, t, -1)
