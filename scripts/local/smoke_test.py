"""FAST smoke test — runs directly on the local machine, auto-detects device (cuda/mps/cpu).

Purpose: verify the compute pipeline (model -> loss -> backward -> optimizer) runs on the
current machine WITHOUT Modal, WITHOUT real data/manifest, and WITHOUT downloading a large
backbone. Uses fake data (random tensors) with the correct interface shapes to catch
shape/device/op errors early.

Usage:
    python scripts/local/smoke_test.py                  # auto-detect device, MQOT on
    python scripts/local/smoke_test.py --device cpu      # force CPU
    python scripts/local/smoke_test.py --steps 5 --no-mqot
"""
import argparse
import os
import sys
from pathlib import Path

# MPS (Apple Silicon) does not support some ops yet (e.g. aten::_ctc_loss). Allow CPU
# fallback for missing ops. Must be set BEFORE importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

# Add project root to sys.path so `import src.*` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.mota import create_model
from src.training.losses import create_loss
from src.utils.device import get_device, supports_amp


def build_config(use_mqot: bool, vocab_size: int) -> dict:
    """Compact config for the smoke test (small model, runs fast)."""
    return {
        "model": {
            "audio_dim": 768,
            "visual_dim": 512,
            "d_model": 128,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "num_heads": 4,
            "vocab_size": vocab_size,
            # blank/pad must stay WITHIN the small smoke vocab (default 50257 would be out of range).
            "blank_id": vocab_size - 1,
            "pad_id": vocab_size - 1,
            "dropout": 0.1,
            "use_mqot": use_mqot,
            "use_backbones": False,
        },
        "loss": {"ctc_weight": 0.3, "ce_weight": 0.7, "quality_loss_weight": 0.1},
    }


def make_batch(B: int, Ta: int, Tv: int, L: int, vocab_size: int, device: torch.device):
    """Fake data matching the MOTA interface shapes."""
    audio = torch.randn(B, Ta, 768, device=device)
    visual = torch.randn(B, Tv, 512, device=device)
    # targets are text tokens (< vocab_size-1 to avoid clashing with blank/pad); mask all 1 (no pad).
    targets = torch.randint(0, vocab_size - 1, (B, L), device=device)
    target_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    return audio, visual, targets, target_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto-detect)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=1000, help="small for speed")
    ap.add_argument("--no-mqot", action="store_true", help="disable the MQOT path")
    args = ap.parse_args()

    device = get_device(args.device)
    use_mqot = not args.no_mqot
    print(f"[smoke] device={device} | amp_supported={supports_amp(device)} | use_mqot={use_mqot}")

    config = build_config(use_mqot, args.vocab_size)
    model = create_model(config["model"]).to(device)
    criterion = create_loss(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] model params={n_params:,}")

    model.train()
    B, Ta, Tv, L = args.batch_size, 40, 20, 8
    for step in range(1, args.steps + 1):
        audio, visual, targets, target_mask = make_batch(B, Ta, Tv, L, args.vocab_size, device)
        outputs = model(audio, visual, targets)

        loss_dict = criterion(
            ctc_logits=outputs["ctc_logits"],
            ar_logits=outputs["ar_logits"],
            targets=targets,
            target_mask=target_mask,
            transport_map=outputs.get("transport_map"),
            mqot_quality=outputs.get("mqot_quality"),
        )
        loss = loss_dict["total_loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"[smoke] FAIL: loss is not finite at step {step}: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f"[smoke] step {step}/{args.steps} | total={loss.item():.4f} "
            f"ctc={loss_dict['ctc_loss'].item():.3f} ce={loss_dict['ce_loss'].item():.3f} "
            f"quality={loss_dict['quality_loss'].item():.3f}"
        )

    print(f"[smoke] PASS — pipeline ran {args.steps} step(s) on {device}, loss finite, gradient OK.")


if __name__ == "__main__":
    main()
