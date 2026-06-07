"""Smoke test NHANH — chạy thẳng local, tự dò device (cuda/mps/cpu).

Mục đích: kiểm tra pipeline compute (model → loss → backward → optimizer) chạy được
trên máy hiện tại MÀ KHÔNG cần Modal, KHÔNG cần data/manifest thật, KHÔNG tải backbone lớn.
Dùng dữ liệu giả (random tensor) đúng shape interface để bắt lỗi shape/device/op sớm.

Usage:
    python scripts/local/smoke_test.py                  # tự dò device, có MQOT
    python scripts/local/smoke_test.py --device cpu      # ép CPU
    python scripts/local/smoke_test.py --steps 5 --no-mqot
"""
import argparse
import os
import sys
from pathlib import Path

# MPS (Apple Silicon) chưa hỗ trợ vài op (vd aten::_ctc_loss). Cho phép fallback CPU cho op thiếu.
# Phải set TRƯỚC khi import torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

# Project root vào sys.path để import src.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.mota import create_model
from src.training.losses import create_loss
from src.utils.device import get_device, supports_amp


def build_config(use_mqot: bool, vocab_size: int) -> dict:
    """Config nhỏ gọn cho smoke test (model bé, chạy nhanh)."""
    return {
        "model": {
            "audio_dim": 768,
            "visual_dim": 512,
            "d_model": 128,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "num_heads": 4,
            "vocab_size": vocab_size,
            # blank/pad nằm TRONG vocab nhỏ của smoke (mặc định 50257 sẽ vượt phạm vi).
            "blank_id": vocab_size - 1,
            "pad_id": vocab_size - 1,
            "dropout": 0.1,
            "use_mqot": use_mqot,
            "use_backbones": False,
        },
        "loss": {"ctc_weight": 0.3, "ce_weight": 0.7, "quality_loss_weight": 0.1},
    }


def make_batch(B: int, Ta: int, Tv: int, L: int, vocab_size: int, device: torch.device):
    """Dữ liệu giả đúng shape interface MOTA."""
    audio = torch.randn(B, Ta, 768, device=device)
    visual = torch.randn(B, Tv, 512, device=device)
    # target là text token (< vocab_size-1 để tránh trùng blank/pad); mask 1 hết (không pad)
    targets = torch.randint(0, vocab_size - 1, (B, L), device=device)
    target_mask = torch.ones(B, L, dtype=torch.bool, device=device)
    return audio, visual, targets, target_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (mặc định: tự dò)")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--vocab-size", type=int, default=1000, help="nhỏ để chạy nhanh")
    ap.add_argument("--no-mqot", action="store_true", help="tắt MQOT path")
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
            raise RuntimeError(f"[smoke] FAIL: loss không hữu hạn ở step {step}: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            f"[smoke] step {step}/{args.steps} | total={loss.item():.4f} "
            f"ctc={loss_dict['ctc_loss'].item():.3f} ce={loss_dict['ce_loss'].item():.3f} "
            f"quality={loss_dict['quality_loss'].item():.3f}"
        )

    print(f"[smoke] PASS — pipeline chạy {args.steps} step trên {device}, loss hữu hạn, gradient OK.")


if __name__ == "__main__":
    main()
