"""Local end-to-end training on a small SYNTHETIC dataset.

Unlike smoke_test.py (which feeds tensors straight to the model), this exercises the FULL
training entrypoint: Dataset -> Collator -> Trainer.train() loop -> validation -> checkpoint,
on the auto-detected device, WITHOUT Modal and WITHOUT real data. It generates fake .pt
feature files + a manifest in a temp dir, then calls the same run_training() used in the cloud.

Heavier than smoke_test (real DataLoader + Whisper tokenizer + full vocab). Use it to validate
the training entrypoint locally before launching a real run on Modal.

Usage:
    python scripts/local/train_local.py                       # synthetic, auto device
    python scripts/local/train_local.py --device cpu --samples 8 --epochs 1
    python scripts/local/train_local.py --keep                # keep temp data to inspect
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# MPS misses some ops (e.g. aten::_ctc_loss) → allow CPU fallback. Set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.run import run_training

# A few Vietnamese strings so the real tokenizer produces meaningful targets.
SAMPLE_TEXTS = [
    "xin chào việt nam",
    "hôm nay trời rất đẹp",
    "tôi đang học nhận dạng tiếng nói",
    "mô hình đa phương thức",
]


def make_synthetic_dataset(root: Path, n_samples: int):
    """Write fake .pt feature files + a 'grid' manifest the existing pipeline can load."""
    feat_dir = root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n_samples):
        ta = 40 + (i % 10)
        tv = ta // 2
        torch.save(
            {"audio": torch.randn(ta, 768), "visual": torch.randn(tv, 512)},
            feat_dir / f"sample_{i}.pt",
        )
        lines.append({"rel_path": f"sample_{i}.pt", "text": SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]})

    # Filename must contain 'grid' so the loader auto-detects GridDataset.
    manifest = root / "grid_synthetic_manifest.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    return str(manifest), str(feat_dir)


def build_config(manifest, data_root, ckpt_dir, device, epochs, use_mqot):
    """Minimal config matching the real Trainer/loader interface (full Whisper vocab)."""
    return {
        "model": {
            "audio_dim": 768,
            "visual_dim": 512,
            "d_model": 256,
            "num_encoder_layers": 2,
            "num_decoder_layers": 2,
            "num_heads": 4,
            "vocab_size": 51865,  # real Whisper multilingual vocab (targets come from tokenizer)
            "dropout": 0.1,
            "use_mqot": use_mqot,
            "use_backbones": False,
        },
        "loss": {"ctc_weight": 0.3, "ce_weight": 0.7, "quality_loss_weight": 0.1},
        "data": {
            "train_manifest": manifest,
            "val_manifest": manifest,
            "data_root": data_root,
            "batch_size": 2,
            "num_workers": 0,
            "use_precomputed_features": True,
        },
        "training": {
            "num_epochs": epochs,
            "learning_rate": 1e-4,
            "min_lr": 1e-6,
            "warmup_steps": 5,
            "weight_decay": 0.01,
            "gradient_clip": 5.0,
            "use_amp": False,
            "device": device,  # None → auto-detect; else force cpu/mps/cuda
            "patience": 10,
            "accum_steps": 1,
        },
        "logging": {"use_wandb": False, "checkpoint_dir": ckpt_dir},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda|mps|cpu (default: auto-detect)")
    ap.add_argument("--samples", type=int, default=8, help="synthetic sample count (ignored with --manifest)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--no-mqot", action="store_true", help="disable the MQOT path")
    ap.add_argument("--keep", action="store_true", help="keep the temp synthetic dataset")
    # Real-data mode: point at an existing manifest + feature dir (e.g. from prep_local.py).
    ap.add_argument("--manifest", default=None, help="train on a REAL manifest instead of synthetic")
    ap.add_argument("--data-root", default=None, help="feature root for --manifest")
    ap.add_argument("--ckpt-dir", default=None, help="checkpoint dir (real-data mode)")
    args = ap.parse_args()

    # --- Real-data mode ---
    if args.manifest:
        data_root = args.data_root or str(Path(args.manifest).parent)
        ckpt = args.ckpt_dir or str(Path(data_root) / "checkpoints_local")
        config = build_config(
            args.manifest, data_root, ckpt, args.device, args.epochs, not args.no_mqot
        )
        print(
            f"[local-train] REAL data | manifest={args.manifest} device={args.device or 'auto'} "
            f"epochs={args.epochs} use_mqot={not args.no_mqot}"
        )
        run_training(config)
        ckpts = sorted(p.name for p in Path(ckpt).glob("*.pt")) if Path(ckpt).exists() else []
        print(f"[local-train] PASS — trained on real data. checkpoints={ckpts} (dir={ckpt})")
        return

    # --- Synthetic mode (self-contained) ---
    workdir = Path(tempfile.mkdtemp(prefix="avsr_local_train_"))
    print(f"[local-train] workdir={workdir}")
    try:
        manifest, data_root = make_synthetic_dataset(workdir, args.samples)
        ckpt = workdir / "checkpoints"
        config = build_config(
            manifest, data_root, str(ckpt), args.device, args.epochs, not args.no_mqot
        )
        print(
            f"[local-train] device={args.device or 'auto'} samples={args.samples} "
            f"epochs={args.epochs} use_mqot={not args.no_mqot}"
        )
        run_training(config)
        ckpts = sorted(p.name for p in ckpt.glob("*.pt")) if ckpt.exists() else []
        print(f"[local-train] PASS — training entrypoint ran end-to-end. checkpoints={ckpts}")
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)
            print("[local-train] cleaned temp workdir")
        else:
            print(f"[local-train] kept temp workdir: {workdir}")


if __name__ == "__main__":
    main()
