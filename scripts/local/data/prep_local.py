"""Local preprocessing of a few ViCocktail samples (raw .tar -> .pt features + manifest).

Runs the real ViCocktailPreprocessor (face-alignment crop + ResNet visual + Whisper audio) on
the local machine (CPU), capped at --max-samples so it finishes quickly. Output feeds
train_local-style training on real data.

Usage:
    python scripts/local/data/prep_local.py --data-root data/vicocktail_raw \\
        --output-dir data/vicocktail_features --max-samples 40
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.preprocessors.vicocktail import ViCocktailPreprocessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/vicocktail_raw", help="dir chứa .tar shards")
    ap.add_argument("--output-dir", default="data/vicocktail_features", help="nơi lưu .pt")
    ap.add_argument(
        "--manifest",
        default="data/vicocktail_features/vicocktail_local_manifest.jsonl",
        help="đường dẫn manifest output (tên chứa 'vicocktail' để loader nhận đúng dataset)",
    )
    ap.add_argument("--max-samples", type=int, default=40)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[prep] data_root={args.data_root} -> output={args.output_dir} | max_samples={args.max_samples}")

    pre = ViCocktailPreprocessor(data_root=args.data_root, use_precropped=False)
    pre.run(
        output_manifest=args.manifest,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

    n = sum(1 for _ in open(args.manifest, encoding="utf-8")) if os.path.exists(args.manifest) else 0
    print(f"[prep] DONE — {n} samples -> manifest {args.manifest}")


if __name__ == "__main__":
    main()
