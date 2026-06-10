"""Verify a frame-based feature shard (downloaded from the Modal volume) before a big run.

Checks BOTH layers:
  1. Raw stored schema  — video.pth (uint8 [T,H,W,C]) + audio.pth (fp16 [T_a,768]) + txt.
  2. Train decode path  — build_webdataset → visual [T,C,H,W] float in [0,1], audio float.

Also dumps a montage PNG of mouth crops per sample so you can EYEBALL that face-alignment
produced real mouths (not center-crop fallbacks / garbage).

Usage:
    python scripts/local/verify_frame_shard.py <shard.tar | glob> [out_dir] [--n N]

Example:
    modal volume get avsr-volume \
        /vicocktail_features/vicocktail-avvn-train-000000-b000-000000.tar /tmp/verify/
    python scripts/local/verify_frame_shard.py /tmp/verify/*.tar /tmp/verify/out --n 6
"""
import argparse
import glob
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import webdataset as wds

# Make `src` importable when run as a standalone script (sys.path[0] is this file's dir).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.shards import build_webdataset  # noqa: E402


def _save_montage(video_uint8: torch.Tensor, path: Path, n_frames: int = 16, cols: int = 8) -> None:
    """video_uint8: [T, H, W, C] RGB uint8 → tiled grid PNG of evenly-spaced frames."""
    T = video_uint8.shape[0]
    idx = np.linspace(0, T - 1, min(n_frames, T)).astype(int)
    frames = [video_uint8[j].numpy() for j in idx]
    h, w = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for k, f in enumerate(frames):
        r, c = divmod(k, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))  # frames are RGB → BGR for cv2


class _FakeTokenizer:
    eot_token_id = 50257

    def encode(self, text):
        return [ord(c) % 97 for c in text] or [0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", help="path or glob to .tar shard(s)")
    ap.add_argument("out_dir", nargs="?", default="verify_out", help="where to write montage PNGs")
    ap.add_argument("--n", type=int, default=6, help="number of samples to visualize")
    args = ap.parse_args()

    shard_paths = sorted(glob.glob(args.shards))
    if not shard_paths:
        print(f"No shards match: {args.shards}")
        return 1
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Shards: {len(shard_paths)} → {shard_paths}\nMontages → {out_dir}/\n")

    # --- Layer 1: raw stored schema ---
    n = 0
    hw_set, t_video, t_audio, audio_dtypes = set(), [], [], set()
    for sample in wds.WebDataset(shard_paths):
        key = sample["__key__"]
        assert "video.pth" in sample, f"{key}: missing video.pth (schema mismatch — old feature shard?)"
        assert "audio.pth" in sample, f"{key}: missing audio.pth"
        video = torch.load(io.BytesIO(sample["video.pth"]))   # [T,H,W,C] uint8
        audio = torch.load(io.BytesIO(sample["audio.pth"]))   # [T_a,768] fp16
        txt = sample["txt"].decode("utf-8") if isinstance(sample["txt"], (bytes, bytearray)) else sample["txt"]

        assert video.dtype == torch.uint8 and video.ndim == 4 and video.shape[-1] == 3, \
            f"{key}: bad video {tuple(video.shape)} {video.dtype}"
        assert audio.ndim == 2 and audio.shape[1] == 768, f"{key}: bad audio {tuple(audio.shape)}"
        hw_set.add(tuple(video.shape[1:3]))
        t_video.append(video.shape[0]); t_audio.append(audio.shape[0]); audio_dtypes.add(audio.dtype)

        if n < args.n:
            _save_montage(video, out_dir / f"{n:02d}_{key}.png")
            print(f"  [{n}] {key}: video={tuple(video.shape)} {video.dtype} "
                  f"audio={tuple(audio.shape)} {audio.dtype} | text={txt[:60]!r}")
        n += 1

    print(f"\nLayer 1 (raw schema) OK — {n} samples")
    print(f"  frame H×W:      {sorted(hw_set)}   (should be one consistent size)")
    print(f"  T (video):      min={min(t_video)} max={max(t_video)} mean={np.mean(t_video):.0f}")
    print(f"  T_a (audio):    min={min(t_audio)} max={max(t_audio)} mean={np.mean(t_audio):.0f}")
    print(f"  audio dtype:    {audio_dtypes}   (expect torch.float16)")

    # --- Layer 2: train decode path (what the model actually receives) ---
    ds = build_webdataset(shard_paths, _FakeTokenizer(), train=False, augment=False)
    s = next(iter(ds))
    v, a = s["visual"], s["audio"]
    print("\nLayer 2 (train decode) OK")
    print(f"  visual: {tuple(v.shape)} {v.dtype}  range=[{float(v.min()):.3f}, {float(v.max()):.3f}]  (expect [T,3,H,W] float 0..1)")
    print(f"  audio:  {tuple(a.shape)} {a.dtype}  (fp16→float)")
    assert v.ndim == 4 and v.shape[1] == 3 and 0.0 <= float(v.min()) and float(v.max()) <= 1.0
    print(f"\n✅ All checks passed. Open the PNGs in {out_dir}/ to eyeball the mouth crops.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
