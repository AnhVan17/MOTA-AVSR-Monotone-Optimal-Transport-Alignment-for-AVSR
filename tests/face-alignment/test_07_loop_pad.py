"""
Step 7: Loop Pad — Pad short videos to target frame count (375 = 15s × 25fps)

Pipeline stage: VideoProcessor._loop_pad_video(video_tensor, target_frames)
  Input  → tensor (T, C, H, W) where T < target_frames
  Output → tensor (target_frames, C, H, W) via loop repetition

Also handles:
  - Trimming if T > target_frames
  - No-op if T == target_frames
"""
import os
import sys
import torch
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

TARGET_FRAMES = 375  # 15s × 25fps


def loop_pad_video(video_tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    """
    Replicate VideoProcessor._loop_pad_video() from base.py (lines 124-158).
    Input:  (T, C, H, W)
    Output: (target_frames, C, H, W)
    """
    T = video_tensor.size(0)
    if T >= target_frames:
        return video_tensor[:target_frames]

    repeats = (target_frames // T) + 1
    looped = video_tensor.repeat(repeats, 1, 1, 1)  # Repeat along T
    return looped[:target_frames]


class TestLoopPad:

    def test_short_video_pad(self):
        """10-frame video padded to 375 frames."""
        short = torch.randn(10, 3, 88, 88)
        padded = loop_pad_video(short, TARGET_FRAMES)

        assert padded.shape == (375, 3, 88, 88), f"Got {padded.shape}"
        print(f"\n📸 INPUT:  shape={tuple(short.shape)} (T=10)")
        print(f"✅ OUTPUT: shape={tuple(padded.shape)} (T=375, padded via loop)")

    def test_long_video_trim(self):
        """500-frame video trimmed to 375."""
        long = torch.randn(500, 3, 88, 88)
        trimmed = loop_pad_video(long, TARGET_FRAMES)

        assert trimmed.shape == (375, 3, 88, 88)
        print(f"\n📸 INPUT:  shape={tuple(long.shape)} (T=500)")
        print(f"✅ OUTPUT: shape={tuple(trimmed.shape)} (T=375, trimmed)")

    def test_exact_length_noop(self):
        """375 frames → no change."""
        exact = torch.randn(375, 3, 88, 88)
        result = loop_pad_video(exact, TARGET_FRAMES)

        assert result.shape == (375, 3, 88, 88)
        assert torch.equal(exact, result), "Exact-length should be identity"
        print(f"\n✅ Exact length (375) → no-op (identity)")

    def test_loop_content_correct(self):
        """Padded frames should be repeats of original frames."""
        # Create 5 distinct frames
        short = torch.zeros(5, 3, 88, 88)
        for i in range(5):
            short[i] = float(i)  # Each frame has unique value

        padded = loop_pad_video(short, 13)  # Pad to 13

        assert padded.shape == (13, 3, 88, 88)

        # Frame 0 should equal frame 5, frame 10
        assert torch.equal(padded[0], padded[5]), "Frame 0 != Frame 5 (loop broken)"
        assert torch.equal(padded[0], padded[10]), "Frame 0 != Frame 10 (loop broken)"
        # Frame 1 should equal frame 6, frame 11
        assert torch.equal(padded[1], padded[6]), "Frame 1 != Frame 6 (loop broken)"
        assert torch.equal(padded[1], padded[11]), "Frame 1 != Frame 11 (loop broken)"

        print(f"\n✅ Loop content verified:")
        print(f"   padded[0] == padded[5] == padded[10] ✓")
        print(f"   padded[1] == padded[6] == padded[11] ✓")

    def test_single_frame_pad(self):
        """Edge case: 1 frame padded to 375 (all frames identical)."""
        single = torch.randn(1, 3, 88, 88)
        padded = loop_pad_video(single, TARGET_FRAMES)

        assert padded.shape == (375, 3, 88, 88)
        # All frames should be identical
        for i in range(1, TARGET_FRAMES):
            assert torch.equal(padded[0], padded[i]), f"Frame {i} differs from frame 0"

        print(f"\n✅ Single frame → 375 identical frames")

    def test_two_frame_pad(self):
        """Edge case: 2 frames looped — alternating pattern."""
        two = torch.zeros(2, 3, 88, 88)
        two[0] = 0.0
        two[1] = 1.0

        padded = loop_pad_video(two, 7)

        assert padded.shape == (7, 3, 88, 88)
        # Pattern: 0, 1, 0, 1, 0, 1, 0
        assert torch.equal(padded[0], padded[2])
        assert torch.equal(padded[1], padded[3])
        print(f"\n✅ 2-frame loop: 0,1,0,1,0,1,0 pattern verified")

    def test_summary(self):
        """Print full summary of loop pad behavior."""
        test_cases = [
            (3, 10, "short → pad"),
            (10, 10, "exact → no-op"),
            (15, 10, "long → trim"),
            (1, 375, "single → repeat"),
        ]

        print(f"\n{'='*60}")
        print(f"Step 7 Summary: Loop Pad Video")
        print(f"{'='*60}")
        for T_in, target, desc in test_cases:
            inp = torch.randn(T_in, 3, 88, 88)
            out = loop_pad_video(inp, target)
            print(f"  {desc:20s}: ({T_in}, 3, 88, 88) → ({out.shape[0]}, 3, 88, 88)")
