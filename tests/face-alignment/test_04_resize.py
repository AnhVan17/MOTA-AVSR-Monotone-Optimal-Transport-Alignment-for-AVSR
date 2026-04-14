"""
Step 4: Resize — cv2.resize() to RESNET_INPUT_SIZE × RESNET_INPUT_SIZE (88×88)

Pipeline stage: cv2.resize(crop, (88, 88))
  Input  → variable-size cropped BGR frame (h', w', 3)
  Output → resized frame (88, 88, 3) uint8
"""
import os
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

RESNET_INPUT_SIZE = 88


class TestResize:

    def test_resize_from_large(self, output_dir):
        """Resize a 200×150 crop down to 88×88."""
        large_crop = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
        resized = cv2.resize(large_crop, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))

        assert resized.shape == (88, 88, 3)
        assert resized.dtype == np.uint8
        print(f"\n📸 INPUT:  shape={large_crop.shape}, dtype={large_crop.dtype}")
        print(f"✅ OUTPUT: shape={resized.shape}, dtype={resized.dtype}")

    def test_resize_from_small(self):
        """Resize a 30×40 crop up to 88×88."""
        small_crop = np.random.randint(0, 255, (30, 40, 3), dtype=np.uint8)
        resized = cv2.resize(small_crop, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))

        assert resized.shape == (88, 88, 3)
        print(f"\n📸 INPUT:  shape={small_crop.shape} (upscale)")
        print(f"✅ OUTPUT: shape={resized.shape}")

    def test_resize_from_exact(self):
        """Already 88×88 — should be identity."""
        exact_crop = np.random.randint(0, 255, (88, 88, 3), dtype=np.uint8)
        resized = cv2.resize(exact_crop, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))

        assert resized.shape == (88, 88, 3)
        assert np.array_equal(exact_crop, resized), "Exact-size resize should be identity"
        print(f"\n✅ Exact size 88×88 → no change (identity resize)")

    def test_resize_nonsquare(self):
        """Non-square crop gets resized to square 88×88."""
        nonsquare = np.random.randint(0, 255, (120, 60, 3), dtype=np.uint8)
        resized = cv2.resize(nonsquare, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))

        assert resized.shape == (88, 88, 3)
        print(f"\n📸 INPUT:  non-square {nonsquare.shape}")
        print(f"✅ OUTPUT: square {resized.shape}")

    def test_resize_preserves_dtype(self):
        """Output dtype should remain uint8."""
        crop = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        resized = cv2.resize(crop, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))
        assert resized.dtype == np.uint8
        print(f"\n✅ dtype preserved: {resized.dtype}")

    def test_visualize_resize(self, sample_cropped_frame, output_dir):
        """Save before/after resize for visual inspection."""
        crop = sample_cropped_frame
        resized = cv2.resize(crop, (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))

        # Save both
        before_path = os.path.join(output_dir, "output_step04_before_resize.jpg")
        after_path = os.path.join(output_dir, "output_step04_resized.jpg")
        cv2.imwrite(before_path, crop)
        cv2.imwrite(after_path, resized)

        print(f"\n📸 INPUT:  shape={crop.shape} → saved {before_path}")
        print(f"✅ OUTPUT: shape={resized.shape} → saved {after_path}")
        assert os.path.exists(before_path)
        assert os.path.exists(after_path)
