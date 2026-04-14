"""
Step 5: Color Conversion — BGR → RGB (or BGR → Grayscale)

Pipeline stage: cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) or cv2.COLOR_BGR2GRAY
  Input  → resized BGR frame (88, 88, 3) uint8
  Output →
    RGB mode:       (88, 88, 3) uint8 — channels reordered
    Grayscale mode: (88, 88)    uint8 — single channel
"""
import os
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class TestColorConversion:

    @pytest.fixture
    def bgr_frame(self):
        """Create a synthetic 88×88 BGR frame with known channel values."""
        frame = np.zeros((88, 88, 3), dtype=np.uint8)
        frame[:, :, 0] = 50   # Blue channel
        frame[:, :, 1] = 100  # Green channel
        frame[:, :, 2] = 200  # Red channel
        return frame

    # ── RGB Conversion ──

    def test_bgr_to_rgb_shape(self, bgr_frame):
        """RGB output shape should be same as BGR."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        assert rgb.shape == (88, 88, 3)
        assert rgb.dtype == np.uint8
        print(f"\n📸 INPUT:  BGR shape={bgr_frame.shape}")
        print(f"✅ OUTPUT: RGB shape={rgb.shape}")

    def test_bgr_to_rgb_channel_swap(self, bgr_frame):
        """B and R channels should be swapped, G stays."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        # BGR[0]=Blue(50) → RGB[2]=Blue(50)
        assert np.all(rgb[:, :, 2] == 50), "Blue should move to channel 2"
        # BGR[1]=Green(100) → RGB[1]=Green(100)
        assert np.all(rgb[:, :, 1] == 100), "Green should stay at channel 1"
        # BGR[2]=Red(200) → RGB[0]=Red(200)
        assert np.all(rgb[:, :, 0] == 200), "Red should move to channel 0"

        print(f"\n✅ Channel swap verified:")
        print(f"   BGR: B={bgr_frame[0,0,0]}, G={bgr_frame[0,0,1]}, R={bgr_frame[0,0,2]}")
        print(f"   RGB: R={rgb[0,0,0]}, G={rgb[0,0,1]}, B={rgb[0,0,2]}")

    # ── Grayscale Conversion ──

    def test_bgr_to_gray_shape(self, bgr_frame):
        """Grayscale output should be (H, W) with no channel dimension."""
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        assert gray.shape == (88, 88), f"Expected (88,88), got {gray.shape}"
        assert gray.dtype == np.uint8
        print(f"\n📸 INPUT:  BGR shape={bgr_frame.shape}")
        print(f"✅ OUTPUT: Gray shape={gray.shape}, dtype={gray.dtype}")

    def test_grayscale_values(self, bgr_frame):
        """Grayscale = 0.114*B + 0.587*G + 0.299*R (approx)."""
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        expected = int(0.114 * 50 + 0.587 * 100 + 0.299 * 200)
        actual = int(gray[44, 44])

        # Allow ±2 for rounding
        assert abs(actual - expected) <= 2, f"Expected ~{expected}, got {actual}"
        print(f"\n✅ Grayscale pixel value = {actual} (expected ~{expected})")

    # ── Visual Output ──

    def test_visualize_rgb(self, output_dir):
        """Save a colorful frame in BGR and RGB for comparison."""
        # Create frame with distinct colors in regions
        frame = np.zeros((88, 88, 3), dtype=np.uint8)
        frame[:44, :44] = [255, 0, 0]     # Blue (top-left in BGR)
        frame[:44, 44:] = [0, 255, 0]     # Green (top-right)
        frame[44:, :44] = [0, 0, 255]     # Red (bottom-left in BGR)
        frame[44:, 44:] = [255, 255, 0]   # Cyan (bottom-right in BGR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bgr_path = os.path.join(output_dir, "output_step05_bgr.jpg")
        rgb_path = os.path.join(output_dir, "output_step05_rgb.jpg")
        gray_path = os.path.join(output_dir, "output_step05_gray.jpg")

        cv2.imwrite(bgr_path, frame)
        # Save RGB as BGR (for correct display in image viewers)
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(gray_path, gray)

        print(f"\n📸 Saved BGR  → {bgr_path}")
        print(f"📸 Saved RGB  → {rgb_path}")
        print(f"📸 Saved Gray → {gray_path}")

    def test_pipeline_uses_rgb_by_default(self):
        """PreprocessConfig.USE_GRAYSCALE = False → pipeline uses BGR→RGB."""
        # Import config to verify default
        from src.data.preprocessors.base import PreprocessConfig
        assert PreprocessConfig.USE_GRAYSCALE is False, \
            "Default should be RGB mode (USE_GRAYSCALE=False)"
        print(f"\n✅ PreprocessConfig.USE_GRAYSCALE = {PreprocessConfig.USE_GRAYSCALE}")
        print(f"   Pipeline uses BGR→RGB conversion by default")
