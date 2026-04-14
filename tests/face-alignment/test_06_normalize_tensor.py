"""
Step 6: Normalize to Tensor — np.ndarray → torch.Tensor, /255.0, permute

Pipeline stage: VideoProcessor.process() final conversion
  Input  → np.ndarray of frames (T, H, W, C) uint8, range [0, 255]
  Output →
    RGB mode:       torch.Tensor (T, 3, 88, 88) float32, range [0.0, 1.0]
    Grayscale mode: torch.Tensor (T, 1, 88, 88) float32, range [0.0, 1.0]
"""
import os
import sys
import cv2
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def normalize_to_tensor_rgb(frames_np: np.ndarray) -> torch.Tensor:
    """
    Replicate VideoProcessor.process() tensor conversion (RGB path).
    Input:  (T, H, W, C) uint8 [0, 255]
    Output: (T, C, H, W) float32 [0.0, 1.0]
    """
    return torch.tensor(frames_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0


def normalize_to_tensor_gray(frames_np: np.ndarray) -> torch.Tensor:
    """
    Replicate VideoProcessor.process() tensor conversion (Grayscale path).
    Input:  (T, H, W) uint8 [0, 255]
    Output: (T, 1, H, W) float32 [0.0, 1.0]
    """
    return torch.tensor(frames_np, dtype=torch.float32).unsqueeze(1) / 255.0


class TestNormalizeTensor:

    @pytest.fixture
    def rgb_frames(self):
        """5 fake RGB frames, shape (5, 88, 88, 3) uint8."""
        return np.random.randint(0, 255, (5, 88, 88, 3), dtype=np.uint8)

    @pytest.fixture
    def gray_frames(self):
        """5 fake grayscale frames, shape (5, 88, 88) uint8."""
        return np.random.randint(0, 255, (5, 88, 88), dtype=np.uint8)

    # ── RGB Path ──

    def test_rgb_input_properties(self, rgb_frames):
        """Verify input array properties before conversion."""
        assert rgb_frames.shape == (5, 88, 88, 3)
        assert rgb_frames.dtype == np.uint8
        print(f"\n📸 INPUT: np.ndarray shape={rgb_frames.shape}, dtype={rgb_frames.dtype}")
        print(f"   Pixel range: [{rgb_frames.min()}, {rgb_frames.max()}]")

    def test_rgb_tensor_shape(self, rgb_frames):
        """RGB: (T, H, W, C) → (T, C, H, W)."""
        tensor = normalize_to_tensor_rgb(rgb_frames)
        assert tensor.shape == (5, 3, 88, 88), f"Got {tensor.shape}"
        print(f"\n📸 INPUT:  np shape = {rgb_frames.shape} (T, H, W, C)")
        print(f"✅ OUTPUT: tensor shape = {tensor.shape} (T, C, H, W)")

    def test_rgb_tensor_dtype(self, rgb_frames):
        """Output should be float32."""
        tensor = normalize_to_tensor_rgb(rgb_frames)
        assert tensor.dtype == torch.float32
        print(f"\n✅ dtype: {tensor.dtype}")

    def test_rgb_tensor_range(self, rgb_frames):
        """After /255.0, values should be in [0.0, 1.0]."""
        tensor = normalize_to_tensor_rgb(rgb_frames)
        assert tensor.min() >= 0.0, f"Min={tensor.min()}"
        assert tensor.max() <= 1.0, f"Max={tensor.max()}"
        print(f"\n✅ Value range: [{tensor.min():.4f}, {tensor.max():.4f}] ⊂ [0.0, 1.0]")

    def test_rgb_permute_correctness(self, rgb_frames):
        """Verify that permute(0,3,1,2) correctly reorders dimensions."""
        tensor = normalize_to_tensor_rgb(rgb_frames)
        # First frame, first channel (R in RGB) should equal original [:,:,0] / 255
        expected = rgb_frames[0, :, :, 0].astype(np.float32) / 255.0
        actual = tensor[0, 0, :, :].numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        print(f"\n✅ Permute correctness verified (frame[0], channel[0])")

    # ── Grayscale Path ──

    def test_gray_tensor_shape(self, gray_frames):
        """Grayscale: (T, H, W) → (T, 1, H, W)."""
        tensor = normalize_to_tensor_gray(gray_frames)
        assert tensor.shape == (5, 1, 88, 88), f"Got {tensor.shape}"
        print(f"\n📸 INPUT:  np shape = {gray_frames.shape} (T, H, W)")
        print(f"✅ OUTPUT: tensor shape = {tensor.shape} (T, 1, H, W)")

    def test_gray_tensor_range(self, gray_frames):
        """Grayscale values also in [0.0, 1.0]."""
        tensor = normalize_to_tensor_gray(gray_frames)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0
        print(f"\n✅ Grayscale range: [{tensor.min():.4f}, {tensor.max():.4f}]")

    # ── Visualization ──

    def test_visualize_tensor(self, rgb_frames, output_dir):
        """Save first frame of tensor as image for visual check."""
        tensor = normalize_to_tensor_rgb(rgb_frames)

        # Convert back: (C, H, W) → (H, W, C) and scale to [0, 255]
        frame_tensor = tensor[0]  # (3, 88, 88)
        frame_np = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(output_dir, "output_step06_tensor.png")
        cv2.imwrite(out_path, frame_bgr)

        print(f"\n📸 Tensor[0] reconstructed and saved → {out_path}")
        print(f"   Tensor shape: {tensor.shape}")
        print(f"   Tensor dtype: {tensor.dtype}")
        print(f"   Tensor min={tensor.min():.4f}, max={tensor.max():.4f}")
        assert os.path.exists(out_path)

    def test_summary(self, rgb_frames, gray_frames):
        """Print full summary of the normalization step."""
        rgb_tensor = normalize_to_tensor_rgb(rgb_frames)
        gray_tensor = normalize_to_tensor_gray(gray_frames)

        print(f"\n{'='*60}")
        print(f"Step 6 Summary: Normalize to Tensor")
        print(f"{'='*60}")
        print(f"RGB path:")
        print(f"  Input:  np.ndarray {rgb_frames.shape} {rgb_frames.dtype} [{rgb_frames.min()},{rgb_frames.max()}]")
        print(f"  Output: Tensor     {tuple(rgb_tensor.shape)} {rgb_tensor.dtype} [{rgb_tensor.min():.3f},{rgb_tensor.max():.3f}]")
        print(f"Gray path:")
        print(f"  Input:  np.ndarray {gray_frames.shape} {gray_frames.dtype} [{gray_frames.min()},{gray_frames.max()}]")
        print(f"  Output: Tensor     {tuple(gray_tensor.shape)} {gray_tensor.dtype} [{gray_tensor.min():.3f},{gray_tensor.max():.3f}]")
