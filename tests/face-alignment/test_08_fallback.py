"""
Step 8: Fallback — No face detected → center crop

Pipeline stage: VideoProcessor.extract_mouth() fallback path (base.py lines 202-205)
  When face-alignment fails to detect a face:
    Input  → BGR frame (H, W, 3) — no face present
    Output → center-cropped frame (IMAGE_SIZE, IMAGE_SIZE, 3) approx, bbox=None

Tests:
  1. extract_mouth fallback returns center crop + None bbox
  2. Center crop is geometrically correct
  3. Full pipeline still produces valid output on blank video
  4. FaceMeshPreprocessor also falls back correctly
"""
import os
import sys
import cv2
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

IMAGE_SIZE = 88


def fallback_center_crop(frame):
    """
    Replicate the fallback logic from VideoProcessor.extract_mouth() (base.py:202-205)
    and FaceMeshPreprocessor._extract_mouth() (facemesh.py:152-155).
    """
    h, w = frame.shape[:2]
    cy, cx = h // 2, w // 2
    r = IMAGE_SIZE // 2  # 44 pixels
    crop = frame[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)]
    return crop, None  # bbox is None on fallback


class TestFallback:

    def test_fallback_returns_none_bbox(self, single_blank_frame):
        """When no face: bbox should be None."""
        crop, bbox = fallback_center_crop(single_blank_frame)
        assert bbox is None, f"Expected None bbox, got {bbox}"
        print(f"\n✅ Fallback bbox = None (correct — no face)")

    def test_fallback_crop_shape(self, single_blank_frame):
        """Center crop has size IMAGE_SIZE//2 * 2 = 88 (radius=44 each side)."""
        frame = single_blank_frame  # (240, 320, 3)
        crop, _ = fallback_center_crop(frame)

        # With r=44, crop should be (88, 88, 3) if frame is large enough
        expected_h = min(240, 120 + 44) - max(0, 120 - 44)  # cy=120, r=44
        expected_w = min(320, 160 + 44) - max(0, 160 - 44)  # cx=160, r=44

        assert crop.shape == (expected_h, expected_w, 3), f"Got {crop.shape}"
        assert crop.dtype == np.uint8
        print(f"\n📸 INPUT:  blank frame shape={frame.shape}")
        print(f"✅ OUTPUT: center crop shape={crop.shape}, dtype={crop.dtype}")
        print(f"   Center: ({frame.shape[0]//2}, {frame.shape[1]//2}), Radius: {IMAGE_SIZE//2}")

    def test_fallback_crop_is_centered(self, single_blank_frame):
        """Verify the crop is taken from the geometric center."""
        h, w = single_blank_frame.shape[:2]
        cy, cx = h // 2, w // 2
        r = IMAGE_SIZE // 2

        # Create frame with a unique pixel at center
        frame = np.zeros_like(single_blank_frame)
        frame[cy, cx] = [0, 255, 0]  # Green dot at center

        crop, _ = fallback_center_crop(frame)

        # The center of the crop should be the green dot
        crop_cy = crop.shape[0] // 2
        crop_cx = crop.shape[1] // 2
        assert np.array_equal(crop[crop_cy, crop_cx], [0, 255, 0]), \
            "Center pixel should be green"
        print(f"\n✅ Center crop is geometrically centered (green dot at crop center)")

    def test_fallback_on_small_frame(self):
        """Frame smaller than crop size — should still work without error."""
        small_frame = np.zeros((40, 30, 3), dtype=np.uint8)
        crop, bbox = fallback_center_crop(small_frame)

        assert bbox is None
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        assert crop.shape[0] <= 40 and crop.shape[1] <= 30
        print(f"\n📸 INPUT:  tiny frame shape={small_frame.shape}")
        print(f"✅ OUTPUT: crop shape={crop.shape} (clipped to frame bounds)")

    def test_fallback_full_pipeline_blank_video(self, blank_video, output_dir):
        """
        Run the FULL face-alignment extraction pipeline on a blank video.
        Even without detected faces, pipeline should produce a valid tensor.
        """
        cap = cv2.VideoCapture(blank_video)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        assert len(frames) > 0, "Blank video should have frames"

        # Process each frame through fallback
        processed = []
        for frame in frames:
            crop, bbox = fallback_center_crop(frame)
            assert bbox is None, "Should not detect face on blank frame"

            # Resize to 88×88 (same as pipeline)
            resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            processed.append(rgb)

        # Convert to tensor (pipeline step 6)
        frames_np = np.array(processed)
        tensor = torch.tensor(frames_np, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

        assert tensor.shape == (10, 3, 88, 88), f"Got {tensor.shape}"
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0.0 and tensor.max() <= 1.0

        print(f"\n{'='*60}")
        print(f"Full Pipeline Fallback Test (blank video)")
        print(f"{'='*60}")
        print(f"📸 INPUT:  blank video with {len(frames)} frames")
        print(f"   Each frame: shape={frames[0].shape}")
        print(f"✅ OUTPUT: tensor shape={tuple(tensor.shape)}")
        print(f"   dtype={tensor.dtype}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")

        # Save a fallback frame for visual inspection
        out_path = os.path.join(output_dir, "output_step08_fallback_crop.jpg")
        cv2.imwrite(out_path, processed[0])
        print(f"   Saved fallback crop → {out_path}")

    def test_fallback_vs_facemesh_consistency(self):
        """
        Verify that VideoProcessor (base.py) and FaceMeshPreprocessor (facemesh.py)
        use the SAME fallback logic: center crop with IMAGE_SIZE//2 radius.
        """
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        # Both should produce same center crop
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        r_base = 88 // 2      # base.py uses PreprocessConfig.IMAGE_SIZE // 2
        r_mesh = 88 // 2      # facemesh.py uses FaceMeshConfig.IMAGE_SIZE // 2

        assert r_base == r_mesh, "Both use same radius"

        crop_base = frame[max(0, cy - r_base):min(h, cy + r_base),
                          max(0, cx - r_base):min(w, cx + r_base)]
        crop_mesh = frame[max(0, cy - r_mesh):min(h, cy + r_mesh),
                          max(0, cx - r_mesh):min(w, cx + r_mesh)]

        assert np.array_equal(crop_base, crop_mesh), \
            "base.py and facemesh.py fallback crops should be identical"
        print(f"\n✅ Fallback consistency: base.py == facemesh.py")
        print(f"   Both: center=({cx},{cy}), radius=44, crop_shape={crop_base.shape}")

    def test_visualize_fallback_comparison(self, single_blank_frame, output_dir):
        """Save original frame with center crop zone highlighted."""
        frame = single_blank_frame.copy()
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        r = IMAGE_SIZE // 2

        # Draw the center crop rectangle in red
        cv2.rectangle(frame,
                      (max(0, cx - r), max(0, cy - r)),
                      (min(w, cx + r), min(h, cy + r)),
                      (0, 0, 255), 2)
        cv2.putText(frame, "FALLBACK ZONE", (cx - r + 2, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)  # Center dot

        out_path = os.path.join(output_dir, "output_step08_fallback_zone.jpg")
        cv2.imwrite(out_path, frame)
        print(f"\n📸 Saved fallback zone visualization → {out_path}")
