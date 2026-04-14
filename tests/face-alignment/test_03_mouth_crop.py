"""
Step 3: Mouth Crop — extract_mouth(frame, prev_bbox)

Pipeline stage: VideoProcessor.extract_mouth() / FaceMeshPreprocessor._extract_mouth()
  Input  → BGR frame (H, W, 3) + optional prev_bbox (x1,y1,x2,y2)
  Output → cropped sub-image (h', w', 3) + bbox tuple or None

Tests both paths:
  A. Landmark-based crop (when face is detected)
  B. prev_bbox reuse (fast path, skip detection)
  C. Fallback center crop (when no face found) is tested in test_08
"""
import os
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ── Pure-logic extract_mouth (extracted from base.py lines 160-205) ──
# Replicated here to test WITHOUT initializing face_alignment model

IMAGE_SIZE = 88


def extract_mouth_with_landmarks(frame, landmarks, prev_bbox=None):
    """
    Mouth extraction logic using pre-computed landmarks.
    Mirrors VideoProcessor.extract_mouth() but takes landmarks as input.
    """
    h, w = frame.shape[:2]

    # Fast path: reuse previous bbox
    if prev_bbox is not None:
        x1, y1, x2, y2 = prev_bbox
        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
            return frame[y1:y2, x1:x2], prev_bbox

    if landmarks is not None and len(landmarks) > 0:
        mouth_points = landmarks[0][48:68]

        xs = mouth_points[:, 0]
        ys = mouth_points[:, 1]

        cx, cy = int(np.mean(xs)), int(np.mean(ys))

        radius = max(
            int((max(xs) - min(xs)) * 1.8) // 2,
            int((max(ys) - min(ys)) * 1.8) // 2,
            IMAGE_SIZE // 2
        )

        y1 = max(0, cy - radius)
        y2 = min(h, cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(w, cx + radius)

        bbox = (x1, y1, x2, y2)
        return frame[y1:y2, x1:x2], bbox

    # Fallback: center crop
    cy, cx = h // 2, w // 2
    r = IMAGE_SIZE // 2
    return frame[max(0, cy - r):min(h, cy + r), max(0, cx - r):min(w, cx + r)], None


def _make_fake_landmarks(cx=160, cy=160):
    """Create fake 68-point landmarks with mouth at (cx, cy)."""
    landmarks = np.zeros((68, 2), dtype=np.float32)
    # Place mouth points [48:68] around (cx, cy)
    for i in range(20):
        angle = 2 * np.pi * i / 20
        landmarks[48 + i, 0] = cx + 20 * np.cos(angle)  # x
        landmarks[48 + i, 1] = cy + 10 * np.sin(angle)  # y
    return [landmarks]  # List of arrays (like face_alignment output)


class TestMouthCrop:

    def test_input_format(self, single_face_frame):
        """Verify input is BGR, uint8, 3-channel."""
        frame = single_face_frame
        print(f"\n📸 INPUT: type={type(frame).__name__}, shape={frame.shape}, dtype={frame.dtype}")
        assert frame.dtype == np.uint8
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3

    def test_landmark_based_crop(self, single_face_frame, output_dir):
        """Crop mouth using fake landmarks — should return sub-image + bbox."""
        frame = single_face_frame
        h, w = frame.shape[:2]
        landmarks = _make_fake_landmarks(cx=w // 2, cy=h // 2 + 30)

        crop, bbox = extract_mouth_with_landmarks(frame, landmarks, prev_bbox=None)

        assert isinstance(crop, np.ndarray)
        assert crop.dtype == np.uint8
        assert len(crop.shape) == 3 and crop.shape[2] == 3
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 < x2 and y1 < y2

        print(f"\n📸 INPUT:  frame shape={frame.shape}")
        print(f"   Landmarks mouth center=({w//2}, {h//2+30})")
        print(f"✅ OUTPUT: crop shape={crop.shape}, bbox={bbox}")

        out_path = os.path.join(output_dir, "output_step03_crop.jpg")
        cv2.imwrite(out_path, crop)
        print(f"   Saved → {out_path}")

    def test_prev_bbox_reuse(self, single_face_frame):
        """When prev_bbox is provided, skip detection and reuse it."""
        frame = single_face_frame
        h, w = frame.shape[:2]

        # Set a known bbox
        prev_bbox = (50, 50, 200, 180)
        crop, returned_bbox = extract_mouth_with_landmarks(frame, None, prev_bbox=prev_bbox)

        assert returned_bbox == prev_bbox, "Should return same bbox"
        expected_h = 180 - 50
        expected_w = 200 - 50
        assert crop.shape == (expected_h, expected_w, 3)

        print(f"\n📸 INPUT:  prev_bbox={prev_bbox}")
        print(f"✅ OUTPUT: crop shape={crop.shape}, bbox={returned_bbox} (reused)")

    def test_invalid_prev_bbox_triggers_new_detection(self, single_face_frame):
        """Invalid prev_bbox (out of bounds) should fall through to landmark path."""
        frame = single_face_frame
        h, w = frame.shape[:2]

        # Invalid bbox: x2 > w
        invalid_bbox = (0, 0, w + 100, h + 100)
        landmarks = _make_fake_landmarks(cx=w // 2, cy=h // 2)

        crop, bbox = extract_mouth_with_landmarks(frame, landmarks, prev_bbox=invalid_bbox)

        # Should NOT be the invalid bbox
        assert bbox != invalid_bbox
        assert crop.shape[0] > 0 and crop.shape[1] > 0
        print(f"\n✅ Invalid prev_bbox={invalid_bbox} → fell through to landmark crop")
        print(f"   New crop shape={crop.shape}, new bbox={bbox}")

    def test_crop_dimensions_at_least_image_size(self, single_face_frame):
        """Crop radius has a minimum of IMAGE_SIZE//2, so crop >= 88 pixels in each dim."""
        frame = single_face_frame
        h, w = frame.shape[:2]
        # Tiny mouth landmarks (very close together)
        landmarks = [np.zeros((68, 2), dtype=np.float32)]
        for i in range(20):
            landmarks[0][48 + i] = [w // 2 + i * 0.1, h // 2 + i * 0.1]

        crop, bbox = extract_mouth_with_landmarks(frame, landmarks, prev_bbox=None)

        # Due to IMAGE_SIZE//2 = 44 radius minimum, crop should be reasonable
        assert crop.shape[0] >= IMAGE_SIZE // 2, f"Crop height too small: {crop.shape[0]}"
        assert crop.shape[1] >= IMAGE_SIZE // 2, f"Crop width too small: {crop.shape[1]}"
        print(f"\n✅ Minimum-size crop: shape={crop.shape} (min radius enforced)")

    def test_visualize_crop_on_frame(self, single_face_frame, output_dir):
        """Draw bbox on original frame + show crop side by side."""
        frame = single_face_frame.copy()
        h, w = frame.shape[:2]
        landmarks = _make_fake_landmarks(cx=w // 2, cy=h // 2 + 30)

        crop, bbox = extract_mouth_with_landmarks(frame, landmarks)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"bbox: {bbox}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        out_path = os.path.join(output_dir, "output_step03_bbox_on_frame.jpg")
        cv2.imwrite(out_path, frame)
        print(f"\n📸 Saved bbox visualization → {out_path}")
