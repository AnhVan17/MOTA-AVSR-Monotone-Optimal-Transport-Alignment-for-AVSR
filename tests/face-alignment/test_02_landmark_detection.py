"""
Step 2: Landmark Detection — face_alignment → 68-point landmarks

Pipeline stage: fa.get_landmarks_from_image(rgb_frame)
  Input  → RGB frame (H, W, 3) uint8
  Output → landmarks array (68, 2) float, or None if no face detected

NOTE: This test requires the `face_alignment` library.
      On synthetic data, detection may fail (which is fine — fallback is tested in step 8).
"""
import os
import sys
import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# face_alignment is a heavy GPU dependency — skip gracefully if unavailable
fa_available = True
try:
    import face_alignment
except ImportError:
    fa_available = False


@pytest.mark.skipif(not fa_available, reason="face_alignment library not installed")
class TestLandmarkDetection:

    @pytest.fixture(scope="class")
    def fa_model(self):
        """Initialize face-alignment model (once per class)."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False,
            face_detector='sfd'
        )
        print(f"\n🔧 face-alignment initialized on: {device}")
        return model

    def test_input_format(self, single_face_frame):
        """Verify input frame is BGR uint8 before conversion."""
        frame = single_face_frame
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert len(frame.shape) == 3 and frame.shape[2] == 3
        print(f"\n✅ INPUT: type={type(frame).__name__}, shape={frame.shape}, dtype={frame.dtype}")

    def test_bgr_to_rgb_conversion(self, single_face_frame):
        """face-alignment expects RGB input — verify BGR→RGB conversion."""
        bgr = single_face_frame
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # B and R channels should be swapped
        assert np.array_equal(bgr[:, :, 0], rgb[:, :, 2]), "Blue channel mismatch"
        assert np.array_equal(bgr[:, :, 2], rgb[:, :, 0]), "Red channel mismatch"
        assert np.array_equal(bgr[:, :, 1], rgb[:, :, 1]), "Green channel unchanged"
        print(f"\n✅ BGR→RGB: shape={rgb.shape}, B↔R swapped correctly")

    def test_landmark_detection_output_type(self, fa_model, single_face_frame):
        """
        Run detection on synthetic face frame.
        Output is either:
          - list of arrays, each (68, 2) → face detected
          - None → no face detected (expected on synthetic data)
        """
        rgb = cv2.cvtColor(single_face_frame, cv2.COLOR_BGR2RGB)
        landmarks = fa_model.get_landmarks_from_image(rgb)

        print(f"\n📸 INPUT:  RGB frame shape={rgb.shape}")

        if landmarks is not None and len(landmarks) > 0:
            lm = landmarks[0]  # First face
            assert lm.shape == (68, 2), f"Expected (68,2), got {lm.shape}"

            mouth_points = lm[48:68]
            assert mouth_points.shape == (20, 2)

            print(f"✅ OUTPUT: landmarks detected!")
            print(f"   Full landmarks shape = {lm.shape}")
            print(f"   Mouth points [48:68] shape = {mouth_points.shape}")
            print(f"   Mouth center = ({np.mean(mouth_points[:,0]):.1f}, {np.mean(mouth_points[:,1]):.1f})")
        else:
            print(f"⚠️  OUTPUT: landmarks = None (no face detected on synthetic data)")
            print(f"   This is expected — fallback will be tested in test_08_fallback.py")
            # Not a failure — synthetic data may not trigger detection
            pytest.skip("No face detected on synthetic data (expected behavior)")

    def test_visualize_landmarks(self, fa_model, single_face_frame, output_dir):
        """Draw landmarks on frame and save for visual inspection."""
        rgb = cv2.cvtColor(single_face_frame, cv2.COLOR_BGR2RGB)
        landmarks = fa_model.get_landmarks_from_image(rgb)

        vis_frame = single_face_frame.copy()

        if landmarks is not None and len(landmarks) > 0:
            lm = landmarks[0]
            # Draw all 68 points as green dots
            for (x, y) in lm:
                cv2.circle(vis_frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            # Highlight mouth points [48:68] as red dots
            for (x, y) in lm[48:68]:
                cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 0, 255), -1)

            print(f"\n📸 Landmarks visualized: 68 green dots, 20 red mouth dots")
        else:
            # Draw text indicating no detection
            cv2.putText(vis_frame, "NO FACE DETECTED", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"\n⚠️  No landmarks to visualize (wrote 'NO FACE DETECTED' on frame)")

        out_path = os.path.join(output_dir, "output_step02_landmarks.jpg")
        cv2.imwrite(out_path, vis_frame)
        print(f"   Saved → {out_path}")
        assert os.path.exists(out_path)
