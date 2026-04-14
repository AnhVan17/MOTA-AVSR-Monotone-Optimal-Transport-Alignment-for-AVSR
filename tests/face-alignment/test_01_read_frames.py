"""
Step 1: Read Frames — cv2.VideoCapture → list of BGR frames

Pipeline stage: VideoProcessor.process() first reads all frames from video.

INPUT:  video file path (str)  →  .mp4 file on disk
OUTPUT: list[np.ndarray]       →  each frame (H, W, 3) dtype=uint8
"""
import os
import sys
import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def read_frames(video_path: str) -> list:
    """
    Replicate the frame-reading logic from VideoProcessor.process().
    Returns list of BGR numpy frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


class TestReadFrames:

    def test_read_returns_list(self, synthetic_face_video):
        """Frames should be returned as a Python list."""
        frames = read_frames(synthetic_face_video)
        assert isinstance(frames, list), f"Expected list, got {type(frames)}"
        print(f"\n✅ Type: {type(frames)}")

    def test_frame_count(self, synthetic_face_video):
        """Synthetic video has 10 frames."""
        frames = read_frames(synthetic_face_video)
        assert len(frames) == 10, f"Expected 10 frames, got {len(frames)}"
        print(f"\n✅ Frame count: {len(frames)}")

    def test_frame_shape_and_dtype(self, synthetic_face_video):
        """Each frame should be (240, 320, 3) uint8 BGR."""
        frames = read_frames(synthetic_face_video)
        for i, frame in enumerate(frames):
            assert isinstance(frame, np.ndarray), f"Frame {i} is not ndarray"
            assert frame.dtype == np.uint8, f"Frame {i} dtype={frame.dtype}"
            assert frame.shape == (240, 320, 3), f"Frame {i} shape={frame.shape}"

        print(f"\n✅ Each frame: shape={frames[0].shape}, dtype={frames[0].dtype}")

    def test_empty_path_returns_empty(self, tmp_data_dir):
        """Non-existent video should return empty list."""
        frames = read_frames(os.path.join(str(tmp_data_dir), "nonexistent.mp4"))
        assert len(frames) == 0
        print(f"\n✅ Non-existent video → {len(frames)} frames (correct)")

    def test_visualize_first_frame(self, synthetic_face_video, output_dir):
        """Save first frame for visual inspection."""
        frames = read_frames(synthetic_face_video)
        assert len(frames) > 0

        first_frame = frames[0]
        out_path = os.path.join(output_dir, "output_step01_frame.jpg")
        cv2.imwrite(out_path, first_frame)

        print(f"\n📸 INPUT:  video_path = {synthetic_face_video}")
        print(f"📸 OUTPUT: {len(frames)} frames")
        print(f"   Frame[0] shape = {first_frame.shape}")
        print(f"   Frame[0] dtype = {first_frame.dtype}")
        print(f"   Frame[0] pixel range = [{first_frame.min()}, {first_frame.max()}]")
        print(f"   Saved → {out_path}")

        assert os.path.exists(out_path)
