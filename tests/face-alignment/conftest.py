"""
Shared fixtures for face-alignment pipeline tests.
Generates synthetic test data (videos, frames) so tests don't depend on real datasets.
"""
import os
import sys
import cv2
import numpy as np
import pytest
import shutil

# ── Project root on sys.path ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Output directory for visualizations ──
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


@pytest.fixture(scope="session", autouse=True)
def setup_output_dir():
    """Create output directory once per test session."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    yield
    # Don't clean up — keep outputs for visual inspection


@pytest.fixture(scope="session")
def output_dir():
    """Return path to the output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


@pytest.fixture(scope="session")
def tmp_data_dir(tmp_path_factory):
    """Session-scoped temp directory for generated test data."""
    return tmp_path_factory.mktemp("face_align_data")


# ────────────────────────────────────────────────────
# Synthetic Video Generation
# ────────────────────────────────────────────────────

def _create_synthetic_face_frame(width=320, height=240):
    """
    Create a single frame with a synthetic 'face' — 
    a skin-colored oval with dark 'eye' and 'mouth' rectangles.
    This helps face-alignment attempt detection (may still fail on synthetic data).
    """
    frame = np.full((height, width, 3), (200, 180, 160), dtype=np.uint8)  # skin-tone BG

    cx, cy = width // 2, height // 2

    # Face oval (skin color)
    cv2.ellipse(frame, (cx, cy), (70, 90), 0, 0, 360, (180, 160, 140), -1)

    # Eyes (dark)
    cv2.rectangle(frame, (cx - 40, cy - 25), (cx - 20, cy - 15), (40, 30, 30), -1)
    cv2.rectangle(frame, (cx + 20, cy - 25), (cx + 40, cy - 15), (40, 30, 30), -1)

    # Mouth (dark red)
    cv2.rectangle(frame, (cx - 25, cy + 25), (cx + 25, cy + 40), (60, 40, 150), -1)

    # Nose dot
    cv2.circle(frame, (cx, cy + 5), 4, (100, 80, 80), -1)

    return frame


def _create_blank_frame(width=320, height=240, color=(255, 255, 255)):
    """Create a solid-color frame with NO face features (for fallback testing)."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def _save_video(frames, path, fps=25):
    """Save list of numpy frames as mp4 video."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


@pytest.fixture(scope="session")
def synthetic_face_video(tmp_data_dir):
    """
    Generate a 10-frame synthetic video with face-like features.
    Returns: path to the .mp4 file
    """
    frames = []
    for i in range(10):
        frame = _create_synthetic_face_frame()
        # Add slight variation (shift mouth slightly) to simulate movement
        noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        frames.append(frame)

    video_path = os.path.join(str(tmp_data_dir), "synthetic_face.mp4")
    _save_video(frames, video_path)
    return video_path


@pytest.fixture(scope="session")
def blank_video(tmp_data_dir):
    """
    Generate a 10-frame blank/white video with NO faces.
    Returns: path to the .mp4 file
    """
    frames = [_create_blank_frame() for _ in range(10)]
    video_path = os.path.join(str(tmp_data_dir), "blank_no_face.mp4")
    _save_video(frames, video_path)
    return video_path


@pytest.fixture(scope="session")
def single_face_frame():
    """Return a single synthetic face frame (BGR, uint8)."""
    return _create_synthetic_face_frame()


@pytest.fixture(scope="session")
def single_blank_frame():
    """Return a single blank frame with no face (BGR, uint8)."""
    return _create_blank_frame()


@pytest.fixture(scope="session")
def sample_cropped_frame():
    """
    Return a pre-made 'cropped mouth' region (small image, variable size).
    Simulates the output of extract_mouth().
    """
    # Simulate a mouth crop: 60×50 region, reddish tones
    crop = np.zeros((50, 60, 3), dtype=np.uint8)
    crop[:, :, 2] = 180  # Red channel (BGR: index 2 = R)
    crop[:, :, 1] = 100  # Green
    crop[:, :, 0] = 80   # Blue
    # Add some variation
    cv2.rectangle(crop, (10, 15), (50, 35), (60, 40, 150), -1)
    return crop
