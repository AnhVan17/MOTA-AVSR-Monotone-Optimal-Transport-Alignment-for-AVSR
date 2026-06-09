"""Shared Modal image / volume definitions for all scripts under ``scripts/modal``.

Centralizes the previously copy-pasted ``modal.Image`` and ``Volume.from_name`` blocks so a
dependency bump (e.g. torch) happens in ONE place instead of ~6 files.

NOTE: this module is only imported when *defining* a Modal app locally (i.e. when you run
``modal run ...``). The remote container does not import it — it only needs ``src`` + ``configs``
which the images add via ``add_local_dir``.

Three image flavors (matching the original per-script definitions exactly):
  - ML_TRAIN_IMAGE     : training + inference (torch + HF + eval/io deps), ships configs + src
  - PREPROC_IMAGE      : data preprocessing (heavier: timm/webdataset/face-alignment/...), ships src
  - FACE_CROP_IMAGE    : CPU-only mouth-ROI crop (face-alignment SFD+FAN)
  - BARE_IMAGE         : plain image for volume-management / debug utilities
"""
import modal

# --- Volumes ---
VOLUME_NAME = "avsr-volume"                  # features + checkpoints
DATASET_VOLUME_NAME = "avsr-dataset-volume"  # raw dataset downloads (separate on purpose)

# --- Shared build constants ---
PY_VERSION = "3.10"
TORCH_INDEX = "https://download.pytorch.org/whl/cu118"
_TORCH = ("torch==2.1.2", "torchaudio==2.1.2", "torchvision==0.16.2", "numpy<2")


def _base() -> "modal.Image":
    return modal.Image.debian_slim(python_version=PY_VERSION)


def get_volume(name: str = VOLUME_NAME) -> "modal.Volume":
    """Modal Volume by name (created if missing)."""
    return modal.Volume.from_name(name, create_if_missing=True)


# Training & inference: torch + HuggingFace + eval/IO deps. Ships configs + src.
ML_TRAIN_IMAGE = (
    _base()
    .apt_install("git", "ffmpeg")
    .pip_install(*_TORCH, index_url=TORCH_INDEX)
    .pip_install(
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "jiwer",
        "matplotlib",
        "soundfile",
        "opencv-python-headless",
    )
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("src", remote_path="/root/src")
)

# Data preprocessing: heavier deps (feature extraction, face detection, audio). Ships src.
PREPROC_IMAGE = (
    _base()
    .apt_install("git", "ffmpeg", "libgl1", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        *_TORCH,
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "timm==0.9.12",
        "webdataset==0.2.79",
        "huggingface_hub",
        "face-alignment>=1.4.0",
        "opencv-python-headless",
        "soundfile",
        "librosa",
        "av",
        "jiwer",
        "matplotlib",
        "pyyaml",
        index_url=TORCH_INDEX,
        extra_index_url="https://pypi.org/simple",
    )
    .add_local_dir("src", remote_path="/root/src")
)

# CPU-only mouth-ROI crop (torch pulled transitively by face-alignment, CPU build).
FACE_CROP_IMAGE = (
    _base()
    .apt_install("ffmpeg", "libgl1-mesa-glx")
    .pip_install("face-alignment>=1.4.0", "opencv-python-headless", "numpy<2", "tqdm", "pyyaml")
    .add_local_dir("src", remote_path="/root/src")
)

# Deprecated alias (was named after MediaPipe FaceMesh; now face-alignment). Remove after migration.
CPU_FACEMESH_IMAGE = FACE_CROP_IMAGE

# Plain image for volume management / debug utilities (no heavy deps).
BARE_IMAGE = _base()
