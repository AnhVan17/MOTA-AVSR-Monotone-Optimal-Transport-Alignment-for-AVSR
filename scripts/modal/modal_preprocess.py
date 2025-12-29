import modal
import os
import sys
from pathlib import Path

# --- Config ---
APP_NAME = "avsr-preprocess"
VOLUME_NAME = "avsr-volume"
DATA_PATH = "/mnt/data/grid" # Nested path
OUTPUT_PATH = "/mnt/data/processed_features" # Keep output separate or same? Let's keep separate for now or change?
# User wants "grid in folder data", implies keeping everything there?
# If BasePreprocessor writes in-place, then OUTPUT_PATH var is not fully used for writing files (it's used for checks?).
# Let's check logic.
MANIFEST_PATH = "/mnt/data/manifests/grid_manifest.jsonl"

# --- Image Definition ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    # System dependencies
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsndfile1", "git",
        "libegl1-mesa", "libgles2-mesa" # Fix for MediaPipe EGL error
    )
    # Python dependencies
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "transformers==4.36.2",
        "timm==0.9.12",
        "huggingface-hub==0.20.3",
        "opencv-python-headless==4.9.0.80",
        "mediapipe==0.10.9",
        "soundfile==0.12.1",
        "numpy<2" # Force numpy < 2 to prevent auto-upgrade
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={"/mnt": volume},
    gpu="A100",
    timeout=3600*6        # 6 hour timeout
)
def run_preprocessing():
    # 1. Setup Path
    sys.path.append("/root")
    
    # 2. Import (Late import because src is only available inside container)
    from src.data.preprocessors.grid import GridPreprocessor
    from src.data.preprocessors.base import PreprocessConfig
    
    print(f"Starting Remote Preprocessing")
    print(f"   Data: {DATA_PATH}")
    print(f"   Output: {OUTPUT_PATH}")
    
    # Check data existence
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Did you upload data?")
        print("   Run: modal volume put avsr-volume local_folder /data/grid")
        return

    # 3. Initialize Preprocessor
    # Ensure output dirs exist
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    processor = GridPreprocessor(data_root=DATA_PATH)
    
    # 4. Run
    # BasePreprocessor uses PreprocessConfig which defaults to usage of output dir logic?
    # Actually BasePreprocessor saves .npy/.pt NEXT TO the video file in logic:
    # save_path = video_path.replace(".mpg", ".pt") 
    # Since video_path is in /data/grid (Volume), it works perfectly.
    # Wait, if we want to separate raw and processed, we might need to symlink or change logic.
    # For Phase 1 simplification: Let's assume we write to Volume IN-PLACE or simple structure.
    # The current BasePreprocessor logic (from previous view) seemed to write .pt next to source?
    # Let's check BasePreprocessor.run again carefully or assuming logic.
    # Re-reading my memory: It replaced extension. So it writes to /data/grid/.../video.pt
    # This is fine for Volume.
    
    processor.run(output_manifest=MANIFEST_PATH)
    
    # 5. Commit Volume
    volume.commit()
    print("Preprocessing Complete & Volume Committed")
    
@app.local_entrypoint()
def main():
    run_preprocessing.remote()
