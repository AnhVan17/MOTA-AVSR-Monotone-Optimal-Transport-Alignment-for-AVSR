import modal
import os
import sys
from pathlib import Path

# --- Config ---
APP_NAME = "avsr-extract"
VOLUME_NAME = "avsr-volume"
DATA_PATH = "/mnt/data/grid_cropped" # Input: Cropped videos
OUTPUT_PATH = "/mnt/data/processed_features" 
MANIFEST_PATH = "/mnt/data/manifests/grid_manifest.jsonl"

# --- Image Definition ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    # System dependencies
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsndfile1", "git",
         # We still need these just in case, though MediaPipe is not used
         "libegl1-mesa", "libgles2-mesa"
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
        "numpy<2" 
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={"/mnt": volume},
    gpu="A100", # Heavy GPU usage here
    timeout=3600*6
)
def run_extraction():
    sys.path.append("/root")
    from src.data.preprocessors.grid import GridPreprocessor
    
    print(f"Starting Remote Extraction (Phase 2)")
    print(f"   Data (Cropped): {DATA_PATH}")
    print(f"   Output: {OUTPUT_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Did you run Phase 1 (Crop) first?")
        return

    # Ensure output dirs exist
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Initialize Preprocessor with use_precropped=True
    # NOTE: GridPreprocessor might need update to search for .mp4 in grid_cropped instead of .mpg
    processor = GridPreprocessor(data_root=DATA_PATH, use_precropped=True)
    
    processor.run(output_manifest=MANIFEST_PATH)
    
    volume.commit()
    print("Extraction Complete & Volume Committed")
    
@app.local_entrypoint()
def main():
    run_extraction.remote()
