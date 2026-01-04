
import modal
import os
import sys
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading

# --- Config ---
APP_NAME = "avsr-preprocess-grid"
VOLUME_NAME = "avsr-volume"

# --- Image Definitions ---
def get_base_image():
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install(
            "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsndfile1", "git"
        )
        .pip_install("numpy<2", "tqdm")
    )

crop_image = (
    get_base_image()
    .pip_install("opencv-python-headless", "mediapipe==0.10.9")
    .add_local_dir("src", remote_path="/root/src")
)

extract_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.1.2", "torchaudio==2.1.2", "torchvision==0.16.2", "numpy<2",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "transformers==4.36.2", "timm==0.9.12", "huggingface-hub==0.20.3",
        "soundfile==0.12.1", "opencv-python-headless", "mediapipe==0.10.9", "av",
        "tqdm"
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# --- GRID PATHS ---
GRID_RAW = "/mnt/grid"
GRID_CROPPED = "/mnt/grid_cropped"
GRID_KEYFRAMES = "/mnt/grid_keyframes"
GRID_FEATURES = "/mnt/processed_features/grid"
GRID_MANIFEST = "/mnt/manifests/grid_manifest.jsonl"


# --- STAGE 1: CROP (Generic for Grid) ---
@app.function(
    image=crop_image,
    volumes={"/mnt": volume},
    cpu=4.0,     # Grid is light
    memory=8192,
    timeout=3600*4
)
def run_crop_grid(output_path: str = GRID_CROPPED, batch_size: int = 8):
    sys.path.append("/root")
    from src.data.preprocessors.cropper import MouthCropper
    from src.utils.logging_utils import setup_logger

    logger = setup_logger("Grid:Crop")
    logger.info(f"Scanning {GRID_RAW}...")
    
    # Grid usually uses .mpg
    video_files = glob.glob(os.path.join(GRID_RAW, "**", "*.mpg"), recursive=True)
    if not video_files: # Fallback
        video_files = glob.glob(os.path.join(GRID_RAW, "**", "*.mp4"), recursive=True)
    
    logger.info(f"Found {len(video_files)} videos.")

    os.makedirs(output_path, exist_ok=True)
    tasks = []
    for src in video_files:
        rel = os.path.relpath(src, GRID_RAW)
        dst = os.path.join(output_path, rel)
        dst = os.path.splitext(dst)[0] + ".mp4"
        if not os.path.exists(dst):
            tasks.append((src, dst))
            
    if not tasks:
        logger.info("Nothing to crop.")
        return

    thread_data = threading.local()
    def get_cropper():
        if not hasattr(thread_data, "c"): thread_data.c = MouthCropper()
        return thread_data.c

    def process(item):
        src, dst = item
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            return get_cropper().process_video(src, dst)
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=batch_size) as ex:
        list(tqdm(ex.map(process, tasks), total=len(tasks)))
        
    volume.commit()


# --- STAGE 2: KEYFRAME (Generic) ---
@app.function(
    image=crop_image,
    volumes={"/mnt": volume},
    cpu=4.0,
    memory=8192,
    timeout=3600
)
def run_keyframe_grid(input_path: str = GRID_CROPPED, output_path: str = GRID_KEYFRAMES, batch_size: int = 8):
    sys.path.append("/root")
    from src.data.preprocessors.base_preprocessor import KeyFrameExtractor
    from src.utils.logging_utils import setup_logger
    
    logger = setup_logger("Grid:KeyFrame")
    if not os.path.exists(input_path): return
    
    video_files = glob.glob(os.path.join(input_path, "**", "*.mp4"), recursive=True)
    os.makedirs(output_path, exist_ok=True)
    
    tasks = []
    for src in video_files:
        rel = os.path.relpath(src, input_path)
        dst = os.path.join(output_path, os.path.splitext(rel)[0])
        if not (os.path.exists(dst) and glob.glob(f"{dst}/*.jpg")):
            tasks.append((src, dst))
            
    if not tasks: return

    thread_data = threading.local()
    def get_ext():
        if not hasattr(thread_data, "e"): 
            thread_data.e = KeyFrameExtractor(threshold=30.0, max_frames=75, min_frames=10)
        return thread_data.e

    def process(item):
        src, dst = item
        try:
            frames = get_ext().extract_from_video(src)
            get_ext().save_as_images(frames, dst)
            return True
        except: return False

    with ThreadPoolExecutor(max_workers=batch_size) as ex:
        list(tqdm(ex.map(process, tasks), total=len(tasks)))
    
    volume.commit()


# --- STAGE 3: EXTRACT (Generic) ---
@app.function(
    image=extract_image,
    volumes={"/mnt": volume},
    gpu="A10G", # Grid is light
    timeout=3600*2
)
def run_extract_grid():
    sys.path.append("/root")
    from src.data.preprocessors.grid import GridPreprocessor
    from src.utils.logging_utils import setup_logger
    
    logger = setup_logger("Grid:Extract")
    if not os.path.exists(GRID_CROPPED): return

    os.makedirs(GRID_FEATURES, exist_ok=True)
    os.makedirs(os.path.dirname(GRID_MANIFEST), exist_ok=True)
    
    proc = GridPreprocessor(data_root=GRID_CROPPED, use_precropped=True)
    proc.run(output_manifest=GRID_MANIFEST, output_dir=GRID_FEATURES)
    volume.commit()


@app.local_entrypoint()
def main(stage: str = "crop"):
    """
    Usage:
        modal run scripts/modal/preprocess_grid.py --stage [crop|keyframe|extract]
    """
    if stage == "crop": run_crop_grid.remote()
    elif stage == "keyframe": run_keyframe_grid.remote()
    elif stage == "extract": run_extract_grid.remote()
    else: print("Valid stages: crop, keyframe, extract")
