import modal
import os
import sys
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading

# !!! DEPRECATED !!!
# This script is a Monolithic Preprocessor that was known to cause EGL/GPU context conflicts
# (Previously: MediaPipe crashing when initialized alongside PyTorch CUDA).
# Now uses face-alignment (GPU-native), no context conflicts.
#
# PLEASE USE THE SPECIALIZED MICROSERVICES INSTEAD:
# 1. scripts/data_prep/prep_facemesh_cpu.py  (CPU-only, stable FaceMesh)
# 2. scripts/data_prep/prep_features_gpu.py  (GPU-only, Fast ResNet/Whisper)
#
# This file is kept only for reference / archival purposes.

# --- Config ---
APP_NAME = "avsr-preprocess-unified-DEPRECATED"
VOLUME_NAME = "avsr-volume"

# --- Image Definitions ---
# 1. Base Image Helper
def get_base_image():
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install(
            "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsndfile1", "git",
            "libegl1-mesa", "libgles2-mesa"
        )
        .pip_install(
            "numpy<2",
            "tqdm"
        )
    )

# 2. CPU Image (For Cropping - Lightweight)
crop_image = (
    get_base_image()
    .pip_install(
        "opencv-python-headless",
        "face-alignment>=1.4.0",  # GPU-native face detection
        "torch>=2.1.0",  # Required by face-alignment
    )
    .add_local_dir("src", remote_path="/root/src")
)

# 3. GPU Image (For Extraction) - Using same pattern as modal_train_phase1.py
extract_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsndfile1", "git",
        "libegl1-mesa", "libgles2-mesa"
    )
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",  # Install with torch (same pattern as modal_train_phase1.py)
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "tqdm",
        "transformers==4.36.2",
        "timm==0.9.12",
        "huggingface-hub==0.20.3",
        "soundfile==0.12.1",
        "opencv-python-headless",
        "face-alignment>=1.4.0",
        "numpy<2",  # Force again to prevent override
        "av",        # PyAV for robust audio extraction
        "jiwer",
        "matplotlib"
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# --- FACTORY: Get Dataset Preprocessor ---
def get_preprocessor(dataset_name, data_root, use_precropped=False):
    sys.path.append("/root")
    dataset_name = dataset_name.lower()
    
    if dataset_name == "grid":
        from src.data.preprocessors.grid import GridPreprocessor
        return GridPreprocessor(data_root=data_root, use_precropped=use_precropped)
    
    elif dataset_name == "vicocktail":
        from src.data.preprocessors.vicocktail import ViCocktailPreprocessor
        return ViCocktailPreprocessor(data_root=data_root, use_precropped=use_precropped)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# --- STAGE 1: CROP (CPU, Parallel) ---
@app.function(
    image=crop_image,
    volumes={"/mnt": volume},
    gpu="T4",      # Use cheap GPU instance for high RAM/CPU availability
    cpu=8.0,       # High CPU for 16 threads
    memory=32768,  # 32GB RAM to prevent OOM
    timeout=3600*12
)
def run_crop_stage(dataset_name: str, input_path: str, output_path: str, batch_size: int = 16):
    sys.path.append("/root")
    from src.data.preprocessors.cropper import MouthCropper
    from src.utils.logging_utils import setup_logger

    logger = setup_logger("Preprocess:Crop")
    logger.info(f"Starting Stage 1: Mouth Cropping ({dataset_name.upper()})")
    logger.info(f"   Input: {input_path}")
    logger.info(f"   Output: {output_path}")

    # 1. Scan Files
    logger.info("Scanning input files...")
    # Supporting recursive search for common video formats
    extensions = ['*.mpg', '*.mp4', '*.webm']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_path, "**", ext), recursive=True))
    
    logger.info(f"   Found {len(video_files)} video files.")

    # 2. Setup Tasks
    os.makedirs(output_path, exist_ok=True)
    tasks = []
    for src_path in video_files:
        rel_path = os.path.relpath(src_path, input_path)
        dest_path = os.path.join(output_path, rel_path)
        # Always save as .mp4 (Compressed, Standard)
        dest_path = os.path.splitext(dest_path)[0] + ".mp4"
        
        if not os.path.exists(dest_path):
            tasks.append((src_path, dest_path))
            
    logger.info(f"   Tasks to process: {len(tasks)} (Skipped {len(video_files) - len(tasks)} already done)")
    
    if not tasks:
        logger.info("All files already cropped!")
        return

    # 3. Thread Local Optimization
    thread_data = threading.local()

    def get_cropper():
        if not hasattr(thread_data, "cropper"):
            # Init ONCE per thread
            thread_data.cropper = MouthCropper()
        return thread_data.cropper
    
    def process_item(item):
        src, dst = item
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            cropper = get_cropper()
            return cropper.process_video(src, dst)
        except Exception as e:
            logger.error(f"Error processing {src}: {e}")
            return False

    # 4. Execute
    logger.info(f"Starting processing with {batch_size} threads...")
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
         list(tqdm(
            executor.map(process_item, tasks),
            total=len(tasks),
            unit="vid"
        ))
    
    volume.commit()
    logger.info("Cropping Complete & Volume Committed")


# --- STAGE 2: KEYFRAME (CPU, Parallel) ---
@app.function(
    image=crop_image,  # Lightweight image, no GPU needed
    volumes={"/mnt": volume},
    cpu=8.0,
    memory=16384,
    timeout=3600*6
)
def run_keyframe_stage(dataset_name: str, input_path: str, output_path: str, batch_size: int = 16):
    """Extract key frames from cropped videos and save as JPEG images."""
    sys.path.append("/root")
    from src.data.preprocessors.base import KeyFrameExtractor
    from src.utils.logging_utils import setup_logger

    logger = setup_logger("Preprocess:KeyFrame")
    logger.info(f"Starting Stage 2: Key Frame Extraction ({dataset_name.upper()})")
    logger.info(f"   Input (Cropped): {input_path}")
    logger.info(f"   Output (KeyFrames): {output_path}")

    if not os.path.exists(input_path):
        logger.error(f"Input path {input_path} not found. Run 'crop' stage first.")
        return

    # Scan cropped videos
    video_files = glob.glob(os.path.join(input_path, "**", "*.mp4"), recursive=True)
    logger.info(f"   Found {len(video_files)} cropped videos.")

    os.makedirs(output_path, exist_ok=True)
    
    # Setup tasks - each video becomes a folder of images
    tasks = []
    for src_path in video_files:
        rel_path = os.path.relpath(src_path, input_path)
        # Create folder for each video's frames
        video_name = os.path.splitext(rel_path)[0]
        dest_folder = os.path.join(output_path, video_name)
        
        # Skip if folder exists and has images
        if os.path.exists(dest_folder) and len(glob.glob(os.path.join(dest_folder, "*.jpg"))) > 0:
            continue
        tasks.append((src_path, dest_folder))

    logger.info(f"   Tasks to process: {len(tasks)} (Skipped {len(video_files) - len(tasks)} already done)")

    if not tasks:
        logger.info("All files already processed!")
        return

    # Thread local for KeyFrameExtractor
    thread_data = threading.local()

    def get_extractor():
        if not hasattr(thread_data, "extractor"):
            thread_data.extractor = KeyFrameExtractor(threshold=30.0, max_frames=75, min_frames=10)
        return thread_data.extractor

    def process_item(item):
        src, dst_folder = item
        try:
            extractor = get_extractor()
            frames = extractor.extract_from_video(src)
            extractor.save_as_images(frames, dst_folder)  # Save as JPEG images
            return True
        except Exception as e:
            logger.error(f"Error {src}: {e}")
            return False

    logger.info(f"Starting processing with {batch_size} threads...")
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        list(tqdm(
            executor.map(process_item, tasks),
            total=len(tasks),
            unit="vid"
        ))

    volume.commit()
    logger.info("KeyFrame Extraction Complete & Volume Committed")


# --- STAGE 3: EXTRACT (GPU, Batch) ---
@app.function(
    image=extract_image,
    volumes={"/mnt": volume},
    gpu="A100", # Need A100 for fast ResNet/Whisper inference
    timeout=3600*6
)
def run_extract_stage(dataset_name: str, input_path: str, output_path: str, manifest_path: str):
    sys.path.append("/root")
    from src.utils.logging_utils import setup_logger
    logger = setup_logger("Preprocess:Extract")
    
    logger.info(f"Starting Stage 3: Feature Extraction ({dataset_name.upper()})")
    logger.info(f"   Input (Cropped): {input_path}")
    logger.info(f"   Output (Features): {output_path}")
    
    if not os.path.exists(input_path):
        logger.error(f"Input path {input_path} not found. Run 'crop' stage first.")
        return

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    
    # Init Processor via Factory
    # use_precropped=True is KEY here to skip face-alignment re-detection
    processor = get_preprocessor(dataset_name, data_root=input_path, use_precropped=True)
    
    # Run Extraction with output_dir to save features in correct location
    processor.run(output_manifest=manifest_path, output_dir=output_path)
    
    volume.commit()
    logger.info("Extraction Complete & Volume Committed")


# --- ENTRYPOINT ---
@app.local_entrypoint()
def main(
    stage: str = "crop",      # 'crop' or 'extract' (keyframe removed - SOTA uses full frames)
    dataset: str = "grid",    # 'grid' or 'vicocktail'
    batch_size: int = 16,     # For crop stage
):
    """
    Unified Preprocessing CLI (SOTA: Full Frames, no keyframe selection)
    Usage:
       modal run scripts/data_prep/preprocess.py --stage crop --dataset grid
       modal run scripts/data_prep/preprocess.py --stage extract --dataset grid
    """
    sys.path.append(os.getcwd())
    from src.utils.logging_utils import setup_logger
    logger = setup_logger("Preprocess:Main")

    # PATH CONFIGURATION
    # You can extend this map for new datasets
    DATA_CONFIG = {
        "grid": {
            "raw": "/mnt/grid",
            "cropped": "/mnt/grid_cropped",
            "features": "/mnt/processed_features/grid",
            "manifest": "/mnt/manifests/grid_manifest.jsonl"
        },
        "vicocktail": {
            "raw": "/mnt/vicocktail/raw",
            "cropped": "/mnt/vicocktail_cropped",
            "features": "/mnt/processed_features/vicocktail",
            "manifest": "/mnt/manifests/vicocktail_manifest.jsonl"
        }
    }
    
    cfg = DATA_CONFIG.get(dataset.lower())
    if not cfg:
        logger.error(f"Dataset '{dataset}' not configured in preprocess.py")
        return

    if stage == "crop":
        run_crop_stage.remote(
            dataset_name=dataset, 
            input_path=cfg["raw"], 
            output_path=cfg["cropped"], 
            batch_size=batch_size
        )
        
    elif stage == "extract":
        run_extract_stage.remote(
            dataset_name=dataset,
            input_path=cfg["cropped"],
            output_path=cfg["features"],
            manifest_path=cfg["manifest"]
        )
    else:
        logger.error(f"Unknown stage: {stage}. Valid: crop, extract")
