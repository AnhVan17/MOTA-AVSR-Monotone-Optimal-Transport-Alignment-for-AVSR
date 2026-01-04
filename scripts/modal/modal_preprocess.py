
import modal
import os
import sys
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading

# --- Config ---
APP_NAME = "avsr-preprocess-unified"
VOLUME_NAME = "avsr-volume"

# --- Image Definitions ---
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

crop_image = (
    get_base_image()
    .pip_install(
        "opencv-python-headless",
        "mediapipe==0.10.9" # Optimized for CPU
    )
    .add_local_dir("src", remote_path="/root/src")
)

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
        "numpy<2",  # Install with torch
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "tqdm",
        "transformers==4.36.2",
        "timm==0.9.12",
        "huggingface-hub==0.20.3",
        "soundfile==0.12.1",
        "opencv-python-headless",
        "mediapipe==0.10.9",
        "numpy<2",  # Force again
        "av"        # PyAV
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


# --- STAGE 1: CROP (Hybrid Logic - The most critical part for optimization) ---
@app.function(
    image=crop_image,
    volumes={"/mnt": volume},
    gpu="T4",      # Cheap GPU support
    cpu=16.0,      # High CPU for parallelism (Needed for ViCocktail)
    memory=65536,  # 64GB RAM for safety with large files
    timeout=3600*12
)
def run_crop_stage(dataset_name: str, input_path: str, output_path: str, batch_size: int = 16):
    sys.path.append("/root")
    from src.utils.logging_utils import setup_logger

    logger = setup_logger("Preprocess:Crop")
    logger.info(f"Starting Stage 1: Mouth Cropping ({dataset_name.upper()})")
    logger.info(f"   Input: {input_path}")
    logger.info(f"   Output: {output_path}")

    # OPTIMIZATION: Check dataset to use the best strategy
    if dataset_name.lower() == "vicocktail":
        # Strategy A: Specialized ViCocktail
        # Use the dedicated class method which handles:
        # - Transcript matching (.txt, .label)
        # - Audio merging (FFmpeg)
        # - Parallel processing internal management
        logger.info(">> Using Optimized ViCocktail Pipeline")
        from src.data.preprocessors.vicocktail import ViCocktailPreprocessor
        
        preprocessor = ViCocktailPreprocessor(data_root=input_path, use_precropped=False)
        preprocessor.phase1_crop_dataset(save_dir=output_path, max_workers=batch_size)
        
    else:
        # Strategy B: Generic Grid (Fast Path)
        # Use simpler file-to-file logic for Grid which is uniform
        logger.info(">> Using Generic Pipeline (Grid/LRS3)")
        from src.data.preprocessors.cropper import MouthCropper
        
        # 1. Scan Files
        logger.info("Scanning input files...")
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
            dest_path = os.path.splitext(dest_path)[0] + ".mp4"
            
            if not os.path.exists(dest_path):
                tasks.append((src_path, dest_path))
        
        if not tasks:
            logger.info("All files already cropped!")
            return

        # 3. Thread Local Optimization
        thread_data = threading.local()

        def get_cropper():
            if not hasattr(thread_data, "cropper"):
                thread_data.cropper = MouthCropper()
            return thread_data.cropper
        
        def process_item(item):
            src, dst = item
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                get_cropper().process_video(src, dst)
                return True
            except Exception as e:
                logger.error(f"Error {src}: {e}")
                return False

        # 4. Execute
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
             list(tqdm(executor.map(process_item, tasks), total=len(tasks)))
    
    volume.commit()
    logger.info("Cropping Complete & Volume Committed")


# --- STAGE 2: KEYFRAME (Optional Utility) ---
@app.function(
    image=crop_image,
    volumes={"/mnt": volume},
    cpu=8.0,
    memory=16384,
    timeout=3600*6
)
def run_keyframe_stage(dataset_name: str, input_path: str, output_path: str, batch_size: int = 16):
    """Extract key frames from cropped videos. Optional stage."""
    sys.path.append("/root")
    from src.data.preprocessors.base_preprocessor import KeyFrameExtractor
    from src.utils.logging_utils import setup_logger

    logger = setup_logger("Preprocess:KeyFrame")
    logger.info(f"Starting Stage 2: Key Frame Extraction ({dataset_name.upper()})")
    
    if not os.path.exists(input_path):
        logger.error("Input path not found. Run 'crop' stage first.")
        return

    # Scan cropped videos
    video_files = glob.glob(os.path.join(input_path, "**", "*.mp4"), recursive=True)
    logger.info(f"Found {len(video_files)} cropped videos.")

    os.makedirs(output_path, exist_ok=True)
    tasks = []
    for src in video_files:
        rel = os.path.relpath(src, input_path)
        video_name = os.path.splitext(rel)[0]
        dst_folder = os.path.join(output_path, video_name)
        # Skip if folder exists and has contents
        if not (os.path.exists(dst_folder) and len(glob.glob(os.path.join(dst_folder, "*.jpg"))) > 0):
            tasks.append((src, dst_folder))

    if not tasks:
        logger.info("All files already processed!")
        return

    thread_data = threading.local()
    def get_extractor():
        if not hasattr(thread_data, "extractor"):
            thread_data.extractor = KeyFrameExtractor(threshold=30.0, max_frames=75, min_frames=10)
        return thread_data.extractor

    def process_item(item):
        src, dst = item
        try:
            frames = get_extractor().extract_from_video(src)
            get_extractor().save_as_images(frames, dst)
            return True
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        list(tqdm(executor.map(process_item, tasks), total=len(tasks)))

    volume.commit()
    logger.info("KeyFrame Extraction Complete")


# --- STAGE 3: EXTRACT (GPU, Batch) ---
@app.function(
    image=extract_image,
    volumes={"/mnt": volume},
    gpu="A100", # Strong GPU required for ResNet + Whisper Feature Extraction
    timeout=3600*12
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
    # use_precropped=True is KEY here to skip MediaPipe re-detection
    processor = get_preprocessor(dataset_name, data_root=input_path, use_precropped=True)
    
    # Run Extraction
    processor.run(output_manifest=manifest_path, output_dir=output_path)
    
    volume.commit()
    logger.info("Extraction Complete & Volume Committed")


# --- ENTRYPOINT ---
@app.local_entrypoint()
def main(
    stage: str = "crop",      # crop | keyframe | extract
    dataset: str = "grid",    # grid | vicocktail
    batch_size: int = 16,     # For crop stage workers
):
    """
    Unified AVSR Preprocessing Pipeline (3-Stage)
    
    Usage:
       modal run scripts/modal/modal_preprocess.py --stage crop --dataset grid
       modal run scripts/modal/modal_preprocess.py --stage keyframe --dataset grid
       modal run scripts/modal/modal_preprocess.py --stage extract --dataset grid
    """
    sys.path.append(os.getcwd())
    from src.utils.logging_utils import setup_logger
    logger = setup_logger("Preprocess:Main")

    # PATH CONFIGURATION
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
            "manifest": "/mnt/manifests/train.jsonl" # Master manifest
        }
    }
    
    cfg = DATA_CONFIG.get(dataset.lower())
    if not cfg:
        logger.error(f"Dataset '{dataset}' not configured in preprocess.py")
        return

    if stage == "crop":
        run_crop_stage.remote(dataset, cfg["raw"], cfg["cropped"], batch_size)
    
    elif stage == "keyframe":
        # Keyframe output path strategy
        kf_out = cfg["cropped"].replace("_cropped", "_keyframes")
        # Fallback if names are similar, just append _kf
        if kf_out == cfg["cropped"]: kf_out += "_kf"
        
        run_keyframe_stage.remote(dataset, cfg["cropped"], kf_out, batch_size)
        
    elif stage == "extract":
        run_extract_stage.remote(dataset, cfg["cropped"], cfg["features"], cfg["manifest"])
        
    else:
        logger.error(f"Unknown stage: {stage}. Valid: crop, keyframe, extract")