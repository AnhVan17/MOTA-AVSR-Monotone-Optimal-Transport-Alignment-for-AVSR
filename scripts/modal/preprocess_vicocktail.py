
import modal
import os
import sys
import glob

# --- Config ---
APP_NAME = "avsr-preprocess-vicocktail-v233"  # Changed to force rebuild
VOLUME_NAME = "avsr-dataset-volume"

# --- Image Definitions ---
def get_base_image():
    return (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install(
            "libgl1-mesa-glx", 
            "libglib2.0-0", 
            "ffmpeg", 
            "libsndfile1", 
            "git",
            "libegl1",      # Required by MediaPipe
            "libgles2-mesa" # Required by MediaPipe
        )
        .pip_install("numpy<2", "tqdm")
    )

crop_image = (
    get_base_image()
    .pip_install(
        "numpy==1.26.4",
        "opencv-python-headless", 
        "mediapipe==0.10.9",
        "torch",            # Required by vicocktail.py import
        "timm", 
        "transformers", 
        "soundfile",
        "requests"
    )
    .add_local_dir("src", remote_path="/root/src")
)

extract_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsndfile1", "git")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.1.2", "torchaudio==2.1.2", "torchvision==0.16.2",
        "transformers==4.36.2", "timm==0.9.12", "huggingface-hub==0.20.3",
        "soundfile==0.12.1", "opencv-python-headless", "mediapipe==0.10.9", "av",
        "tqdm",
        extra_index_url="https://download.pytorch.org/whl/cu118"
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)

# --- Volumes ---
# Vol 1: INPUT (Raw Data) - High Inode Usage (~493k)
vol_raw = modal.Volume.from_name("avsr-dataset-volume", create_if_missing=True)

# Vol 2: OUTPUT (processed data) - Fresh Volume
vol_processed = modal.Volume.from_name("avsr-vicocktail-processed", create_if_missing=True)

# --- PATHS ---
# Raw data is in: /mnt/raw_data/vicocktail/raw
VC_RAW_MOUNT = "/mnt/raw_data"
VC_RAW = "/mnt/raw_data/vicocktail/raw"

# Output goes to new volume
VC_PROCESSED_MOUNT = "/mnt/processed"
VC_CROPPED = "/mnt/processed/vicocktail_cropped"
VC_KEYFRAMES = "/mnt/processed/vicocktail_keyframes"
VC_FEATURES = "/mnt/processed/features/vicocktail"
VC_MANIFEST = "/mnt/processed/manifests/train.jsonl"

# --- STAGE 1: CROP (Specialized & Distributed) ---

@app.function(
    image=crop_image,
    volumes={
        VC_RAW_MOUNT: vol_raw,
        VC_PROCESSED_MOUNT: vol_processed
    },
    cpu=2.0,           # Lower CPU per worker
    memory=8192,       # 8GB RAM is plenty for 1 video
    timeout=600        # 10 mins per video max
)
def crop_video_task(video_path: str, data_root: str, save_dir: str):
    sys.path.append("/root")
    from src.data.preprocessors.vicocktail import _process_single_video_wrapper
    return _process_single_video_wrapper((video_path, data_root, save_dir))

@app.function(
    image=crop_image,
    volumes={
        VC_RAW_MOUNT: vol_raw,
        VC_PROCESSED_MOUNT: vol_processed
    },
    cpu=16.0,          # High CPU for Single-Container Mode
    memory=65536,      # High RAM for Single-Container Mode
    timeout=3600*24
)
def run_crop_vicocktail_legacy(output_path: str = VC_CROPPED, max_workers: int = 8, limit: int = None):
    sys.path.append("/root")
    from src.data.preprocessors.vicocktail import ViCocktailPreprocessor
    from src.utils.logging_utils import setup_logger
    
    logger = setup_logger("ViCocktail:Crop:Single")
    logger.info(f"Starting Single-Container Cropping (16 CPU, {max_workers} Workers)...")
    
    # This uses ProcessPoolExecutor inside the 16-CPU container
    # Since we fixed Streaming I/O, this is now safe from RAM OOM
    proc = ViCocktailPreprocessor(data_root=VC_RAW, use_precropped=False)
    # Note: We don't have a direct 'limit' in phase1_crop_dataset but we can filter before
    proc.phase1_crop_dataset(save_dir=output_path, max_workers=max_workers)
    
    vol_processed.commit()

@app.function(
    image=crop_image,
    volumes={
        VC_RAW_MOUNT: vol_raw,
        VC_PROCESSED_MOUNT: vol_processed
    },
    cpu=2.0,
    memory=4096,
    timeout=3600*12
)
def run_crop_vicocktail_dist(output_path: str = VC_CROPPED, limit: int = None, concurrency: int = 50):
    sys.path.append("/root")
    from src.utils.logging_utils import setup_logger
    from tqdm import tqdm
    
    logger = setup_logger("ViCocktail:Crop:Dist")
    logger.info(f"Starting Distributed ViCocktail Cropping (Concurrency: {concurrency})...")
    
    # 1. Collect all video files
    extensions = ['*.mp4', '*.mkv', '*.webm', '*.avi', '*.video']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(VC_RAW, "**", ext), recursive=True))
    
    if limit:
        video_files = video_files[:limit]
        
    logger.info(f"Found {len(video_files)} videos. Distributing...")
    
    # 2. Prepare mapping tasks
    tasks = [(v, VC_RAW, output_path) for v in video_files]
    
    # 3. Execute distributed map
    success_count = 0
    # concurrency_limit prevents overloading volume
    for res in tqdm(crop_video_task.starmap(tasks, order_outputs=False, concurrency_limit=concurrency), total=len(tasks), desc="Distributed Crop"):
        if res:
            success_count += 1
            
    logger.info(f"✅ Completed: {success_count}/{len(video_files)}")
    vol_processed.commit()


# --- STAGE 2: KEYFRAME (Generic) ---
@app.function(
    image=crop_image,
    volumes={
        VC_RAW_MOUNT: vol_raw,
        VC_PROCESSED_MOUNT: vol_processed
    },
    cpu=8.0,
    memory=16384,
    timeout=3600*4
)
def run_keyframe_vicocktail(input_path: str = VC_CROPPED, output_path: str = VC_KEYFRAMES, batch_size: int = 16):
    sys.path.append("/root")
    # Same generic logic as Grid for Keyframes, just different paths/resources
    from concurrent.futures import ThreadPoolExecutor
    from src.data.preprocessors.base_preprocessor import KeyFrameExtractor
    from src.utils.logging_utils import setup_logger
    import threading
    
    logger = setup_logger("ViCocktail:KeyFrame")
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
    
    vol_processed.commit()


# --- STAGE 3: EXTRACT (Specialized) ---
@app.function(
    image=extract_image,
    volumes={
        VC_RAW_MOUNT: vol_raw,
        VC_PROCESSED_MOUNT: vol_processed
    },
    gpu="A100",        # Vicocktail needs A100
    timeout=3600*12
)
def run_extract_vicocktail():
    sys.path.append("/root")
    from src.data.preprocessors.vicocktail import ViCocktailPreprocessor
    from src.utils.logging_utils import setup_logger
    import json
    import random
    import os
    
    logger = setup_logger("ViCocktail:Extract")
    if not os.path.exists(VC_CROPPED): 
        logger.error(f"Input dir {VC_CROPPED} not found!")
        return

    os.makedirs(VC_FEATURES, exist_ok=True)
    manifest_dir = os.path.dirname(VC_MANIFEST)
    os.makedirs(manifest_dir, exist_ok=True)
    
    # 1. Run Extraction -> Save to TEMP manifest
    temp_manifest = os.path.join(manifest_dir, "all_temp.jsonl")
    
    logger.info(f"Extracting features from {VC_CROPPED}...")
    proc = ViCocktailPreprocessor(data_root=VC_CROPPED, use_precropped=True)
    proc.run(output_manifest=temp_manifest, output_dir=VC_FEATURES)
    
    # 2. Post-Process: Split Train/Val/Test
    logger.info("Splitting dataset into Train (90%) / Val (10%) / Test ...")
    
    if not os.path.exists(temp_manifest):
        logger.error("Manifest generation failed.")
        return

    with open(temp_manifest, 'r', encoding='utf-8') as f:
        all_data = [json.loads(line) for line in f if line.strip()]
        
    test_set = []
    train_val_candidates = []
    
    for item in all_data:
        # Heuristic: If 'test' is in the file path (case-insensitive) -> Test Set
        # This handles cases where data is in 'Test' folder
        rel_path = item.get('rel_path', '').lower()
        if 'test' in rel_path.split(os.sep) or 'test' in os.path.basename(rel_path).lower():
            test_set.append(item)
        else:
            train_val_candidates.append(item)
            
    # Shuffle and Split Train/Val (90/10)
    random.seed(42)
    random.shuffle(train_val_candidates)
    
    split_idx = int(len(train_val_candidates) * 0.9)
    train_set = train_val_candidates[:split_idx]
    val_set = train_val_candidates[split_idx:]
    
    # Helper to save
    def save_split(name, data):
        path = os.path.join(manifest_dir, name)
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return path
        
    save_split("train.jsonl", train_set)
    save_split("val.jsonl", val_set)
    save_split("test.jsonl", test_set)
    
    logger.info(f"✅ Split Summary:")
    logger.info(f"   Train: {len(train_set)}")
    logger.info(f"   Val:   {len(val_set)}")
    logger.info(f"   Test:  {len(test_set)}")
    
    # Cleanup
    if os.path.exists(temp_manifest):
        os.remove(temp_manifest)
        
    vol_processed.commit()


@app.local_entrypoint()
def main(stage: str = "crop", mode: str = "dist", limit: int = None, workers: int = 8):
    """
    Usage:
        modal run scripts/modal/preprocess_vicocktail.py --stage crop --mode dist
        modal run scripts/modal/preprocess_vicocktail.py --stage crop --mode single --workers 8
    """
    if stage == "crop":
        if mode == "dist":
            run_crop_vicocktail_dist.remote(output_path=VC_CROPPED, limit=limit, concurrency=workers*10)
        else:
            run_crop_vicocktail_legacy.remote(output_path=VC_CROPPED, max_workers=workers, limit=limit)
    elif stage == "keyframe": 
        run_keyframe_vicocktail.remote(batch_size=workers)
    elif stage == "extract": 
        run_extract_vicocktail.remote()
    else: 
        print("Valid stages: crop, keyframe, extract")
