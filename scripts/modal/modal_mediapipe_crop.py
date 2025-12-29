import modal
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# --- Config ---
APP_NAME = "avsr-crop"
VOLUME_NAME = "avsr-volume"
DATA_PATH = "/mnt/data/grid" 
OUTPUT_PATH = "/mnt/data/grid_cropped" 
BATCH_SIZE = 16 # Number of videos to process in parallel (Concurrency) 

# --- Image Definition ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsndfile1", "git",
        "libegl1-mesa", "libgles2-mesa"
    )
    .pip_install(
        "opencv-python-headless",
        "mediapipe==0.10.9", # Pin version for consistency
        "numpy<2",
        "tqdm"
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={"/mnt": volume},
    gpu="T4",      
    timeout=3600*12
)
def run_crop():
    import sys
    sys.path.append("/root")
    from src.data.preprocessors.cropper import MouthCropper

    print(f"Starting Massive Cropping Job")
    print(f"   Input: {DATA_PATH}")
    print(f"   Output: {OUTPUT_PATH}")

    # 1. Scan Files
    print("Scanning input files...")
    search_pattern = os.path.join(DATA_PATH, "**", "*.mpg")
    video_files = glob.glob(search_pattern, recursive=True)
    print(f"   Found {len(video_files)} video files.")

    # 2. Metadata/State Check
    # We can skip files already cropped to support resuming
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # We need to replicate directory structure? Or flat?
    # User said "save after crop".
    # Best practice: Replicate folder structure s1/..., s2/... inside grid_cropped
    
    tasks = []
    for src_path in video_files:
        # /mnt/data/grid/s1/bbaf2n.mpg -> /mnt/data/grid_cropped/s1/bbaf2n.mp4
        rel_path = os.path.relpath(src_path, DATA_PATH)
        dest_path = os.path.join(OUTPUT_PATH, rel_path)
        dest_path = os.path.splitext(dest_path)[0] + ".mp4"
        
        if not os.path.exists(dest_path):
            tasks.append((src_path, dest_path))
            
    print(f"   Tasks to process: {len(tasks)} (Skipped {len(video_files) - len(tasks)} already done)")
    
    if not tasks:
        print("All files already cropped!")
        return

    # 3. Processing with ThreadPool
    # OPTIMIZATION: Use thread_local to reuse MediaPipe instances
    # instantiating FaceMesh 30k times causes OOM and slowness.
    import threading
    thread_data = threading.local()

    def get_cropper():
        if not hasattr(thread_data, "cropper"):
            # Init ONCE per thread
            thread_data.cropper = MouthCropper()
        return thread_data.cropper
    
    def process_item(item):
        src, dst = item
        
        # Ensure dest dir exists
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        
        try:
            # Reuse the cropper instance for this thread
            cropper = get_cropper()
            success = cropper.process_video(src, dst)
            return success
        except Exception as e:
            print(f"Error processing {src}: {e}")
            return False

    # Use ThreadPool
    print(f"Starting processing with {BATCH_SIZE} threads...")
    completed = 0
    # Note: BATCH_SIZE determines max_workers (threads).
    # so we will have at most BATCH_SIZE active FaceMesh instances.
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
         results = list(tqdm(
            executor.map(process_item, tasks),
            total=len(tasks),
            unit="vid"
        ))
    
    volume.commit()
    print("Cropping Complete & Volume Committed")

@app.local_entrypoint()
def main():
    run_crop.remote()
