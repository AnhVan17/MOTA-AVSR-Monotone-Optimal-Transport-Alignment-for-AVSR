import modal
import os
import sys
import glob
import shutil
from pathlib import Path

# --- Config ---
APP_NAME = "avsr-prep-facecrop-gpu"
VOLUME_NAME = "avsr-volume"
DATA_ROOT = "/mnt/vicocktail_raw"
OUTPUT_ROOT = "/mnt/vicocktail_cropped"

# --- Image Definition ---
# GPU image for face-alignment processin
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1-mesa-glx")  # System libs for OpenCV
    .pip_install(
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "face-alignment>=1.4.0",  # GPU-native face detection
        "opencv-python-headless",
        "numpy<2",
        "tqdm",
        "pyyaml"
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Global worker function (Picklable)
def process_video_task(args):
    """
    Worker function to process a single video.
    Args:
        args: tuple(vid_path, temp_dir, shard_out_dir)
    """
    vid_path, temp_dir, shard_out_dir = args
    import os
    from src.data.preprocessors.facemesh import FaceMeshPreprocessor
    
    try:
        # SAFETY: Rename .video to .mp4 to help OpenCV detect container format
        work_path = vid_path
        if vid_path.endswith(".video"):
            new_path = vid_path[:-6] + ".mp4" # Replaces .video with .mp4
            try:
                os.rename(vid_path, new_path)
                work_path = new_path
            except OSError:
                pass

        # Singleton handles Init once per process
        processor = FaceMeshPreprocessor() 
        
        # Construct output path relative to temp_dir
        rel_path = os.path.relpath(vid_path, temp_dir)
        out_path = os.path.join(shard_out_dir, rel_path)
        # Ensure output extension is .mp4
        out_path = os.path.splitext(out_path)[0] + ".mp4"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        if os.path.exists(out_path):
            return "Skipped"
            
        result = processor.process_video(work_path, output_path=out_path)
        return "Success" if result else "Failed"
    except Exception as e:
        return f"Error: {e}"

@app.function(
    image=image,
    volumes={"/mnt": volume},
    gpu="T4",          # GPU for face-alignment (fast & cheap)
    cpu=4,
    timeout=7200,      # 2 hours
    memory=16384       # 16GB RAM for parallel workers
)
def process_shard(tar_path):
    import tarfile
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    
    # 1. Setup paths
    file_name = os.path.basename(tar_path)
    shard_id = file_name.split(".")[0]
    
    # Extract to local temp (fast SSD on key-value modal runner)
    temp_dir = f"/tmp/{shard_id}"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Extracting {file_name} to {temp_dir}...")
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=temp_dir)
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return
        
    # 2. Find Videos using os.walk
    video_files = []
    print(f"Scanning {temp_dir} for videos and transcripts...")
    video_files = []
    text_files = [] 
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            if file.endswith(".video") or file.endswith(".mp4") or file.endswith(".webm") or file.endswith(".mkv"):
                video_files.append(os.path.join(root, file))
            elif file.endswith(".txt") or file.endswith(".json") or file.endswith(".label"):
                # Copy text/json immediately to output to preserve structure
                src_path = os.path.join(root, file)
                # Compute relative path to keep structure (e.g. key.txt next to key.mp4)
                rel = os.path.relpath(src_path, temp_dir)
                dst_path = os.path.join(OUTPUT_ROOT, shard_id, rel)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                text_files.append(dst_path)
    
    print(f"Found {len(video_files)} videos and {len(text_files)} transcripts in {shard_id}")
    
    if len(text_files) == 0:
        print(f"⚠️  WARNING: No transcript files (.txt/.json) found in {shard_id}! Verify raw data structure.")
    else:
        print(f"✅ Verified: Capturing {len(text_files)} transcripts.")
    
    # 3. Output dir
    shard_out_dir = os.path.join(OUTPUT_ROOT, shard_id)
    os.makedirs(shard_out_dir, exist_ok=True)
    
    # 4. Prepare Args for Parallel Execution
    task_args = [(f, temp_dir, shard_out_dir) for f in video_files]

    # 5. Parallel Execution
    print(f"Processing with Pool...")
    success_count = 0
    with ProcessPoolExecutor(max_workers=6) as executor:
        results = list(tqdm(
            executor.map(process_video_task, task_args),
            total=len(video_files)
        ))
        
    for r in results:
        if r == "Success": success_count += 1
        
    # Cleanup
    shutil.rmtree(temp_dir)
    volume.commit()
    return f"Shard {shard_id}: Processed {success_count}/{len(video_files)} videos."

@app.local_entrypoint()
def main(subset: str = "train", limit_ratio: float = 0.1): # Default 10% batch
    # 1. State Tracking
    state_file = "/mnt/processed_log.txt"
    processed_shards = set()
    
    # Needs a way to read volume file from local entrypoint? 
    # Actually, local_entrypoint runs on user machine, but volume is remote.
    # We need a helper function to read the log.
    try:
        log_content = read_log.remote(state_file)
        if log_content:
            processed_shards = set(log_content.strip().split("\n"))
            print(f"📖 Found {len(processed_shards)} processed shards in history.")
    except Exception:
        print("🆕 No history found. Starting fresh.")

    # 2. List all available shards
    print(f"Launching CPU FaceMesh Processing for {subset} (Batch Ratio: {limit_ratio})...")
    all_tars = list_tars.remote(subset)
    all_tars.sort() # Ensure consistent order
    print(f"   Found {len(all_tars)} total shards available.")
    
    # 3. Filter pending
    # Helper to get shard ID from path
    def get_id(path): return os.path.basename(path).split(".")[0]
    
    pending_tars = [t for t in all_tars if get_id(t) not in processed_shards]
    print(f"   Pending: {len(pending_tars)}/{len(all_tars)} shards.")
    
    if not pending_tars:
        print("✅ All shards processed! Nothing to do.")
        return

    # 4. Select Batch
    # Calculate batch size based on TOTAL shards (fixed size)
    # e.g. 10% of 100 = 10 shards per run
    batch_size = max(1, int(len(all_tars) * limit_ratio))
    batch_tars = pending_tars[:batch_size]
    
    print(f"🚀 Starting Batch: Processing {len(batch_tars)} shards ({len(pending_tars) - len(batch_tars)} will remain pending).")

    # 5. Process
    results = []
    for result in process_shard.map(batch_tars):
        print(result)
        results.append(result)
        
    # 6. Update Log (Append new IDs)
    # Extract IDs from processed batch
    new_ids = [get_id(t) for t in batch_tars]
    append_log.remote(state_file, "\n".join(new_ids))
    print(f"📝 Updated log with {len(new_ids)} shards.")

@app.function(volumes={"/mnt": volume})
def read_log(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()
    return ""

@app.function(volumes={"/mnt": volume})
def append_log(path, content):
    with open(path, 'a') as f:
        f.write(content + "\n")
    volume.commit()

@app.function(volumes={"/mnt": volume})
def list_tars(subset):
    files = glob.glob(f"{DATA_ROOT}/**/*{subset}*.tar", recursive=True)
    return files
