import modal
import os
import glob
import sys
import shutil

# --- Config ---
APP_NAME = "avsr-prep-features-gpu"
VOLUME_NAME = "avsr-volume"
CROPPED_ROOT = "/mnt/vicocktail_cropped"
OUTPUT_ROOT = "/mnt/vicocktail_features"

# --- Image Definition ---
# Heavy image for GPU processing (PyTorch, Timm, etc)
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "timm==0.9.12",
        "webdataset==0.2.79", 
        "huggingface_hub",
        "face-alignment>=1.4.0",
        "opencv-python-headless", # Still needed for VideoProcessor cv2
        "soundfile",
        "librosa",
        "av",
        "jiwer",
        "matplotlib",
        "pyyaml",
        index_url="https://download.pytorch.org/whl/cu118",
        extra_index_url="https://pypi.org/simple"
    )
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={"/mnt": volume},
    gpu="A100",        # Request A100 GPU
    timeout=7200,      # 2 hours
    cpu=2,
    memory=16384,
    max_containers=2 # Reduced to 2 as requested (and renamed from concurrency_limit)
)
def extract_features_shard(subset_path):
    """
    Process a single subset directory containing cropped .mp4 files.
    """
    from src.data.preprocessors.base import BasePreprocessor
    from src.utils.logging_utils import setup_logger
    import torch
    
    logger = setup_logger("FeatureExtractor")
    
    subset_name = os.path.basename(subset_path)
    logger.info(f"Processing subset: {subset_name}")
    
    # Custom Preprocessor for Filesystem
    class FileSystemPreprocessor(BasePreprocessor):
        def collect_metadata(self):
            # Scan for .mp4 files
            mp4_files = glob.glob(f"{self.data_root}/**/*.mp4", recursive=True)
            meta = []
            for f in mp4_files:
                rel_path = os.path.relpath(f, self.data_root)
                
                # Resolving Transcript (Text Label)
                text = ""
                base_path = os.path.splitext(f)[0]
                
                # Try .txt
                txt_path = base_path + ".txt"
                if os.path.exists(txt_path):
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as tf:
                            text = tf.read().strip()
                    except Exception as e:
                        logger.warning(f"Failed to read text {txt_path}: {e}")
                
                # Try .label (common in ViCocktail)
                if not text:
                     label_path = base_path + ".label"
                     if os.path.exists(label_path):
                         try:
                             with open(label_path, 'r', encoding='utf-8') as lf:
                                 text = lf.read().strip()
                         except Exception as e:
                             logger.warning(f"Failed to read label {label_path}: {e}")
                
                # Try .json (if not found in txt)
                if not text:
                    json_path = base_path + ".json"
                    if os.path.exists(json_path):
                        try:
                            import json
                            with open(json_path, 'r', encoding='utf-8') as jf:
                                data = json.load(jf)
                                # Adapt key based on actual json structure
                                text = data.get('text', data.get('transcript', ''))
                        except Exception as e:
                            logger.warning(f"Failed to read json {json_path}: {e}")

                meta.append({
                    "full_path": f,
                    "rel_path": rel_path,
                    "text": text,
                    "id": os.path.splitext(os.path.basename(f))[0]
                })
            return meta

    # Run Extraction
    # use_precropped=True is CRITICAL to skip FaceMesh and just resize
    processor = FileSystemPreprocessor(data_root=subset_path, use_precropped=True)
    
    output_dir = os.path.join(OUTPUT_ROOT, subset_name)
    output_manifest = os.path.join(OUTPUT_ROOT, f"{subset_name}_manifest.jsonl")
    
    os.makedirs(output_dir, exist_ok=True)
    
    processor.run(
        output_manifest=output_manifest,
        output_dir=output_dir,
        extract_features=True
    )
    
    volume.commit()
    
    # --- Self-Verification ---
    try:
        with open(output_manifest, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            empty_count = 0
            for line in lines:
                if '"text": ""' in line:
                    empty_count += 1
            
            if empty_count > 0:
                logger.error(f"❌ CRITICAL: Found {empty_count}/{len(lines)} samples with EMPTY LABELS in {subset_name}!")
                logger.error("   -> Check if prep_facemesh_cpu.py successfully copied .txt files.")
            else:
                logger.info(f"✅ Verified: All {len(lines)} samples have valid text labels.")
    except Exception as e:
        logger.warning(f"Verification failed: {e}")

    return f"Subset {subset_name} completed. Labels Verified: {'FAIL' if empty_count > 0 else 'PASS'}"

@app.local_entrypoint()
def main(subset: str = "train", limit_ratio: float = 0.1): # Default 10% batch
    # 1. State Tracking
    state_file = "/mnt/processed_features.log"
    processed_shards = set()
    
    try:
        log_content = read_log.remote(state_file)
        if log_content:
            processed_shards = set(log_content.strip().split("\n"))
            print(f"📖 Found {len(processed_shards)} processed shards in history.")
    except Exception:
        print("🆕 No history found. Starting fresh.")

    print(f"Launching GPU Feature Extraction for {subset} (Batch Ratio: {limit_ratio})...")
    
    # 2. List all available directories
    all_dirs = list_directories.remote(subset)
    all_dirs.sort()
    
    if not all_dirs:
        print(f"No directories found for subset {subset} in {CROPPED_ROOT}")
        return

    print(f"   Found {len(all_dirs)} total directories available.")
    
    # 3. Filter Pending
    # Helper to get shard ID from path (folder name)
    def get_id(path): return os.path.basename(path)
    
    pending_dirs = [d for d in all_dirs if get_id(d) not in processed_shards]
    print(f"   Pending: {len(pending_dirs)}/{len(all_dirs)} shards.")
    
    if not pending_dirs:
        print("✅ All shards processed! Nothing to do.")
        return

    # 4. Select Batch
    # Batch size based on TOTAL available (consistent sizing)
    batch_size = max(1, int(len(all_dirs) * limit_ratio))
    batch_dirs = pending_dirs[:batch_size]
    
    print(f"🚀 Starting Batch: Processing {len(batch_dirs)} shards ({len(pending_dirs) - len(batch_dirs)} will remain pending).")

    # 5. Process
    for result in extract_features_shard.map(batch_dirs):
        print(result)
        
    # 6. Update Log (Append new IDs)
    new_ids = [get_id(d) for d in batch_dirs]
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
def list_directories(subset):
    # Find all directories in CROPPED_ROOT that match the subset name
    target_pattern = f"{CROPPED_ROOT}/*{subset}*"
    dirs = glob.glob(target_pattern)
    return [d for d in dirs if os.path.isdir(d)]
