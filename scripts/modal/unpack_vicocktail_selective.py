"""
Selective ViCocktail Unpacker
=============================
Unpack specific number of train files + all test files

Usage:
    # 50 train + all test (default)
    modal run scripts/modal/unpack_vicocktail_selective.py
    
    # Custom train limit
    modal run scripts/modal/unpack_vicocktail_selective.py --train-limit 100
    
    # All files (no limit)
    modal run scripts/modal/unpack_vicocktail_selective.py --train-limit 0
"""
import modal
import os
import tarfile
import glob
import shutil

APP_NAME = "avsr-unpack-selective-v2"
VOLUME_NAME = "avsr-dataset-volume"

# Define Mount Paths
VOLUME_MOUNT = "/mnt/dataset"
MIRROR_DIR = f"{VOLUME_MOUNT}" 
RAW_OUTPUT_DIR = f"{VOLUME_MOUNT}/vicocktail/raw"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("tqdm")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={
        VOLUME_MOUNT: volume
    },
    timeout=86400,
    cpu=4.0,
    memory=4096
)
def unpack_selective(train_limit: int = 50, clean_first: bool = False):
    """
    Unpack vicocktail tar files selectively.
    
    Args:
        train_limit: Max number of train tars. 0 = no limit (all train files)
        clean_first: If True, delete existing unpacked data before starting
    """
    from tqdm import tqdm
    
    print("="*60)
    print("📦 SELECTIVE VICOCKTAIL UNPACKER")
    print(f"   Train limit: {train_limit if train_limit > 0 else 'ALL'}")
    print(f"   Clean first: {clean_first}")
    print(f"   Mount: {VOLUME_MOUNT}")
    print("="*60)
    
    # 1. Optional: Clean old unpacked data
    if clean_first and os.path.exists(RAW_OUTPUT_DIR):
        print(f"🧹 Cleaning old unpacked dir: {RAW_OUTPUT_DIR}")
        try:
            shutil.rmtree(RAW_OUTPUT_DIR)
        except Exception as e:
            print(f"Warning cleaning dir: {e}")
    
    os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
    
    # 2. Scan Tar Files
    print(f"🔍 Scanning .tar files in {MIRROR_DIR}...")
    tar_files = glob.glob(os.path.join(MIRROR_DIR, "*.tar"))
    if not tar_files:
        tar_files = glob.glob(os.path.join(MIRROR_DIR, "**", "*.tar"), recursive=True)
    
    if not tar_files:
        print("❌ No .tar files found! Did you run download.py?")
        return
        
    print(f"   Found {len(tar_files)} total .tar files.")
    
    # 3. Separate train vs test
    tar_files.sort()
    
    train_tars = []
    test_tars = []
    other_tars = []
    
    for tf in tar_files:
        filename = os.path.basename(tf).lower()
        
        if "test" in filename:
            test_tars.append(tf)
        elif "train" in filename:
            train_tars.append(tf)
        else:
            other_tars.append(tf)
    
    # 4. Apply train limit
    if train_limit > 0 and len(train_tars) > train_limit:
        train_tars = train_tars[:train_limit]
    
    all_targets = train_tars + test_tars + other_tars
    
    print(f"\n📋 Selection Summary:")
    print(f"   Train: {len(train_tars)} files" + (f" (limited from {len([t for t in tar_files if 'train' in os.path.basename(t).lower()])})" if train_limit > 0 else ""))
    print(f"   Test:  {len(test_tars)} files (ALL)")
    print(f"   Other: {len(other_tars)} files")
    print(f"   Total: {len(all_targets)} files to unpack")
    
    # 5. Unpack Loop
    print("\n🚀 Starting Unpack...")
    success_count = 0
    
    for idx, tar_path in enumerate(all_targets):
        fname = os.path.basename(tar_path)
        
        # Check if already unpacked (folder exists with files)
        extract_dir = os.path.join(RAW_OUTPUT_DIR, os.path.splitext(fname)[0])
        if os.path.exists(extract_dir) and os.listdir(extract_dir):
            print(f"⏩ [{idx+1}/{len(all_targets)}] Skipping {fname} (exists)")
            success_count += 1
            continue
        
        print(f"📦 [{idx+1}/{len(all_targets)}] Unpacking {fname}...")
        
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=RAW_OUTPUT_DIR)
            success_count += 1
            
            # Periodic commit
            if success_count % 10 == 0:
                print("   💾 Committing...")
                volume.commit()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # 6. Rename .video -> .mp4
    print("\n🔄 Renaming .video -> .mp4...")
    video_files = glob.glob(os.path.join(RAW_OUTPUT_DIR, "**", "*.video"), recursive=True)
    for vf in video_files:
        os.rename(vf, vf.replace(".video", ".mp4"))
    print(f"   Renamed {len(video_files)} files.")
    
    # 7. Final stats
    mp4_count = len(glob.glob(os.path.join(RAW_OUTPUT_DIR, "**", "*.mp4"), recursive=True))
    
    print("="*60)
    print(f"✅ Complete: {success_count}/{len(all_targets)} tars unpacked")
    print(f"   Total .mp4 files: {mp4_count}")
    print("="*60)
    
    print("💾 Committing volume...")
    volume.commit()
    print("✅ Done!")

@app.local_entrypoint()
def main(train_limit: int = 50, clean_first: bool = False):
    """
    Usage Examples:
        # Default: 50 train + all test
        modal run scripts/modal/unpack_vicocktail_selective.py
        
        # 100 train + all test
        modal run scripts/modal/unpack_vicocktail_selective.py --train-limit 100
        
        # All files (no limit)
        modal run scripts/modal/unpack_vicocktail_selective.py --train-limit 0
        
        # Clean existing data first
        modal run scripts/modal/unpack_vicocktail_selective.py --clean-first
    """
    unpack_selective.remote(train_limit=train_limit, clean_first=clean_first)
