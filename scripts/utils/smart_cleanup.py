import os
import shutil
import modal
from tqdm import tqdm

APP_NAME = "avsr-volume-cleanup"
VOLUME_NAME = "avsr-volume"

image = modal.Image.debian_slim().pip_install("tqdm")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(image=image, volumes={"/mnt": volume})
def smart_cleanup():
    """
    Frees up Inodes by deleting intermediate 'cropped' files 
    THAT HAVE ALREADY BEEN PROCESSED into 'features'.
    """
    CROPPED_ROOT = "/mnt/vicocktail_cropped"
    FEATURES_ROOT = "/mnt/vicocktail_features"
    LEGACY_ROOT = "/mnt/_legacy_archive"
    
    print("🧹 Starting Smart Cleanup...")
    
    # 1. Delete Legacy Archive (If exists and user agrees - forcing true here for recovery)
    if os.path.exists(LEGACY_ROOT):
        print(f"🗑️  Found Legacy Archive: {LEGACY_ROOT}")
        try:
            shutil.rmtree(LEGACY_ROOT)
            print("   ✅ Deleted Legacy Archive.")
        except Exception as e:
            print(f"   ❌ Failed to delete Legacy: {e}")

    # 2. Find Processed Shards
    # Logic: If folder exists in FEATURES_ROOT, we can delete it from CROPPED_ROOT
    if not os.path.exists(FEATURES_ROOT):
        print("⚠️  Features directory not found. Skipping smart crop cleanup.")
        return

    # List all feature directories
    feature_dirs = [d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))]
    print(f"🔍 Found {len(feature_dirs)} completed feature shards.")
    
    deleted_count = 0
    reclaimed_inodes = 0
    
    for shard_name in tqdm(feature_dirs):
        cropped_path = os.path.join(CROPPED_ROOT, shard_name)
        
        if os.path.exists(cropped_path):
            # Count files roughly (for reporting)
            try:
                # fast count not easy, just delete
                # num_files = len(os.listdir(cropped_path)) 
                shutil.rmtree(cropped_path)
                deleted_count += 1
                # reclaimed_inodes += num_files
                # print(f"   deleted {shard_name} ({num_files} files)")
            except Exception as e:
                print(f"   ❌ Failed to delete {shard_name}: {e}")
                
    print(f"✨ Cleanup Complete!")
    print(f"   - Deleted {deleted_count} shards from {CROPPED_ROOT}")
    # print(f"   - Reclaimed approx {reclaimed_inodes} inodes (files).")
    volume.commit()
