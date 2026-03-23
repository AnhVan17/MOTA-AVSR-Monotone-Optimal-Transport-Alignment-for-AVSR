import modal
import os
import glob

APP_NAME = "debug-crop-output"
VOLUME_NAME = "avsr-vicocktail-processed"
MOUNT_PATH = "/mnt/processed"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(volumes={MOUNT_PATH: volume})
def check_crop_content():
    # 1. Check RAW Input
    # Note: Volume is mounted at MOUNT_PATH (/mnt/processed). 
    # But based on previous scripts, Raw might be at /mnt/dataset/vicocktail/raw 
    # We need to mount the volume to check Raw too or use relative if mounted at root
    # Let's assume we can look around if we mount volume at root /mnt/vol
    pass

# Re-define app with root mount to check everything
app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)
MOUNT_ROOT = "/mnt/vol"

@app.function(volumes={MOUNT_ROOT: volume})
def check_data_flow():
    print("="*60)
    print("🕵️ DATA FLOW INSPECTION")
    print("="*60)
    
    # 1. Check Unpacked RAW
    raw_dir = f"{MOUNT_ROOT}/vicocktail/raw"
    print(f"1️⃣ Checking RAW Input: {raw_dir}")
    if os.path.exists(raw_dir):
        raw_files = glob.glob(os.path.join(raw_dir, "**", "*.mp4"), recursive=True)
        # Also check .video if not renamed
        if not raw_files:
             raw_files = glob.glob(os.path.join(raw_dir, "**", "*.video"), recursive=True)
             
        print(f"   ✅ Found {len(raw_files)} raw video files.")
        if len(raw_files) > 0:
            print(f"   Sample: {raw_files[:3]}")
    else:
        print("   ❌ RAW Directory does not exist!")

    # 2. Check CROPPED Output
    # Based on preprocess script: /mnt/processed/vicocktail_cropped
    # In our root mount, this is /mnt/vol/vicocktail_cropped (since processed mount logic usually puts it at root/vicocktail_cropped)
    # Let's check typical paths
    
    crop_dir_1 = f"{MOUNT_ROOT}/vicocktail_cropped"
    crop_dir_2 = f"{MOUNT_ROOT}/processed/vicocktail_cropped" # If mounted specifically
    
    print(f"\n2️⃣ Checking CROPPED Output...")
    
    final_crop_dir = None
    if os.path.exists(crop_dir_1): final_crop_dir = crop_dir_1
    elif os.path.exists(crop_dir_2): final_crop_dir = crop_dir_2
    
    if final_crop_dir:
        crop_files = glob.glob(os.path.join(final_crop_dir, "**", "*.mp4"), recursive=True)
        print(f"   Path: {final_crop_dir}")
        print(f"   ✅ Found {len(crop_files)} cropped video files.")
    else:
        print(f"   ❌ Cropped directory not found at {crop_dir_1} or {crop_dir_2}")
        
    print("="*60)
    
@app.local_entrypoint()
def main():
    check_data_flow.remote()
