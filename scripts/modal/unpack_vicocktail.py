import modal
import os
import subprocess
import glob
import shutil
from concurrent.futures import ThreadPoolExecutor

# --- Config ---
APP_NAME = "avsr-unpack-vicocktail-v2"
VOLUME_NAME = "avsr-dataset-volume" 
MOUNT_PATH = "/mnt"

image = modal.Image.debian_slim(python_version="3.10")

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={MOUNT_PATH: volume},
    cpu=4.0,       
    memory=4096,   # Increased memory for safety
    timeout=7200   # 2 hours
)
def unpack_dataset():
    input_dir = f"{MOUNT_PATH}/raw_mirror"
    output_dir = f"{MOUNT_PATH}/vicocktail/raw"
    
    print(f"🚀 Starting Robust Unpack V2...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    if not os.path.exists(input_dir):
        print("❌ Critical: Input directory missing.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Get Tar Files
    tar_files = glob.glob(os.path.join(input_dir, "**", "*.tar"), recursive=True)
    if not tar_files:
        print("❌ No tar files found.")
        return
        
    print(f"✅ Found {len(tar_files)} tar files.")

    def process_single_tar(tar_path):
        """
        Unpack a single tar file and rename .video -> .mp4
        """
        tar_name = os.path.basename(tar_path)
        # Create a specific subdir for each tar to avoid collisions
        # e.g. /mnt/vicocktail/raw/avvn-train-000000/
        sub_dir_name = os.path.splitext(tar_name)[0]
        target_dir = os.path.join(output_dir, sub_dir_name)
        
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            # 1. Unpack
            # Use --no-same-owner to avoid permission issues
            cmd = ["tar", "-xf", tar_path, "-C", target_dir, "--no-same-owner"]
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
            
            # 2. Rename .video -> .mp4
            renamed_count = 0
            for root, _, files in os.walk(target_dir):
                for file in files:
                    if file.endswith(".video"):
                        old_path = os.path.join(root, file)
                        new_path = os.path.join(root, file.replace(".video", ".mp4"))
                        os.rename(old_path, new_path)
                        renamed_count += 1
            
            return True, f"Unpacked {tar_name} ({renamed_count} videos renamed)"
            
        except subprocess.CalledProcessError as e:
            return False, f"Failed {tar_name}: {e.stderr.decode().strip()}"
        except Exception as e:
            return False, f"Error {tar_name}: {str(e)}"

    # Parallel Execution
    max_workers = 8
    print(f"📦 Unpacking with {max_workers} processes...")
    
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_tar, t) for t in tar_files]
        
        # Simple progress tracking
        finished = 0
        total = len(futures)
        
        for future in futures:
            success, msg = future.result()
            finished += 1
            if success:
                success_count += 1
                if finished % 10 == 0: print(f"[{finished}/{total}] {msg}")
            else:
                print(f"❌ [{finished}/{total}] {msg}")

    print(f"\n🏁 Finished: {success_count}/{len(tar_files)} successful.")
    
    # Validation
    mp4_count = len(glob.glob(os.path.join(output_dir, "**", "*.mp4"), recursive=True))
    print(f"🔍 Total .mp4 files ready: {mp4_count}")
    
    if mp4_count > 0:
        print("💾 Committing Volume...")
        volume.commit()
        print("✅ Done!")

@app.local_entrypoint()
def main():
    unpack_dataset.remote()
