import modal
import os
import shutil

APP_NAME = "cleanup-volume"
VOLUME_NAME = "avsr-dataset-volume"
# Mount to root to see everything
MOUNT_PATH = "/mnt"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)

@app.function(volumes={MOUNT_PATH: volume}, timeout=3600)
def cleanup(clean_all: bool = False):
    print(f"🧹 STARTING CLEANUP on {MOUNT_PATH}...")
    
    # Check Stats
    def print_stats(label):
        try:
            stat = os.statvfs(MOUNT_PATH)
            inodes = stat.f_files - stat.f_ffree
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            print(f"[{label}] Inodes: {inodes} used | Free Space: {free_gb:.2f}/{total_gb:.2f} GB")
        except: pass

    print_stats("BEFORE")

    if clean_all:
        print("\n⚠️ WARNING: WIPING ENTIRE VOLUME (RESET MODE)...")
        # List items in root mount
        items = os.listdir(MOUNT_PATH)
        for item in items:
            # Skip hidden system files if any, though usually safe to delete all in /mnt
            full_path = os.path.join(MOUNT_PATH, item)
            try:
                print(f"🗑️ Deleting {item} ...")
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.unlink(full_path)
            except Exception as e:
                print(f"❌ Failed to delete {item}: {e}")
        
        print("✅ Volume Reset Complete.")
    
    else:
        # Normal cleanup (Partial)
        print("ℹ Partial Cleanup Mode (Debug files & Raw Mirror)")
        targets = [
            f"{MOUNT_PATH}/raw_mirror",             
            f"{MOUNT_PATH}/vicocktail_cropped_debug", 
            f"{MOUNT_PATH}/processed_features/vicocktail_debug", 
            f"{MOUNT_PATH}/manifests/debug"         
        ]
        
        for target in targets:
            if os.path.exists(target):
                print(f"Stef🗑️ Deleting {target}...")
                try:
                    if os.path.isdir(target):
                        shutil.rmtree(target)
                    else:
                        os.remove(target)
                except Exception as e:
                    print(f"   ❌ Failed: {e}")

    # Commit
    print("💾 Committing changes...")
    volume.commit()
    print_stats("AFTER")

@app.local_entrypoint()
def main(all: bool = False):
    """
    Usage:
      modal run scripts/modal/cleanup_volume.py 
      modal run scripts/modal/cleanup_volume.py --all
    """
    msg = "WARNING: This will delete specific folders."
    if all:
        msg = "🚨 CRITICAL WARNING: THIS WILL WIPE THE ENTIRE VOLUME! 🚨"
    
    print(msg)
    confirm = input("Type 'yes' to proceed: ")
    if confirm.lower() == "yes":
        cleanup.remote(clean_all=all)
    else:
        print("Aborted.")
