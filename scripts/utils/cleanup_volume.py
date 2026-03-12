import modal
import os
import shutil

image = modal.Image.debian_slim()
app = modal.App("cleanup-volume")
volume = modal.Volume.from_name("avsr-volume")

@app.function(image=image, volumes={"/mnt": volume})
def move_legacy_data():
    print("=== STARTING VOLUME CLEANUP ===")
    
    archive_root = "/mnt/_legacy_archive"
    legacy_data = "/mnt/data"
    legacy_manifests = "/mnt/manifests"
    
    # Create Archive Root
    if not os.path.exists(archive_root):
        print(f"Creating archive root: {archive_root}")
        os.makedirs(archive_root, exist_ok=True)
        
    # Move /mnt/data -> /mnt/_legacy_archive/grid_data
    if os.path.exists(legacy_data):
        dest = os.path.join(archive_root, "grid_data")
        print(f"Moving {legacy_data} -> {dest}...")
        try:
            shutil.move(legacy_data, dest)
            print("✅ Move complete.")
        except Exception as e:
            print(f"❌ Error moving data: {e}")
    else:
        print(f"Legacy data {legacy_data} not found (already moved?).")

    # Move /mnt/manifests -> /mnt/_legacy_archive/legacy_manifests
    if os.path.exists(legacy_manifests):
        dest = os.path.join(archive_root, "legacy_manifests")
        print(f"Moving {legacy_manifests} -> {dest}...")
        try:
            shutil.move(legacy_manifests, dest)
            print("✅ Move complete.")
        except Exception as e:
            print(f"❌ Error moving manifests: {e}")
    else:
        print(f"Legacy manifests {legacy_manifests} not found.")
        
    volume.commit()
    print("=== CLEANUP COMPLETE ===")

@app.local_entrypoint()
def main():
    move_legacy_data.remote()
