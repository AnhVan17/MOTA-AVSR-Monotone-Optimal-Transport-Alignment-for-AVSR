import modal
import os
import subprocess

# Define image
image = modal.Image.debian_slim()
app = modal.App("audit-volume")
volume = modal.Volume.from_name("avsr-volume")

@app.function(image=image, volumes={"/mnt": volume})
def audit_volume():
    print("=== AVSR VOLUME AUDIT REPORT ===")
    print("Root: /mnt")
    
    # 1. Top-Level Listing
    print("\n[1] Top-Level Directories & Sizes:")
    try:
        # Run du -sh for top level items
        result = subprocess.run(["du", "-sh", "/mnt/*"], capture_output=True, text=True, shell=True)
        print(result.stdout)
    except Exception as e:
        print(f"Error running du: {e}")
        
    # 2. Detailed Structure Analysis
    print("\n[2] Detailed Content Analysis:")
    
    important_dirs = ["/mnt/data", "/mnt/manifests", "/mnt/vicocktail_raw", "/mnt/vicocktail_cropped", "/mnt/vicocktail_features", "/mnt/checkpoints", "/mnt/logs"]
    
    # Walk top level only
    try:
        root_items = os.listdir("/mnt")
    except OSError:
        print("Could not list /mnt")
        return

    for item in root_items:
        full_path = os.path.join("/mnt", item)
        if os.path.isdir(full_path):
            print(f"\n--- Directory: {item} ---")
            
            # Count files recursively
            file_count = 0
            dir_count = 0
            ext_counts = {}
            
            # Limit traversal to avoid timeout on huge dirs
            for root, dirs, files in os.walk(full_path):
                dir_count += len(dirs)
                file_count += len(files)
                for f in files:
                    ext = os.path.splitext(f)[1]
                    ext_counts[ext] = ext_counts.get(ext, 0) + 1
                    
                # Break if too huge to keep listing fast
                if file_count > 10000:
                    break
            
            print(f"  Approx File Count: {file_count}{'+' if file_count >= 10000 else ''}")
            print(f"  Subdirectories: {dir_count}")
            print(f"  File Types: {ext_counts}")
            
            # If it looks like a dataset, show first few subfolders/files
            if "vicocktail" in item or "data" in item:
                 print("  Sample Contents:")
                 try:
                     sub_items = os.listdir(full_path)[:5]
                     for sub in sub_items:
                         print(f"    - {sub}")
                 except: 
                     pass
        else:
             print(f"\n--- File: {item} ---")
             
    print("\n=== END AUDIT ===")

@app.local_entrypoint()
def main():
    audit_volume.remote()
