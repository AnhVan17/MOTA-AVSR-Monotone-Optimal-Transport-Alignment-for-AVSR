import modal
from pathlib import Path

app = modal.App("inspect-data")
volume = modal.Volume.from_name("avsr-dataset-volume")

image = modal.Image.debian_slim().pip_install("webdataset")

@app.function(image=image, volumes={"/data": volume})
def inspect():
    import os
    import webdataset as wds
    import json
    
    raw_path = "/data/raw_mirror"
    print(f"Inspecting {raw_path}...")
    
    tars = []
    for root, dirs, files in os.walk(raw_path):
        for f in files:
            if f.endswith(".tar"):
                tars.append(os.path.join(root, f))
    
    if not tars:
        print("No TAR files found in /data/raw_mirror")
        # Let's check root
        print("Checking /data root...")
        for root, dirs, files in os.walk("/data"):
             for f in files:
                if f.endswith(".tar"):
                    print(f"Found: {os.path.join(root, f)}")
        return

    target_tar = [t for t in tars if "snr_-5" in t][0]
    print(f"\n==========================================")
    print(f"Opening {target_tar}...")
    print(f"==========================================")
    
    try:
        ds = wds.WebDataset(f"file://{target_tar}")
        for i, sample in enumerate(ds):
            if i != 1: continue # Only Sample 1
            print(f"\n--- {target_tar.split('/')[-1]} Sample {i} ---")
            print(f"Key: {sample.get('__key__')}")
            print(f"Keys present: {list(sample.keys())}")
            
            for k, v in sample.items():
                if k.startswith('__'): continue
                if isinstance(v, bytes):
                    try:
                        decoded = v.decode('utf-8')
                        print(f"  [{k}] (decoded): {decoded[:200]}...")
                    except:
                        print(f"  [{k}] (binary, {len(v)} bytes)")
                else:
                    print(f"  [{k}]: {str(v)[:150]}...")
            break
    except Exception as e:
        print(f"Error reading TAR: {e}")

if __name__ == "__main__":
    with app.run():
        inspect.remote()
