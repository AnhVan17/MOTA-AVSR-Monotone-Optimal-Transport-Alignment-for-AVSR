import modal
import torch
import os
import glob
import sys

# --- Config ---
APP_NAME = "avsr-inspect-features"
VOLUME_NAME = "avsr-volume"
FEATURES_ROOT = "/mnt/vicocktail_features"

# --- Image ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch",
        "numpy"
    )
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={"/mnt": volume}
)
def inspect_subset(subset_name):
    print(f"Inspecting features for subset: {subset_name}")
    subset_dir = os.path.join(FEATURES_ROOT, subset_name)
    
    if not os.path.exists(subset_dir):
        print(f"Directory not found: {subset_dir}")
        # List root to see what exists
        print(f"Contents of {FEATURES_ROOT}:")
        try:
            print(os.listdir(FEATURES_ROOT))
        except Exception as e:
            print(e)
        return

    files = glob.glob(os.path.join(subset_dir, "*.pt"))
    print(f"Found {len(files)} .pt files.")
    
    if not files:
        return

    # Inspect first 3 files
    for fpath in files[:3]:
        print(f"\n--- Checking {os.path.basename(fpath)} ---")
        try:
            data = torch.load(fpath)
            print(f"Keys: {list(data.keys())}")
            
            if 'visual' in data:
                v = data['visual']
                print(f"Visual Shape: {v.shape} | Type: {v.dtype} | Range: [{v.min():.3f}, {v.max():.3f}]")
                if v.shape[-1] != 512:
                    print("WARNING: Visual feature dim is not 512!")
                    
            if 'audio' in data:
                a = data['audio']
                print(f"Audio Shape: {a.shape} | Type: {a.dtype} | Range: [{a.min():.3f}, {a.max():.3f}]")
                if a.shape[-1] != 768:
                    print("WARNING: Audio feature dim is not 768!")

            if 'text' in data:
                print(f"Text: '{data['text']}'")
                
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

@app.local_entrypoint()
def main(subset: str = "test_snr_0_interferer_1"):
    # The output struct from prep_features_gpu was /mnt/vicocktail_features/{subset_name}
    # And prep_features_gpu passed os.path.basename(subset_path) as subset_name.
    # The input path was /mnt/vicocktail_cropped/{shard_id}
    # So the subset_name in features dir is likely the shard_id name.
    
    inspect_subset.remote(subset)
