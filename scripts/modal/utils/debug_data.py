import modal
import os
import glob
import json

APP_NAME = "avsr-debug-data"
VOLUME_NAME = "avsr-volume"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(image=modal.Image.debian_slim(), volumes={"/mnt": volume})
def check_data():
    print("=== DEBUG REPORT ===")
    
    # 1. Check Cropped Dir for .txt/.json
    print("\n[1] Checking /mnt/vicocktail_cropped for labels...")
    txt_files = []
    for root, dirs, files in os.walk("/mnt/vicocktail_cropped"):
        for f in files:
            if f.endswith(".txt") or f.endswith(".json"):
                txt_files.append(os.path.join(root, f))
                if len(txt_files) >= 5: break
        if len(txt_files) >= 5: break
        
    if not txt_files:
        print("❌ NO .txt or .json files found in /mnt/vicocktail_cropped")
        print("   -> Conclusion: prep_facemesh_cpu.py needs to be re-run.")
    else:
        print(f"✅ Found label files (Sample): {txt_files}")
        
    # 2. Check Manifest Content
    print("\n[2] Checking Manifest content...")
    manifests = glob.glob("/mnt/vicocktail_features/*manifest.jsonl")
    if not manifests:
        print("❌ No manifests found in /mnt/vicocktail_features")
    else:
        for m in manifests:
            print(f"File: {os.path.basename(m)}")
            with open(m, 'r') as f:
                for i, line in enumerate(f):
                    if i >= 3: break
                    try:
                        entry = json.loads(line)
                        text = entry.get('text', 'N/A')
                        print(f"  - Sample {i}: text='{text}'") 
                        if not text:
                             print("    ⚠️  TEXT IS EMPTY!")
                    except:
                        print(f"  - Sample {i}: (Invalid JSON)")

@app.local_entrypoint()
def main():
    check_data.remote()
