import modal
import os
import tarfile
import glob

APP_NAME = "avsr-inspect-tar"
VOLUME_NAME = "avsr-volume"
DATA_ROOT = "/mnt/vicocktail_raw"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(image=modal.Image.debian_slim(), volumes={"/mnt": volume})
def inspect_tar_contents(subset="test_snr_0_interferer_1"):
    print(f"Searching for tar files matching '{subset}' in {DATA_ROOT}...")
    tars = glob.glob(f"{DATA_ROOT}/**/*{subset}*.tar", recursive=True)
    
    if not tars:
        print("❌ No matching tar files found.")
        return

    target_tar = tars[0]
    print(f"🔍 Inspecting: {target_tar}")
    
    try:
        with tarfile.open(target_tar, "r") as tar:
            members = tar.getmembers()
            print(f"Total files in archive: {len(members)}")
            
            print("\nFirst 20 files:")
            for m in members[:20]:
                print(f" - {m.name} ({m.size} bytes)")
                
            print("\nSearching for potential text/metadata files:")
            found = False
            for m in members:
                if m.name.endswith(('.txt', '.json', '.csv', '.srt', '.vtt')):
                    print(f" -> FOUND: {m.name}")
                    found = True
            
            if not found:
                print("❌ No standard text files (.txt, .json, .csv, .srt, .vtt) found in archive.")
                
            # Check for directories structure
            print("\nDirectory Structure (Top Level):")
            dirs = set(m.name.split('/')[0] for m in members)
            for d in dirs:
                print(f"  [{d}]")
                
    except Exception as e:
        print(f"Error reading tar: {e}")

@app.local_entrypoint()
def main(subset: str = "test_snr_0_interferer_1"):
    inspect_tar_contents.remote(subset)
