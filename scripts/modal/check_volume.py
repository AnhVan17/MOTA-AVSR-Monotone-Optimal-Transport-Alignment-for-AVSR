import modal
import os
import shutil

APP_NAME = "check-volume-structure"
VOLUME_NAME = "avsr-dataset-volume"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)

# Mount 2 paths to see everything
vol_mounts = {
    "/mnt/raw_mirror": volume,        # Where tars usually are
    "/mnt/raw_data": volume           # Where unpacked data usually is
}

@app.function(volumes=vol_mounts)
def inspect_volume():
    print("="*60)
    print("🔍 CHECK VOLUME CONTENT (Smart Check)")
    print("="*60)
    
    # 1. Check Mirror (Tar files)
    mirror_path = "/mnt/raw_mirror"
    # Note: If volume structure is flat, raw_mirror might be a subdir or root. 
    # Let's check root structure first if mirror is empty.
    
    # Check what's actually in the volume root by listing the mount point content
    # Modal mounts the volume to the path. If we mounted same volume to 2 paths, 
    # checking one reveals the content.
    
    if os.path.exists(mirror_path):
        try:
            files = sorted(os.listdir(mirror_path))
            print(f"📁 Root/Mirror of Volume: {mirror_path} ({len(files)} items)")
            
            # Filter Tar files
            tars = [f for f in files if f.endswith('.tar')]
            print(f"   found {len(tars)} .tar files")
            if tars:
                print(f"   First 3: {tars[:3]}")
                print(f"   Last 3:  {tars[-3:]}")
                
            # Find Pattern for TEST
            test_tars = [f for f in tars if "test" in f.lower()]
            if test_tars:
                print(f"   🎯 TEST Tars found ({len(test_tars)}): {test_tars}")
            else:
                print("   ⚠️ No specific 'test' tar found. Test data might be inside numbered tars.")
        except Exception as e:
            print(f"Error listing mirror: {e}")
    
    # 2. Check Unpacked (Raw)
    # Based on previous scripts: /mnt/raw_data/vicocktail/raw
    unpacked_path = "/mnt/raw_data/vicocktail/raw"
    if os.path.exists(unpacked_path):
        try:
            subdirs = sorted(os.listdir(unpacked_path))
            print(f"\n📁 Unpacked Data: {unpacked_path} ({len(subdirs)} dirs)")
            if subdirs:
                print(f"   Sample: {subdirs[:3]} ... {subdirs[-3:]}")
        except:
             print(f"Error listing unpacked at {unpacked_path}")
    else:
         print(f"❌ Path {unpacked_path} does not exist.")

    # 3. Stats
    try:
        stat = os.statvfs("/mnt/raw_mirror")
        inodes_used = stat.f_files - stat.f_ffree
        print(f"\n🔢 INODES: {inodes_used} used / {stat.f_files} total ({inodes_used/stat.f_files*100:.1f}%)")
    except: pass
    print("="*60)

@app.local_entrypoint()
def main():
    inspect_volume.remote()
