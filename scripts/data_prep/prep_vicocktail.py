import modal
import sys
import os
from pathlib import Path

# Face detection via face-alignment (GPU-native, no EGL issues)

from modal import App, Image, Volume

# --- Config ---
APP_NAME = "avsr-prep-vicocktail"
VOLUME_NAME = "avsr-volume"
DATA_ROOT = "/mnt/vicocktail_raw"
OUTPUT_ROOT = "/mnt/vicocktail_features"

# --- Image ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libgl1", "libgl1-mesa-glx", "libglib2.0-0") # ffmpeg
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "timm==0.9.12",             # Required by BasePreprocessor
        "webdataset==0.2.79",
        "huggingface_hub",
        "face-alignment>=1.4.0",   # GPU-native face detection (replaces MediaPipe)
        "opencv-python-headless",
        "soundfile",
        "librosa",
        "av",                       # PyAV for robust audio
        "jiwer",
        "matplotlib",
        index_url="https://download.pytorch.org/whl/cu118",
        extra_index_url="https://pypi.org/simple" # Fallback for non-torch packages
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={"/mnt": volume},
    timeout=3600, # 1 hour per shard download usually enough
    secrets=[modal.Secret.from_name("hf-token")] # Ensure HF_TOKEN is available
)
def download_shard_subset(subset):
    sys.path.append("/root")
    from scripts.data_prep.download_vicocktail import download_vicocktail
    
    print(f"Downloading subset: {subset}")
    # Download to raw folder
    subsets_arg = [subset] if subset != 'all' else None
    download_vicocktail(DATA_ROOT, subsets=subsets_arg)
    volume.commit()
    return f"Downloaded {subset}"

@app.function(
    image=image,
    volumes={"/mnt": volume},
    timeout=7200, # 2 hours processing time
    gpu="T4" # GPU needed for FaceMesh + Whisper
)
def process_data(subset_name, limit_ratio: float = 1.0):
    sys.path.append("/root")
    from src.data.preprocessors.vicocktail import ViCocktailPreprocessor
    
    # We assume data is downloaded in DATA_ROOT
    # But ViCocktailPreprocessor scans widely. 
    # To be safe, we point it to DATA_ROOT
    
    print(f"Processing data in {DATA_ROOT}...")
    
    # Init Preprocessor
    processor = ViCocktailPreprocessor(
        data_root=DATA_ROOT,
        use_precropped=False # It's raw in tar
    )
    
    # Run
    # Output manifest
    manifest_name = f"vicocktail_{subset_name}.jsonl"
    manifest_path = f"/mnt/manifests/{manifest_name}"
    os.makedirs("/mnt/manifests", exist_ok=True)
    
    processor.run(
        output_manifest=manifest_path, 
        output_dir=OUTPUT_ROOT,
        limit_ratio=limit_ratio,
        filter_keyword=subset_name
    )
    
    volume.commit()
    return f"Processed {subset_name}. Manifest: {manifest_path}"

@app.function(
    image=image,
    volumes={"/mnt": volume}
)
def inspect_data(subset):
    """
    Peek inside the first .tar file to see valid keys.
    """
    import tarfile
    import glob
    
    print(f"Inspecting subset: {subset}")
    # Find the file recursively (files are likely in a subdir like 'data/')
    pattern = f"{DATA_ROOT}/**/*{subset}*.tar"
    files = glob.glob(pattern, recursive=True)
    if not files:
        print(f"No files found matching {pattern} in {DATA_ROOT}")
        return
        
    tar_path = files[0]
    print(f"Inspecting file: {tar_path}")
    
    try:
        with tarfile.open(tar_path, "r") as tar:
            print("First 10 members in tar:")
            for i, member in enumerate(tar):
                if i >= 10: break
                print(f" - {member.name} (Size: {member.size})")
    except Exception as e:
        print(f"Failed to read tar: {e}")


@app.function(
    image=image,
    volumes={"/mnt": volume},
    gpu="A10G", # Strong GPU for ResNet/Whisper
    timeout=7200,
    cpu=4,
    memory=16384
)
def extract_features_shard(subset_name):
    """
    Run Feature Extraction (ResNet + Whisper) on previously cropped videos.
    Input: /mnt/vicocktail_cropped/{subset_name}
    Output: /mnt/vicocktail_features/{subset_name}
    """
    sys.path.append("/root")
    from src.data.preprocessors.base import BasePreprocessor
    from src.utils.logging_utils import setup_logger
    import glob
    
    logger = setup_logger("FeatureExtractor")
    logger.info(f"Starting Feature Extraction for {subset_name}...")
    
    # Input/Output Config
    input_root = "/mnt/vicocktail_cropped"
    output_root = "/mnt/vicocktail_features"
    
    # Find the specific shard folder (it might be named slightly differently or exactly matches)
    # The CPU script output to /mnt/vicocktail_cropped/{shard_id}
    # We need to process ALL shards that belong to this subset
    
    # Find all shards matching the subset
    search_pattern = f"{input_root}/*{subset_name}*"
    shard_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
    
    if not shard_dirs:
        return f"No cropped data found for {subset_name} in {input_root}"
    
    logger.info(f"Found {len(shard_dirs)} shards to process: {[os.path.basename(d) for d in shard_dirs]}")

    results = []
    
    # Define a Custom Preprocessor to read .mp4 files from filesystem
    class FileSystemPreprocessor(BasePreprocessor):
        def collect_metadata(self):
            # Scan for .mp4 files in the specific input_dir passed to constructor
            mp4_files = glob.glob(f"{self.data_root}/**/*.mp4", recursive=True)
            meta = []
            for f in mp4_files:
                rel_path = os.path.relpath(f, self.data_root)
                meta.append({
                    "full_path": f,
                    "rel_path": rel_path,
                    "text": "", # Placeholder (Merged later via Manifest)
                    "id": os.path.splitext(os.path.basename(f))[0]
                })
            return meta

    for shard_dir in shard_dirs:
        shard_id = os.path.basename(shard_dir)
        shard_out_dir = os.path.join(output_root, shard_id)
        os.makedirs(shard_out_dir, exist_ok=True)
        
        # Manifest for this shard
        manifest_path = os.path.join(output_root, f"{shard_id}.jsonl")
        
        # Init & Run
        processor = FileSystemPreprocessor(data_root=shard_dir, use_precropped=True)
        processor.run(
            output_manifest=manifest_path,
            output_dir=shard_out_dir,
            extract_features=True
        )
        results.append(f"Processed {shard_id}")
        
    volume.commit()
    return "\n".join(results)


@app.local_entrypoint()
def main(action: str = "download", subset: str = "train", limit_ratio: float = 1.0):
    """
    Args:
        action: 'download', 'process' (features), 'inspect', 'inspect_output'
        subset: 'train' or 'test_snr_...' or 'all'
    """
    if action == "download":
        print(f"Starting Download for {subset}...")
        download_shard_subset.remote(subset)
        
    elif action == "process":
        # Note: 'process' in this context now means GPU Feature Extraction
        # The CPU FaceMesh step is handled by 'prep_facemesh_cpu.py' (kept separate for env isolation)
        print(f"Starting GPU Processing (FaceMesh + Audio) for {subset} (Ratio: {limit_ratio})...")
        # Reuse process_data for the GPU-heavy part (FaceMesh + Extraction)
        # Note: Previous 'process_data' in this file was CPU-based or assumed raw tars.
        # Wait, lines 64-98 define process_data with GPU=T4.
        # It calls ViCocktailPreprocessor.
        # So we should call process_data, NOT extract_features_shard (which reads cropped folders).
        # extract_features_shard is legacy or for 2-step pipeline.
        
        result = process_data.remote(subset, limit_ratio)
        print(result)
        
    elif action == "inspect":
         print(f"Inspecting data for {subset}...")
         inspect_data.remote(subset)
         
    elif action == "inspect_output":
         print(f"Inspecting output for {subset}...")
         inspect_output.remote(subset)
         
    else:
        print("Invalid action. Use 'download', 'process', 'inspect', or 'inspect_output'.")

@app.function(
    image=image,
    volumes={"/mnt": volume}
)
def inspect_output(subset):
    """Check if output files exist."""
    import glob
    import os
    
    out_dir = "/mnt/vicocktail_features"
    print(f"Checking output directory: {out_dir}")
    if not os.path.exists(out_dir):
        print("Output directory does not exist.")
        return

    pattern = f"{out_dir}/**/*.pt"
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} .pt files.")
    for f in files[:5]:
        print(f" - {f} ({os.path.getsize(f)} bytes)")

