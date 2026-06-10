import modal
import sys
import os
from pathlib import Path

# Face detection via face-alignment (GPU-native, no EGL issues)


# --- Config ---
APP_NAME = "avsr-prep-vicocktail"
VOLUME_NAME = "avsr-volume"
DATA_ROOT = "/mnt/vicocktail_raw"
OUTPUT_ROOT = "/mnt/vicocktail_features"

# --- Image ---
# Make `src` importable. Locally: add repo root (for app-build). In Modal containers the
# script is flattened to /root and src lives at /root/src, so parents[N] would be out of range.
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import PREPROC_IMAGE, get_volume

# Shared preprocessing image + this script also ships scripts/ for its helpers.
image = PREPROC_IMAGE.add_local_dir("scripts", remote_path="/root/scripts")

app = modal.App(APP_NAME)
volume = get_volume()

@app.function(
    image=image,
    volumes={"/mnt": volume},
    timeout=3600, # 1 hour per shard download usually enough
    secrets=[modal.Secret.from_name("huggingface-secret")] # Provides HF_TOKEN (gated/rate-limit)
)
def download_shard_subset(subset):
    sys.path.append("/root")
    from scripts.local.data.download_vicocktail import download_vicocktail

    print(f"Downloading subset: {subset}")
    # Download to raw folder
    subsets_arg = [subset] if subset != 'all' else None
    download_vicocktail(DATA_ROOT, subsets=subsets_arg)
    volume.commit()
    return f"Downloaded {subset}"

@app.function(
    image=image,
    volumes={"/mnt": volume},
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def download_one_shard(shard):
    """Download ONE .tar shard — cheap re-fetch smoke test. E.g. shard='avvn-train-000098.tar'."""
    sys.path.append("/root")
    from scripts.local.data.download_vicocktail import download_single_shard
    path = download_single_shard(DATA_ROOT, shard)
    volume.commit()
    return f"Downloaded shard to {path}"

@app.function(
    image=image,
    volumes={"/mnt": volume},
)
def list_raw_shards(subset):
    """List raw .tar shard basenames matching `subset` (or 'all'), sorted — for batching."""
    import glob
    files = glob.glob(f"{DATA_ROOT}/**/*.tar", recursive=True)
    names = sorted(
        os.path.basename(f) for f in files
        if subset == "all" or subset in os.path.basename(f)
    )
    return names


@app.function(
    image=image,
    volumes={"/mnt": volume},
    gpu="T4",          # rẻ nhất; bottleneck là face-align per-frame (launch-bound), GPU to KHÔNG giúp (đã đo)
    cpu=4,             # giải mã video (ffmpeg/opencv)
    memory=16384,      # 16 GB — headroom cho frame buffer + models
    timeout=21600,     # 6h — đủ cho 1 batch (≤5 raw shard × ~40min) + biên an toàn
)
def process_data(subset_name, shard_names=None, out_tag=None, limit_ratio: float = 1.0, max_samples: int = 0):
    """Raw .tar shards → feature WebDataset shards (.tar of audio/visual/text).

    shard_names: nếu set, chỉ xử lý đúng các raw shard này (1 batch). out_tag: nhãn output
    riêng cho batch để counter %06d KHÔNG đè batch khác. Resume: bỏ qua nếu _meta.json đã có.
    max_samples>0 limits total samples (cheap smoke).
    """
    sys.path.append("/root")
    from src.data.preprocessors.vicocktail import ViCocktailPreprocessor
    from src.data.shards import _meta_path

    tag = out_tag or subset_name
    shard_pattern = f"{OUTPUT_ROOT}/vicocktail-{tag}-%06d.tar"

    # Resume: a finished batch wrote its _meta.json → skip on rerun (idempotent).
    if os.path.exists(_meta_path(shard_pattern)):
        print(f"SKIP batch '{tag}' — already done ({_meta_path(shard_pattern)} exists).")
        return f"SKIP '{tag}' (already done)"

    print(f"Processing batch '{tag}' in {DATA_ROOT}: {shard_names or subset_name} (max_samples={max_samples or 'ALL'})...")
    processor = ViCocktailPreprocessor(data_root=DATA_ROOT, use_precropped=False)
    processor.run(
        shard_pattern=shard_pattern,
        filter_keyword=subset_name,
        shard_names=shard_names,
        limit_ratio=limit_ratio,
        max_samples=(max_samples or None),
    )
    volume.commit()
    return f"Sharded batch '{tag}' → {shard_pattern}"

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
def main(action: str = "download", subset: str = "train", limit_ratio: float = 1.0, max_samples: int = 0, batch_size: int = 5):
    """
    Args:
        action: 'download', 'download_one', 'process' (raw→feature shards), 'inspect', 'inspect_output'
        subset: 'train' / 'avvn-test-000000' / 'test_snr_...' / 'all' (filter keyword on raw shards)
        max_samples: >0 limits total samples for a cheap smoke (process action).
        batch_size: số raw shard mỗi container (process). ≤5 để không chạm timeout 6h (~40min/shard).
    """
    if action == "download_one":
        print(f"Smoke test: downloading ONE shard '{subset}'...")
        print(download_one_shard.remote(subset))

    elif action == "download":
        print(f"Starting Download for {subset}...")
        download_shard_subset.remote(subset)

    elif action == "process":
        # Raw .tar shards → feature WebDataset shards. Chia thành batch ≤batch_size shard,
        # mỗi batch = 1 container (commit volume riêng + resume-skip). Gọi TUẦN TỰ (không song song).
        names = list_raw_shards.remote(subset)
        if not names:
            print(f"No raw shards match subset='{subset}' in {DATA_ROOT}.")
            return
        batches = [names[i:i + batch_size] for i in range(0, len(names), batch_size)]
        print(f"Processing '{subset}': {len(names)} raw shards → {len(batches)} batch(es) of ≤{batch_size}.")
        for bi, batch in enumerate(batches):
            tag = f"{subset}-b{bi:03d}"
            print(f"\n[{bi + 1}/{len(batches)}] batch '{tag}': {batch}")
            result = process_data.remote(subset, batch, tag, limit_ratio, max_samples)
            print(f"   → {result}")

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

