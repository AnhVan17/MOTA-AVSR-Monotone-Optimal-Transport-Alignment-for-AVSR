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

# Modal pins the GPU on the function decorator at app-build (import) time — it can't be
# overridden per-call in this client. Default T4: face-align is CPU-bound (SFD postprocess in
# numpy + per-frame FAN loop), so measured L40S+batching gave ~0% on the 80% bottleneck and
# only sped up Whisper (~16% overall) for 3.3× cost. Override with: PROCESS_GPU=L40S modal run ...
PROCESS_GPU = os.environ.get("PROCESS_GPU", "T4")

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
    gpu=PROCESS_GPU,   # default T4 (env PROCESS_GPU=L40S để mạnh hơn)
    cpu=12,            # nhiều core cho multiprocessing (face-align CPU-bound)
    memory=49152,      # 48 GB — mp_workers × frame buffer + models
    timeout=21600,     # 6h
)
def process_data(shard_names, mp_workers: int = 4, limit_ratio: float = 1.0, max_samples: int = 0, detect_batch: int = 1):
    """Process a GROUP of raw shards IN PARALLEL inside one container.

    mp_workers subprocess (multiprocessing 'spawn') chạy đồng thời, mỗi worker xử lý 1 raw shard
    với model riêng, CHIA SẺ 1 GPU (face-align CPU-bound → GPU rảnh đủ nuôi nhiều worker → rẻ hơn
    spawn nhiều GPU container). Mỗi shard ghi output tag riêng (= stem raw shard) + resume-skip riêng.
    """
    sys.path.append("/root")
    import multiprocessing as mp

    from scripts.modal.data_prep.shard_worker import prewarm_model_cache, process_one_shard

    shard_names = list(shard_names or [])
    if not shard_names:
        return "no shards"
    print(f"Container: {len(shard_names)} shards × mp_workers={mp_workers} "
          f"(gpu={PROCESS_GPU}, detect_batch={detect_batch}): {shard_names}")

    # spawn children re-launch python fresh and read sys.path from PYTHONPATH (NOT our runtime
    # sys.path edits). Export /root so they can import scripts.* and src.* .
    os.environ["PYTHONPATH"] = "/root" + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

    print("Prewarming model cache (one CPU download → workers don't race it)...")
    prewarm_model_cache()

    tasks = [(DATA_ROOT, OUTPUT_ROOT, s, limit_ratio, max_samples, detect_batch) for s in shard_names]
    ctx = mp.get_context("spawn")  # CUDA requires 'spawn' (fork corrupts the CUDA context)
    with ctx.Pool(processes=max(1, min(mp_workers, len(tasks)))) as pool:
        results = pool.map(process_one_shard, tasks)

    volume.commit()  # workers wrote into the mounted volume; commit once from the parent
    for r in results:
        print(f"  {r}")
    n_ok = sum(r.startswith("OK") for r in results)
    n_skip = sum(r.startswith("SKIP") for r in results)
    return f"Container: {n_ok} ok, {n_skip} skip, {len(results) - n_ok - n_skip} fail / {len(results)}"

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
def main(action: str = "download", subset: str = "train", limit_ratio: float = 1.0, max_samples: int = 0,
         containers: int = 4, mp_workers: int = 4, detect_batch: int = 1, max_shards: int = 0):
    """
    Args:
        action: 'download', 'download_one', 'process' (raw→frame shards), 'inspect', 'inspect_output'
        subset: 'train' / 'avvn-test-000000' / 'test_snr_...' / 'all' (filter keyword on raw shards)
        max_samples: >0 limits samples PER shard for a cheap smoke (process action).
        containers: số container GPU chạy SONG SONG (process). ~3-4 để lịch sự với volume dùng chung.
        mp_workers: số worker multiprocessing TRONG mỗi container (chia sẻ 1 GPU). Hạ nếu OOM VRAM.
        detect_batch: gom frame qua SFD/GPU 1 lần. DEFAULT 1 — đo thực tế batching KHÔNG giúp (CPU-bound).
        max_shards: >0 chỉ xử lý N raw shard đầu (cho smoke test mp trước khi chạy full).

    GPU: mặc định T4 (xem PROCESS_GPU ở đầu file); thử L40S bằng `PROCESS_GPU=L40S modal run ...`.
    Tổng song song = containers × mp_workers. Mỗi raw shard → output tag riêng + resume-skip riêng.
    """
    if action == "download_one":
        print(f"Smoke test: downloading ONE shard '{subset}'...")
        print(download_one_shard.remote(subset))

    elif action == "download":
        print(f"Starting Download for {subset}...")
        download_shard_subset.remote(subset)

    elif action == "process":
        # Raw shards → frame shards. 2 tầng song song: `containers` GPU container chạy đồng thời,
        # mỗi container multiprocessing `mp_workers` shard cùng lúc trên 1 GPU. Resume per-shard.
        names = list_raw_shards.remote(subset)
        if not names:
            print(f"No raw shards match subset='{subset}' in {DATA_ROOT}.")
            return
        if max_shards:
            names = names[:max_shards]
        # Round-robin split → cân tải giữa các container.
        groups = [g for g in (names[i::containers] for i in range(containers)) if g]
        total_par = len(groups) * mp_workers
        print(f"Processing '{subset}': {len(names)} shards across {len(groups)} container(s) × "
              f"{mp_workers} workers = {total_par} parallel (gpu={PROCESS_GPU}, detect_batch={detect_batch}).")
        # Spawn all containers concurrently, then gather.
        handles = [process_data.spawn(g, mp_workers, limit_ratio, max_samples, detect_batch) for g in groups]
        for h in handles:
            print(f"   → {h.get()}")

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

