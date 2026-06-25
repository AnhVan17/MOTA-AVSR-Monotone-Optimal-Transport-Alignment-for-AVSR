import modal
import os
import sys
from pathlib import Path

# Make `src` importable. Locally: add repo root (for app-build). In Modal containers the
# script is flattened to /root and src lives at /root/src, so parents[N] would be out of range.
if modal.is_local():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
else:
    sys.path.insert(0, "/root")
from src.infra.modal_image import ML_TRAIN_IMAGE, get_volume

# --- Config ---
APP_NAME = "avsr-train-phase1"
MANIFEST_PATH = "/mnt/vicocktail_features/avvn-test_snr_0_interferer_1-000000_manifest.jsonl"

app = modal.App(APP_NAME)
volume = get_volume()


@app.function(
    image=ML_TRAIN_IMAGE,
    volumes={"/mnt": volume},
    gpu=["L40S", "A100", "A10G"],  # prefer faster/larger GPUs, fall back to the old A10G pool
    cpu=8.0,            # dedicated cores for num_workers=4 frame decode
    memory=65536,       # 64GB headroom for raw-frame batches + DataLoader prefetch queues
    timeout=86400,      # 24h (Modal max) — fits ~14 epochs/container → far fewer manual relaunches
    # Secret 'wandb' gắn VÔ ĐIỀU KIỆN → deterministic (local==remote, tránh lệch dependency gây
    # crash-loop). Secret PHẢI tồn tại trong env (modal secret create wandb WANDB_API_KEY=...).
    # use_wandb trong config điều khiển việc thực sự dùng wandb hay không.
    secrets=[modal.Secret.from_name("wandb")],
)
def train_remote(manifest_path: str = None, config_path: str = None, max_epochs_per_run: int = None):
    sys.path.append("/root")
    from src.training.run import run_training
    from src.utils.logging_utils import setup_logger
    from src.utils.config_utils import load_config
    
    logger = setup_logger("Train:Phase1")

    logger.info("Starting Remote Phase 1 Training")
    
    # Defaults
    final_manifest = manifest_path if manifest_path else MANIFEST_PATH
    final_config_path = config_path if config_path else "/root/configs/phase1.yaml"

    # Load config first so we can detect the data format.
    config = load_config(final_config_path)

    # Cap số epoch cho LẦN CHẠY này (treo --detach từng chặng → tự dừng sạch; relaunch resume tiếp).
    if max_epochs_per_run is not None:
        config.setdefault("training", {})["max_epochs_per_run"] = int(max_epochs_per_run)
        logger.info(f"max_epochs_per_run={max_epochs_per_run} (dừng sau {max_epochs_per_run} epoch lần này)")

    # WebDataset (frame shards): no jsonl manifest — data.train_shards/val_shards drive the loader.
    if str(config.get("data", {}).get("format", "")).lower() == "webdataset":
        logger.info(f"Loaded config from {final_config_path} (format=webdataset)")
        run_training(config)
        return

    if not os.path.exists(final_manifest):
        logger.error(f"Manifest {final_manifest} not found. Run preprocessing first.")
        return
    
    # Override manifest in config if provided override differs
    if manifest_path:
        config['data']['train_manifest'] = manifest_path
        config['data']['val_manifest'] = manifest_path # Use same for test run
        logger.info(f"Overridden manifest to: {manifest_path}")
        
        # Heuristic: If overriding manifest for a subset run, the data files are likely 
        # in a subdirectory matching the subset name (which is the stem of the manifest).
        # OR they are in the same directory.
        # Check prep_features_gpu.py: 
        #   output_dir = OUTPUT_ROOT/subset_name
        #   manifest = OUTPUT_ROOT/subset_name_manifest.jsonl
        #   rel_path in manifest = simple filename (e.g. video.pt)
        # So effective data_root must be OUTPUT_ROOT/subset_name.
        
        manifest_p = Path(manifest_path)
        # E.g. /mnt/.../avvn-test_snr_0_interferer_1-000000_manifest.jsonl
        # Stem: avvn-test_snr_0_interferer_1-000000_manifest
        # We want: avvn-test_snr_0_interferer_1-000000
        
        if "_manifest" in manifest_p.stem:
             subset_name = manifest_p.stem.replace("_manifest", "")
             # Check if a directory with this name exists in the same parent dir
             possible_data_root = manifest_p.parent / subset_name
             if os.path.exists(possible_data_root):
                 config['data']['data_root'] = str(possible_data_root)
                 logger.info(f"Auto-detected data_root for subset: {possible_data_root}")
             else:
                 logger.warning(f"Could not auto-detect data root for {subset_name} at {possible_data_root}")
        
    logger.info(f"Loaded config from {final_config_path}")

    # Pure training logic (device-agnostic) — shared with scripts/local.
    run_training(config)
    

@app.local_entrypoint()
def main(manifest_path: str = None, config_path: str = None, max_epochs_per_run: int = None,
         spawn: bool = False):
    # spawn=True → .spawn() (fire-and-forget): KHÔNG bị Modal huỷ khi client local ngắt → dùng cho
    #   BACKGROUND. PHẢI kèm `modal run --detach ... --spawn` (không --detach thì app ephemeral chết
    #   khi entrypoint return → hàm spawn bị giết). .remote() (mặc định) là BLOCKING + stream log,
    #   nhưng bị huỷ khi client ngắt kể cả --detach → chỉ hợp foreground/smoke.
    kwargs = dict(manifest_path=manifest_path, config_path=config_path,
                  max_epochs_per_run=max_epochs_per_run)
    if spawn:
        call = train_remote.spawn(**kwargs)
        print(f"Spawned train_remote (detach-safe). FunctionCall id: {call.object_id}")
        print("Theo dõi: `modal app logs <app-id>` hoặc WandB run theo config (xem `modal app list`).")
    else:
        train_remote.remote(**kwargs)
