import modal
import os
import sys
import yaml
from pathlib import Path

# --- Config ---
APP_NAME = "avsr-train-vicocktail-phase1-ctcfix-v1"  # Force rebuild with CTC fixes
# Processed Volume containing Features & Manifests
VOLUME_PROCESSED = "avsr-vicocktail-processed" 

# --- Image Definition (Robust Numpy Fix) ---
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg", "libsndfile1")
    # 1. Force Uninstall Numpy
    .run_commands("pip uninstall -y numpy || true")
    # 2. Install numpy<2 FIRST along with Torch
    .pip_install(
        "numpy==1.26.4",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        extra_index_url="https://download.pytorch.org/whl/cu118"
    )
    # 3. Install other deps
    .pip_install(
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "soundfile==0.12.1",
        "wandb", 
        "pyyaml",
        "jiwer"  # Required for WER/CER calculation
    )
    .env({"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"})
    # Be sure to include src and configs
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
vol_processed = modal.Volume.from_name(VOLUME_PROCESSED, create_if_missing=True)

# Mount paths
MOUNT_PROCESSED = "/mnt/processed"
CONFIG_PATH = "/root/configs/vicocktail_phase1.yaml"


@app.function(
    image=image,
    volumes={MOUNT_PROCESSED: vol_processed},
    gpu="A100-40GB",         # Good balance for training
    timeout=86400,      # 24 hours
    secrets=[modal.Secret.from_name("wandb-secret")] # Ensure you have this secret for WandB
)
def train_remote():
    sys.path.append("/root")
    import torch
    from src.training.trainer import Trainer
    from src.utils.logging_utils import setup_logger
    
    logger = setup_logger("Train:ViCocktail:Phase1")
    logger.info("🚀 Starting ViCocktail Phase 1 Training (Features)...")
    
    # 1. Load Config
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"Config not found at {CONFIG_PATH}")
        return

    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override Config paths with mounted paths if needed
    # (Though we aligned the YAML with the mount points, double check is good)
    config['data']['train_manifest'] = f"{MOUNT_PROCESSED}/manifests/train.jsonl"
    config['data']['val_manifest'] = f"{MOUNT_PROCESSED}/manifests/val.jsonl"
    # Note: features might be in /mnt/processed/features/vicocktail or similar
    # Check preprocess script output for exact path. 
    # Usually preprocess_vicocktail.py outputs to /mnt/processed/features/vicocktail
    config['data']['data_root'] = f"{MOUNT_PROCESSED}/features/vicocktail"
    config['checkpoint_dir'] = f"{MOUNT_PROCESSED}/checkpoints/phase1"
    
    logger.info(f"Loaded Configuration:\n{yaml.dump(config)}")
    
    # 2. Check Data Existence
    if not os.path.exists(config['data']['train_manifest']):
        logger.error(f"❌ Train manifest not found: {config['data']['train_manifest']}")
        # Fallback to debug manifest for testing if main is missing?
        # debug_manifest = f"{MOUNT_PROCESSED}/manifests/debug/train.jsonl"
        # if os.path.exists(debug_manifest):
        #     logger.warning("⚠️ Falling back to DEBUG manifest!")
        #     config['data']['train_manifest'] = debug_manifest
        #     config['data']['val_manifest'] = debug_manifest # Use same for val
        #     config['data']['data_root'] = f"{MOUNT_PROCESSED}/features/vicocktail_debug"
        # else:
        return
            
    # 3. Initialize Trainer
    try:
        trainer = Trainer(config)
        logger.info("✅ Trainer initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Trainer: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Start Training
    try:
        trainer.train()
        logger.info("🎉 Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Commit checkpoints to volume
        logger.info("Committing volume changes...")
        vol_processed.commit()
    

@app.local_entrypoint()
def main():
    train_remote.remote()
