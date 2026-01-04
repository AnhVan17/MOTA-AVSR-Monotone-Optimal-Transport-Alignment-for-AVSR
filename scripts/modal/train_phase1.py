import modal
import os
import sys
import yaml
from pathlib import Path

# --- Config ---
APP_NAME = "avsr-train-phase1"
VOLUME_NAME = "avsr-volume"
MANIFEST_PATH = "/mnt/manifests/grid_manifest.jsonl"

# --- Image ---
# Same image dependencies as preprocess
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        "numpy<2",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "numpy<2" # Force numpy < 2
    )
    # Add configs folder
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={"/mnt": volume},
    gpu="A10G",         # A10G is good balance
    timeout=7200        # 2 hours
)
def train_remote():
    sys.path.append("/root")
    import torch
    import torch.optim as optim
    from tqdm import tqdm
    import yaml
    
    from src.training.trainer import Trainer
    from src.utils.logging_utils import setup_logger
    
    logger = setup_logger("Train:Phase1")

    logger.info("Starting Remote Phase 1 Training")
    
    if not os.path.exists(MANIFEST_PATH):
        logger.error(f"Manifest {MANIFEST_PATH} not found. Run preprocessing first.")
        return

    # Load Config from YAML
    config_path = "/root/configs/phase1_base.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded config from {config_path}")
    
    # Initialize Trainer
    trainer = Trainer(config)
    
    # Start Training
    trainer.train()
    


@app.local_entrypoint()
def main():
    train_remote.remote()