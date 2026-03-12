import modal
import os
import sys
import yaml
from pathlib import Path

# --- Config ---
APP_NAME = "avsr-train-phase2-mqot"
VOLUME_NAME = "avsr-volume"
MANIFEST_PATH = "/mnt/manifests/grid_manifest.jsonl"

# --- Image ---
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
        "numpy<2",
        "jiwer",
        "matplotlib",
        "soundfile",
        "opencv-python-headless"
    )
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("src", remote_path="/root/src")
)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

@app.function(
    image=image,
    volumes={"/mnt": volume},
    gpu="A10G",
    timeout=7200
)
def train_remote():
    sys.path.append("/root")
    import torch
    import yaml
    from src.training.trainer import Trainer
    from src.utils.logging_utils import setup_logger
    from src.utils.config_utils import load_config

    logger = setup_logger("Train:Phase2")
    logger.info("Starting Phase 2: MQOT Integration Training")
    
    if not os.path.exists(MANIFEST_PATH):
        logger.error(f"Manifest {MANIFEST_PATH} not found. Run preprocessing first.")
        return

    # Load Config
    config_path = "/root/configs/phase2_mqot.yaml"
    if not os.path.exists(config_path):
         logger.error(f"Config {config_path} not found.")
         return

    logger.info(f"Loading config from {config_path}")
    # Load with inheritance
    config = load_config(config_path)
    
    # Initialize Trainer
    # Trainer handles the complexities of creating the MQOT-enabled model
    # because config['model']['use_mqot'] is True
    trainer = Trainer(config)
    
    # Start Training
    trainer.train()

@app.local_entrypoint()
def main():
    train_remote.remote()
