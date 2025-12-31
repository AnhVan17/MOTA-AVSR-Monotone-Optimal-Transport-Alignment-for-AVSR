import sys
import os
import yaml
import torch
from pathlib import Path

# Add project root
sys.path.append(os.getcwd())

from src.training.trainer import Trainer
from src.utils.logging_utils import setup_logger

def main():
    logger = setup_logger("Local:Train")
    logger.info("Starting Local Phase 1 Training (Debug)")
    
    # 1. Load Config
    config_path = "configs/phase1_base.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config {config_path} not found.")
        return
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Override Paths for Local Mac Environment
    # Assuming running from project root
    current_dir = os.getcwd()
    
    # These paths depend on where your data is locally.
    # Adjusting based on user workspace info: /Volumes/Kingston_XS1000_Media/...
    # Ideally we keep them relative or absolute based on known structure.
    # config['data']['data_root'] and 'manifest' are usually /data/... in YAML for Modal.
    # We strip the /data prefix and prepend local root?
    # Or simply explicit overrides if we know the user's path.
    
    # Using relative paths for safety if data is in ./data or ./processed_features
    # User path: /Volumes/Kingston_XS1000_Media/Audio-Visual-Speech-Recognition/processed_features/grid
    
    local_data_root = os.path.join(current_dir, "processed_features", "grid")
    local_manifest = os.path.join(current_dir, "manifests", "grid_manifest.jsonl")
    
    if os.path.exists(local_data_root):
        config['data']['data_root'] = local_data_root
        logger.info(f"Overriding data_root: {local_data_root}")
    
    if os.path.exists(local_manifest):
        config['data']['train_manifest'] = local_manifest
        config['data']['val_manifest'] = local_manifest
        logger.info(f"Overriding manifests: {local_manifest}")
        
    # 3. Reduce parameters for quick local check
    config['data']['batch_size'] = 2
    config['data']['num_workers'] = 0 # Fix encoding error on Mac multiprocessing
    config['training']['num_epochs'] = 1
    config['data']['max_samples'] = 10 # Only run 10 samples
    config['logging']['use_wandb'] = False
    
    # 4. Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # Disable Mixed Precision on CPU
        config['training']['use_amp'] = False
    
    # 5. Run Trainer
    trainer = Trainer(config)
    trainer.train()
    
    logger.info("Local Training Check Finished")

if __name__ == "__main__":
    main()
