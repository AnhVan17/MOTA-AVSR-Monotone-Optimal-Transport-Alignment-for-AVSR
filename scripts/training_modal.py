"""
Training Script for AURORA-XT - Modal Pipeline
================================================
Modal wrapper - uses shared Trainer class
"""

import modal
from pathlib import Path
import yaml
import sys

# MODAL CONFIG
APP_NAME = "avsr-training-v3"
VOLUME_NAME = "avsr-dataset-volume"
CHECKPOINT_VOLUME = "avsr-checkpoints"
VOL_MOUNT_PATH = "/data"
CHECKPOINT_PATH = "/checkpoints"

# DOCKER IMAGE
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install(
        "transformers==4.36.2",
        "tqdm==4.66.1",
        "numpy<2",
        "pyyaml==6.0.1",
        "wandb==0.16.2",
        "jiwer==3.0.3"
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)

# Mount src directory
image_with_src = image.add_local_dir(
    local_path=Path(__file__).parent.parent / "src",
    remote_path="/root/src"
)


@app.cls(
    image=image_with_src,
    volumes={
        VOL_MOUNT_PATH: data_volume,
        CHECKPOINT_PATH: checkpoint_volume
    },
    gpu="A100-40GB",
    cpu=8.0,
    memory=65536,
    timeout=28800,  # 8 hours
    secrets=[modal.Secret.from_name("wandb-secret")]
)
class ModalTrainer:
    """Modal Trainer - uses shared Trainer class"""
    
    @modal.enter()
    def initialize(self):
        """Add src to path"""
        import sys
        sys.path.insert(0, "/root")
        print("✅ Modal trainer initialized")
    
    @modal.method()
    def train(self, config: dict):
        """Run training using shared Trainer"""
        # Import inside method to use correct path
        from src.training.trainer import Trainer
        
        # Override paths for Modal
        config['data']['train_manifest'] = f"{VOL_MOUNT_PATH}/manifests/train.jsonl"
        config['data']['val_manifest'] = f"{VOL_MOUNT_PATH}/manifests/val.jsonl"
        config['data']['data_root'] = VOL_MOUNT_PATH
        config['checkpoint_dir'] = CHECKPOINT_PATH
        
        # Create trainer and run
        trainer = Trainer(config)
        result = trainer.train()
        
        # Commit checkpoints
        checkpoint_volume.commit()
        
        return result


@app.local_entrypoint()
def main(config_path: str = "configs/model/config.yaml"):
    """Launch training"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("🚀 AURORA-XT Training (Modal)")
    print("="*70)
    print(f"\n📋 Config: {config_path}")
    print(f"   d_model: {config['model']['d_model']}")
    print(f"   Encoder: {config['model']['num_encoder_layers']} layers")
    print(f"   Decoder: {config['model']['num_decoder_layers']} layers")
    print(f"   Batch: {config['data']['batch_size']}")
    print(f"   Epochs: {config['training']['num_epochs']}")
    
    trainer = ModalTrainer()
    result = trainer.train.remote(config)
    
    print("\n✅ Training complete!")
    print(f"   Best WER: {result['best_wer']:.2f}%")
    print(f"   Steps: {result['steps']}")