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
    .apt_install("git", "libsndfile1", "ffmpeg")
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
        "jiwer==3.0.3",
        "librosa==0.10.1"
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
checkpoint_volume = modal.Volume.from_name(CHECKPOINT_VOLUME, create_if_missing=True)

# Mount src and configs directory
image_final = (
    image
    .add_local_dir(Path(__file__).parent.parent / "src", remote_path="/root/src")
    .add_local_dir(Path(__file__).parent.parent / "configs", remote_path="/root/configs")
)


@app.cls(
    image=image_final,
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
    print("🚀 AURORA-XT Training (Modal - Optimized)")
    print("="*70)
    print(f"📋 Config: {config_path}")
    print(f"   Model: d_model={config['model']['d_model']}, heads={config['model']['num_heads']}")
    print(f"   Optim: LR={config['training']['learning_rate']}, Accum={config['training'].get('accumulation_steps', 1)}")
    print(f"   Regularization: Smoothing={config['training'].get('label_smoothing', 0)}, Dropout={config['model']['dropout']}")
    print(f"   Hardware: A100-40GB x 1")
    print("="*70 + "\n")
    
    trainer = ModalTrainer()
    result = trainer.train.remote(config)
    
    print("\n✅ Training complete!")
    print(f"   Best WER: {result['best_wer']:.2f}%")
    print(f"   Total Global Steps: {result['steps']}")