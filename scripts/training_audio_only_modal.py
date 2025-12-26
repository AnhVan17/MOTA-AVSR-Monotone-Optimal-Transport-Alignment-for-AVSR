"""
Audio-Only Training Script - Modal Pipeline
============================================
Train Audio-Only ASR model trên Modal platform.

Mô hình:
- Audio: Whisper features [T, 768]
- Encoder: Conformer (6 layers)
- Decoder: CTC + Attention Hybrid

→ Giống AVSR nhưng KHÔNG dùng visual features
→ So sánh để đánh giá hiệu quả của multimodal
"""

import modal
from pathlib import Path
import yaml
import sys

# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

APP_NAME = "audio-only-training-v1"
VOLUME_NAME = "avsr-dataset-volume"
CHECKPOINT_VOLUME = "audio-only-checkpoints"
VOL_MOUNT_PATH = "/data"
CHECKPOINT_PATH = "/checkpoints"

# ============================================================================
# DOCKER IMAGE
# ============================================================================

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    # PyTorch với CUDA 11.8
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "torchvision==0.16.2",
        index_url="https://download.pytorch.org/whl/cu118"
    )
    # Dependencies
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


# ============================================================================
# MODAL TRAINER CLASS
# ============================================================================

@app.cls(
    image=image_with_src,
    volumes={
        VOL_MOUNT_PATH: data_volume,
        CHECKPOINT_PATH: checkpoint_volume
    },
    gpu="A100-40GB",
    cpu=8.0,
    memory=65536,  # 64GB RAM
    timeout=28800,  # 8 hours
    secrets=[modal.Secret.from_name("wandb-secret")]
)
class AudioOnlyModalTrainer:
    """Modal Trainer for Audio-Only ASR Model"""
    
    @modal.enter()
    def initialize(self):
        """Initialize paths"""
        import sys
        sys.path.insert(0, "/root")
        print("✅ Audio-Only Modal trainer initialized")
    
    @modal.method()
    def train(self, config: dict):
        """
        Run training using AudioOnlyTrainer
        
        Args:
            config: Configuration dict from YAML
            
        Returns:
            Training results dict
        """
        # Import AudioOnlyTrainer
        from src.models.audio_only import AudioOnlyTrainer
        
        # Override paths for Modal environment
        config['data']['train_manifest'] = f"{VOL_MOUNT_PATH}/manifests/train.jsonl"
        config['data']['val_manifest'] = f"{VOL_MOUNT_PATH}/manifests/val.jsonl"
        config['data']['data_root'] = VOL_MOUNT_PATH
        config['checkpoint_dir'] = CHECKPOINT_PATH
        
        print("\n" + "="*70)
        print("🎤 Audio-Only ASR Training on Modal")
        print("="*70)
        print(f"\n📋 Configuration:")
        print(f"   Model: Audio-Only (Whisper + Conformer)")
        print(f"   d_model: {config['model']['d_model']}")
        print(f"   Encoder layers: {config['model']['num_encoder_layers']}")
        print(f"   Decoder layers: {config['model']['num_decoder_layers']}")
        print(f"   Batch size: {config['data']['batch_size']}")
        print(f"   Epochs: {config['training']['num_epochs']}")
        print(f"   Learning rate: {config['training']['learning_rate']}")
        print("="*70 + "\n")
        
        # Create trainer and run
        trainer = AudioOnlyTrainer(config)
        result = trainer.train()
        
        # Commit checkpoints to volume
        checkpoint_volume.commit()
        
        print("\n" + "="*70)
        print("✅ Training Complete!")
        print(f"   Best WER: {result['best_wer']:.2f}%")
        print(f"   Total steps: {result['steps']}")
        print("="*70)
        
        return result


# ============================================================================
# LOCAL ENTRYPOINT
# ============================================================================

@app.local_entrypoint()
def main(config_path: str = "configs/model/audio_only_config.yaml"):
    """
    Launch Audio-Only training on Modal
    
    Usage:
        modal run scripts/training_audio_only_modal.py
        
    hoặc với custom config:
        modal run scripts/training_audio_only_modal.py --config-path path/to/config.yaml
    """
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("🚀 AUDIO-ONLY ASR TRAINING (Modal)")
    print("="*70)
    print(f"\n📋 Config file: {config_path}")
    print(f"\n🎯 Mục tiêu: So sánh Audio-Only vs AVSR")
    print(f"   → Đánh giá hiệu quả của visual features")
    print(f"\n📊 Model Settings:")
    print(f"   Audio dim: {config['model']['audio_dim']}")
    print(f"   d_model: {config['model']['d_model']}")
    print(f"   Encoder: {config['model']['num_encoder_layers']} layers (Conformer)")
    print(f"   Decoder: {config['model']['num_decoder_layers']} layers (Hybrid CTC+Attention)")
    print(f"\n🔧 Training Settings:")
    print(f"   Batch size: {config['data']['batch_size']}")
    print(f"   Epochs: {config['training']['num_epochs']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   Mixed precision: {config['training']['use_amp']}")
    print(f"\n💾 Checkpoints: {config['checkpoint_dir']}")
    print("="*70 + "\n")
    
    # Launch training on Modal
    print("🚀 Launching training on Modal...")
    trainer = AudioOnlyModalTrainer()
    result = trainer.train.remote(config)
    
    print("\n" + "="*70)
    print("🎉 Training Complete!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   Best WER: {result['best_wer']:.2f}%")
    print(f"   Total steps: {result['steps']}")
    print(f"\n💡 Next Steps:")
    print(f"   1. So sánh với AVSR WER")
    print(f"   2. Phân tích khi nào visual features giúp ích")
    print(f"   3. Kiểm tra trên noisy audio")
    print("="*70 + "\n")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@app.function(
    image=image_with_src,
    volumes={CHECKPOINT_PATH: checkpoint_volume}
)
def list_checkpoints():
    """List all saved checkpoints"""
    import os
    
    checkpoint_dir = Path(CHECKPOINT_PATH)
    if not checkpoint_dir.exists():
        print("⚠️ No checkpoint directory found")
        return []
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    print("\n📦 Saved Checkpoints:")
    print("="*70)
    for ckpt in checkpoints:
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"   {ckpt.name} ({size_mb:.1f} MB)")
    print("="*70)
    
    return [str(c) for c in checkpoints]


@app.function(
    image=image_with_src,
    volumes={CHECKPOINT_PATH: checkpoint_volume}
)
def download_checkpoint(checkpoint_name: str, output_path: str):
    """
    Download checkpoint từ Modal volume
    
    Args:
        checkpoint_name: Tên file checkpoint (e.g., 'best_model.pt')
        output_path: Đường dẫn local để save
    """
    import shutil
    
    src = Path(CHECKPOINT_PATH) / checkpoint_name
    if not src.exists():
        print(f"❌ Checkpoint not found: {checkpoint_name}")
        return False
    
    # Copy to local
    shutil.copy2(src, output_path)
    print(f"✅ Downloaded: {checkpoint_name} → {output_path}")
    return True
