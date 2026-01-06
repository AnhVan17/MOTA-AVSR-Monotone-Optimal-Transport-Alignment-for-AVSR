"""
Local Training Script for MOTA
==============================
Wrapper for local training - uses shared Trainer class
"""

import argparse
import yaml
import logging
import sys
import os

# Add project root to path for absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train MOTA locally")
    parser.add_argument('--config', type=str, default='configs/model/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (not implemented yet)')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load config
    config = load_config(args.config)
    
    print("\n" + "="*70)
    print("🚀 MOTA Training (Local)")
    print("="*70)
    print(f"\n📋 Config: {args.config}")
    print(f"   d_model: {config['model']['d_model']}")
    print(f"   Encoder: {config['model']['num_encoder_layers']} layers")
    print(f"   Decoder: {config['model']['num_decoder_layers']} layers")
    print(f"   Batch: {config['data']['batch_size']}")
    print(f"   Epochs: {config['training']['num_epochs']}")
    
    # Create trainer and train
    trainer = Trainer(config)
    result = trainer.train()
    
    print(f"\n✅ Training complete! Best WER: {result['best_wer']:.2f}%")


if __name__ == "__main__":
    main()