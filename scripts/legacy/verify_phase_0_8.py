import torch
import sys
import os
import shutil
from pathlib import Path

# Add project root
sys.path.append(os.getcwd())

from src.models.mota import create_model
from src.training.trainer import Trainer
from src.evaluation.decoding import CTCDecoder

def verify_phase_0_8():
    print("="*50)
    print("VERIFICATION: Phase 0.8 Critical Patch")
    print("="*50)
    
    # 1. Architecture Checks
    print("\n[1] Checking Architecture Fixes...")
    config = {
        'audio_dim': 768, 
        'visual_dim': 512, 
        'd_model': 16, 
        'use_mqot': True, # Enable MQOT to check init
        'num_heads': 2,
        'vocab_size': 100 # Smal vocab
    }
    
    model = create_model(config)
    
    # Check Gate Init
    gate_val = model.fine_align_gate.item()
    print(f"   Gate Value: {gate_val}")
    if abs(gate_val - 0.01) < 1e-4:
        print("✅ Gate Init Fixed (0.01)")
    else:
        print(f"❌ Gate Init Incorrect ({gate_val} != 0.01)")
        sys.exit(1)
        
    # 2. Config & Data Isolation Check (Read config file)
    print("\n[2] Checking Config Isolation...")
    with open("configs/base.yaml", 'r') as f:
        content = f.read()
        if 'train_manifest.jsonl' in content and 'val_manifest.jsonl' in content:
            print("✅ Config points to Split Manifests")
        else:
            print("❌ Config still uses 'grid_manifest.jsonl' (Duplicate)")
            sys.exit(1)
            
    # 3. Real WER Validation Check
    print("\n[3] Checking Real WER Integration...")
    
    # Mock Config for Trainer
    trainer_config = {
        'model': config,
        'loss': {'ctc_weight': 0.5, 'ce_weight': 0.5, 'quality_loss_weight': 0.1},
        'data': {
            'train_manifest': 'dummy_manifest.jsonl',
            'val_manifest': 'dummy_manifest.jsonl',
            'data_root': './',
            'batch_size': 2,
            'num_workers': 0,
            'use_precomputed_features': True # We'll mock loader anyway
        },
        'training': {
            'learning_rate': 1e-4, 
            'num_epochs': 1,
            'pretrained_path': None
        },
        'logging': {
            'checkpoint_dir': 'dummy_checkpoints',
            'use_wandb': False
        }
    }
    
    # Mock Dataset & Loader
    class MockDataset:
        def __init__(self):
            # Tokenizer needs to support decode
            class MockTokenizer:
                def decode(self, ids, **kwargs): return "hello"
                def encode(self, text): return [1, 2, 3]
            self.tokenizer = MockTokenizer()
            
    class MockLoader:
        dataset = MockDataset()
        def __iter__(self):
            # Yield one batch
            yield {
                'audio': torch.randn(2, 50, 768),
                'visual': torch.randn(2, 25, 512),
                'tokens': torch.randint(0, 10, (2, 10))
            }
        def __len__(self): return 1

    # Instantiate Trainer (Mocking build_dataloader)
    # We need to temporarily patch build_dataloader
    import src.training.trainer
    src.training.trainer.build_dataloader = lambda cfg, split: MockLoader()
    
    trainer = Trainer(trainer_config)
    
    # Run Validate Epoch
    metrics = trainer.validate_epoch(epoch=0)
    print(f"   Validation Metrics: {metrics}")
    
    if 'wer' in metrics and metrics['wer'] >= 0:
        print(f"✅ Real WER Calculated: {metrics['wer']}")
    else:
        print("❌ Real WER missing or invalid")
        sys.exit(1)
        
    # Cleanup
    if os.path.exists("dummy_checkpoints"):
        shutil.rmtree("dummy_checkpoints")
        
    print("\n✅ PHASE 0.8 VERIFIED SUCCESS")

if __name__ == "__main__":
    verify_phase_0_8()
