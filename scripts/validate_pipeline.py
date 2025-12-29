"""
Validate Pipeline - Quick test of all modules
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all imports"""
    print("Testing imports...")
    
    from src.utils import load_config, set_seed, save_checkpoint
    print("  ✓ utils.py")
    
    from data.tokenizer_old import VietnameseCharTokenizer
    print("  ✓ tokenizer.py")
    
    from data.dataset_old import create_dataloaders, AuroraDataset
    print("  ✓ dataset.py")
    
    from src.training.losses import create_loss, HybridLoss
    print("  ✓ losses.py")
    
    from src.models.aurora_xt import create_model, AuroraXT
    print("  ✓ aurora_xt.py")
    
    from src.evaluation.evaluator import Evaluator
    print("  ✓ evaluator.py")
    
    from src.training.trainer import Trainer
    print("  ✓ trainer.py")
    
    print("\n✅ All imports OK!")


def test_model():
    """Test model forward pass"""
    import torch
    from src.models.aurora_xt import create_model
    
    print("\nTesting model forward pass...")
    
    config = {
        'audio_dim': 768,
        'visual_dim': 512,
        'd_model': 256,
        'num_encoder_layers': 6,
        'num_decoder_layers': 4,
        'num_heads': 4,
        'vocab_size': 220,
        'dropout': 0.1
    }
    
    model = create_model(config)
    print(f"  Model created: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Test forward (15s: 450 audio frames, 375 visual frames)
    audio = torch.randn(2, 450, 768)
    visual = torch.randn(2, 375, 512)
    
    with torch.no_grad():
        outputs = model(audio, visual)
    
    print(f"  CTC logits: {outputs['ctc_logits'].shape}")
    print(f"  Gate weights: {outputs['gate_weights'].shape}")
    
    print("\n✅ Model forward OK!")


def test_loss():
    """Test loss function"""
    import torch
    from src.training.losses import create_loss
    
    print("\nTesting loss function...")
    
    config = {
        'model': {'vocab_size': 220},
        'loss': {'ctc_weight': 0.3, 'ce_weight': 0.7}
    }
    
    criterion = create_loss(config)
    
    # Fake inputs (15s CTC output ~450 frames)
    ctc_logits = torch.randn(2, 450, 220)
    ar_logits = torch.randn(2, 80, 220)
    targets = torch.randint(0, 220, (2, 80))
    target_mask = torch.ones(2, 80, dtype=torch.bool)
    
    loss_dict = criterion(
        ctc_logits=ctc_logits,
        ar_logits=ar_logits,
        targets=targets,
        target_mask=target_mask,
        epoch=0,
        max_epochs=20
    )
    
    print(f"  Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  CTC loss: {loss_dict['ctc_loss'].item():.4f}")
    print(f"  CE loss: {loss_dict['ce_loss'].item():.4f}")
    
    print("\n✅ Loss function OK!")


def test_tokenizer():
    """Test tokenizer"""
    from data.tokenizer_old import VietnameseCharTokenizer
    
    print("\nTesting tokenizer...")
    
    tokenizer = VietnameseCharTokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    text = "xin chào việt nam"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    
    print(f"  Input: '{text}'")
    print(f"  IDs: {ids}")
    print(f"  Decoded: '{decoded}'")
    
    print("\n✅ Tokenizer OK!")


def test_dataset_path():
    """Test dataset path resolution"""
    from data.dataset_old import AuroraDataset
    from pathlib import Path
    
    print("\nTesting dataset path resolution...")
    
    dataset = AuroraDataset(
        ["data/processed_features/test.pt", "processed_features/test2.pt"],
        data_root="./data"
    )
    
    # Test path resolution
    path1 = dataset._resolve_path("data/processed_features/test.pt")
    path2 = dataset._resolve_path("processed_features/test2.pt")
    
    print(f"  'data/processed_features/test.pt' -> {path1}")
    print(f"  'processed_features/test2.pt' -> {path2}")
    
    # Check no double "data"
    assert "data\\data" not in str(path1).lower() and "data/data" not in str(path1).lower(), "Double data/ detected!"
    
    print("\n✅ Path resolution OK (no double data/)!")


if __name__ == "__main__":
    print("="*60)
    print("AURORA-XT Pipeline Validation")
    print("="*60 + "\n")
    
    test_imports()
    test_tokenizer()
    test_model()
    test_loss()
    test_dataset_path()
    
    print("\n" + "="*60)
    print("🎉 ALL TESTS PASSED!")
    print("="*60)
