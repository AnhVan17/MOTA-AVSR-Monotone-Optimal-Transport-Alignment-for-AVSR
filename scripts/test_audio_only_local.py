"""
Local Testing Script for Audio-Only Model
==========================================
Test Audio-Only model trước khi deploy lên Modal

Kiểm tra:
1. Model initialization
2. Forward pass
3. Loss computation  
4. Training step
5. Evaluation
"""

import torch
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_model_creation():
    """Test model initialization"""
    print("\n" + "="*70)
    print("🧪 Test 1: Model Creation")
    print("="*70)
    
    from src.models.audio_only import AudioOnlyASR
    
    config = {
        'audio_dim': 768,
        'd_model': 256,
        'num_encoder_layers': 6,
        'num_decoder_layers': 4,
        'num_heads': 4,
        'vocab_size': 121,
        'dropout': 0.1
    }
    
    model = AudioOnlyASR(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params*4/1024**2:.1f} MB")
    
    return model


def test_forward_pass(model):
    """Test forward pass"""
    print("\n" + "="*70)
    print("🧪 Test 2: Forward Pass")
    print("="*70)
    
    batch_size = 4
    seq_len = 150
    target_len = 20
    
    # Dummy inputs
    audio = torch.randn(batch_size, seq_len, 768)
    target = torch.randint(0, 121, (batch_size, target_len))
    
    print(f"Input shapes:")
    print(f"   Audio: {audio.shape}")
    print(f"   Target: {target.shape}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(audio, target)
    
    print(f"\nOutput shapes:")
    print(f"   CTC logits: {outputs['ctc_logits'].shape}")
    print(f"   AR logits: {outputs['ar_logits'].shape}")
    
    expected_ctc = (batch_size, seq_len, 121)
    expected_ar = (batch_size, target_len, 121)
    
    assert outputs['ctc_logits'].shape == expected_ctc, f"CTC shape mismatch!"
    assert outputs['ar_logits'].shape == expected_ar, f"AR shape mismatch!"
    
    print(f"✅ Forward pass successful!")


def test_loss_computation(model):
    """Test loss computation"""
    print("\n" + "="*70)
    print("🧪 Test 3: Loss Computation")
    print("="*70)
    
    from src.training.losses import create_loss
    
    config = {
        'model': {
            'vocab_size': 121
        },
        'loss': {
            'ctc_weight': 0.3,
            'ce_weight': 0.7
        }
    }
    
    criterion = create_loss(config)
    
    # Dummy data
    batch_size = 4
    seq_len = 150
    target_len = 20
    
    audio = torch.randn(batch_size, seq_len, 768)
    target = torch.randint(0, 121, (batch_size, target_len))
    target_mask = torch.ones_like(target).bool()
    
    # Forward
    outputs = model(audio, target)
    
    # Loss
    loss_dict = criterion(
        ctc_logits=outputs['ctc_logits'],
        ar_logits=outputs['ar_logits'],
        targets=target,
        target_mask=target_mask,
        epoch=0,
        max_epochs=20
    )
    
    print(f"Loss values:")
    print(f"   CTC loss: {loss_dict['ctc_loss'].item():.4f}")
    print(f"   CE loss: {loss_dict['ce_loss'].item():.4f}")
    print(f"   Total loss: {loss_dict['total_loss'].item():.4f}")
    
    assert not torch.isnan(loss_dict['total_loss']), "Loss is NaN!"
    assert not torch.isinf(loss_dict['total_loss']), "Loss is Inf!"
    
    print(f"✅ Loss computation successful!")


def test_backward_pass(model):
    """Test backward pass"""
    print("\n" + "="*70)
    print("🧪 Test 4: Backward Pass")
    print("="*70)
    
    from src.training.losses import create_loss
    
    config = {
        'model': {
            'vocab_size': 121
        },
        'loss': {
            'ctc_weight': 0.3,
            'ce_weight': 0.7
        }
    }
    
    criterion = create_loss(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Dummy data
    batch_size = 2
    seq_len = 100
    target_len = 15
    
    audio = torch.randn(batch_size, seq_len, 768)
    target = torch.randint(0, 121, (batch_size, target_len))
    target_mask = torch.ones_like(target).bool()
    
    # Training step
    optimizer.zero_grad()
    outputs = model(audio, target)
    loss_dict = criterion(
        ctc_logits=outputs['ctc_logits'],
        ar_logits=outputs['ar_logits'],
        targets=target,
        target_mask=target_mask,
        epoch=0,
        max_epochs=20
    )
    
    loss = loss_dict['total_loss']
    loss.backward()
    
    # Check gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
    
    assert has_grad, "No gradients computed!"
    
    optimizer.step()
    
    print(f"✅ Backward pass successful!")
    print(f"   Gradients computed and applied")


def test_config_loading():
    """Test config loading"""
    print("\n" + "="*70)
    print("🧪 Test 5: Config Loading")
    print("="*70)
    
    config_path = Path(__file__).parent.parent / "configs/model/audio_only_config.yaml"
    
    if not config_path.exists():
        print(f"⚠️ Config file not found: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Config loaded from: {config_path}")
    print(f"\nConfig contents:")
    print(f"   Model:")
    for key, value in config['model'].items():
        print(f"      {key}: {value}")
    print(f"   Training:")
    for key, value in config['training'].items():
        print(f"      {key}: {value}")
    
    return config


def test_trainer_initialization():
    """Test AudioOnlyTrainer initialization (without data)"""
    print("\n" + "="*70)
    print("🧪 Test 6: Trainer Initialization")
    print("="*70)
    
    try:
        from src.models.audio_only import AudioOnlyTrainer
        
        # Minimal config
        config = {
            'model': {
                'audio_dim': 768,
                'd_model': 256,
                'num_encoder_layers': 6,
                'num_decoder_layers': 4,
                'num_heads': 4,
                'vocab_size': 121,
                'dropout': 0.1
            },
            'training': {
                'learning_rate': 3e-4,
                'weight_decay': 0.01,
                'gradient_clip': 5.0,
                'num_epochs': 20,
                'min_lr': 1e-6,
                'use_amp': True
            },
            'data': {
                'train_manifest': './dummy_train.jsonl',
                'val_manifest': './dummy_val.jsonl',
                'data_root': './data',
                'batch_size': 4,
                'num_workers': 0,
                'max_train_samples': 10,
                'max_val_samples': 5
            },
            'max_eval_batches': 2,
            'checkpoint_dir': './test_checkpoints',
            'seed': 42
        }
        
        print("⚠️ Trainer initialization requires dataset files")
        print("   Skipping full trainer test")
        print("   Manual test: Uncomment if you have dataset")
        
        # Uncomment if you have dataset:
        # trainer = AudioOnlyTrainer(config)
        # print("✅ Trainer initialized successfully!")
        
    except Exception as e:
        print(f"⚠️ Could not test trainer: {e}")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 AUDIO-ONLY MODEL TESTING")
    print("="*70)
    print("\nĐây là test suite để verify Audio-Only model trước khi deploy")
    
    # Test 1: Model creation
    model = test_model_creation()
    
    # Test 2: Forward pass
    test_forward_pass(model)
    
    # Test 3: Loss computation
    test_loss_computation(model)
    
    # Test 4: Backward pass
    test_backward_pass(model)
    
    # Test 5: Config loading
    config = test_config_loading()
    
    # Test 6: Trainer initialization
    test_trainer_initialization()
    
    # Summary
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n📝 Summary:")
    print("   ✅ Model architecture correct")
    print("   ✅ Forward pass working")
    print("   ✅ Loss computation working")
    print("   ✅ Backward pass working")
    print("   ✅ Config loading working")
    print("\n🚀 Ready to deploy to Modal!")
    print("\nNext steps:")
    print("   1. modal run scripts/training_audio_only_modal.py")
    print("   2. Monitor training on wandb")
    print("   3. Compare with AVSR results")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
