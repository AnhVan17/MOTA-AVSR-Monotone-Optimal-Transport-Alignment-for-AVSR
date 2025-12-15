"""
AURORA-XT Training Module
==========================
Core training logic - shared by training.py và training_modal.py
Uses absolute imports for Modal compatibility
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import yaml
from tqdm import tqdm
import logging

# Absolute imports - works both local and Modal
from src.data.tokenizer import VietnameseCharTokenizer
from src.data.dataset import create_dataloaders
from src.models.aurora_xt import create_model
from src.training.losses import create_loss
from src.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """Core Trainer class - dùng chung cho local và Modal"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Dict config (không phải path)
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Initialize tokenizer
        print("📝 Initializing tokenizer...")
        self.tokenizer = VietnameseCharTokenizer()
        self.config['model']['vocab_size'] = self.tokenizer.vocab_size
        
        # Create model
        print("🏗️ Creating model...")
        self.model = create_model(self.config['model']).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"   Total params: {total_params:,} (~{total_params*4/1024**2:.1f}MB)")
        
        # Create loss
        self.criterion = create_loss(self.config)
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=float(self.config['training'].get('min_lr', 1e-6))
        )
        
        # Mixed precision
        self.use_amp = self.config['training'].get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Create dataloaders
        print("📊 Loading data...")
        self.dataloaders = create_dataloaders(
            train_manifest=self.config['data']['train_manifest'],
            val_manifest=self.config['data']['val_manifest'],
            batch_size=self.config['data']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            data_root=self.config['data']['data_root'],
            max_train_samples=self.config['data'].get('max_train_samples'),
            max_val_samples=self.config['data'].get('max_val_samples')
        )
        
        print(f"   Train batches: {len(self.dataloaders['train'])}")
        print(f"   Val batches: {len(self.dataloaders['val'])}")
        
        # Create evaluator
        self.evaluator = Evaluator(self.tokenizer)
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_wer = float('inf')
        
        # Checkpoint dir
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Trainer initialized")
    
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        
        # Metrics
        total_loss = 0.0
        total_ctc = 0.0
        total_ce = 0.0
        
        pbar = tqdm(
            self.dataloaders['train'],
            desc=f"Epoch {self.epoch+1}/{self.config['training']['num_epochs']}"
        )
        
        for batch in pbar:
            # Move to device
            audio = batch['audio'].to(self.device)
            visual = batch['visual'].to(self.device)
            target = batch['target'].to(self.device)
            target_mask = batch['target_mask'].to(self.device)
            
            # Forward with mixed precision
            with autocast(enabled=self.use_amp):
                outputs = self.model(audio, visual, target)
                
                loss_dict = self.criterion(
                    ctc_logits=outputs['ctc_logits'],
                    ar_logits=outputs['ar_logits'],
                    targets=target,
                    target_mask=target_mask,
                    epoch=self.epoch,
                    max_epochs=self.config['training']['num_epochs']
                )
                
                loss = loss_dict['total_loss']
            
            # Backward
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss_dict['total_loss'].item()
            total_ctc += loss_dict['ctc_loss'].item()
            total_ce += loss_dict['ce_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'ctc': f"{loss_dict['ctc_loss'].item():.4f}",
                'ce': f"{loss_dict['ce_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            self.step += 1
        
        # Scheduler step
        self.scheduler.step()
        
        # Epoch averages
        n = len(self.dataloaders['train'])
        return {
            'loss': total_loss / n,
            'ctc_loss': total_ctc / n,
            'ce_loss': total_ce / n
        }
    
    @torch.no_grad()
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        # Evaluate
        results = self.evaluator.evaluate(
            self.model,
            self.dataloaders['val'],
            self.device,
            max_batches=self.config.get('max_eval_batches', 20)
        )
        
        print(f"\n📊 Validation:")
        print(f"   WER: {results['wer']:.2f}%")
        print(f"   CER: {results['cer']:.2f}%")
        
        if results.get('samples'):
            print(f"\n📝 Samples:")
            for i, (pred, ref) in enumerate(results['samples'][:2]):
                print(f"   [{i}] Pred: {pred}")
                print(f"       Ref:  {ref}")
        
        return results
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_wer': self.best_wer,
            'config': self.config
        }, checkpoint_path)
        
        print(f"💾 Saved: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("🚀 Starting Training")
        print("="*70)
        
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            print(f"\n📊 Epoch {epoch+1}:")
            print(f"   Loss: {train_metrics['loss']:.4f}")
            print(f"   CTC: {train_metrics['ctc_loss']:.4f}")
            print(f"   CE: {train_metrics['ce_loss']:.4f}")
            
            # Validate
            val_metrics = self.validate()
            
            # Save best model
            if val_metrics['wer'] < self.best_wer:
                self.best_wer = val_metrics['wer']
                self.save_checkpoint('best_model.pt')
                print(f"   ✅ New best WER: {self.best_wer:.2f}%")
            
            # Periodic save
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        # Final save
        self.save_checkpoint('final_model.pt')
        
        print("\n" + "="*70)
        print(f"🎉 Training Complete!")
        print(f"   Best WER: {self.best_wer:.2f}%")
        print("="*70)
        
        return {'best_wer': self.best_wer, 'steps': self.step}