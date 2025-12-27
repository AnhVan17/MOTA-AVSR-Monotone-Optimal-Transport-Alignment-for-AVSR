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
import math

# Absolute imports - works both local and Modal
from src.data.dataset import create_dataloaders
from src.data.tokenizers.whisper import WhisperProcessor
from src.models.fusion.aurora_xt_model import create_model
from src.training.losses import create_loss
from src.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


class Trainer:
    """Optimized Trainer class for AURORA-XT"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        print("📝 Initializing WhisperProcessor...")
        self.processor = WhisperProcessor(model_name="openai/whisper-small", language="vi")
        self.tokenizer = self.processor
        
        # Update config with vocab info
        self.config['model']['vocab_size'] = self.processor.vocab_size
        self.config['model']['pad_token_id'] = self.processor.tokenizer.pad_token_id
        
        # Create model
        print("🏗️ Creating model...")
        self.model = create_model(self.config['model']).to(self.device)
        
        # Loss & Optimizer
        self.criterion = create_loss(self.config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        # Gradient Accumulation
        self.accumulation_steps = self.config['training'].get('accumulation_steps', 1)
        
        # Dataloaders
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
        
        # Scheduler with Warmup
        num_epochs = self.config['training']['num_epochs']
        num_steps = len(self.dataloaders['train']) * num_epochs // self.accumulation_steps
        warmup_steps = int(num_steps * self.config['training'].get('warmup_ratio', 0.1))
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(progress * math.pi))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # AMP
        self.use_amp = self.config['training'].get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Evaluator
        self.evaluator = Evaluator(self.tokenizer)
        
        # State
        self.epoch = 0
        self.step = 0
        self.global_step = 0
        self.best_wer = float('inf')
        
        # WandB - robust initialization
        self.use_wandb = self.config.get('use_wandb', False)
        if self.use_wandb:
            import wandb
            import os
            if os.environ.get('WANDB_API_KEY'):
                try:
                    wandb.init(
                        project=self.config.get('wandb_project', 'aurora-xt'),
                        name=self.config.get('wandb_run', 'run'),
                        config=self.config
                    )
                except Exception as e:
                    print(f"⚠️ Failed to initialize WandB: {e}")
                    self.use_wandb = False
            else:
                print("\n" + "!"*50)
                print("⚠️ WANDB_API_KEY not found in environment!")
                print("   Logging will be LOCAL ONLY.")
                print("   To fix this, create a Modal secret with 'WANDB_API_KEY'.")
                print("!"*50 + "\n")
                self.use_wandb = False
            
        # Checkpointing
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Trainer initialized. Steps per epoch: {len(self.dataloaders['train'])}")

    def train_epoch(self):
        self.model.train()
        metrics = {'loss': 0.0, 'ctc': 0.0, 'ce': 0.0}
        
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {self.epoch+1}")
        
        for i, batch in enumerate(pbar):
            audio = batch['audio'].to(self.device)
            audio_mask = batch['audio_mask'].to(self.device)
            visual = batch['visual'].to(self.device)
            target = batch['target'].to(self.device)
            target_mask = batch['target_mask'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(audio, visual, target, audio_mask=audio_mask)
                loss_dict = self.criterion(
                    ctc_logits=outputs['ctc_logits'],
                    ar_logits=outputs['ar_logits'],
                    targets=target,
                    target_mask=target_mask,
                    epoch=self.epoch,
                    max_epochs=self.config['training']['num_epochs']
                )
                loss = loss_dict['total_loss'] / self.accumulation_steps
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (i + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['gradient_clip'])
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Metrics update
            metrics['loss'] += loss.item() * self.accumulation_steps
            metrics['ctc'] += loss_dict['ctc_loss'].item()
            metrics['ce'] += loss_dict['ce_loss'].item()
            
            pbar.set_postfix({'l': f"{loss.item()*self.accumulation_steps:.3f}", 'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"})
            
            if self.use_wandb and self.global_step % 10 == 0:
                import wandb
                wandb.log({
                    'train/loss': loss.item() * self.accumulation_steps,
                    'train/ctc_loss': loss_dict['ctc_loss'].item(),
                    'train/ce_loss': loss_dict['ce_loss'].item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/ctc_weight': loss_dict.get('ctc_weight', 0),
                    'train/ce_weight': loss_dict.get('ce_weight', 0)
                }, step=self.global_step)
            
            self.step += 1
            
        n = len(self.dataloaders['train'])
        return {k: v/n for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        results = self.evaluator.evaluate(
            self.model, self.dataloaders['val'], self.device, 
            max_batches=self.config.get('max_eval_batches')
        )
        
        if self.use_wandb:
            import wandb
            log_dict = {
                'val/wer': results['wer'],
                'val/cer': results['cer'],
            }
            # Log samples as table
            if results.get('samples'):
                table = wandb.Table(columns=["Prediction", "Reference"])
                for p, r in results['samples']:
                    table.add_data(p, r)
                log_dict['val/samples'] = table
            
            wandb.log(log_dict, step=self.global_step)
            
        return results

    def save_checkpoint(self, filename: str, metrics: dict = None):
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': self.epoch,
            'step': self.step,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_wer': self.best_wer,
            'metrics': metrics,
            'config': self.config
        }, path)
        print(f"💾 Saved: {path}")

    def train(self):
        print(f"\n🚀 Starting Training on {self.device}")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            print(f"\n📊 Epoch {epoch+1}: WER={val_metrics['wer']:.2f}% loss={train_metrics['loss']:.4f}")
            
            if val_metrics['wer'] < self.best_wer:
                self.best_wer = val_metrics['wer']
                self.save_checkpoint('best_model.pt', val_metrics)
                
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', val_metrics)
        
        self.save_checkpoint('final_model.pt')
        return {'best_wer': self.best_wer, 'steps': self.global_step}