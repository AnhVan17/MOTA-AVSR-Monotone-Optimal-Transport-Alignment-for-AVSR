"""
Audio-Only ASR Model for Comparison
====================================
Uses SAME preprocessed features as AVSR
→ Fair comparison!

Architecture:
- Audio: Whisper features [T, 768] (from preprocessing)
- Encoder: Conformer (6 layers) 
- Decoder: CTC + Attention Hybrid (same as AVSR)

→ Chỉ khác AVSR ở chỗ: KHÔNG dùng visual features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


# ============================================================================
# REUSE CONFORMER & DECODER từ AURORA-XT
# ============================================================================

# Import từ aurora_xt_model.py
from src.models.aurora_xt import ConformerBlock, HybridDecoder


# ============================================================================
# AUDIO-ONLY MODEL
# ============================================================================

class AudioOnlyASR(nn.Module):
    """
    Audio-Only ASR Model
    
    Architecture:
    1. Project audio features: 768 → d_model
    2. Conformer encoder (6 layers)
    3. Hybrid CTC + Attention decoder
    
    → SAME components as AVSR, just NO visual!
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        audio_dim = config.get('audio_dim', 768)
        d_model = config.get('d_model', 256)
        num_encoder_layers = config.get('num_encoder_layers', 6)
        num_decoder_layers = config.get('num_decoder_layers', 4)
        num_heads = config.get('num_heads', 4)
        vocab_size = config.get('vocab_size', 220)
        dropout = config.get('dropout', 0.1)
        
        # 1. Audio projection
        self.audio_proj = nn.Linear(audio_dim, d_model)
        
        # 2. Conformer encoder (SAME as AVSR)
        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel=31, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 3. Hybrid decoder (SAME as AVSR)
        self.decoder = HybridDecoder(
            d_model, num_heads, num_decoder_layers,
            vocab_size, dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Audio-Only ASR initialized")
        print(f"   Parameters: {total_params:,} (~{total_params*4/1024**2:.1f}MB)")
    
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        audio: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass - Audio ONLY
        
        Args:
            audio: [B, T, 768] Whisper features (from preprocessing)
            target: [B, L] target token IDs (optional)
            
        Returns:
            dict with ctc_logits, ar_logits
        """
        # Project audio
        audio_feat = self.audio_proj(audio)  # [B, T, D]
        
        # Conformer encoding
        encoded = audio_feat
        for layer in self.encoder:
            encoded = layer(encoded)
        
        # Hybrid decoding
        decoder_out = self.decoder(encoded, target)
        
        return {
            'ctc_logits': decoder_out['ctc_logits'],
            'ar_logits': decoder_out['ar_logits']
        }


def create_audio_only_model(config: Dict) -> AudioOnlyASR:
    """Factory function for audio-only model"""
    return AudioOnlyASR(config)


# ============================================================================
# AUDIO-ONLY TRAINER (Modified from Trainer)
# ============================================================================

class AudioOnlyTrainer:
    """
    Trainer for Audio-Only model
    
    Modified from main Trainer class to:
    - Only pass audio features to model
    - Same loss, optimizer, scheduler as AVSR
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: Dict config
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed
        seed = config.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Initialize tokenizer
        print("🔤 Initializing tokenizer...")
        from src.data.tokenizer import VietnameseCharTokenizer
        self.tokenizer = VietnameseCharTokenizer()
        self.config['model']['vocab_size'] = self.tokenizer.vocab_size
        
        # Create Audio-Only model
        print("🗣️ Creating Audio-Only model...")
        self.model = create_audio_only_model(self.config['model']).to(self.device)
        
        # Create loss (SAME as AVSR)
        from src.training.losses import create_loss
        self.criterion = create_loss(self.config)
        
        # Create optimizer (SAME as AVSR)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay'])
        )
        
        # Create scheduler (SAME as AVSR)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=float(self.config['training'].get('min_lr', 1e-6))
        )
        
        # Mixed precision (SAME as AVSR)
        from torch.cuda.amp import autocast, GradScaler
        self.use_amp = self.config['training'].get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Create dataloaders (SAME dataset!)
        print("📊 Loading data...")
        from src.data.dataset import create_dataloaders
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
        
        # Create evaluator (Audio-Only specific!)
        # Use AudioOnlyEvaluator instead of generic Evaluator
        self.evaluator = AudioOnlyEvaluator(self.tokenizer)
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_wer = float('inf')
        
        # Checkpoint dir
        from pathlib import Path
        self.checkpoint_dir = Path(self.config.get('checkpoint_dir', './checkpoints_audio_only'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print("✅ Audio-Only Trainer initialized")
    
    def train_epoch(self):
        """Train one epoch"""
        from torch.cuda.amp import autocast
        from tqdm import tqdm
        
        self.model.train()
        
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
            # Visual NOT used! ✅
            target = batch['target'].to(self.device)
            target_mask = batch['target_mask'].to(self.device)
            
            # Forward (ONLY audio!)
            with autocast(enabled=self.use_amp):
                outputs = self.model(audio, target)
                
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
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'ctc': f"{loss_dict['ctc_loss'].item():.4f}",
                'ce': f"{loss_dict['ce_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            self.step += 1
        
        self.scheduler.step()
        
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
        print("🚀 Starting Audio-Only Training")
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
        print(f"🎉 Audio-Only Training Complete!")
        print(f"   Best WER: {self.best_wer:.2f}%")
        print("="*70)
        
        return {'best_wer': self.best_wer, 'steps': self.step}


# ============================================================================
# MODIFIED EVALUATOR (works with audio-only model)
# ============================================================================

class AudioOnlyEvaluator:
    """
    Evaluator modified for audio-only model
    
    Calls model with ONLY audio (no visual)
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def ctc_greedy_decode(self, logits: torch.Tensor, blank_id: int = 4):
        """CTC greedy decode"""
        pred_ids = logits.argmax(dim=-1)
        decoded = []
        for seq in pred_ids:
            unique_tokens = []
            prev = None
            for token in seq:
                token_id = token.item()
                if token_id != prev:
                    unique_tokens.append(token_id)
                    prev = token_id
            
            unique_tokens = [t for t in unique_tokens if t != blank_id]
            decoded.append(unique_tokens)
        
        return decoded
    
    def decode_predictions(self, logits: torch.Tensor):
        """Decode logits to text"""
        pred_ids = self.ctc_greedy_decode(logits, blank_id=4)
        
        texts = []
        for ids in pred_ids:
            if len(ids) > 0:
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
            else:
                text = ""
            texts.append(text.strip())
        
        return texts
    
    def decode_targets(self, targets: torch.Tensor):
        """Decode targets to text"""
        texts = []
        for seq in targets:
            text = self.tokenizer.decode(seq, skip_special_tokens=True)
            texts.append(text.strip())
        
        return texts
    
    def compute_wer(self, predictions, references):
        """Compute WER"""
        from jiwer import wer
        try:
            error_rate = wer(references, predictions)
            return error_rate * 100
        except:
            return 100.0
    
    def compute_cer(self, predictions, references):
        """Compute CER"""
        from jiwer import cer
        try:
            error_rate = cer(references, predictions)
            return error_rate * 100
        except:
            return 100.0
    
    @torch.no_grad()
    def evaluate(self, model, dataloader, device, max_batches=None):
        """
        Evaluate audio-only model
        
        Modified: Pass ONLY audio to model (no visual!)
        """
        model.eval()
        
        all_preds = []
        all_refs = []
        
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            audio = batch['audio'].to(device)
            # Visual NOT used! ✅
            target = batch['target'].to(device)
            
            # Call model with ONLY audio
            outputs = model(audio, target=None)
            
            preds = self.decode_predictions(outputs['ctc_logits'])
            refs = self.decode_targets(target)
            
            all_preds.extend(preds)
            all_refs.extend(refs)
        
        wer_score = self.compute_wer(all_preds, all_refs)
        cer_score = self.compute_cer(all_preds, all_refs)
        
        return {
            'wer': wer_score,
            'cer': cer_score,
            'samples': list(zip(all_preds[:5], all_refs[:5]))
        }


# Update AudioOnlyTrainer to use modified evaluator
AudioOnlyTrainer.evaluator = None  # Will be set in __init__