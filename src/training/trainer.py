import torch
import torch.optim as optim
from typing import Dict
from pathlib import Path
from tqdm import tqdm

# Project Modules
from src.models.mota import create_model
from src.data.loader import build_dataloader
from src.data.tokenizers.whisper import WhisperTokenizer
from src.training.losses import create_loss
from src.evaluation.metrics import MetricCalculator
from src.evaluation.decoding import CTCDecoder
from src.utils.logging_utils import setup_logger
from src.utils.common import (
    AverageMeter,
    save_checkpoint,
    load_checkpoint,
    get_lr,
    EarlyStopping
)

# Initialize Logger
logger = setup_logger(__name__)

_wandb_logger = None  # Lazy init


def _get_wandb_logger(config: Dict):
    """Lazy WandB init — only when use_wandb: true in config."""
    global _wandb_logger
    if _wandb_logger is not None:
        return _wandb_logger

    if not config.get('logging', {}).get('use_wandb', False):
        return None

    try:
        from src.utils.wandb_logger import WandbLogger
        _wandb_logger = WandbLogger(
            project=config['logging'].get('wandb_project', 'mota-avsr'),
            name=config['logging'].get('wandb_name', None),
            config=config,
        )
        return _wandb_logger
    except ImportError:
        logger.warning("wandb not installed. Install with: pip install wandb")
        return None

class Trainer:
    """
    Unified Trainer for AURORA-XT (Phase 1 & 2)
    
    Features:
    - Metric-based Curriculum Learning (Adaptive Loss Weights)
    - Adaptive Learning Rate (ReduceLROnPlateau)
    - Robust Checkpointing (Flexible State Dict Loading)
    - Defensive Programming (NaN/Inf Checks)
    - E2E Backbone Support (Optional)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup Checkpoint Directory
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing Trainer on {self.device}")
        
        # 1. Tokenizer (required for DataLoader)
        logger.info("Initializing Tokenizer...")
        self.tokenizer = WhisperTokenizer(model="openai/whisper-small", language="vi")

        # 2. Data Loaders
        logger.info("Building DataLoaders...")
        self.train_loader = build_dataloader(config, tokenizer=self.tokenizer, mode='train')
        self.val_loader = build_dataloader(config, tokenizer=self.tokenizer, mode='val')
        
        # 3. Model Initialization
        logger.info("Creating Model...")
        self.model = create_model(config['model']).to(self.device)
        logger.info(f"Model Params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 3. Optimization Setup
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training'].get('weight_decay', 0.01))
        )
        
        # Warmup + Adaptive LR via ChainedScheduler
        # LinearLR ramps up LR during warmup_steps; ReduceLROnPlateau kicks in after
        warmup_steps = config['training'].get('warmup_steps', 1000)
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-4,      # start at 0.01% of base LR
            end_factor=1.0,
            total_iters=warmup_steps
        )
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=float(config['training'].get('min_lr', 1e-6))
        )
        self.scheduler = optim.lr_scheduler.ChainedScheduler(
            [warmup_scheduler, plateau_scheduler]
        )
        
        # 4. Loss Function & Metrics
        self.criterion = create_loss(config).to(self.device)
        self.early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 10), 
            mode='min'
        )
        
        # Mixed Precision
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Training State
        self.start_epoch = 0
        self.step = 0
        self.best_metric = float('inf')

        # Load Pretrained / Resume
        if config['training'].get('pretrained_path'):
            self._load_checkpoint(config['training']['pretrained_path'])

        # 5. Validation Tools
        self.metric_calc = MetricCalculator()
        self.tokenizer = self.train_loader.dataset.tokenizer
        blank_id = config['model'].get('blank_id', 50257)
        self.decoder = CTCDecoder(self.tokenizer, blank_id=blank_id)

        # 6. WandB (lazy — only if use_wandb: true in config)
        self.wandb = _get_wandb_logger(config)
        if self.wandb:
            logger.info(f"WandB enabled: {self.wandb.run.url}")


    def _load_checkpoint(self, path: str):
        """
        Robust checkpoint loading.
        Handles:
        1. Dimension mismatches (Phase 1 -> Phase 2)
        2. Missing keys (Feature -> E2E variables)
        """
        logger.info(f"Loading checkpoint from {path}...")
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load state dict with strict=False to allow architecture changes
            missing, unexpected = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            
            if missing:
                logger.warning(f"Missing keys: {len(missing)} (Normal for P1->P2 or Feature->E2E transtion)")
            if unexpected:
                logger.warning(f"Unexpected keys: {len(unexpected)}")
                
            logger.info("Checkpoint loaded successfully (flexible mode).")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            # Decision: Don't crash, just start fresh if load fails? 
            # Better to Crash explicitely if path was provided but invalid.
            raise e

    def train(self):
        """Main Training Loop"""
        num_epochs = self.config['training']['num_epochs']
        logger.info(f"Starting Training for {num_epochs} epochs")
        
        for epoch in range(self.start_epoch, num_epochs):
            # 1. Train One Epoch
            train_metrics = self.train_epoch(epoch)
            
            # 2. Validate
            val_metrics = self.validate_epoch(epoch)
            
            # 3. Update Learning Rate (Adaptive)
            # Use Validation WER as the primary metric for scheduler
            current_metric = val_metrics.get('wer', val_metrics['loss'])
            self.scheduler.step(current_metric)
            
            # 4. Save Checkpoint
            is_best = current_metric < self.best_metric
            if is_best:
                self.best_metric = current_metric
                
            # Save latest
            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                epoch, self.step, self.best_metric,
                str(self.checkpoint_dir),
                filename=f"epoch_{epoch}.pt"
            )
            
            # Save best
            if is_best:
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, self.step, self.best_metric,
                    str(self.checkpoint_dir),
                    filename="best_model.pt"
                )
            
            # 5. Logging
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val WER: {val_metrics.get('wer', 0):.2f}% | "
                f"LR: {get_lr(self.optimizer):.2e}"
            )
            
            # 6. Early Stopping
            if self.early_stopping(current_metric, epoch):
                logger.info("Early stopping triggered. Training finished.")
                break

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meter = AverageMeter()
        
        
        # Training Configs
        accum_steps = self.config['training'].get('accum_steps', 1)
        assert accum_steps > 0, f"accum_steps must be > 0, got {accum_steps}"
        
        # 0.9.5 Fix: Ensure gradients are zeroed before loop starts
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}")
        grad_norm = 0.0  # Always defined for consistent tqdm postfix
        for batch_idx, batch in enumerate(pbar):
            self.step += 1

            # Move data to device
            # Ensure collate_fn produces this structure
            audio = batch['audio'].to(self.device)
            visual = batch['visual'].to(self.device) 
            targets = batch['target'].to(self.device)
            target_mask = batch.get('target_mask', None)
            if target_mask is not None:
                target_mask = target_mask.to(self.device)
            
            # Zero Gradients handled at step boundary now (Moved to after step in 0.9.5)
            # if (batch_idx % accum_steps) == 0:
            #      self.optimizer.zero_grad()
            
            # Forward & Loss (with Mixed Precision)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # E2E Support: forward() handles raw/features internally
                outputs = self.model(audio, visual, targets)
                
                # Compute Loss
                # Pass epoch for Curriculum Learning (if used inside Loss)
                loss_dict = self.criterion(
                    ctc_logits=outputs['ctc_logits'],
                    ar_logits=outputs['ar_logits'],
                    targets=targets,
                    target_mask=target_mask, # Required for CTC
                    epoch=epoch,
                    max_epochs=self.config['training']['num_epochs']
                )
                loss = loss_dict['total_loss']
                
                # Normalize loss for accumulation (Fix 0.9.2)
                loss = loss / accum_steps
            
            # Strict Defensive Check: NaN/Inf
            if not torch.isfinite(loss):
                logger.critical(f"Loss Diverged (NaN/Inf) at step {self.step}: {loss.item()}")
                # Dump batch for debugging
                dump_path = self.checkpoint_dir / "nan_batch_dump.pt"
                torch.save(batch, dump_path)
                logger.critical(f"Failing batch dumped to {dump_path}")
                raise ValueError(f"Loss Diverged at step {self.step}")
            
            # Backward Pass
            self.scaler.scale(loss).backward()
            
            # Optimizer Step (Accumulated)
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # Gradient Clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training'].get('gradient_clip', 5.0)
                )
                
                # Optimizer Step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Zero Gradients (0.9.5 Fix: Clear after update)
                self.optimizer.zero_grad()
            
            # Update Metrics (Scale back up for logging)
            loss_val = loss.item() * accum_steps
            meter.update(loss_val)
            
            # Log Norm (0.9.5)
            postfix = {'loss': f"{meter.avg:.4f}", 'lr': f"{get_lr(self.optimizer):.2e}"}
            if grad_norm > 0:
                 postfix['norm'] = f"{grad_norm:.2f}"
            pbar.set_postfix(postfix)
            
        return {'loss': meter.avg}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        wer_meter = AverageMeter()
        cer_meter = AverageMeter()
        
        logged_samples = False # Flag to log only first batch
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val E{epoch}"):
                audio = batch['audio'].to(self.device)
                visual = batch['visual'].to(self.device)
                targets = batch['target'].to(self.device)
                target_mask = batch.get('target_mask', None)
                if target_mask is not None:
                    target_mask = target_mask.to(self.device)
                
                # Single Forward Pass (Optimization 0.8.3)
                outputs = self.model(audio, visual, targets)
                
                # 1. Loss Calculation
                loss_dict = self.criterion(
                    outputs['ctc_logits'],
                    outputs['ar_logits'],
                    targets,
                    target_mask,
                    epoch=epoch,
                    max_epochs=self.config['training']['num_epochs']
                )
                loss_meter.update(loss_dict['total_loss'].item())
                
                # 2. Real WER/CER Calculation
                # Use CTC Logits for Greedy/Beam decoding
                decoded_text = self.decoder.decode_batch(outputs['ctc_logits'], method='greedy')
                target_text = self.decoder.decode_targets(targets)
                
                # Log samples for first batch
                if not logged_samples:
                    logger.info(f"--- Epoch {epoch} Validation Samples ---")
                    for i in range(min(3, len(target_text))):
                        logger.info(f"Ref:  {target_text[i]}")
                        logger.info(f"Pred: {decoded_text[i]}")
                    logger.info("----------------------------------------")
                    logged_samples = True
                
                # Calculate Metrics
                wer = self.metric_calc.compute_wer(target_text, decoded_text)
                cer = self.metric_calc.compute_cer(target_text, decoded_text)
                
                wer_meter.update(wer)
                cer_meter.update(cer)
                
        return {
            'loss': loss_meter.avg,
            'wer': wer_meter.avg,
            'cer': cer_meter.avg
        }