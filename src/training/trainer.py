import os
import torch
import torch.optim as optim
from typing import Dict
from pathlib import Path
from tqdm import tqdm

# Project Modules
from src.models.mota import create_model
from src.data.loader import build_dataloader
from src.data.tokenizers import build_tokenizer
from src.training.losses import create_loss
from src.evaluation.metrics import MetricCalculator
from src.evaluation.decoding import CTCDecoder
from src.utils.logging_utils import setup_logger
from src.utils.device import get_device, supports_amp, make_grad_scaler
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
            id=config['logging'].get('wandb_id', None),
            resume=config['logging'].get('wandb_resume', 'allow'),
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
        # Optional override via config['training']['device']; else auto cuda → mps → cpu.
        self.device = get_device(config.get('training', {}).get('device'))
        
        # Setup Checkpoint Directory
        self.checkpoint_dir = Path(config['logging']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing Trainer on {self.device}")
        
        # 1. Tokenizer (required for DataLoader)
        logger.info("Initializing Tokenizer...")
        self.tokenizer = build_tokenizer(config)

        # 2. Data Loaders
        logger.info("Building DataLoaders...")
        self.train_loader = build_dataloader(config, tokenizer=self.tokenizer, mode='train')
        self.val_loader = build_dataloader(config, tokenizer=self.tokenizer, mode='val')
        
        # 3. Model Initialization
        logger.info("Creating Model...")
        # Pass the (sibling) `mqot` block into the model config so MQOTLayer reads the YAML
        # hyperparams (lambda_time/epsilon/n_iters); otherwise it silently falls back to defaults.
        self.model = create_model({**config['model'], 'mqot': config.get('mqot', {})}).to(self.device)
        logger.info(f"Model Params: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 3. Optimization Setup
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training'].get('weight_decay', 0.01))
        )
        
        # Warmup + Adaptive LR — quản lý 2 scheduler RIÊNG.
        # ChainedScheduler không bọc được ReduceLROnPlateau (step() cần metric),
        # nên warmup chạy per-step, plateau chạy per-epoch.
        self.warmup_steps = config['training'].get('warmup_steps', 1000)
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=float(config['training'].get('min_lr', 1e-6))
        )

        # 4. Loss Function & Metrics
        self.criterion = create_loss(config, tokenizer=self.tokenizer).to(self.device)
        self.early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 10), 
            mode='min'
        )
        
        # Mixed Precision
        # AMP chỉ ổn định trên CUDA — tự tắt trên MPS/CPU để khỏi lỗi/giảm tốc.
        self.use_amp = config['training'].get('use_amp', False) and supports_amp(self.device)
        if config['training'].get('use_amp', False) and not self.use_amp:
            logger.warning(f"AMP requested nhưng device {self.device.type} không hỗ trợ → tắt AMP.")
        self.scaler = make_grad_scaler(self.device, enabled=self.use_amp)

        # Training State
        self.start_epoch = 0
        self.step = 0
        self.best_metric = float('inf')
        self.log_interval = config['logging'].get('log_interval', 50)  # wandb per-step cadence
        self.train_wer_interval = config['logging'].get('train_wer_interval', 500)  # periodic train WER (0=off)
        # Progress-bar total: WebDataset is an IterableDataset (no __len__), so derive steps/epoch
        # from the known clip count → tqdm shows "210/12129 [ETA]" instead of a bare "210it".
        _spe = config['data'].get('samples_per_epoch')
        self.train_steps = (_spe // config['data']['batch_size']) if _spe else None

        # Load Pretrained / Resume.
        # True resume (full state) takes precedence over pretrained (weights-only warm-start):
        # a relaunched/preempted container picks up its own in-progress checkpoint, which already
        # contains the warm-started weights — re-applying pretrained would clobber training progress.
        resume_path = self._resolve_resume_path(config['training'], self.checkpoint_dir)
        if resume_path:
            self._resume_from(resume_path)
        elif config['training'].get('pretrained_path'):
            self._load_checkpoint(config['training']['pretrained_path'])

        # 5. Validation Tools
        # (self.tokenizer đã tạo ở trên và được truyền vào dataloader → không gán lại.)
        self.metric_calc = MetricCalculator()
        # Blank for greedy CTC decode = tokenizer's eot/blank id (consistent with create_loss).
        blank_id = getattr(self.tokenizer, 'eot_token_id', config['model'].get('blank_id', 50257))
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

    def _resume_from(self, path: str):
        """True crash-resume: restore weights + optimizer + scheduler + epoch/step/best_metric.

        Unlike _load_checkpoint (weights-only warm-start for cross-phase P1→P2), this restores the
        full training state via load_checkpoint so a preempted run continues exactly where it left
        off — same LR schedule, same optimizer moments, next epoch.
        """
        logger.info(f"Resuming training from {path}...")
        ckpt = load_checkpoint(path, self.model, self.optimizer, self.plateau_scheduler, self.device)
        self.start_epoch, self.step, self.best_metric = self._resume_state(ckpt)
        logger.info(
            f"Resumed at epoch {self.start_epoch} (step {self.step}, best_metric {self.best_metric:.4f})."
        )

    @staticmethod
    def _resume_state(checkpoint: Dict) -> tuple:
        """(start_epoch, step, best_metric) to resume from a loaded checkpoint dict.

        The saved epoch already completed, so resume at epoch+1; step/best_metric carry over.
        """
        return checkpoint['epoch'] + 1, checkpoint['step'], checkpoint['best_metric']

    @staticmethod
    def _resolve_resume_path(train_cfg: Dict, checkpoint_dir):
        """Checkpoint to resume from, or None (→ cold start / pretrained warm-start).

        - explicit ``resume_path`` → that file if it exists (else None, don't crash);
        - ``resume: true`` → latest ``epoch_*.pt`` in checkpoint_dir (auto multi-container recovery;
          empty on first launch → None);
        - neither → None.
        """
        explicit = train_cfg.get('resume_path')
        if explicit:
            return explicit if os.path.exists(explicit) else None
        if train_cfg.get('resume'):
            ckpts = sorted(Path(checkpoint_dir).glob('epoch_*.pt'),
                           key=lambda p: int(p.stem.split('_')[1]))
            return str(ckpts[-1]) if ckpts else None
        return None

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
            self.plateau_scheduler.step(current_metric)
            
            # 4. Save Checkpoint
            is_best = current_metric < self.best_metric
            if is_best:
                self.best_metric = current_metric
                
            # Save latest
            save_checkpoint(
                self.model, self.optimizer, self.plateau_scheduler,
                epoch, self.step, self.best_metric,
                str(self.checkpoint_dir),
                filename=f"epoch_{epoch}.pt"
            )
            
            # Save best
            if is_best:
                save_checkpoint(
                    self.model, self.optimizer, self.plateau_scheduler,
                    epoch, self.step, self.best_metric,
                    str(self.checkpoint_dir),
                    filename="best_model.pt"
                )
            
            # 5. Logging — expose plateau state so the LR-drop countdown is visible (num_bad/patience)
            plateau = self.plateau_scheduler
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val WER: {val_metrics.get('wer', 0):.2f}% | "
                f"LR: {get_lr(self.optimizer):.2e} | "
                f"Plateau: {plateau.num_bad_epochs}/{plateau.patience} (best {plateau.best:.2f})"
            )

            # WandB per-epoch
            if self.wandb:
                self.wandb.log({
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_metrics['loss'],
                    "val/loss": val_metrics['loss'],
                    "val/wer": val_metrics.get('wer', 0),
                    "val/cer": val_metrics.get('cer', 0),
                    "lr": get_lr(self.optimizer),
                    "plateau/num_bad_epochs": plateau.num_bad_epochs,
                }, step=self.step)

            # 6. Early Stopping
            if self.early_stopping(current_metric, epoch):
                logger.info("Early stopping triggered. Training finished.")
                break

        if self.wandb:
            self.wandb.finish()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        meter = AverageMeter()
        
        
        # Training Configs
        accum_steps = self.config['training'].get('accum_steps', 1)
        assert accum_steps > 0, f"accum_steps must be > 0, got {accum_steps}"
        
        # 0.9.5 Fix: Ensure gradients are zeroed before loop starts
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}", total=self.train_steps)
        grad_norm = 0.0  # Always defined for consistent tqdm postfix
        for batch_idx, batch in enumerate(pbar):
            if batch is None:   # cả batch hỏng (mọi sample lỗi load) → skip
                continue
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

                # Warmup LR
                if self.step <= self.warmup_steps:
                    self.warmup_scheduler.step()
                
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

            # WandB per-step (every log_interval steps; no-op when wandb disabled)
            if self.wandb and self.step % self.log_interval == 0:
                self.wandb.log({
                    "train/loss": loss_val,
                    "train/ctc_loss": loss_dict['ctc_loss'].item(),
                    "train/ce_loss": loss_dict['ce_loss'].item(),
                    "train/lr": get_lr(self.optimizer),
                    "train/grad_norm": float(grad_norm),
                }, step=self.step)

            # WandB train WER (periodic, CTC-greedy on the current batch — early in-epoch signal).
            # CTC greedy ONLY (not AR): the AR decoder runs teacher-forced here, so its WER would be
            # unrealistically optimistic. Cheap at this cadence; val/wer remains the source of truth.
            if self.wandb and self.train_wer_interval > 0 and self.step % self.train_wer_interval == 0:
                with torch.no_grad():
                    pred_text = self.decoder.decode_batch(outputs['ctc_logits'], method='greedy')
                    ref_text = self.decoder.decode_targets(targets)
                    train_wer = self.metric_calc.compute_wer(pred_text, ref_text)
                self.wandb.log({"train/wer": train_wer}, step=self.step)
            
        return {'loss': meter.avg}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        wer_meter = AverageMeter()
        cer_meter = AverageMeter()
        
        logged_samples = False # Flag to log only first batch
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val E{epoch}"):
                if batch is None:   # cả batch hỏng → skip
                    continue
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
                
                # Calculate Metrics — compute_wer(predictions, references) order matters!
                wer = self.metric_calc.compute_wer(decoded_text, target_text)
                cer = self.metric_calc.compute_cer(decoded_text, target_text)
                
                wer_meter.update(wer)
                cer_meter.update(cer)
                
        return {
            'loss': loss_meter.avg,
            'wer': wer_meter.avg,
            'cer': cer_meter.avg
        }