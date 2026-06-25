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
from src.utils.warn_once import warn_once

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
        self.model = create_model({**config['model'], 'mqot': config.get('mqot', {}), 'rgf': config.get('rgf', {})}).to(self.device)
        logger.info(f"Model Params: {sum(p.numel() for p in self.model.parameters()):,}")
        if config['model'].get('use_mqot') and not config.get('mqot'):
            warn_once(logger, "mqot_defaults",
                      "use_mqot=True nhưng thiếu block 'mqot' trong config → MQOT dùng default hyperparams.")
        self._freeze_quality_gate = bool(
            config['training'].get('freeze_quality_gate_for_visual_force', False)
        )
        if self._freeze_quality_gate:
            self._freeze_quality_gate_for_visual_force()
        
        # 3. Optimization Setup — param groups (head @ main lr; optional visual last-block @ low lr
        # for gradual unfreeze). Groups are FIXED at construction so ReduceLROnPlateau's per-group
        # min_lrs length matches (it cannot accept param groups added mid-run).
        self.optimizer = optim.AdamW(
            self._build_param_groups(self.model, config['training']),
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
        # Gradual-unfreeze (visual backbone last block) — active only when configured. Own plateau
        # tracker, independent of plateau_scheduler; the fallback epoch makes it resume-safe.
        self._uf_patience = config['training'].get('visual_unfreeze_patience')
        self._uf_epoch = config['training'].get('visual_unfreeze_epoch', float('inf'))
        self._uf_min_delta = float(config['training'].get('visual_unfreeze_min_delta', 0.1))
        self._uf_best = float('inf')
        self._uf_bad = 0
        self._unfrozen = False
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
        if config['training'].get('visual_unfreeze_at_start', False):
            self._unfreeze_visual_now(reason="visual_unfreeze_at_start=True")
        elif self._uf_epoch != float('inf') and self.start_epoch >= int(self._uf_epoch):
            self._unfreeze_visual_now(
                reason=f"resume/start_epoch={self.start_epoch} >= visual_unfreeze_epoch={self._uf_epoch}"
            )

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
        elif config.get('logging', {}).get('use_wandb', False):
            logger.warning("use_wandb=True nhưng WandB DISABLED (thiếu secret/key hoặc chưa cài wandb) "
                           "→ monitor qua log + checkpoint.")


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
        ckpt = load_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.plateau_scheduler,
            self.device,
            warmup_scheduler=self.warmup_scheduler,
            scaler=self.scaler,
        )
        self._sync_warmup_scheduler_after_resume(ckpt)
        self.start_epoch, self.step, self.best_metric = self._resume_state(ckpt)
        logger.info(
            f"Resumed at epoch {self.start_epoch} (step {self.step}, best_metric {self.best_metric:.4f})."
        )

    def _sync_warmup_scheduler_after_resume(self, checkpoint: Dict) -> None:
        """Keep LinearLR from restarting after a full-state resume.

        Older checkpoints only stored the plateau scheduler. If we restore the optimizer LR from such
        a checkpoint while LinearLR still thinks it is at the first warmup step, the next
        ``LinearLR.step()`` can multiply the already-warmed LR and explode training. New checkpoints
        store warmup state exactly; old ones infer optimizer-step progress from the micro-step count.
        """
        if checkpoint.get('warmup_scheduler_state_dict'):
            logger.info(
                f"Warmup scheduler restored from checkpoint "
                f"(last_epoch={self.warmup_scheduler.last_epoch})."
            )
            return

        accum_steps = max(1, int(self.config['training'].get('accum_steps', 1)))
        optimizer_steps = int(checkpoint.get('step', 0)) // accum_steps
        inferred_last_epoch = min(int(self.warmup_steps), optimizer_steps)
        self.warmup_scheduler.last_epoch = inferred_last_epoch
        if hasattr(self.warmup_scheduler, "_step_count"):
            self.warmup_scheduler._step_count = max(1, inferred_last_epoch + 1)
        self.warmup_scheduler._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        logger.info(
            "Warmup scheduler state missing in checkpoint; inferred "
            f"optimizer_steps={optimizer_steps}, last_epoch={inferred_last_epoch}, "
            f"lr={get_lr(self.optimizer):.2e}."
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

    @staticmethod
    def _build_param_groups(model, train_cfg: Dict):
        """AdamW param groups. With `visual_backbone_lr` set AND a visual backbone exposing
        `last_block_parameters` (the lip-reading frontend), the unfreezable last block gets its own
        low-lr group (gradual unfreeze); the rest ('head') trains at the main lr. The permanently
        frozen stem/early-trunk params are in NO group (never trained). Else a single group."""
        main_lr = float(train_cfg['learning_rate'])
        vb_lr = train_cfg.get('visual_backbone_lr')
        backbone = getattr(model, 'visual_backbone', None)
        if vb_lr is None or not hasattr(backbone, 'last_block_parameters'):
            return [{'params': list(model.parameters()), 'lr': main_lr, 'name': 'all'}]
        vb_ids = {id(p) for p in backbone.parameters()}
        head = [p for p in model.parameters() if id(p) not in vb_ids]
        last_block = list(backbone.last_block_parameters())
        logger.info(f"Param groups: head={len(head)} @ {main_lr:.1e}, "
                    f"visual_last_block={len(last_block)} @ {float(vb_lr):.1e}")
        return [
            {'params': head, 'lr': main_lr, 'name': 'head'},
            {'params': last_block, 'lr': float(vb_lr), 'name': 'visual_backbone'},
        ]

    def _maybe_unfreeze_visual(self, epoch: int, val_metric: float):
        """Gradual unfreeze: enable the visual backbone's last block once the head plateaus (own
        tracker) or the fallback epoch is reached. One-shot; resets the visual group's lr to the
        intended fine-tune lr (undoing any pre-unfreeze plateau decay). No-op unless configured."""
        if self._unfrozen or self._uf_patience is None:
            return
        backbone = getattr(self.model, 'visual_backbone', None)
        if not hasattr(backbone, 'unfreeze_last_block'):
            return
        if val_metric < self._uf_best - self._uf_min_delta:
            self._uf_best, self._uf_bad = val_metric, 0
        else:
            self._uf_bad += 1
        if self._uf_bad >= int(self._uf_patience) or (epoch + 1) >= self._uf_epoch:
            self._unfreeze_visual_now(reason=f"epoch={epoch}, val={val_metric:.2f}")
            vb_lr = float(self.config['training']['visual_backbone_lr'])
            logger.info(f"[gradual-unfreeze] visual last block unfrozen at epoch {epoch} "
                        f"(val={val_metric:.2f}, lr={vb_lr:.1e}).")

    def _unfreeze_visual_now(self, reason: str = ""):
        """One-shot unfreeze for the visual frontend's last block."""
        if self._unfrozen:
            return
        backbone = getattr(self.model, 'visual_backbone', None)
        if not hasattr(backbone, 'unfreeze_last_block'):
            return
        backbone.unfreeze_last_block()
        vb_lr = float(self.config['training'].get('visual_backbone_lr', get_lr(self.optimizer)))
        for group in self.optimizer.param_groups:
            if group.get('name') == 'visual_backbone':
                group['lr'] = vb_lr
        self._unfrozen = True
        logger.info(f"[visual-unfreeze] visual last block unfrozen ({reason}); lr={vb_lr:.1e}.")

    def _freeze_quality_gate_for_visual_force(self) -> None:
        """Freeze Stage-1 QualityGate so visual forcing cannot destroy the E15 audio behavior.

        The frozen module remains differentiable with respect to its inputs, so gradients still train
        visual_proj, the visual last block, encoder, and decoder. It just prevents the gate itself
        from racing to an all-visual solution before the visual stream can actually read speech.
        """
        qg = getattr(self.model, 'quality_gate', None)
        if qg is None:
            return
        qg.requires_grad_(False)
        qg.eval()
        logger.info("[force-visual] QualityGate frozen; visual forcing trains through a fixed fusion gate.")

    def _keep_force_frozen_modules_eval(self) -> None:
        if self._freeze_quality_gate:
            qg = getattr(self.model, 'quality_gate', None)
            if qg is not None:
                qg.eval()

    @staticmethod
    def _normalize_max_epochs_per_run(value):
        """Return a positive per-launch epoch cap, or None when unset."""
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError(f"max_epochs_per_run must be > 0, got {value}")
        return value

    @staticmethod
    def _reached_max_epochs_per_run(start_epoch: int, epoch: int, max_per_run: int) -> bool:
        """Whether this launch has completed its requested number of epochs."""
        return max_per_run is not None and (epoch - start_epoch + 1) >= max_per_run

    @staticmethod
    def _ramped_aux_weight(epoch: int, base_weight: float, warmup_epochs: int = 0,
                           start_epoch: int = 0) -> float:
        """Linearly ramp an auxiliary loss from 0 to base_weight after start_epoch."""
        base_weight = float(base_weight or 0.0)
        if base_weight <= 0.0 or epoch < int(start_epoch):
            return 0.0
        warmup_epochs = int(warmup_epochs or 0)
        if warmup_epochs <= 0:
            return base_weight
        progress = min(1.0, float(epoch - int(start_epoch) + 1) / float(warmup_epochs))
        return base_weight * progress

    def _loss_from_outputs(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor,
                           target_mask: torch.Tensor, input_lengths: torch.Tensor,
                           epoch: int) -> Dict[str, torch.Tensor]:
        """Shared loss call for normal and auxiliary modality-specific forwards."""
        return self.criterion(
            ctc_logits=outputs['ctc_logits'],
            ar_logits=outputs['ar_logits'],
            targets=targets,
            target_mask=target_mask,
            input_lengths=input_lengths,
            epoch=epoch,
            max_epochs=self.config['training']['num_epochs'],
            transport_map=outputs.get('transport_map'),
            mqot_quality=outputs.get('mqot_quality'),
            router_probs=outputs.get('router_probs')
        )

    def _gate_diagnostics(self, outputs: Dict[str, torch.Tensor], prefix: str) -> Dict[str, float]:
        """Small scalar diagnostics for whether the model is actually opening the visual path."""
        metrics = {}
        gate = outputs.get('gate_weights')
        if gate is not None:
            metrics[f"{prefix}/gate_audio_mean"] = gate[..., 0].detach().float().mean().item()
            metrics[f"{prefix}/gate_visual_mean"] = gate[..., 1].detach().float().mean().item()
        q_audio = outputs.get('q_audio')
        if q_audio is not None:
            metrics[f"{prefix}/q_audio_mean"] = q_audio.detach().float().mean().item()
        q_visual = outputs.get('q_visual')
        if q_visual is not None:
            metrics[f"{prefix}/q_visual_mean"] = q_visual.detach().float().mean().item()
        return metrics

    def _model_gate_diagnostics(self) -> Dict[str, float]:
        """Model-level gate scalars; useful for force-visual and later MQOT runs."""
        metrics = {}
        quality_gate = getattr(self.model, 'quality_gate', None)
        if quality_gate is not None and hasattr(quality_gate, 'residual_gate'):
            metrics["train/qg_residual_gate"] = torch.sigmoid(
                quality_gate.residual_gate.detach()
            ).float().item()
        if hasattr(self.model, 'fine_align_gate'):
            metrics["train/mqot_fine_align_gate"] = torch.sigmoid(
                self.model.fine_align_gate.detach()
            ).float().item()
        return metrics

    def _optimizer_accum_step(self) -> torch.Tensor:
        """Apply one optimizer step after gradient accumulation.

        WebDataset loaders are iterable and do not implement ``len()``, so the train loop cannot
        detect the final partial accumulation window with ``batch_idx == len(loader) - 1``.
        Keeping the step logic here lets the loop flush by counter instead.
        """
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config['training'].get('gradient_clip', 5.0)
        )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if torch.isfinite(grad_norm) and self.warmup_scheduler.last_epoch < self.warmup_steps:
            self.warmup_scheduler.step()
        elif not torch.isfinite(grad_norm):
            warmup_note = (
                "warmup đã hoàn tất" if self.warmup_scheduler.last_epoch >= self.warmup_steps
                else "warmup chưa advance ở step này"
            )
            warn_once(
                logger,
                "nonfinite_grad_norm",
                "Gradient norm không finite; AMP GradScaler có thể skip optimizer step và giảm scale "
                f"({warmup_note}). Nếu loss/LR vẫn ổn sau vài batch thì không phải lỗi divergence."
            )

        self.optimizer.zero_grad()
        return grad_norm

    def train(self):
        """Main Training Loop"""
        num_epochs = self.config['training']['num_epochs']
        max_per_run = self._normalize_max_epochs_per_run(
            self.config['training'].get('max_epochs_per_run')
        )  # None = chạy tới num_epochs
        logger.info(
            f"Starting Training for {num_epochs} epochs"
            + (f" (cap {max_per_run} epoch cho LẦN CHẠY này)" if max_per_run else "")
        )
        
        for epoch in range(self.start_epoch, num_epochs):
            # 1. Train One Epoch
            train_metrics = self.train_epoch(epoch)
            
            # 2. Validate
            val_metrics = self.validate_epoch(epoch)
            
            # 3. Update Learning Rate (Adaptive)
            # Use Validation WER as the primary metric for scheduler
            current_metric = val_metrics.get('wer', val_metrics['loss'])
            self.plateau_scheduler.step(current_metric)

            # 3b. Gradual unfreeze of the visual backbone's last block (no-op unless configured).
            self._maybe_unfreeze_visual(epoch, current_metric)
            
            # 4. Save Checkpoint
            is_best = current_metric < self.best_metric
            if is_best:
                self.best_metric = current_metric
                
            # Save latest
            save_checkpoint(
                self.model, self.optimizer, self.plateau_scheduler,
                epoch, self.step, self.best_metric,
                str(self.checkpoint_dir),
                filename=f"epoch_{epoch}.pt",
                warmup_scheduler=self.warmup_scheduler,
                scaler=self.scaler,
            )
            
            # Save best
            if is_best:
                save_checkpoint(
                    self.model, self.optimizer, self.plateau_scheduler,
                    epoch, self.step, self.best_metric,
                    str(self.checkpoint_dir),
                    filename="best_model.pt",
                    warmup_scheduler=self.warmup_scheduler,
                    scaler=self.scaler,
                )
            
            # 5. Logging — expose plateau state so the LR-drop countdown is visible (num_bad/patience)
            plateau = self.plateau_scheduler
            visual_ctc_note = (
                f"Val visualCTC WER: {val_metrics['visual_ctc_wer']:.2f}% | "
                if 'visual_ctc_wer' in val_metrics else ""
            )
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val WER: {val_metrics.get('wer', 0):.2f}% | "
                f"{visual_ctc_note}"
                f"LR: {get_lr(self.optimizer):.2e} | "
                f"Plateau: {plateau.num_bad_epochs}/{plateau.patience} (best {plateau.best:.2f})"
            )

            # WandB per-epoch
            if self.wandb:
                epoch_log = {
                    "epoch": epoch + 1,
                    "train/epoch_loss": train_metrics['loss'],
                    "val/loss": val_metrics['loss'],
                    "val/wer": val_metrics.get('wer', 0),
                    "val/cer": val_metrics.get('cer', 0),
                    "lr": get_lr(self.optimizer),
                    "plateau/num_bad_epochs": plateau.num_bad_epochs,
                }
                if 'visual_ctc_loss' in val_metrics:
                    epoch_log.update({
                        "val/visual_ctc_direct_loss": val_metrics['visual_ctc_loss'],
                        "val/visual_ctc_direct_wer": val_metrics['visual_ctc_wer'],
                        "val/visual_ctc_direct_cer": val_metrics['visual_ctc_cer'],
                    })
                self.wandb.log(epoch_log, step=self.step)

            # 6. Early Stopping
            if self.early_stopping(current_metric, epoch):
                logger.info("Early stopping triggered. Training finished.")
                break

            # 7. Cap epoch/LẦN-CHẠY (treo detach từng chặng): dừng sạch sau N epoch của lần chạy này.
            #    Checkpoint epoch hiện tại đã lưu ở trên → relaunch (resume:true) tiếp từ epoch+1.
            if self._reached_max_epochs_per_run(self.start_epoch, epoch, max_per_run):
                logger.info(f"Đã chạy {max_per_run} epoch lần này (dừng sau epoch {epoch}; "
                            f"relaunch sẽ resume từ epoch {epoch + 1}).")
                break

        if self.wandb:
            self.wandb.finish()

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self._keep_force_frozen_modules_eval()
        meter = AverageMeter()
        
        
        # Training Configs
        accum_steps = self.config['training'].get('accum_steps', 1)
        assert accum_steps > 0, f"accum_steps must be > 0, got {accum_steps}"
        visual_aux_weight = self._ramped_aux_weight(
            epoch,
            self.config['training'].get('visual_only_loss_weight', 0.0),
            self.config['training'].get('visual_only_warmup_epochs', 0),
            self.config['training'].get('visual_only_start_epoch', 0),
        )
        visual_ctc_aux_weight = self._ramped_aux_weight(
            epoch,
            self.config['training'].get('visual_ctc_loss_weight', 0.0),
            self.config['training'].get('visual_ctc_warmup_epochs', 0),
            self.config['training'].get('visual_ctc_start_epoch', 0),
        )
        audio_aux_weight = self._ramped_aux_weight(
            epoch,
            self.config['training'].get('audio_only_loss_weight', 0.0),
            self.config['training'].get('audio_only_warmup_epochs', 0),
            self.config['training'].get('audio_only_start_epoch', 0),
        )
        if visual_aux_weight > 0:
            logger.info(f"Train E{epoch}: visual-only auxiliary loss active (weight={visual_aux_weight:.3f}).")
        if visual_ctc_aux_weight > 0:
            logger.info(f"Train E{epoch}: visual CTC auxiliary loss active (weight={visual_ctc_aux_weight:.3f}).")
        if audio_aux_weight > 0:
            logger.info(f"Train E{epoch}: audio-only auxiliary loss active (weight={audio_aux_weight:.3f}).")
        
        # 0.9.5 Fix: Ensure gradients are zeroed before loop starts
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f"Train E{epoch}", total=self.train_steps)
        grad_norm = 0.0  # Always defined for consistent tqdm postfix
        n_skipped = 0
        accum_counter = 0
        for batch_idx, batch in enumerate(pbar):
            if batch is None:   # cả batch hỏng (mọi sample lỗi load) → skip
                n_skipped += 1
                warn_once(logger, "train_batch_skip",
                          "Bỏ batch hỏng khi train (mọi sample lỗi load) → tổng hợp cuối epoch.")
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
            # Masks (F3) lên device + độ dài audio thật cho CTC (F1).
            audio_mask = batch.get('audio_mask')
            visual_mask = batch.get('visual_mask')
            audio_mask = audio_mask.to(self.device) if audio_mask is not None else None
            visual_mask = visual_mask.to(self.device) if visual_mask is not None else None
            audio_lengths = audio_mask.sum(1) if audio_mask is not None else None
            visual_lengths = visual_mask.sum(1) if visual_mask is not None else None

            # Zero Gradients handled at step boundary now (Moved to after step in 0.9.5)
            # if (batch_idx % accum_steps) == 0:
            #      self.optimizer.zero_grad()
            
            # Forward & Loss (with Mixed Precision)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                # E2E Support: forward() handles raw/features internally
                outputs = self.model(audio, visual, targets, audio_mask=audio_mask, visual_mask=visual_mask)
                
                # Compute Loss
                # Pass epoch for Curriculum Learning (if used inside Loss)
                loss_dict = self._loss_from_outputs(
                    outputs, targets, target_mask, audio_lengths, epoch
                )
                loss = loss_dict['total_loss']

                # Visual bootstrap: direct CTC supervision on the visual timeline before fusion.
                # This asks the lip frontend/projection to become transcript-aware without turning
                # off audio in the main AV objective.
                if visual_ctc_aux_weight > 0 and outputs.get('visual_ctc_logits') is not None:
                    visual_ctc_loss_dict = self.criterion.compute_ctc_only(
                        outputs['visual_ctc_logits'], targets, target_mask, visual_lengths
                    )
                    loss = loss + visual_ctc_aux_weight * visual_ctc_loss_dict['loss']
                    loss_dict['visual_ctc_aux_loss'] = visual_ctc_loss_dict['loss'].detach()
                    loss_dict['visual_ctc_aux_empty'] = visual_ctc_loss_dict['empty'].detach()
                    loss_dict['visual_ctc_aux_weight'] = torch.tensor(
                        visual_ctc_aux_weight, device=loss.device
                    )

                # Force-visual bet: an auxiliary visual-only forward directly trains the fusion +
                # decoder stack to read from lips instead of letting Whisper features solve every
                # batch. This is intentionally opt-in via config because it roughly doubles compute.
                if visual_aux_weight > 0:
                    visual_only_outputs = self.model(
                        torch.zeros_like(audio), visual, targets,
                        audio_mask=audio_mask, visual_mask=visual_mask
                    )
                    visual_loss_dict = self._loss_from_outputs(
                        visual_only_outputs, targets, target_mask, audio_lengths, epoch
                    )
                    loss = loss + visual_aux_weight * visual_loss_dict['total_loss']
                    loss_dict['visual_only_loss'] = visual_loss_dict['total_loss'].detach()
                    loss_dict['visual_only_ctc_loss'] = visual_loss_dict['ctc_loss'].detach()
                    loss_dict['visual_only_ce_loss'] = visual_loss_dict['ce_loss'].detach()
                    loss_dict['visual_only_weight'] = torch.tensor(
                        visual_aux_weight, device=loss.device
                    )

                # Optional symmetry / diagnostics: keep audio-only performance explicit if desired.
                if audio_aux_weight > 0:
                    audio_only_outputs = self.model(
                        audio, torch.zeros_like(visual), targets,
                        audio_mask=audio_mask, visual_mask=visual_mask
                    )
                    audio_loss_dict = self._loss_from_outputs(
                        audio_only_outputs, targets, target_mask, audio_lengths, epoch
                    )
                    loss = loss + audio_aux_weight * audio_loss_dict['total_loss']
                    loss_dict['audio_only_loss'] = audio_loss_dict['total_loss'].detach()
                    loss_dict['audio_only_weight'] = torch.tensor(
                        audio_aux_weight, device=loss.device
                    )
                
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
            accum_counter += 1
            
            # Optimizer Step (Accumulated)
            if accum_counter >= accum_steps:
                grad_norm = self._optimizer_accum_step()
                accum_counter = 0
            
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
                if 'visual_only_loss' in loss_dict:
                    visual_log = {
                        "train/visual_only_loss": loss_dict['visual_only_loss'].item(),
                        "train/visual_only_ctc_loss": loss_dict['visual_only_ctc_loss'].item(),
                        "train/visual_only_ce_loss": loss_dict['visual_only_ce_loss'].item(),
                        "train/visual_only_weight": loss_dict['visual_only_weight'].item(),
                    }
                    visual_log.update(self._gate_diagnostics(visual_only_outputs, "train/visual_only"))
                    self.wandb.log(visual_log, step=self.step)
                if 'visual_ctc_aux_loss' in loss_dict:
                    self.wandb.log({
                        "train/visual_ctc_aux_loss": loss_dict['visual_ctc_aux_loss'].item(),
                        "train/visual_ctc_aux_empty": loss_dict['visual_ctc_aux_empty'].item(),
                        "train/visual_ctc_aux_weight": loss_dict['visual_ctc_aux_weight'].item(),
                    }, step=self.step)
                if 'audio_only_loss' in loss_dict:
                    self.wandb.log({
                        "train/audio_only_loss": loss_dict['audio_only_loss'].item(),
                        "train/audio_only_weight": loss_dict['audio_only_weight'].item(),
                    }, step=self.step)
                diag_log = self._gate_diagnostics(outputs, "train/av")
                diag_log.update(self._model_gate_diagnostics())
                if diag_log:
                    self.wandb.log(diag_log, step=self.step)

            # WandB train WER (periodic, CTC-greedy on the current batch — early in-epoch signal).
            # CTC greedy ONLY (not AR): the AR decoder runs teacher-forced here, so its WER would be
            # unrealistically optimistic. Cheap at this cadence; val/wer remains the source of truth.
            if self.wandb and self.train_wer_interval > 0 and self.step % self.train_wer_interval == 0:
                with torch.no_grad():
                    pred_text = self.decoder.decode_batch(outputs['ctc_logits'], method='greedy')
                    ref_text = self.decoder.decode_targets(targets)
                    train_wer = self.metric_calc.compute_wer(pred_text, ref_text)
                self.wandb.log({"train/wer": train_wer}, step=self.step)
            
        if accum_counter > 0:
            grad_norm = self._optimizer_accum_step()
            logger.info(f"Train E{epoch}: flushed {accum_counter} leftover accumulated batch(es).")
        if n_skipped:
            logger.warning(f"Train E{epoch}: bỏ {n_skipped} batch hỏng (toàn sample lỗi load).")
        return {'loss': meter.avg}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        loss_meter = AverageMeter()
        wer_meter = AverageMeter()
        cer_meter = AverageMeter()
        visual_ctc_loss_meter = AverageMeter()
        visual_ctc_wer_meter = AverageMeter()
        visual_ctc_cer_meter = AverageMeter()
        
        logged_samples = False # Flag to log only first batch
        
        n_skipped = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val E{epoch}"):
                if batch is None:   # cả batch hỏng → skip
                    n_skipped += 1
                    warn_once(logger, "val_batch_skip", "Bỏ batch hỏng khi val → tổng hợp cuối epoch.")
                    continue
                audio = batch['audio'].to(self.device)
                visual = batch['visual'].to(self.device)
                targets = batch['target'].to(self.device)
                target_mask = batch.get('target_mask', None)
                if target_mask is not None:
                    target_mask = target_mask.to(self.device)
                audio_mask = batch.get('audio_mask')
                visual_mask = batch.get('visual_mask')
                audio_mask = audio_mask.to(self.device) if audio_mask is not None else None
                visual_mask = visual_mask.to(self.device) if visual_mask is not None else None
                audio_lengths = audio_mask.sum(1) if audio_mask is not None else None
                visual_lengths = visual_mask.sum(1) if visual_mask is not None else None

                # Single Forward Pass (Optimization 0.8.3)
                outputs = self.model(audio, visual, targets, audio_mask=audio_mask, visual_mask=visual_mask)
                
                # 1. Loss Calculation
                loss_dict = self._loss_from_outputs(
                    outputs, targets, target_mask, audio_lengths, epoch
                )
                loss_meter.update(loss_dict['total_loss'].item())
                
                # 2. Real WER/CER Calculation
                # Use CTC Logits for Greedy/Beam decoding
                decoded_text = self.decoder.decode_batch(outputs['ctc_logits'], method='greedy')
                target_text = self.decoder.decode_targets(targets)
                if outputs.get('visual_ctc_logits') is not None:
                    visual_ctc = self.criterion.compute_ctc_only(
                        outputs['visual_ctc_logits'], targets, target_mask, visual_lengths
                    )
                    visual_ctc_loss_meter.update(visual_ctc['loss'].item())
                    visual_decoded_text = self.decoder.decode_batch(
                        outputs['visual_ctc_logits'], method='greedy'
                    )
                    visual_ctc_wer_meter.update(
                        self.metric_calc.compute_wer(visual_decoded_text, target_text)
                    )
                    visual_ctc_cer_meter.update(
                        self.metric_calc.compute_cer(visual_decoded_text, target_text)
                    )
                
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
                
        if n_skipped:
            logger.warning(f"Val E{epoch}: bỏ {n_skipped} batch hỏng.")
        metrics = {
            'loss': loss_meter.avg,
            'wer': wer_meter.avg,
            'cer': cer_meter.avg
        }
        if visual_ctc_loss_meter.count:
            metrics.update({
                'visual_ctc_loss': visual_ctc_loss_meter.avg,
                'visual_ctc_wer': visual_ctc_wer_meter.avg,
                'visual_ctc_cer': visual_ctc_cer_meter.avg,
            })
        return metrics
