"""
WandB Logger for MOTA AVSR Training.
"""
import wandb
import numpy as np
from typing import Dict, Any, Optional


class WandbLogger:
    """
    Thin wrapper quanh wandb.init().

    Log được:
    - Loss components (total, CTC, CE, quality)
    - Metrics (WER, CER)
    - Learning rate
    - Gradient norm
    - MQOT diagnostics (transport entropy, quality scores)
    - Alignment map (first batch, image)

    Usage:
        logger = WandbLogger(project="mota-avsr", name="phase1-v2")
        logger.log({"train/loss": 2.5, "epoch": 1})
        logger.log_alignment_map(alignment_weights, step=100)
        logger.finish()
    """

    def __init__(
        self,
        project: str = "mota-avsr",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        resume=False,
        id: Optional[str] = None,
    ):
        """
        Args:
            project: WandB project name
            name: Run name (defaults to timestamp)
            config: Hyperparameters to log
            resume: Resume from previous run
        """
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            raise ImportError(
                "wandb not installed. Run: pip install wandb"
            )

        self.enabled = True
        self.run = wandb.init(
            project=project,
            name=name,
            id=id,                 # fixed id + resume="allow" → reconnect to the same run on relaunch
            config=config,
            resume=resume,
            settings=wandb.Settings(_disable_stats=True),
        )
        self._step = 0
        # Decouple wandb's internal monotonic step from our global_step. On resume our step
        # rewinds (we replay the interrupted epoch), which otherwise makes wandb DROP every
        # replayed log as "step out of order". Logging global_step as a custom step metric —
        # and NOT passing step= to wandb.log — lets the x-axis go backwards without warnings.
        self.run.define_metric("global_step")
        self.run.define_metric("*", step_metric="global_step")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log scalar metrics."""
        if not self.enabled:
            return
        self._step = step if step is not None else self._step + 1
        self._log(metrics, self._step)

    def _log(self, data: Dict[str, Any], step: Optional[int]):
        """Send to wandb using global_step as the x-axis metric. We do NOT pass step= so
        wandb's internal step stays monotonic (no 'out of order' drops on resume/replay)."""
        if step is not None:
            data = {**data, "global_step": step}
        self._wandb.log(data)

    def log_mqot_diagnostics(
        self,
        transport_map: np.ndarray,   # [B, Ta, Tv]
        quality_scores: np.ndarray,   # [B, Tv]
        step: int,
    ):
        """
        Log MQOT diagnostics:
        - Transport plan entropy (alignment sharpness)
        - Per-frame quality distribution
        - Transport plan heatmap (first sample)
        """
        if not self.enabled:
            return

        # Entropy per audio frame: H(j) = -sum(P_ij * log P_ij)
        # P already row-stochastic: P.sum(dim=-1) = 1
        P = transport_map.astype(np.float64)
        P = np.clip(P, 1e-10, 1.0)
        entropy = -np.sum(P * np.log(P), axis=-1)  # [B, Ta]
        max_entropy = np.log(P.shape[2])  # = log(Tv)
        sharpness = 1.0 - entropy / max_entropy   # [B, Ta]

        diagnostics = {
            "mqot/entropy_mean": float(np.mean(entropy)),
            "mqot/entropy_std": float(np.std(entropy)),
            "mqot/sharpness_mean": float(np.mean(sharpness)),
            "mqot/sharpness_std": float(np.std(sharpness)),
            "mqot/quality_mean": float(np.mean(quality_scores)),
            "mqot/quality_std": float(np.std(quality_scores)),
            "mqot/quality_min": float(np.min(quality_scores)),
            "mqot/quality_max": float(np.max(quality_scores)),
        }
        self._log(diagnostics, step)

        # Alignment heatmap (first sample)
        if transport_map.shape[0] > 0:
            heatmap = wandb.Image(
                transport_map[0],
                caption="Transport Plan (sample 0)"
            )
            self._log({"mqot/transport_heatmap": heatmap}, step)

    def log_alignment_map(
        self,
        alignment_weights: np.ndarray,  # [B, Ta, Tv]
        step: int,
        caption: str = "QualityGate Cross-Attention"
    ):
        """Log alignment map từ QualityGate."""
        if not self.enabled or alignment_weights is None:
            return
        if alignment_weights.shape[0] > 0:
            img = wandb.Image(
                alignment_weights[0].astype(np.float32),
                caption=caption
            )
            self._log({"quality/alignment_map": img}, step)

    def finish(self):
        """Close WandB run."""
        if self.enabled:
            self._wandb.finish()
