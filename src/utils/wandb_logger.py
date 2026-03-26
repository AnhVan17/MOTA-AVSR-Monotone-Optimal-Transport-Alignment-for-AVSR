"""
WandB Logger for MOTA AVSR Training.
"""
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
        resume: bool = False,
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
            config=config,
            resume=resume,
            settings=wandb.Settings(_disable_stats=True),
        )
        self._step = 0

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log scalar metrics."""
        if not self.enabled:
            return
        self._step = step if step is not None else self._step + 1
        self._wandb.log(metrics, step=self._step)

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
        self._wandb.log(diagnostics, step=step)

        # Alignment heatmap (first sample)
        if transport_map.shape[0] > 0:
            heatmap = wandb.Image(
                transport_map[0],
                caption="Transport Plan (sample 0)"
            )
            self._wandb.log({"mqot/transport_heatmap": heatmap}, step=step)

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
            self._wandb.log({"quality/alignment_map": img}, step=step)

    def finish(self):
        """Close WandB run."""
        if self.enabled:
            self._wandb.finish()
