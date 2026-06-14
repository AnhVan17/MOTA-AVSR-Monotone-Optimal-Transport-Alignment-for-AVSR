"""WandB wiring guard: training must work with wandb OFF (no secret / not installed).

The trainer builds the logger lazily via _get_wandb_logger and guards every .log()/.finish()
call with `if self.wandb`. This pins the off-path so a run without WANDB_API_KEY (or without the
wandb package) never crashes — it just skips logging.
"""
from src.training.trainer import _get_wandb_logger


def test_wandb_disabled_returns_none():
    assert _get_wandb_logger({"logging": {"use_wandb": False}}) is None


def test_wandb_absent_logging_key_returns_none():
    # missing logging block → treated as disabled, not an error
    assert _get_wandb_logger({}) is None
