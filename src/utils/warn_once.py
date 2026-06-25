"""De-duplicated warnings for training-loop fallbacks.

Any code path that silently falls back to a degraded behavior (no mask, full CTC length, skipped
batch, disabled wandb, dropped sample, ...) should call ``warn_once`` so the fallback is VISIBLE in
the logs exactly once per process/worker — loud enough to notice, quiet enough not to spam a 30-epoch
run. Principle: never let a behavior-changing fallback happen without a signal.
"""
import logging

_seen: set = set()


def warn_once(logger: logging.Logger, key: str, msg: str, level: int = logging.WARNING) -> None:
    """Log ``msg`` once per unique ``key`` (per process). Subsequent calls with the same key no-op."""
    if key not in _seen:
        _seen.add(key)
        logger.log(level, f"[fallback/once:{key}] {msg}")


def reset_warn_once() -> None:
    """Clear the seen-keys set (tests only)."""
    _seen.clear()
