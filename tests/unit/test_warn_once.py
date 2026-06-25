"""warn_once fires exactly once per key (so loop fallbacks are visible but not spammy)."""
import logging

from src.utils.warn_once import reset_warn_once, warn_once


def test_fires_once_per_key(caplog):
    reset_warn_once()
    log = logging.getLogger("t")
    with caplog.at_level(logging.WARNING):
        warn_once(log, "k1", "one")
        warn_once(log, "k1", "one")  # suppressed
        warn_once(log, "k2", "two")
    msgs = [r.message for r in caplog.records]
    assert sum("once:k1" in m for m in msgs) == 1
    assert sum("once:k2" in m for m in msgs) == 1


def test_reset_reallows(caplog):
    reset_warn_once()
    log = logging.getLogger("t")
    with caplog.at_level(logging.WARNING):
        warn_once(log, "k", "a")
        reset_warn_once()
        warn_once(log, "k", "a")
    assert sum("once:k" in r.message for r in caplog.records) == 2
