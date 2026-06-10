"""Unit tests for device selection helpers (src/utils/device.py)."""
import torch

from src.utils.device import get_device, supports_amp


def test_get_device_prefer_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_autodetect_returns_valid_device():
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in {"cuda", "mps", "cpu"}


def test_supports_amp_only_cuda():
    # torch.device("cuda") is a descriptor; supports_amp only checks .type, no CUDA needed.
    assert supports_amp(torch.device("cuda")) is True
    assert supports_amp(torch.device("cpu")) is False
    assert supports_amp(torch.device("mps")) is False
