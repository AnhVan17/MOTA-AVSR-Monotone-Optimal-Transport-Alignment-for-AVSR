"""Device selection — auto-detect accelerator cho cả local lẫn cloud.

Thứ tự ưu tiên: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU.
Dùng chung cho training/inference/smoke-test để code device-agnostic.
"""
from typing import Optional

import torch


def get_device(prefer: Optional[str] = None) -> torch.device:
    """Trả về device tốt nhất hiện có.

    Args:
        prefer: ép thủ công ("cuda" | "mps" | "cpu"). Hữu ích khi smoke-test.
                Nếu None → tự dò.

    Returns:
        torch.device theo thứ tự CUDA → MPS → CPU.
    """
    if prefer:
        return torch.device(prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # MPS = Apple Metal (Mac M-series). Kiểm tra cả tồn tại lẫn build hỗ trợ.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def supports_amp(device: torch.device) -> bool:
    """AMP (mixed precision) hiện chỉ ổn định trên CUDA.

    MPS/CPU chưa hỗ trợ mixed-precision GradScaler ổn định → tránh bật để khỏi lỗi/giảm tốc.
    """
    return device.type == "cuda"


def make_grad_scaler(device: torch.device, enabled: bool):
    """GradScaler tương thích nhiều phiên bản torch.

    API hợp nhất ``torch.amp.GradScaler`` chỉ có từ torch 2.3. Các build cũ hơn
    (vd image cloud ghim torch 2.1.2) chỉ expose ``torch.cuda.amp.GradScaler``.
    """
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)
