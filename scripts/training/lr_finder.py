"""
LR Range Test — tìm optimal learning rate cho MOTA.

Chạy 1 epoch với LR tăng dần từ min_lr → max_lr.
Plot loss theo LR → chọn LR ở "elbow" (nơi loss bắt đầu giảm mạnh).
Output: optimal_lr, loss_curve.png

Usage:
    python scripts/training/lr_finder.py --config configs/phase1_base.yaml --max_lr 1e-2 --num_batches 100
"""
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.mota import create_model
from src.data.loader import build_dataloader
from src.data.tokenizers.whisper import WhisperTokenizer
from src.training.losses import create_loss


def find_optimal_lr(
    config: dict,
    min_lr: float = 1e-7,
    max_lr: float = 1e-2,
    num_batches: int = 100,
    save_path: str = "lr_curve.png",
):
    """
    LR Range Test (Leslie Smith, 2018).

    Args:
        min_lr: Start learning rate
        max_lr: End learning rate (after num_batches)
        num_batches: Number of batches to test
        save_path: Path to save plot

    Returns:
        optimal_lr: Best learning rate (at minimum loss slope)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Setup ---
    print("Building model...")
    model = create_model(config['model']).to(device)
    criterion = create_loss(config).to(device)

    # Get tokenizer from dataloader (avoid double init)
    print("Building dataloader...")
    tok = WhisperTokenizer(model="openai/whisper-small", language="vi")
    train_loader = build_dataloader(config, tokenizer=tok, mode='train')

    # --- LR Schedule ---
    # Exponential LR increase: lr(i) = min_lr * (max_lr/min_lr)^(i/num_batches)
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_batches)

    # --- Training Loop ---
    model.train()
    losses = []

    # Use a single optimizer with a dummy LR; we'll set LR manually per step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)

    print(f"Running LR range test: {num_batches} batches, [{min_lr:.2e}, {max_lr:.2e}]")
    iterator = iter(train_loader)

    for step, lr in enumerate(tqdm(lrs, desc="LR Test")):
        try:
            batch = next(iterator)
        except StopIteration:
            break

        # Set LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        targets = batch['target'].to(device)
        target_mask = batch.get('target_mask', None)
        if target_mask is not None:
            target_mask = target_mask.to(device)

        optimizer.zero_grad()
        outputs = model(audio, visual, targets)
        loss_dict = criterion(
            outputs['ctc_logits'],
            outputs['ar_logits'],
            targets,
            target_mask,
            epoch=0,
            max_epochs=1,
        )
        loss = loss_dict['total_loss']
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step + 1 >= num_batches:
            break

    losses = np.array(losses[:len(lrs)])

    # --- Plot ---
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses)
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("LR Range Test — MOTA AVSR")
    plt.grid(True, alpha=0.3)

    # Mark optimal region (largest loss decrease)
    # Heuristic: LR where |gradient of log(loss)| is maximum
    log_loss = np.log(losses + 1e-8)
    loss_slope = np.gradient(log_loss, np.log10(lrs))

    # Find elbow: minimum of smoothed slope
    from scipy.ndimage import uniform_filter1d
    slope_smooth = uniform_filter1d(loss_slope, size=5, mode='nearest')
    # Skip first 10% (unstable) and last 10% (divergence)
    skip = int(len(slope_smooth) * 0.1)
    eligible = slope_smooth[skip:-skip]
    best_idx = skip + np.argmin(eligible)

    optimal_lr = lrs[best_idx]
    plt.axvline(optimal_lr, color='r', linestyle='--', label=f"optimal_lr={optimal_lr:.2e}")
    plt.legend()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
    print(f"Optimal LR: {optimal_lr:.2e}")

    # --- Suggest config ---
    print()
    print("=== Suggested Config Update ===")
    print(f"  learning_rate: {optimal_lr:.6f}  # ~{optimal_lr:.1e}")
    print(f"  # Suggested warmup_steps based on dataset size:")
    dataset_size = len(train_loader.dataset)
    print(f"  #   ~{max(100, dataset_size // 10)} (10% of dataset)")

    return optimal_lr


def main():
    parser = argparse.ArgumentParser(description="LR Range Test for MOTA")
    parser.add_argument("--config", default="configs/phase1_base.yaml")
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--max_lr", type=float, default=1e-2)
    parser.add_argument("--num_batches", type=int, default=100)
    parser.add_argument("--save", default="lr_curve.png")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Limit batches for speed
    if 'data' not in config:
        config['data'] = {}
    config['data']['max_samples'] = args.num_batches * 4  # ~4 samples/batch

    find_optimal_lr(
        config,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_batches=args.num_batches,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
