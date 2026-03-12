import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step: int,
    best_metric: float,
    checkpoint_dir: str,
    filename: str = "checkpoint.pt"
):
    """Save training checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_metric': best_metric
    }
    
    torch.save(checkpoint, checkpoint_dir / filename)
    logger.info(f"💾 Checkpoint saved: {checkpoint_dir / filename}")
    
    # Cleanup (Fix 0.9.2): Keep only last 5 epochs
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split('_')[1]))
    if len(checkpoints) > 5:
        for old_ckpt in checkpoints[:-5]:
            try:
                old_ckpt.unlink()
                logger.info(f"Deleted old checkpoint: {old_ckpt.name}")
            except OSError as e:
                logger.warning(f"Failed to delete {old_ckpt.name}: {e}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None
) -> Dict[str, Any]:
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f" Checkpoint loaded from: {checkpoint_path}")
    logger.info(f"   Epoch: {checkpoint['epoch']}, Step: {checkpoint['step']}")
    
    return checkpoint


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0) -> float:
    """
    Compute token-level accuracy
    
    Args:
        logits: [B, L_logits, vocab_size]
        targets: [B, L_target]
        ignore_index: Padding token to ignore
    
    Returns:
        accuracy: float (0-100)
    """
    B, L_logits, V = logits.shape
    L_target = targets.size(1)
    
    min_len = min(L_logits, L_target)
    logits = logits[:, :min_len, :]
    targets = targets[:, :min_len]
    
    predictions = logits.argmax(dim=-1)  
    
    mask = targets != ignore_index
    
    if mask.sum() == 0:
        return 0.0
    correct = (predictions == targets) & mask
    accuracy = correct.sum().item() / mask.sum().item() * 100
    
    return accuracy


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def format_time(seconds: float) -> str:
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class EarlyStopping:
    """Early stopping to stop training when metric stops improving"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: How many checks to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if should stop training
        
        Args:
            score: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def state_dict(self):
        """Get state for saving"""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch
        }
    
    def load_state_dict(self, state):
        """Load state from checkpoint"""
        self.counter = state['counter']
        self.best_score = state['best_score']
        self.best_epoch = state['best_epoch']
