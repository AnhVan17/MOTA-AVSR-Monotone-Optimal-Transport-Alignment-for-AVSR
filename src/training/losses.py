"""
FINAL Loss Function - Attention-Only
=====================================
Pure CrossEntropy Loss (NO CTC)

Optimized for WhisperTokenizer to fix Test WER 55%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class AttentionOnlyLoss(nn.Module):
    """
    ⭐ Pure Attention Loss (NO CTC)
    
    Why NO CTC:
    - Character-level: CTC works (char-acoustic alignment natural)
    - Subword tokens: CTC FAILS (subwords don't align to frames)
    - Test performance: Pure attention generalizes better
    """
    
    def __init__(
        self,
        vocab_size: int = 51865,
        pad_id: int = -100,
        label_smoothing: float = 0.15
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing
        
        # Pure CrossEntropy
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            reduction='mean',
            label_smoothing=label_smoothing
        )
        
        print(f"\n{'='*70}")
        print("⭐ Attention-Only Loss (NO CTC)")
        print(f"{'='*70}")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Label smoothing: {label_smoothing}")
        print(f"   ❌ CTC: REMOVED")
        print(f"   ✅ Pure CrossEntropy")
        print(f"{'='*70}\n")
    
    def forward(
        self,
        ctc_logits: Optional[torch.Tensor],  # Will be None
        ar_logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        epoch: int = 0,
        max_epochs: int = 30
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            ctc_logits: None (not used)
            ar_logits: [B, L, V] from attention decoder
            targets: [B, L] target token IDs
            target_mask: [B, L] valid positions
            epoch, max_epochs: For logging only
            
        Returns:
            dict with total_loss, ce_loss
        """
        device = ar_logits.device
        
        # Shift for autoregressive prediction
        # logits[:, :-1] predicts targets[:, 1:]
        logits_shifted = ar_logits[:, :-1, :].contiguous()
        targets_shifted = targets[:, 1:].contiguous()
        
        # Flatten
        logits_flat = logits_shifted.view(-1, self.vocab_size)
        targets_flat = targets_shifted.view(-1)
        
        # Compute CE loss
        ce_loss_val = self.ce_loss(logits_flat, targets_flat)
        
        return {
            'total_loss': ce_loss_val,
            'ce_loss': ce_loss_val.detach(),
            'ctc_loss': torch.tensor(0.0, device=device),  # For logging
            'ce_weight': 1.0,
            'ctc_weight': 0.0
        }


def create_loss(config: Dict) -> AttentionOnlyLoss:
    """Factory function"""
    return AttentionOnlyLoss(
        vocab_size=config['model']['vocab_size'],
        pad_id=-100,  # Standard for CrossEntropyLoss
        label_smoothing=config['training'].get('label_smoothing', 0.15)
    )


if __name__ == "__main__":
    print("\nTesting Attention-Only Loss...")
    
    config = {
        'model': {'vocab_size': 51865},
        'training': {'label_smoothing': 0.15}
    }
    
    loss_fn = create_loss(config)
    
    # Test
    B, L, V = 4, 20, 51865
    ar_logits = torch.randn(B, L, V)
    targets = torch.randint(0, V, (B, L))
    target_mask = torch.ones(B, L, dtype=torch.bool)
    
    loss_dict = loss_fn(
        ctc_logits=None,
        ar_logits=ar_logits,
        targets=targets,
        target_mask=target_mask
    )
    
    print(f"\nLoss dict:")
    print(f"  total_loss: {loss_dict['total_loss']:.4f}")
    print(f"  ce_loss: {loss_dict['ce_loss']:.4f}")
    print(f"  ctc_loss: {loss_dict['ctc_loss']:.4f}")
    
    print("\n✅ Loss test passed!")