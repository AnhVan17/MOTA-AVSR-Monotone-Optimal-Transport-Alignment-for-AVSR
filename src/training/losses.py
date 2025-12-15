"""
Hybrid Loss for AURORA-XT
==========================
CTC + CrossEntropy with curriculum learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class HybridLoss(nn.Module):
    """
    Hybrid CTC + CrossEntropy Loss
    
    Features:
    - CTC for alignment
    - CE for language model
    - Curriculum weighting
    """
    
    def __init__(
        self,
        vocab_size: int = 220,
        ctc_weight: float = 0.3,
        ce_weight: float = 0.7,
        pad_id: int = 0,
        blank_id: int = 4
    ):
        super().__init__()
        
        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight
        self.pad_id = pad_id
        self.blank_id = blank_id
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id,
            reduction='mean',
            zero_infinity=True
        )
        
        # CrossEntropy loss
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            reduction='mean'
        )
    
    def forward(
        self,
        ctc_logits: torch.Tensor,
        ar_logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        epoch: int = 0,
        max_epochs: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss
        
        Args:
            ctc_logits: [B, T_enc, V] from CTC head
            ar_logits: [B, L, V] from AR decoder (or None)
            targets: [B, L] target token IDs
            target_mask: [B, L] valid target positions
            epoch: Current epoch (for curriculum)
            max_epochs: Total epochs
            
        Returns:
            dict with total_loss, ctc_loss, ce_loss
        """
        device = ctc_logits.device
        B = ctc_logits.size(0)
        
        # ============================
        # CTC Loss
        # ============================
        ctc_loss_val = torch.tensor(0.0, device=device)
        
        if ctc_logits is not None:
            # Log probs: [T, B, V]
            log_probs = F.log_softmax(ctc_logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)
            
            # Input lengths (CTC encoder output length)
            input_lengths = torch.full(
                (B,), log_probs.size(0),
                dtype=torch.long,
                device=device
            )
            
            # Target lengths (non-padded)
            target_lengths_orig = target_mask.sum(dim=1)
            
            # Flatten targets (remove padding AND special tokens)
            targets_list = []
            actual_lengths = []
            for i in range(B):
                L = target_lengths_orig[i].item()
                target_seq = targets[i, :int(L)]
                # Remove special tokens (PAD=0, BOS=1, EOS=2, UNK=3, BLANK=4)
                target_seq = target_seq[target_seq >= 5]
                targets_list.append(target_seq)
                actual_lengths.append(len(target_seq))
            
            # Use actual lengths after filtering
            target_lengths = torch.tensor(actual_lengths, dtype=torch.long, device=device)
            targets_flat = torch.cat(targets_list) if targets_list else torch.tensor([], dtype=torch.long, device=device)
            
            # Skip CTC if targets are empty
            if len(targets_flat) == 0:
                ctc_loss_val = torch.tensor(0.0, device=device)
            else:
                try:
                    ctc_loss_val = self.ctc_loss(
                        log_probs,
                        targets_flat,
                        input_lengths,
                        target_lengths
                    )
                except RuntimeError as e:
                    print(f"⚠️ CTC loss failed: {e}")
                    ctc_loss_val = torch.tensor(0.0, device=device)
        
        # ============================
        # CE Loss
        # ============================
        ce_loss_val = torch.tensor(0.0, device=device)
        
        if ar_logits is not None:
            # Shift: predict next token
            # logits[:, :-1] predicts targets[:, 1:]
            logits_shifted = ar_logits[:, :-1, :].contiguous()
            targets_shifted = targets[:, 1:].contiguous()
            
            # Flatten
            logits_flat = logits_shifted.view(-1, ar_logits.size(-1))
            targets_flat = targets_shifted.view(-1)
            
            ce_loss_val = self.ce_loss(logits_flat, targets_flat)
        
        # ============================
        # Curriculum Weighting
        # ============================
        # Start with CTC, gradually increase CE
        progress = min(1.0, epoch / max(1, max_epochs // 2))
        ctc_w = self.ctc_weight * (1 - 0.3 * progress)
        ce_w = self.ce_weight * (1 + 0.3 * progress)
        
        total_loss = ctc_w * ctc_loss_val + ce_w * ce_loss_val
        
        return {
            'total_loss': total_loss,
            'ctc_loss': ctc_loss_val.detach(),
            'ce_loss': ce_loss_val.detach(),
            'ctc_weight': ctc_w,
            'ce_weight': ce_w
        }


# Factory
def create_loss(config: Dict) -> HybridLoss:
    """Create loss from config"""
    return HybridLoss(
        vocab_size=config['model']['vocab_size'],
        ctc_weight=config['loss']['ctc_weight'],
        ce_weight=config['loss']['ce_weight'],
        pad_id=0,
        blank_id=4
    )