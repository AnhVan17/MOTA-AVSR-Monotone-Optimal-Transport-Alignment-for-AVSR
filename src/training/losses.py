"""
Hybrid Loss for MOTA
====================
CTC + CrossEntropy with curriculum learning

FIXED: Correct blank_id configuration for Whisper tokenizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class HybridLoss(nn.Module):
    """
    Hybrid CTC + CrossEntropy Loss
    
    Features:
    - CTC for alignment
    - CE for language model
    - Curriculum weighting
    
    CRITICAL FIX: blank_id placement for Whisper
    """
    
    def __init__(
        self,
        vocab_size: int = 51865,  # Full Whisper vocab
        ctc_weight: float = 0.3,
        ce_weight: float = 0.7,
        pad_id: int = -100,
        blank_id: Optional[int] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight
        self.pad_id = pad_id
        
        # CRITICAL FIX: Use blank_id = vocab_size (outside valid token range)
        #
        # WHY THIS MATTERS:
        # - Whisper tokens: [0, 51864] (51865 total)
        # - Vietnamese content tokens: mostly [0-50256]
        # - Special tokens: [50257-51864]
        # - blank_id = 0 CONFLICTS with valid token 0!
        # - Solution: Place blank at vocab_size (51865) - outside all valid tokens
        
        if blank_id is None:
            self.blank_id = vocab_size  # Place blank outside vocab
        else:
            self.blank_id = blank_id
            
        print("="*60)
        print("Loss Configuration:")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Blank ID: {self.blank_id}")
        print(f"   Pad ID: {pad_id}")
        print(f"   CTC weight: {ctc_weight}")
        print(f"   CE weight: {ce_weight}")
        print("="*60)
        
        # Validate configuration
        if self.blank_id < vocab_size:
            print(f"WARNING: blank_id ({self.blank_id}) overlaps with valid vocab!")
            print(f"   This may cause training issues. Recommended: blank_id = {vocab_size}")
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_id,
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
        input_lengths: Optional[torch.Tensor] = None,  # ADDED: Actual encoder output lengths
        epoch: int = 0,
        max_epochs: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss
        
        Args:
            ctc_logits: [B, T_enc, V+1] from CTC head (V+1 includes blank)
            ar_logits: [B, L, V] from AR decoder (or None)
            targets: [B, L] target token IDs
            target_mask: [B, L] valid target positions
            input_lengths: [B] ACTUAL encoder output lengths (not padded max!)
            epoch: Current epoch (for curriculum)
            max_epochs: Total epochs
            
        Returns:
            dict with total_loss, ctc_loss, ce_loss
        """
        device = ctc_logits.device
        B = ctc_logits.size(0)
        T_enc = ctc_logits.size(1)  # Encoder sequence length
        
        # ============================
        # CTC Loss
        # ============================
        ctc_loss_val = torch.tensor(0.0, device=device)
        
        if ctc_logits is not None:
            # Log probs: [T, B, V+1]
            log_probs = F.log_softmax(ctc_logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)
            
            # CRITICAL FIX: Use actual input lengths if provided
            if input_lengths is not None:
                # Use provided lengths (from audio_len in trainer)
                encoder_input_lengths = input_lengths.clone()
            else:
                # Fallback: assume all sequences are max length (not recommended)
                encoder_input_lengths = torch.full(
                    (B,), T_enc,
                    dtype=torch.long,
                    device=device
                )
            
            # Target lengths (non-padded)
            target_lengths_orig = target_mask.sum(dim=1)
            
            # Validate and filter targets
            targets_list = []
            actual_lengths = []
            valid_input_lengths = []  # Track input lengths for valid samples
            
            for i in range(B):
                L = target_lengths_orig[i].item()
                target_seq = targets[i, :int(L)]
                
                # Filter out invalid tokens:
                # 1. Must be >= 0
                # 2. Must be < vocab_size (exclude blank_id)
                # 3. Must not be pad_id
                valid_mask = (
                    (target_seq >= 0) & 
                    (target_seq < self.vocab_size) & 
                    (target_seq != self.pad_id)
                )
                target_seq = target_seq[valid_mask]
                
                if len(target_seq) == 0:
                    continue
                
                targets_list.append(target_seq)
                actual_lengths.append(len(target_seq))
                valid_input_lengths.append(encoder_input_lengths[i].item())
            
            # DEBUG: Log CTC samples info (first batch only)
            if not hasattr(self, '_logged_ctc_debug'):
                print("\n" + "="*80)
                print("🔍 [CTC LOSS DEBUG] First batch analysis:")
                print("="*80)
                print(f"   Batch size: {B}")
                print(f"   Valid samples for CTC: {len(targets_list)} / {B}")
                
                if len(targets_list) > 0:
                    print(f"   Target lengths (first 5): {actual_lengths[:5]}")
                    print(f"   Input lengths (first 5): {valid_input_lengths[:5]}")
                    print(f"   Input/Target ratio (first 5): {[f'{i/t:.1f}' for i, t in zip(valid_input_lengths[:5], actual_lengths[:5])]}")
                    print(f"   First target tokens: {targets_list[0][:20].tolist()}")
                else:
                    print("   ❌ NO VALID SAMPLES! All targets filtered out!")
                    print(f"   Original target_lengths: {target_lengths_orig.tolist()}")
                    print(f"   First target raw: {targets[0][:30].tolist()}")
                
                print("="*80 + "\n")
                self._logged_ctc_debug = True
            
            # Skip CTC if no valid targets in batch
            if len(targets_list) == 0:
                ctc_loss_val = torch.tensor(0.0, device=device)
            else:
                # Concatenate valid targets
                target_lengths = torch.tensor(actual_lengths, dtype=torch.long, device=device)
                targets_flat = torch.cat(targets_list)
                
                # Get input lengths for valid samples only
                ctc_input_lengths = torch.tensor(valid_input_lengths, dtype=torch.long, device=device)
                
                # CRITICAL CHECK: input_length must be >= target_length for CTC
                if (ctc_input_lengths < target_lengths).any():
                    print("❌ ERROR: Some input_lengths < target_lengths! CTC will fail!")
                    print(f"   Input lengths: {ctc_input_lengths.tolist()}")
                    print(f"   Target lengths: {target_lengths.tolist()}")
                
                try:
                    ctc_loss_val = self.ctc_loss(
                        log_probs[:, :len(targets_list), :],
                        targets_flat,
                        ctc_input_lengths,
                        target_lengths
                    )
                except RuntimeError as e:
                    print(f"CTC loss failed: {e}")
                    ctc_loss_val = torch.tensor(0.0, device=device)
        
        # ============================
        # CE Loss
        # ============================
        ce_loss_val = torch.tensor(0.0, device=device)
        
        if ar_logits is not None:
            # Shift: predict next token
            logits_shifted = ar_logits[:, :-1, :].contiguous()
            targets_shifted = targets[:, 1:].contiguous()
            
            # Flatten
            logits_flat = logits_shifted.view(-1, ar_logits.size(-1))
            targets_flat = targets_shifted.view(-1)
            
            # Clamp invalid targets
            invalid_mask = (targets_flat >= 0) & (targets_flat >= self.vocab_size)
            if invalid_mask.any():
                targets_flat = torch.clamp(targets_flat, max=self.vocab_size - 1)
            
            ce_loss_val = self.ce_loss(logits_flat, targets_flat)
        
        # ============================
        # Curriculum Weighting
        # ============================
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
    vocab_size = config['model']['vocab_size']
    
    # Use blank_id = vocab_size (outside valid token range)
    return HybridLoss(
        vocab_size=vocab_size,
        ctc_weight=config['loss']['ctc_weight'],
        ce_weight=config['loss']['ce_weight'],
        pad_id=-100,
        blank_id=vocab_size  # Place blank outside vocab
    )