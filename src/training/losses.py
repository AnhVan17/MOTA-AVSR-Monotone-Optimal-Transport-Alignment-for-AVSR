"""
Hybrid Loss for AURORA-XT
==========================
CTC + CrossEntropy + Quality Supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HybridLoss(nn.Module):
    """
    Hybrid CTC + CrossEntropy + Quality Supervision Loss
    
    Features:
    - CTC for alignment
    - CE for language model / decoding
    - Curriculum weighting (CTC -> CE)
    - Quality Supervision (Auxiliary)
    """
    
    def __init__(
        self,
        vocab_size: int = 220,
        ctc_weight: float = 0.3,
        ce_weight: float = 0.7,
        quality_loss_weight: float = 0.1, # Weight for aux loss
        pad_id: int = 0,
        blank_id: int = 4
    ):
        super().__init__()
        
        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight
        self.quality_loss_weight = quality_loss_weight
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
        target_mask: Optional[torch.Tensor] = None,
        epoch: int = 0,
        max_epochs: int = 20,
        transport_map: Optional[torch.Tensor] = None, # [B, Ta, Tv]
        mqot_quality: Optional[torch.Tensor] = None   # [B, Tv]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss
        
        Args:
            ctc_logits: [B, T_enc, V] from CTC head
            ar_logits: [B, L, V] from AR decoder (or None)
            targets: [B, L] target token IDs
            target_mask: [B, L] valid target positions (optional)
            epoch: Current epoch (for curriculum)
            max_epochs: Total epochs
            transport_map: Optimal transport plan from MQOT
            mqot_quality: Predicted quality scores from QualityEstimator
            
        Returns:
            dict with total_loss, ctc_loss, ce_loss, quality_loss
        """
        device = ctc_logits.device
        B = ctc_logits.size(0)
        
        # Handle defaults
        if target_mask is None:
             target_mask = (targets != self.pad_id)

        # ============================
        # 1. CTC Loss
        # ============================
        ctc_loss_val = torch.tensor(0.0, device=device)
        
        if ctc_logits is not None:
            # Log probs: [T, B, V] required for CTC
            log_probs = F.log_softmax(ctc_logits, dim=-1)
            log_probs = log_probs.transpose(0, 1) # [T_enc, B, V]
            
            # Input lengths: Assume full length for now (or pass via batch if variable)
            # Robust checking: shape 0 of log_probs is T_enc
            input_lengths = torch.full(
                (B,), log_probs.size(0),
                dtype=torch.long,
                device=device
            )
            
            # Target lengths calculation
            # Remove padding and special tokens logic
            targets_list = []
            actual_lengths = []
            
            # Pre-calculate lengths from mask
            raw_lengths = target_mask.sum(dim=1).long()
            
            targets_flat_list = []
            
            for i in range(B):
                L = raw_lengths[i].item()
                # Slice valid target
                target_seq = targets[i, :L]
                # Filter special tokens for CTC
                # CRITICAL FIX for Whisper: 
                # Blank ID is 50257 (EOT). Targets MUST NOT contain blank_id.
                # Also filter pad_id.
                # We do NOT filter by range < 50257 because valid text tokens can be > 50257 in some vocabs.
                valid_mask = (target_seq != self.blank_id) & (target_seq != self.pad_id)
                
                # Optional: Filter SOT if known? (50258). 
                # For now, just ensuring Blank is gone is the most critical fix.
                valid_tokens = target_seq[valid_mask]
                
                targets_list.append(valid_tokens)
                actual_lengths.append(len(valid_tokens))
                targets_flat_list.append(valid_tokens)
            
            # Prepare CTC tensors
            target_lengths = torch.tensor(actual_lengths, dtype=torch.long, device=device)
            if targets_flat_list:
                targets_flat = torch.cat(targets_flat_list)
            else:
                targets_flat = torch.tensor([], dtype=torch.long, device=device)
            
            # Compute CTC
            # Check for empty targets to avoid NaN
            if len(targets_flat) > 0 and target_lengths.sum() > 0:
                ctc_loss_val = self.ctc_loss(
                    log_probs,
                    targets_flat,
                    input_lengths,
                    target_lengths
                )
            else:
                # Fallback if no valid text (e.g. empty audio)
                ctc_loss_val = torch.tensor(0.0, device=device)
        
        # ============================
        # 2. CE Loss
        # ============================
        ce_loss_val = torch.tensor(0.0, device=device)
        
        if ar_logits is not None:
            # Shift: predict next token logic
            # logits[:, :-1] predicts targets[:, 1:]
            
            # We assume logits are [B, L_out, V] and targets are [B, L_tgt]
            # Usually aligned or Teacher Forcing. 
            # Assuming standard AR: Output L matches Target L.
            
            # Slicing for Next Token Prediction
            if ar_logits.size(1) == targets.size(1):
                 logits_shifted = ar_logits[:, :-1, :].contiguous()
                 targets_shifted = targets[:, 1:].contiguous()
            else:
                 # Fallback/Mismatch shape handling or direct
                 # Assume user logic ensures shape alignment
                 logits_shifted = ar_logits.transpose(1, 2) # [B, V, L] for CE if needed?
                 # No, CE expects [B, V, d1...] or [B, C] for simple
                 # Flattening strategy is safest
                 logits_shifted = ar_logits[:, :-1, :].contiguous()
                 targets_shifted = targets[:, 1:].contiguous()

            # Flatten
            logits_flat = logits_shifted.view(-1, ar_logits.size(-1))
            targets_flat = targets_shifted.view(-1)
            
            ce_loss_val = self.ce_loss(logits_flat, targets_flat)
        
        # ============================
        # 3. Quality Supervision (Aux)
        # ============================
        quality_loss_val = torch.tensor(0.0, device=device)
        
        if self.quality_loss_weight > 0 and transport_map is not None and mqot_quality is not None:
            # Proxy Target: Entropy of Transport Map
            # transport_map P: [B, Ta, Tv]
            # Normalize along audio axis (how much attention each visual frame gets)
            # P is usually normalized by rows/cols in Sinkhorn, but let's re-norm col-wise
            P_col = F.normalize(transport_map, p=1, dim=1) + 1e-8 # [B, Ta, Tv]
            
            # Entropy per visual frame j: H(j) = -sum(P_ij * log P_ij) over i
            entropy = -torch.sum(P_col * torch.log(P_col), dim=1) # [B, Tv]
            
            # Max possible entropy is log(Ta)
            max_ent = torch.log(torch.tensor(transport_map.size(1), dtype=torch.float, device=device))
            
            # Sharpness (Quality Proxy) = 1.0 - (Entropy / MaxEntropy)
            # Low Entropy -> High Sharpness -> High Quality
            target_quality = 1.0 - (entropy / (max_ent + 1e-8))
            target_quality = torch.clamp(target_quality, 0.0, 1.0).detach() # Don't backprop to P
            
            # MSE Loss
            quality_loss_val = F.mse_loss(mqot_quality, target_quality)

        # ============================
        # 4. Total Loss & Curriculum
        # ============================
        
        # Fixed Weights (Disabled Curriculum for Stability)
        ctc_w = self.ctc_weight
        ce_w = self.ce_weight
        
        total_loss = (ctc_w * ctc_loss_val) + (ce_w * ce_loss_val) + (self.quality_loss_weight * quality_loss_val)
        
        return {
            'total_loss': total_loss,
            'ctc_loss': ctc_loss_val.detach(),
            'ce_loss': ce_loss_val.detach(),
            'quality_loss': quality_loss_val.detach(),
            'ctc_weight': torch.tensor(ctc_w),
            'ce_weight': torch.tensor(ce_w)
        }


# Factory
def create_loss(config: Dict) -> HybridLoss:
    """Create loss from config"""
    return HybridLoss(
        vocab_size=config['model']['vocab_size'],
        ctc_weight=config['loss']['ctc_weight'],
        ce_weight=config['loss']['ce_weight'],
        quality_loss_weight=config['loss'].get('quality_loss_weight', 0.0), # Default 0
        pad_id=0,
        blank_id=config['model'].get('blank_id', 50257) # Default to EOT if missing
    )