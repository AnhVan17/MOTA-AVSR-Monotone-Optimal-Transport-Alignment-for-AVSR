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

from src.utils.warn_once import warn_once

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
        lambda_bal: float = 0.0, # RGF load-balancing weight (Switch Transformer); 0 = off
        pad_id: int = 50257,
        blank_id: int = 50257,
        special_ids: Optional[list] = None,  # token điều khiển cần lọc khỏi CTC target
        label_smoothing: float = 0.0,
    ):
        super().__init__()

        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight
        self.quality_loss_weight = quality_loss_weight
        self.lambda_bal = lambda_bal
        self.pad_id = pad_id
        self.blank_id = blank_id
        self.sot_id = 50258  # Whisper Start-of-Transcript token (fallback khi không có special_ids)
        self.special_ids = special_ids
        logger.info(f"HybridLoss: ctc_w={ctc_weight} ce_w={ce_weight} label_smoothing={label_smoothing}")

        # CTC loss
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id,
            reduction='mean',
            zero_infinity=True
        )

        # CrossEntropy loss — ignore theo VỊ TRÍ (target_mask → -100), không theo giá trị pad_id.
        # Lý do: pad_id == eot_id (50257) ở Whisper; nếu ignore theo giá trị thì EOT thật cũng bị
        # bỏ → model không học được token dừng. Mask theo vị trí giữ EOT, chỉ bỏ padding (idiom HF).
        # label_smoothing (0.1 chuẩn ASR Transformer): giảm overconfident, lợi khi train có nhiễu.
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=-100,
            reduction='mean',
            label_smoothing=label_smoothing,
        )

    def _resolve_target_mask(
        self,
        targets: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a valid-position mask that keeps the first EOT/pad token."""
        if target_mask is not None:
            return target_mask

        warn_once(logger, "ce_no_target_mask",
                  "target_mask=None → tự suy ra mask từ pad_id (Collator nên truyền sẵn).")
        is_pad = (targets == self.pad_id)
        seq_len = targets.size(1)
        first_pad = torch.where(
            is_pad.any(dim=1),
            is_pad.float().argmax(dim=1),                          # vị trí EOT đầu tiên
            torch.full((targets.size(0),), seq_len, device=targets.device),
        )
        positions = torch.arange(seq_len, device=targets.device).unsqueeze(0)
        return positions <= first_pad.unsqueeze(1)                  # gồm EOT đầu tiên

    def compute_ctc_only(
        self,
        ctc_logits: Optional[torch.Tensor],
        targets: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute a standalone CTC loss for any encoder timeline.

        Used by the main audio-timeline CTC and by visual bootstrap CTC. The target filtering is
        intentionally shared so both paths learn from the same non-control transcript tokens.
        """
        device = ctc_logits.device if ctc_logits is not None else targets.device
        if ctc_logits is None:
            return {
                'loss': torch.tensor(0.0, device=device),
                'empty': torch.tensor(0.0, device=device),
            }

        B = ctc_logits.size(0)
        target_mask = self._resolve_target_mask(targets, target_mask)

        # Log probs: [T, B, V] required for CTC
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)

        # Input lengths cho CTC. Ưu tiên độ dài THẬT của timeline đang giám sát; nếu không có
        # → fallback full padded length + cảnh báo 1 lần.
        if input_lengths is not None:
            input_lengths = input_lengths.to(device).long().clamp(max=log_probs.size(0))
        else:
            warn_once(logger, "ctc_input_lengths",
                      "input_lengths=None → dùng full padded length cho CTC (kém chuẩn; "
                      "trainer nên truyền *_mask.sum).")
            input_lengths = torch.full((B,), log_probs.size(0), dtype=torch.long, device=device)

        raw_lengths = target_mask.sum(dim=1).long()
        targets_flat_list = []
        actual_lengths = []

        # Dựng 1 lần: tập token điều khiển cần lọc khỏi CTC target.
        special_tensor = (
            torch.tensor(self.special_ids, device=device)
            if self.special_ids is not None else None
        )

        for i in range(B):
            L = raw_lengths[i].item()
            target_seq = targets[i, :L]
            if special_tensor is not None:
                valid_mask = ~torch.isin(target_seq, special_tensor)
            else:
                valid_mask = (
                    (target_seq != self.blank_id) &
                    (target_seq != self.pad_id) &
                    (target_seq != self.sot_id)
                )
            valid_tokens = target_seq[valid_mask]
            actual_lengths.append(len(valid_tokens))
            targets_flat_list.append(valid_tokens)

        target_lengths = torch.tensor(actual_lengths, dtype=torch.long, device=device)
        targets_flat = (
            torch.cat(targets_flat_list)
            if targets_flat_list else torch.tensor([], dtype=torch.long, device=device)
        )

        if len(targets_flat) > 0 and target_lengths.sum() > 0:
            loss = self.ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)
            empty = torch.tensor(0.0, device=device)
        else:
            warn_once(logger, "ctc_empty_target",
                      "CTC gặp batch không còn token hợp lệ sau lọc special → ctc_loss=0 "
                      "(kiểm tra dữ liệu/tokenizer nếu lặp lại).")
            loss = torch.tensor(0.0, device=device)
            empty = torch.tensor(1.0, device=device)

        return {'loss': loss, 'empty': empty}
    
    def forward(
        self,
        ctc_logits: torch.Tensor,
        ar_logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,  # [B] số frame audio THẬT cho CTC
        epoch: int = 0,
        max_epochs: int = 20,
        transport_map: Optional[torch.Tensor] = None, # [B, Ta, Tv]
        mqot_quality: Optional[torch.Tensor] = None,  # [B, Tv]
        router_probs: Optional[torch.Tensor] = None   # [B, n, 3] from RGF (load-balancing)
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
        
        # Handle defaults — Collator thường truyền target_mask theo độ dài (gồm cả EOT).
        # Fallback khi None: GIỮ tới (và gồm) pad_id ĐẦU TIÊN (= EOT thật), bỏ padding sau đó.
        # Không dùng (targets != pad_id) vì pad_id == eot_id sẽ loại nhầm EOT.
        target_mask = self._resolve_target_mask(targets, target_mask)

        # ============================
        # 1. CTC Loss
        # ============================
        ctc = self.compute_ctc_only(ctc_logits, targets, target_mask, input_lengths)
        ctc_loss_val = ctc['loss']
        
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
            
            # Shift for next-token prediction: logits[:, t] predicts targets[:, t+1]
            logits_shifted = ar_logits[:, :-1, :].contiguous()
            targets_shifted = targets[:, 1:].contiguous()

            # Ignore theo VỊ TRÍ: chỉ bỏ padding (target_mask==0), GIỮ EOT (token thật).
            # Tránh việc ignore theo giá trị pad_id==eot_id làm model không học token dừng.
            if target_mask is not None:
                mask_shifted = target_mask[:, 1:].contiguous()
                targets_shifted = targets_shifted.masked_fill(mask_shifted == 0, -100)

            # Flatten: [B*L, V] for CrossEntropy (ignore_index=-100)
            logits_flat = logits_shifted.view(-1, ar_logits.size(-1))
            ce_targets_flat = targets_shifted.view(-1)

            ce_loss_val = self.ce_loss(logits_flat, ce_targets_flat)
        
        # ============================
        # 3. Quality Supervision (Aux)
        # ============================
        quality_loss_val = torch.tensor(0.0, device=device)
        
        if self.quality_loss_weight > 0 and transport_map is not None and mqot_quality is not None:
            # MQOT return [B, H, Ta, Tv]. combine heads and take mean: [B, Ta, Tv] -> dim=1 for audio normalization
            if transport_map.dim() == 4:
                transport_map = transport_map.mean(dim=1) # [B, Ta, Tv]
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
        
        # Load-balancing (Switch Transformer, Fedus 2021): keep RGF's 3 modes used ~evenly across
        # the batch → prevents global mode-collapse (router always picking one expert).
        bal_loss_val = torch.tensor(0.0, device=device)
        if router_probs is not None and self.lambda_bal > 0:
            n_exp = router_probs.size(-1)
            f_i = F.one_hot(router_probs.argmax(dim=-1), n_exp).to(router_probs.dtype).mean(dim=(0, 1))
            p_i = router_probs.mean(dim=(0, 1))
            bal_loss_val = self.lambda_bal * n_exp * (f_i * p_i).sum()

        # Fixed Weights (Disabled Curriculum for Stability)
        ctc_w = self.ctc_weight
        ce_w = self.ce_weight
        
        total_loss = (ctc_w * ctc_loss_val) + (ce_w * ce_loss_val) + (self.quality_loss_weight * quality_loss_val) + bal_loss_val
        
        return {
            'total_loss': total_loss,
            'ctc_loss': ctc_loss_val.detach(),
            'ce_loss': ce_loss_val.detach(),
            'quality_loss': quality_loss_val.detach(),
            'bal_loss': bal_loss_val.detach(),
            'ctc_empty': ctc['empty'].detach(),
            'ctc_weight': torch.tensor(ctc_w),
            'ce_weight': torch.tensor(ce_w)
        }


# Factory
def create_loss(config: Dict, tokenizer=None) -> HybridLoss:
    """Create loss from config.

    Nếu truyền tokenizer: derive pad/blank/special_ids TỪ tokenizer (không hardcode 50257).
    Nếu không: fallback về giá trị trong config (backward-compatible).
    """
    if tokenizer is not None:
        pad_id = tokenizer.pad_token_id
        blank_id = tokenizer.eot_token_id      # blank CTC = EOT (derive)
        special_ids = tokenizer.all_special_ids
    else:
        pad_id = config['model'].get('pad_id', 50257)
        blank_id = config['model'].get('blank_id', 50257)
        special_ids = None

    return HybridLoss(
        vocab_size=config['model']['vocab_size'],
        ctc_weight=config['loss']['ctc_weight'],
        ce_weight=config['loss']['ce_weight'],
        quality_loss_weight=config['loss'].get('quality_loss_weight', 0.0),
        lambda_bal=config.get('rgf', {}).get('lambda_bal', 0.0),
        pad_id=pad_id,
        blank_id=blank_id,
        special_ids=special_ids,
        label_smoothing=config['loss'].get('label_smoothing', 0.0),
    )
