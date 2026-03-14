"""
Hybrid Loss for MOTA AVSR
=========================
CTC + CrossEntropy + Length Prediction

🔧 KEY FIXES:
1. Length Predictor: Học độ dài target → chặn CTC babbling
2. EOS Token: Thêm token kết thúc → model biết khi nào dừng
3. Stronger Repetition Penalty: Phạt lặp từ mạnh hơn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class HybridLoss(nn.Module):
    """
    Hybrid CTC + CrossEntropy + Length Prediction Loss
    
    KEY: Thêm Length Predictor để model học khi nào nên dừng
    """
    
    def __init__(
        self,
        vocab_size: int = 51865,
        ctc_weight: float = 0.7,
        ce_weight: float = 0.3,
        length_weight: float = 0.5,  # 🔧 NEW: Weight cho length prediction
        pad_id: int = -100,
        blank_id: Optional[int] = None,
        eos_id: Optional[int] = None  # 🔧 NEW: EOS token
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.ctc_weight = ctc_weight
        self.ce_weight = ce_weight
        self.length_weight = length_weight
        self.pad_id = pad_id
        self.blank_id = blank_id if blank_id is not None else vocab_size
        
        # 🔧 NEW: EOS token (sử dụng blank_id - 1 nếu không chỉ định)
        self.eos_id = eos_id if eos_id is not None else (vocab_size - 1)
        
        print(f"🔧 [Loss] CTC: {ctc_weight}, CE: {ce_weight}, Length: {length_weight}")
        print(f"   Blank ID: {self.blank_id}, EOS ID: {self.eos_id}")
        
        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_id,
            reduction='mean',
            zero_infinity=True
        )
        
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            reduction='mean',
            label_smoothing=0.1
        )
        
        # 🔧 NEW: Length prediction loss (MAE)
        self.length_loss = nn.L1Loss(reduction='mean')
        
        self._debug_logged = False
    
    def forward(
        self,
        ctc_logits: torch.Tensor,
        ar_logits: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None,
        predicted_lengths: Optional[torch.Tensor] = None,  # 🔧 NEW: From model
        epoch: int = 0,
        max_epochs: int = 20
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            ctc_logits: [B, T, V+1]
            ar_logits: [B, L, V] or None
            targets: [B, L]
            target_mask: [B, L]
            input_lengths: [B] - encoder output lengths
            predicted_lengths: [B] - predicted target lengths from model
        """
        device = ctc_logits.device
        B = ctc_logits.size(0)
        T = ctc_logits.size(1)
        
        # ========== 1. CTC Loss ==========
        ctc_loss_val = torch.tensor(0.0, device=device)
        
        if ctc_logits is not None:
            log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # [T, B, V+1]
            
            if input_lengths is not None:
                enc_lens = input_lengths.clone().long()
            else:
                enc_lens = torch.full((B,), T, dtype=torch.long, device=device)
            
            # Target lengths
            tgt_lens = target_mask.sum(dim=1).long()
            
            # Collect valid samples
            valid_targets = []
            valid_tgt_lens = []
            valid_enc_lens = []
            skipped = 0
            
            for i in range(B):
                tgt_len = tgt_lens[i].item()
                enc_len = enc_lens[i].item()
                
                tgt = targets[i, :tgt_len]
                tgt = tgt[(tgt >= 0) & (tgt < self.vocab_size)]
                
                if len(tgt) == 0:
                    skipped += 1
                    continue
                
                if enc_len < len(tgt):
                    skipped += 1
                    continue
                
                valid_targets.append(tgt)
                valid_tgt_lens.append(len(tgt))
                valid_enc_lens.append(enc_len)
            
            if not self._debug_logged and epoch == 0:
                print(f"\n📊 [CTC Loss Debug]")
                print(f"   Valid: {len(valid_targets)}/{B}, Skipped: {skipped}")
                if len(valid_targets) > 0:
                    print(f"   Enc lengths: {valid_enc_lens[:5]}")
                    print(f"   Tgt lengths: {valid_tgt_lens[:5]}")
            
            if len(valid_targets) > 0:
                targets_cat = torch.cat(valid_targets)
                tgt_lens_t = torch.tensor(valid_tgt_lens, dtype=torch.long, device=device)
                enc_lens_t = torch.tensor(valid_enc_lens, dtype=torch.long, device=device)
                
                log_probs_valid = log_probs[:, :len(valid_targets), :]
                
                try:
                    ctc_loss_val = self.ctc_loss(
                        log_probs_valid, targets_cat, enc_lens_t, tgt_lens_t
                    )
                except RuntimeError as e:
                    print(f"❌ CTC error: {e}")
                    ctc_loss_val = torch.tensor(0.0, device=device)
        
        # ========== 2. CE Loss ==========
        ce_loss_val = torch.tensor(0.0, device=device)
        
        if ar_logits is not None:
            logits = ar_logits[:, :-1, :].contiguous()
            tgt = targets[:, 1:].contiguous()
            
            logits_flat = logits.view(-1, ar_logits.size(-1))
            tgt_flat = torch.clamp(tgt.view(-1), min=-100, max=self.vocab_size - 1)
            
            ce_loss_val = self.ce_loss(logits_flat, tgt_flat)
        
        # ========== 🔧 3. LENGTH PREDICTION LOSS (NEW) ==========
        length_loss_val = torch.tensor(0.0, device=device)
        
        if predicted_lengths is not None:
            # True target lengths (số từ không phải padding)
            true_lengths = target_mask.sum(dim=1).float()  # [B]
            
            # Predicted lengths from model
            pred_lengths = predicted_lengths.squeeze(-1)  # [B]
            
            # MAE loss
            length_loss_val = self.length_loss(pred_lengths, true_lengths)
            
            if not self._debug_logged and epoch == 0:
                print(f"\n🔧 [Length Loss Debug]")
                print(f"   True lengths (first 5): {true_lengths[:5].tolist()}")
                print(f"   Pred lengths (first 5): {pred_lengths[:5].tolist()}")
                print(f"   MAE: {length_loss_val.item():.2f}")
        
        # ========== 🔧 4. STRONGER REPETITION PENALTY ==========
        rep_penalty = torch.tensor(0.0, device=device)
        
        if ctc_logits is not None:
            probs = torch.softmax(ctc_logits, dim=-1)  # [B, T, V+1]
            
            # Penalize consecutive same non-blank tokens
            # Method: KL divergence between adjacent distributions
            p_t = probs[:, :-1, :self.blank_id]    # [B, T-1, V] (exclude blank)
            p_t1 = probs[:, 1:, :self.blank_id]    # [B, T-1, V]
            
            # KL(p_t || p_t1) = sum(p_t * log(p_t / p_t1))
            kl_div = F.kl_div(
                p_t1.log(),  # log(q)
                p_t,         # p
                reduction='none'
            ).sum(dim=-1)  # [B, T-1]
            
            # We want HIGH KL (different distributions) → penalize LOW KL
            # So loss = -KL or 1/(KL + eps)
            # Let's use: penalty = exp(-KL) → high when KL is low (similar)
            similarity = torch.exp(-kl_div)  # [B, T-1]
            rep_penalty = similarity.mean() * 2.0  # Weight
            
            if not self._debug_logged and epoch == 0:
                print(f"🔧 [Rep Penalty] Val: {rep_penalty.item():.4f}")
                print(f"   Mean KL divergence: {kl_div.mean().item():.4f}")
        
        # ========== 🔧 5. EOS PREDICTION LOSS (Optional) ==========
        # Encourage model to predict EOS at the END of sequence
        eos_loss = torch.tensor(0.0, device=device)
        
        if ctc_logits is not None and self.eos_id < self.vocab_size:
            # Get probabilities
            probs = torch.softmax(ctc_logits, dim=-1)  # [B, T, V+1]
            
            # For each sample, encourage EOS at position = target_length
            for b in range(B):
                tgt_len = target_mask[b].sum().item()
                if 0 < tgt_len < T:
                    # Encourage high prob for EOS at position tgt_len
                    eos_prob_at_end = probs[b, int(tgt_len), self.eos_id]
                    eos_loss += -torch.log(eos_prob_at_end + 1e-8)
            
            eos_loss = eos_loss / B * 0.1  # Small weight
        
        self._debug_logged = True
        
        # ========== Combine ==========
        total = (self.ctc_weight * ctc_loss_val + 
                 self.ce_weight * ce_loss_val + 
                 self.length_weight * length_loss_val +
                 rep_penalty + 
                 eos_loss)
        
        return {
            'total_loss': total,
            'ctc_loss': ctc_loss_val.detach(),
            'ce_loss': ce_loss_val.detach(),
            'length_loss': length_loss_val.detach(),
            'rep_penalty': rep_penalty.detach(),
            'eos_loss': eos_loss.detach()
        }


def create_loss(config: Dict) -> HybridLoss:
    """Create loss from config"""
    return HybridLoss(
        vocab_size=config['model']['vocab_size'],
        ctc_weight=config['loss']['ctc_weight'],
        ce_weight=config['loss']['ce_weight'],
        length_weight=config['loss'].get('length_weight', 0.5),
        pad_id=-100,
        blank_id=config['model']['vocab_size'],
        eos_id=config['model'].get('eos_id', config['model']['vocab_size'] - 1)
    )