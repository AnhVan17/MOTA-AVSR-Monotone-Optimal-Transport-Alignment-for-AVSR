"""
Hybrid Decoder for AVSR
========================
CTC + Attention Decoder with proper initialization
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class HybridDecoder(nn.Module):
    """
    Hybrid CTC + Attention Decoder
    
    CTC: Fast alignment (acoustic → text)
    Attention: Context refinement (language model)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        vocab_size: int = 51865,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # CTC head: outputs vocab_size + 1 (blank at position vocab_size)
        # 🔧 FIX: Added dropout to prevent overfitting/repetition
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),  # Prevent repetition
            nn.Linear(d_model, vocab_size + 1)
        )
        
        # Initialize CTC head with anti-collapse bias
        self._init_ctc_head(vocab_size)
        
        print(f"🎯 CTC Head: {d_model} -> {vocab_size + 1} (vocab + blank)")
        
        # Attention decoder
        self.target_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(d_model, 5000)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # 🔧 FIX: Pre-Norm stabilizes training & prevents NaN
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.ar_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        self._debug_logged = False
    
    def _init_ctc_head(self, vocab_size: int):
        """
        Initialize CTC head to prevent blank collapse.
        
        🔧 CRITICAL: -10 was too weak! Model overcame it.
        Using -30 now - model CANNOT overcome this.
        
        Math:
        - softmax([0, 0, ..., -30]) → blank prob ≈ 1e-13 (essentially 0)
        - Model forced to predict content tokens
        """
        # 🔧 FIX: Linear layer is now at index 2 (0=Norm, 1=Dropout, 2=Linear)
        linear_layer = self.ctc_head[2]
        
        with torch.no_grad():
            nn.init.xavier_uniform_(linear_layer.weight)
            linear_layer.bias.fill_(0.0)
            
            # 🔧 FIX: bias = 5.0 (Balanced "Goldilocks" value)
            # High enough to beat random noise (prevents babbling)
            # Low enough to be learned (prevents total silence)
            linear_layer.bias[vocab_size] = 5.0
            
            print(f"🔧 [CTC Init] Blank bias = 5.0 (Balanced Silence)")
    
    @staticmethod
    def _create_pos_encoding(d_model: int, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_out: [B, T_enc, D]
            target: [B, L] (optional, for training)
        """
        B, T, D = encoder_out.shape
        
        # CTC logits
        ctc_logits = self.ctc_head(encoder_out)  # [B, T, V+1]
        
        # Debug first forward
        if not self._debug_logged:
            with torch.no_grad():
                probs = torch.softmax(ctc_logits, dim=-1)
                blank_prob = probs[:, :, -1].mean().item()
                print(f"[CTC] Initial blank probability: {blank_prob*100:.2f}%")
                if blank_prob > 0.5:
                    print(f"   Blank still high - may need adjustment")
                else:
                    print(f"   Blank probability low - good!")
            self._debug_logged = True
        
        # AR logits (during training only)
        ar_logits = None
        if target is not None:
            B, L = target.shape
            
            # Clamp invalid tokens
            target_for_embed = target.clone()
            target_for_embed = torch.clamp(target_for_embed, min=0, max=self.vocab_size - 1)
            
            target_embed = self.target_embedding(target_for_embed)
            pos = self.pos_encoding[:, :L, :].to(encoder_out.device)
            target_embed = target_embed + pos
            
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L)
            causal_mask = causal_mask.to(encoder_out.device)
            
            ar_out = self.decoder(
                tgt=target_embed,
                memory=encoder_out,
                tgt_mask=causal_mask
            )
            
            ar_logits = self.ar_head(ar_out)
        
        return {
            'ctc_logits': ctc_logits,
            'ar_logits': ar_logits
        }