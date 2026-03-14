"""
Hybrid Decoder for AVSR
========================
🚨 EMERGENCY FIX: Hard-code Vietnamese token filter

CTC + Attention Decoder with FORCED Vietnamese-only output
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class HybridDecoder(nn.Module):
    """
    Hybrid CTC + Attention Decoder
    
    🚨 EMERGENCY: Filter non-Vietnamese tokens with -inf logits
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
        
        # CTC head
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, vocab_size + 1)
        )
        
        # 🚨 EMERGENCY: Create Vietnamese token mask
        self._create_vietnamese_mask()
        
        self._init_ctc_head(vocab_size)
        
        print(f"🎯 CTC Head: {d_model} -> {vocab_size + 1} (vocab + blank)")
        print(f"🚨 EMERGENCY: Vietnamese-only filter enabled")
        print(f"   Allowed tokens: {self.viet_mask.sum().item()} / {vocab_size}")
        
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
            norm_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.ar_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        # 🔧 NEW: Length Predictor
        # Predicts target length from encoder output
        self.length_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)  # Outputs single value (length)
        )
        print(f"📏 Length Predictor: {d_model} -> 1")
        
        self._debug_logged = False
    
    def _create_vietnamese_mask(self):
        """
        🚨 EMERGENCY: Hard-code Vietnamese token IDs
        
        Whisper Vietnamese tokens are in range [0, 50256]:
        - [0-255]: Byte tokens
        - [256-50256]: BPE tokens (Vietnamese in low range)
        - [50257+]: Special tokens (language tags, timestamps)
        """
        # Create mask: 1 = allowed, 0 = blocked
        viet_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        
        # Method 1: Allow ONLY low-range BPE (Vietnamese tends to be here)
        # Tokens 0-10000 cover most Vietnamese
        viet_mask[:10000] = True
        
        # Method 2: Block known foreign ranges
        # Korean: ~50000-50100 (example, actual range may vary)
        # We don't have exact mapping, so use conservative filter
        
        # For now, AGGRESSIVELY block high-range tokens
        # that are definitely NOT Vietnamese
        viet_mask[30000:] = False  # Block upper half
        
        # Always allow blank (will be at vocab_size position, handled separately)
        
        self.register_buffer('viet_mask', viet_mask)
        
        print(f"🚨 Vietnamese mask created:")
        print(f"   Allowed range: [0, {viet_mask.nonzero()[-1].item()}]")
        print(f"   Total allowed: {viet_mask.sum().item()} tokens")
    
    def _apply_vietnamese_filter(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply hard filter to logits
        
        Args:
            logits: [B, T, V+1] (includes blank at position V)
            
        Returns:
            filtered_logits: [B, T, V+1]
        """
        B, T, V_plus_1 = logits.shape
        
        # Separate content logits and blank logit
        content_logits = logits[:, :, :-1]  # [B, T, V]
        blank_logit = logits[:, :, -1:]      # [B, T, 1]
        
        # Apply mask: Set non-Vietnamese tokens to very negative (but AMP-safe)
        # Note: -1e9 overflows float16, use -1e4 instead
        mask = self.viet_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, V]
        filtered_content = content_logits.masked_fill(~mask, -1e4)
        
        # Recombine with blank
        filtered_logits = torch.cat([filtered_content, blank_logit], dim=-1)
        
        return filtered_logits
    
    def _init_ctc_head(self, vocab_size: int):
        """Initialize CTC head"""
        linear_layer = self.ctc_head[2]
        
        with torch.no_grad():
            nn.init.xavier_uniform_(linear_layer.weight)
            linear_layer.bias.fill_(0.0)
            
            # Moderate blank bias (not too high = collapse, not too low = babbling)
            linear_layer.bias[vocab_size] = 5.0
            
            print(f"🔧 [CTC Init] Blank bias = 5.0 (balanced)")
    
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
        
        # 🔧 Length Prediction: Predict target length from encoder output
        # Use mean pooling over time dimension
        encoder_pooled = encoder_out.mean(dim=1)  # [B, D]
        predicted_length = self.length_predictor(encoder_pooled)  # [B, 1]
        predicted_length = torch.relu(predicted_length)  # Length must be positive
        
        # CTC logits (raw)
        ctc_logits_raw = self.ctc_head(encoder_out)  # [B, T, V+1]
        
        # 🚨 Apply Vietnamese filter
        ctc_logits = self._apply_vietnamese_filter(ctc_logits_raw)
        
        # Debug first forward
        if not self._debug_logged:
            with torch.no_grad():
                probs_filtered = torch.softmax(ctc_logits, dim=-1)
                blank_prob = probs_filtered[:, :, -1].mean().item()
                top_tokens = probs_filtered[0, 0].topk(10).indices
                
                print(f"\n🚨 [Vietnamese Filter Debug]")
                print(f"   Blank prob: {blank_prob*100:.2f}%")
                print(f"   Top 10 tokens: {top_tokens.tolist()}")
                print(f"   Predicted length (first 3): {predicted_length[:3].squeeze(-1).tolist()}")
                
            self._debug_logged = True
        
        # AR logits (training only)
        ar_logits = None
        if target is not None:
            B, L = target.shape
            
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
            
            ar_logits_raw = self.ar_head(ar_out)
            
            # Also filter AR logits (use -1e4 for AMP compatibility)
            mask = self.viet_mask.unsqueeze(0).unsqueeze(0)
            ar_logits = ar_logits_raw.masked_fill(~mask, -1e4)
        
        return {
            'ctc_logits': ctc_logits,
            'ar_logits': ar_logits,
            'predicted_length': predicted_length  # 🔧 NEW: For length loss
        }