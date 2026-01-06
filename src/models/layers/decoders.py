import torch
import torch.nn as nn
import math
from typing import Dict, Optional

class HybridDecoder(nn.Module):
    """
    Hybrid CTC + Attention Decoder
    
    CTC: Fast alignment (acoustic → char)
    Attention: Context refinement (language model)
    
    FIXED: CTC head outputs vocab_size + 1 (including blank)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        vocab_size: int = 51865,  # Whisper full vocab
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # CRITICAL FIX: CTC head should output vocab_size + 1
        # The +1 is for the CTC blank token (at position vocab_size)
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size + 1)  # +1 for blank
        )
        
        # ============================================================
        # FIX CTC COLLAPSE: Initialize blank bias to negative value
        # This discourages the model from predicting blank at start
        # ============================================================
        self._init_ctc_bias(vocab_size)
        
        print(f"CTC Head initialized: {d_model} -> {vocab_size + 1} (vocab + blank)")
        
        # Attention decoder (uses vocab_size only, no blank needed)
        self.target_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(d_model, 5000)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.ar_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)  # AR uses vocab_size only (no blank)
        )
    
    def _init_ctc_bias(self, vocab_size: int):
        """
        CTC Kickstart Technique: Set blank token bias to large negative value.
        
        This forces the model to initially predict non-blank tokens,
        preventing CTC collapse where model outputs only blanks.
        """
        with torch.no_grad():
            # Get the Linear layer from Sequential
            linear_layer = self.ctc_head[1]
            
            # Initialize all biases to 0
            linear_layer.bias.fill_(0)
            
            # Set blank token bias (at position vocab_size) to -4.0
            # This makes softmax output low probability for blank initially
            linear_layer.bias[vocab_size] = -4.0
            
            print(f"🔧 [CTC Init] Set Blank Token Bias (idx={vocab_size}) to -4.0 to prevent collapse.")
    
    @staticmethod
    def _create_pos_encoding(d_model: int, max_len: int) -> torch.Tensor:
        """Sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2) *
            -(math.log(10000.0) / d_model)
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
            
        Returns:
            dict with ctc_logits, ar_logits
        """
        # CTC logits - outputs [B, T_enc, V+1] where V+1 includes blank
        ctc_logits = self.ctc_head(encoder_out)
        
        # DEBUG: Log CTC head stats (first forward only)
        if not hasattr(self, '_logged_ctc_head_debug'):
            print("\n" + "="*70)
            print("🎯 [CTC HEAD DEBUG]:")
            print("="*70)
            print(f"   Input mean: {encoder_out.mean().item():.4f}, std: {encoder_out.std().item():.4f}")
            print(f"   Output (logits) mean: {ctc_logits.mean().item():.4f}, std: {ctc_logits.std().item():.4f}")
            
            # Check blank logit vs non-blank logits
            blank_logit = ctc_logits[:, :, -1].mean().item()  # Last index = blank
            nonblank_logit = ctc_logits[:, :, :-1].mean().item()
            print(f"   Blank logit mean: {blank_logit:.4f}")
            print(f"   Non-blank logit mean: {nonblank_logit:.4f}")
            print(f"   Difference (blank - nonblank): {blank_logit - nonblank_logit:.4f}")
            
            if blank_logit > nonblank_logit + 2:
                print("   ⚠️ WARNING: Blank logit is MUCH higher than non-blank!")
            
            print("="*70 + "\n")
            self._logged_ctc_head_debug = True
        
        # AR logits (only during training)
        ar_logits = None
        if target is not None:
            B, L = target.shape
            
            # Handle invalid tokens for embedding
            target_for_embed = target.clone()
            
            # Create mask for invalid tokens
            invalid_mask = (target_for_embed < 0) | (target_for_embed >= self.vocab_size)
            
            if invalid_mask.any():
                # Replace invalid tokens with 0
                target_for_embed[invalid_mask] = 0
                
                if not hasattr(self, '_logged_invalid_tokens'):
                    num_invalid = invalid_mask.sum().item()
                    print(f"Warning: Found {num_invalid} invalid tokens in target. Replacing with 0.")
                    self._logged_invalid_tokens = True
            
            # Embed targets
            target_embed = self.target_embedding(target_for_embed)
            
            # Add positional encoding
            pos = self.pos_encoding[:, :L, :].to(encoder_out.device)
            target_embed = target_embed + pos
            
            # Create causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L)
            causal_mask = causal_mask.to(encoder_out.device)
            
            # Decode
            ar_out = self.decoder(
                tgt=target_embed,
                memory=encoder_out,
                tgt_mask=causal_mask
            )
            
            ar_logits = self.ar_head(ar_out)  # [B, L, V]
        
        return {
            'ctc_logits': ctc_logits,    # [B, T, V+1]
            'ar_logits': ar_logits       # [B, L, V]
        }