import torch
import torch.nn as nn
import math
from typing import Dict, Optional

class HybridDecoder(nn.Module):
    """
    Hybrid CTC + Attention Decoder
    
    CTC: Fast alignment (acoustic → char)
    Attention: Context refinement (language model)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        vocab_size: int = 220,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # CTC head (direct projection)
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        # Attention decoder
        self.target_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(d_model, 512)
        
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
            nn.Linear(d_model, vocab_size)
        )
    
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
        # CTC logits (always computed)
        ctc_logits = self.ctc_head(encoder_out)  # [B, T_enc, V]
        
        # AR logits (only during training)
        ar_logits = None
        if target is not None:
            B, L = target.shape
            
            # Embed targets
            target_embed = self.target_embedding(target)
            
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
            'ctc_logits': ctc_logits,
            'ar_logits': ar_logits
        }
