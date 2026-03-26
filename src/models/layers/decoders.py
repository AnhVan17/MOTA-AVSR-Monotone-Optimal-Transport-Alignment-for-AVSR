import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class HybridDecoder(nn.Module):
    """
    Hybrid CTC + Attention Decoder.

    Thay đổi so với v1:
    - pos_encoding: plain tensor → register_buffer (tự động .to(device) khi model chuyển GPU)
    - Causal mask: tạo mỗi forward → cache trong __init__ (max_len cố định)
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        vocab_size: int = 220,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # CTC head
        self.ctc_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )

        # Attention decoder
        self.target_embedding = nn.Embedding(vocab_size, d_model)

        # PE as register_buffer — tự động .to(device) khi model.to('cuda')
        pe = self._create_pos_encoding(max_len, d_model)
        self.register_buffer('pos_encoding', pe)

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

        # Cached causal mask (max_len cố định → tạo 1 lần)
        causal = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer('causal_mask', causal)

    @staticmethod
    def _create_pos_encoding(max_len: int, d_model: int) -> torch.Tensor:
        """Sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

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
        # CTC logits (always)
        ctc_logits = self.ctc_head(encoder_out)  # [B, T_enc, V]

        # AR logits (training only)
        ar_logits = None
        if target is not None:
            B, L = target.shape

            # Embed + positional encoding (PE buffer tự động lên GPU)
            target_embed = self.target_embedding(target)
            target_embed = target_embed + self.pos_encoding[:, :L, :]

            # Causal mask — dùng buffer đã cache
            causal_mask = self.causal_mask[:L, :L]

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
