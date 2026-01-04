"""
AURORA-XT Complete Model
=========================
Architecture:
- Audio: Whisper encoder (frozen) → 768
- Visual: ResNet18/ViViT → 512/768
- Fusion: Quality gating + Cross-attention
- Encoder: Conformer (6 layers)
- Decoder: CTC + Attention Hybrid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


# ============================================================================
# CONFORMER ENCODER
# ============================================================================

class ConformerBlock(nn.Module):
    """
    Conformer block: Conv + Attention
    
    Best for speech: captures acoustic (conv) + context (attention)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Feed-forward 1
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # 2. Multi-head attention
        self.norm_mha = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_mha = nn.Dropout(dropout)
        
        # 3. Depthwise convolution (KEY!)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(
                d_model, d_model,
                kernel_size=conv_kernel,
                padding=conv_kernel // 2,
                groups=d_model  # Depthwise
            ),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. Feed-forward 2
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            x: [B, T, D]
        """
        # FFN 1
        x = x + 0.5 * self.ffn1(x)
        
        # Multi-head attention
        x_norm = self.norm_mha(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.dropout_mha(attn_out)
        
        # Convolution
        x_norm = self.norm_conv(x)
        x_conv = x_norm.transpose(1, 2)  # [B, D, T]
        x_conv = self.conv(x_conv)
        x = x + x_conv.transpose(1, 2)
        
        # FFN 2
        x = x + 0.5 * self.ffn2(x)
        
        return x


# ============================================================================
# QUALITY GATING & FUSION
# ============================================================================

class QualityGate(nn.Module):
    """
    Learn audio/visual quality → adaptive fusion
    """
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        
        # Quality scorers (per-frame)
        self.audio_quality = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.visual_quality = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        audio_feat: torch.Tensor,
        visual_feat: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_feat: [B, T_a, D]
            visual_feat: [B, T_v, D]
            
        Returns:
            dict with fused, gate_weights, qualities
        """
        B = audio_feat.size(0)
        T_a = audio_feat.size(1)
        
        # Align visual to audio length
        if visual_feat.size(1) != T_a:
            visual_feat = F.interpolate(
                visual_feat.transpose(1, 2),
                size=T_a,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # Per-frame quality scores
        q_audio = self.audio_quality(audio_feat)  # [B, T, 1]
        q_visual = self.visual_quality(visual_feat)  # [B, T, 1]
        
        # Fusion gate
        combined = torch.cat([
            audio_feat,
            visual_feat,
            q_audio,
            q_visual
        ], dim=-1)
        
        gate_weights = self.gate(combined)  # [B, T, 2]
        
        # Weighted fusion
        fused = (gate_weights[:, :, 0:1] * audio_feat +
                gate_weights[:, :, 1:2] * visual_feat)
        
        return {
            'fused': fused,
            'gate_weights': gate_weights,
            'q_audio': q_audio.mean(dim=1),  # [B, 1]
            'q_visual': q_visual.mean(dim=1)  # [B, 1]
        }


# ============================================================================
# HYBRID DECODER (CTC + ATTENTION)
# ============================================================================

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
        self.pos_encoding = self._create_pos_encoding(d_model, 2048)
        
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
            
            # Embed targets (handle -100 padding for CE loss compatibility)
            target_input = target.masked_fill(target == -100, 0)
            target_input = target_input.clamp(0, self.vocab_size - 1)
            target_embed = self.target_embedding(target_input)
            
            # Add positional encoding (Safe slice)
            seq_len = target_embed.size(1)
            max_pos = self.pos_encoding.size(1)
            if seq_len > max_pos:
                target_embed = target_embed[:, :max_pos, :]
                seq_len = max_pos
            
            pos = self.pos_encoding[:, :seq_len, :].to(encoder_out.device)
            target_embed = target_embed + pos
            
            # Create causal mask
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
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


# ============================================================================
# COMPLETE AURORA-XT MODEL
# ============================================================================

class AuroraXT(nn.Module):
    """
    Complete AURORA-XT Model
    
    Pipeline:
    1. Project audio/visual to d_model
    2. Quality gating & fusion
    3. Conformer encoder
    4. Hybrid CTC + Attention decoder
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        audio_dim = config.get('audio_dim', 768)
        visual_dim = config.get('visual_dim', 512)
        d_model = config.get('d_model', 256)
        num_encoder_layers = config.get('num_encoder_layers', 6)
        num_decoder_layers = config.get('num_decoder_layers', 4)
        num_heads = config.get('num_heads', 4)
        vocab_size = config.get('vocab_size', 220)
        dropout = config.get('dropout', 0.1)
        
        # 1. Projections
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.visual_proj = nn.Linear(visual_dim, d_model)
        
        # 2. Quality gating & fusion
        self.quality_gate = QualityGate(d_model)
        
        # 3. Conformer encoder
        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel=31, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # 4. Hybrid decoder
        self.decoder = HybridDecoder(
            d_model, num_heads, num_decoder_layers,
            vocab_size, dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            audio: [B, T_a, 768] Whisper features
            visual: [B, T_v, 512] ResNet features
            target: [B, L] target token IDs (optional)
            
        Returns:
            dict with ctc_logits, ar_logits, gate_weights
        """
        # Project to d_model
        audio_feat = self.audio_proj(audio)   # [B, T_a, D]
        visual_feat = self.visual_proj(visual)  # [B, T_v, D]
        
        # Quality gating & fusion
        gate_out = self.quality_gate(audio_feat, visual_feat)
        fused = gate_out['fused']  # [B, T, D]
        
        # Conformer encoding
        encoded = fused
        for layer in self.encoder:
            encoded = layer(encoded)
        
        # Hybrid decoding
        decoder_out = self.decoder(encoded, target)
        
        return {
            'ctc_logits': decoder_out['ctc_logits'],
            'ar_logits': decoder_out['ar_logits'],
            'gate_weights': gate_out['gate_weights'],
            'q_audio': gate_out['q_audio'],
            'q_visual': gate_out['q_visual']
        }


# Factory function
def create_model(config: Dict) -> AuroraXT:
    """Create AURORA-XT model from config"""
    return AuroraXT(config)


# Test
if __name__ == "__main__":
    print("="*60)
    print("Testing AURORA-XT Model")
    print("="*60)
    
    config = {
        'audio_dim': 768,
        'visual_dim': 512,
        'd_model': 256,
        'num_encoder_layers': 6,
        'num_decoder_layers': 4,
        'num_heads': 4,
        'vocab_size': 220,
        'dropout': 0.1
    }
    
    model = create_model(config)
    
    # Test forward
    B, T_a, T_v, L = 2, 450, 375, 80  # 15s audio/visual
    audio = torch.randn(B, T_a, 768)
    visual = torch.randn(B, T_v, 512)
    target = torch.randint(0, 220, (B, L))
    
    with torch.no_grad():
        outputs = model(audio, visual, target)
    
    print(f"\nInput:")
    print(f"  Audio: {audio.shape}")
    print(f"  Visual: {visual.shape}")
    print(f"  Target: {target.shape}")
    
    print(f"\nOutput:")
    print(f"  CTC logits: {outputs['ctc_logits'].shape}")
    print(f"  AR logits: {outputs['ar_logits'].shape}")
    print(f"  Gate weights: {outputs['gate_weights'].shape}")
    
    # Count params
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal params: {total:,} (~{total*4/1024**2:.1f}MB)")
    
    print("\n✅ Model test passed!")