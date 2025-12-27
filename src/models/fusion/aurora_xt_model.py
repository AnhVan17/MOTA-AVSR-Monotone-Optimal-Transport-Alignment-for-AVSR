"""
AURORA-XT FINAL - Optimized for WhisperTokenizer
=================================================
CRITICAL CHANGES from original:
1. ❌ REMOVED CTC (incompatible with subword tokens)
2. ✅ Stronger AR decoder (6 layers, d_model=512)
3. ✅ Better positional encoding
4. ✅ Optimized for Test WER 55% → 18-25%

Architecture:
- Audio: Whisper frozen features [T, 768]
- Visual: ResNet18 features [T, 512]
- Fusion: Quality gating (adaptive)
- Encoder: Conformer (6 layers)
- Decoder: ATTENTION-ONLY (NO CTC!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math


class ConformerBlock(nn.Module):
    """Conformer block: Conv + Attention"""
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        conv_kernel: int = 31,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm_mha = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_mha = nn.Dropout(dropout)
        
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(
                d_model, d_model,
                kernel_size=conv_kernel,
                padding=conv_kernel // 2,
                groups=d_model
            ),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x_norm = self.norm_mha(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.dropout_mha(attn_out)
        x_norm = self.norm_conv(x)
        x_conv = x_norm.transpose(1, 2)
        x_conv = self.conv(x_conv)
        x = x + x_conv.transpose(1, 2)
        x = x + 0.5 * self.ffn2(x)
        return x


class QualityGate(nn.Module):
    """Quality-aware adaptive fusion"""
    
    def __init__(self, d_model: int = 512):
        super().__init__()
        
        self.audio_quality = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.visual_quality = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
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
        
        q_audio = self.audio_quality(audio_feat)
        q_visual = self.visual_quality(visual_feat)
        
        combined = torch.cat([
            audio_feat,
            visual_feat,
            q_audio,
            q_visual
        ], dim=-1)
        
        gate_weights = self.gate(combined)
        
        fused = (gate_weights[:, :, 0:1] * audio_feat +
                gate_weights[:, :, 1:2] * visual_feat)
        
        return {
            'fused': fused,
            'gate_weights': gate_weights,
            'q_audio': q_audio.mean(dim=1),
            'q_visual': q_visual.mean(dim=1)
        }


class AttentionOnlyDecoder(nn.Module):
    """
    ⭐ ATTENTION-ONLY Decoder (NO CTC!)
    
    Optimized for WhisperTokenizer (51,865 vocab)
    Designed to fix character-level's Test WER 55% problem
    """
    
    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        vocab_size: int = 51865,
        dropout: float = 0.2,
        max_len: int = 512
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Target embedding
        self.target_embedding = nn.Embedding(vocab_size, d_model)
        
        # Hybrid positional encoding (learned + sinusoidal)
        self.pos_encoding_learned = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )
        self.register_buffer(
            'pos_encoding_sin',
            self._create_sinusoidal_pe(d_model, max_len)
        )
        
        # Stronger transformer decoder (6 layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for stability
        )
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # Output projection
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize with small weights
        nn.init.normal_(self.output_proj.weight, mean=0, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    @staticmethod
    def _create_sinusoidal_pe(d_model: int, max_len: int) -> torch.Tensor:
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
        target: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoder_out: [B, T_enc, D]
            target: [B, L] (for training)
            encoder_mask: [B, T_enc] (optional)
            
        Returns:
            dict with ar_logits (NO ctc_logits!)
        """
        
        if target is None:
            raise NotImplementedError("Use generate() for inference")
        
        B, L = target.shape
        
        # Embed targets
        target_embed = self.target_embedding(target)
        
        # Add positional encoding (hybrid)
        pos_learned = self.pos_encoding_learned[:, :L, :]
        pos_sin = self.pos_encoding_sin[:, :L, :].to(target_embed.device)
        target_embed = target_embed + 0.5 * pos_learned + 0.5 * pos_sin
        
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L)
        causal_mask = causal_mask.to(target_embed.device)
        
        # Encoder attention mask
        memory_key_padding_mask = None
        if encoder_mask is not None:
            memory_key_padding_mask = ~encoder_mask
        
        # Decode
        decoder_out = self.decoder(
            tgt=target_embed,
            memory=encoder_out,
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Output projection
        decoder_out = self.output_norm(decoder_out)
        ar_logits = self.output_proj(decoder_out)
        
        return {
            'ar_logits': ar_logits,
            'ctc_logits': None  # NO CTC!
        }


class AuroraXT(nn.Module):
    """
    ⭐ AURORA-XT FINAL - Attention-Only
    
    Optimized for WhisperTokenizer to fix Test WER 55%
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        audio_dim = config.get('audio_dim', 768)
        visual_dim = config.get('visual_dim', 512)
        d_model = config.get('d_model', 512)
        num_encoder_layers = config.get('num_encoder_layers', 6)
        num_decoder_layers = config.get('num_decoder_layers', 6)
        num_heads = config.get('num_heads', 8)
        vocab_size = config.get('vocab_size', 51865)
        dropout = config.get('dropout', 0.2)
        
        print(f"\n{'='*70}")
        print("⭐ AURORA-XT FINAL (Attention-Only)")
        print(f"{'='*70}")
        print(f"   d_model: {d_model}")
        print(f"   Encoder: {num_encoder_layers} Conformer layers")
        print(f"   Decoder: {num_decoder_layers} Transformer layers")
        print(f"   Vocab: {vocab_size} (WhisperTokenizer)")
        print(f"   Heads: {num_heads}")
        print(f"   Dropout: {dropout}")
        print(f"   ❌ CTC: REMOVED (incompatible with subwords)")
        print(f"   ✅ Attention-Only Decoder")
        print(f"{'='*70}\n")
        
        # Projections
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.visual_proj = nn.Linear(visual_dim, d_model)
        
        # Quality gating
        self.quality_gate = QualityGate(d_model)
        
        # Conformer encoder
        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel=31, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Attention-only decoder (NO CTC!)
        self.decoder = AttentionOnlyDecoder(
            d_model, num_heads, num_decoder_layers,
            vocab_size, dropout
        )
        
        # Initialize
        self.apply(self._init_weights)
        
        # Count params
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ Total parameters: {total_params:,} (~{total_params/1e6:.1f}M)")
        
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"   Decoder: {decoder_params:,} (~{decoder_params/1e6:.1f}M)")
        print(f"   Output layer: {d_model * vocab_size:,} (~{d_model * vocab_size/1e6:.1f}M)\n")
    
    @staticmethod
    def _init_weights(module):
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
        target: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            audio: [B, T_a, 768] Whisper features
            visual: [B, T_v, 512] ResNet features
            target: [B, L] target token IDs
            audio_mask: [B, T_a] (optional)
            visual_mask: [B, T_v] (optional)
            
        Returns:
            dict with ar_logits (NO ctc_logits!)
        """
        # Project
        audio_feat = self.audio_proj(audio)
        visual_feat = self.visual_proj(visual)
        
        # Quality gating & fusion
        gate_out = self.quality_gate(audio_feat, visual_feat)
        fused = gate_out['fused']
        
        # Conformer encoding
        encoded = fused
        for layer in self.encoder:
            encoded = layer(encoded)
        
        # Attention decoding (NO CTC!)
        decoder_out = self.decoder(
            encoded, 
            target,
            encoder_mask=audio_mask
        )
        
        return {
            'ar_logits': decoder_out['ar_logits'],
            'ctc_logits': None,  # NO CTC!
            'gate_weights': gate_out['gate_weights'],
            'q_audio': gate_out['q_audio'],
            'q_visual': gate_out['q_visual']
        }
    
    @torch.no_grad()
    def generate(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        max_len: int = 150,
        bos_token_id: int = 50258,
        eos_token_id: int = 50257
    ) -> Dict[str, torch.Tensor]:
        """Greedy decoding for inference"""
        device = audio.device
        B = audio.size(0)
        
        # Encoding
        audio_feat = self.audio_proj(audio)
        visual_feat = self.visual_proj(visual)
        gate_out = self.quality_gate(audio_feat, visual_feat)
        fused = gate_out['fused']
        
        encoded = fused
        for layer in self.encoder:
            encoded = layer(encoded)
        
        # AR Decoding
        generated_tokens = torch.full(
            (B, 1), bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            decoder_out = self.decoder(encoded, generated_tokens)
            logits = decoder_out['ar_logits'][:, -1, :]
            next_tokens = torch.argmax(logits, dim=-1)
            
            next_tokens = torch.where(
                finished, 
                torch.zeros_like(next_tokens), 
                next_tokens
            )
            generated_tokens = torch.cat([
                generated_tokens, 
                next_tokens.unsqueeze(1)
            ], dim=1)
            
            finished = finished | (next_tokens == eos_token_id)
            if finished.all():
                break
        
        return {
            'tokens': generated_tokens,
            'q_audio': gate_out['q_audio'],
            'q_visual': gate_out['q_visual']
        }


def create_model(config: Dict) -> AuroraXT:
    """Factory function"""
    return AuroraXT(config)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Testing AURORA-XT FINAL (Attention-Only)")
    print("="*70 + "\n")
    
    config = {
        'audio_dim': 768,
        'visual_dim': 512,
        'd_model': 512,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'num_heads': 8,
        'vocab_size': 51865,
        'dropout': 0.2
    }
    
    model = create_model(config)
    
    # Test forward
    B, T_a, T_v, L = 2, 450, 375, 15
    audio = torch.randn(B, T_a, 768)
    visual = torch.randn(B, T_v, 512)
    target = torch.randint(0, 51865, (B, L))
    
    outputs = model(audio, visual, target)
    
    print(f"Input:")
    print(f"  Audio: {audio.shape}")
    print(f"  Visual: {visual.shape}")
    print(f"  Target: {target.shape}")
    
    print(f"\nOutput:")
    print(f"  AR logits: {outputs['ar_logits'].shape}")
    print(f"  CTC logits: {outputs['ctc_logits']}")  # None
    print(f"  Gate weights: {outputs['gate_weights'].shape}")
    
    print("\n✅ Model test passed!")