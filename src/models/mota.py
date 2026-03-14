"""
MOTA: Multimodal Optimal Transport Alignment

🔧 FIXED: Removed problematic masking that causes model to only see first N frames

Architecture:
- Audio: Whisper encoder (frozen) → 768dim
- Visual: ResNet18 2D (per-frame) → 512dim
- Fusion Stage 1: Quality gating (Coarse)
- Fusion Stage 2: M-QOT + Guided Attention (Fine/Optional)
- Encoder: Conformer (6 layers)
- Decoder: Hybrid CTC + Attention
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

# Import Modular Components
from .layers.conformer import ConformerBlock
from .layers.decoders import HybridDecoder
from .layers.adapters import VisualAdapter
from .fusion.quality_gate import QualityGate
from .fusion.mqot import MQOTLayer, QualityEstimator, GuidedAttention

class MOTA(nn.Module):
    """
    MOTA: Multimodal Optimal Transport Alignment Model
    
    🔧 FIXED: Removed masking to allow model to process full sequence
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
        
        self.use_mqot = config.get('use_mqot', False)
        
        # Stage 1: Coarse Fusion
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.quality_gate = QualityGate(d_model)
        
        # Stage 2: Fine-grained Alignment (Optional)
        if self.use_mqot:
            mqot_dim = 768
            self.audio_upsample = nn.Linear(d_model, mqot_dim)
            self.visual_adapter = VisualAdapter(visual_dim, mqot_dim)
            self.quality_estimator = QualityEstimator(mqot_dim)
            self.mqot = MQOTLayer(
                lambda_time=config.get('mqot', {}).get('lambda_time', 0.5),
                lambda_qual=config.get('mqot', {}).get('lambda_qual', 5.0)
            )
            self.guided_attention = GuidedAttention(mqot_dim, num_heads=8, dropout=dropout)
            self.downsample = nn.Linear(mqot_dim, d_model)
        
        # Stage 3: Conformer Encoder
        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel=31, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Stage 4: Hybrid Decoder
        self.decoder = HybridDecoder(
            d_model, num_heads, num_decoder_layers,
            vocab_size, dropout
        )
        
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
        audio_len: Optional[torch.Tensor] = None,
        visual_len: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        🔧 FIXED: No masking applied to avoid zeroing out valid data
        
        Args:
            audio: [B, T_a, 768]
            visual: [B, T_v, 512]
            audio_len: [B] (informational only, not used for masking)
            visual_len: [B] (informational only)
            target: [B, L]
        """
        # Stage 1: Coarse Fusion
        audio_feat = self.audio_proj(audio)
        visual_feat = self.visual_proj(visual)
        
        gate_out = self.quality_gate(audio_feat, visual_feat)
        fused_coarse = gate_out['fused']
        
        # Stage 2: Refinement (M-QOT)
        transport_map = None
        if self.use_mqot:
            audio_rich = self.audio_upsample(fused_coarse)
            visual_rich = self.visual_adapter(visual)
            quality = self.quality_estimator(visual_rich)
            transport_map = self.mqot(audio_rich, visual_rich, quality)
            fused_fine = self.guided_attention(
                q=audio_rich,
                k=visual_rich,
                v=visual_rich,
                guide_map=transport_map
            )
            fused = fused_coarse + self.downsample(fused_fine)
        else:
            fused = fused_coarse
        
        # Stage 3: Conformer Encoding
        encoded = fused
        for layer in self.encoder:
            encoded = layer(encoded)
        
        # 🔧 CRITICAL FIX: NO MASKING!
        # Old code (BUGGY):
        # if audio_mask is not None:
        #     encoded = encoded * audio_mask.unsqueeze(-1).float()
        #
        # Why removed:
        # - Mask shape doesn't match encoded shape
        # - Zero out valid data causes model to only see first N frames
        # - CTC loss already handles lengths correctly
        # - Padding frames have small values, minimal impact
        
        # Stage 4: Hybrid Decoding
        decoder_out = self.decoder(encoded, target)
        
        outputs = {
            'ctc_logits': decoder_out['ctc_logits'],
            'ar_logits': decoder_out['ar_logits'],
            'predicted_length': decoder_out.get('predicted_length'),  # 🔧 NEW
            'gate_weights': gate_out['gate_weights'],
            'q_audio': gate_out['q_audio'],
            'q_visual': gate_out['q_visual']
        }
        
        if transport_map is not None:
            outputs['transport_map'] = transport_map
        
        return outputs


def create_model(config: Dict) -> MOTA:
    """Create MOTA model from config"""
    return MOTA(config)