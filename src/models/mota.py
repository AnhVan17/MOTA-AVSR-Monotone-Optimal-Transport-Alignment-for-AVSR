"""
Architecture:
- Audio: Whisper encoder (frozen) → 768
- Visual: ResNet18/ViViT → 512/768
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
    
    Hybrid Approaches:
    1. Baseline: QualityGate fusion only.
    2. MQOT: QualityGate + Optimal Transport Refinement.
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
        
        # Toggle for Phase 2
        self.use_mqot = config.get('use_mqot', False)
        
        # ============================================================
        # STAGE 1: COARSE FUSION (QualityGate - Baseline)
        # ============================================================
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.quality_gate = QualityGate(d_model)
        
        # ============================================================
        # STAGE 2: FINE-GRAINED ALIGNMENT (M-QOT - Refinement)
        # ============================================================
        if self.use_mqot:
            mqot_dim = 768  # Whisper dimension
            
            # Adapters to upscale/transform for MQOT space
            self.audio_upsample = nn.Linear(d_model, mqot_dim)
            self.visual_adapter = VisualAdapter(visual_dim, mqot_dim)
            
            # MQOT Components
            self.quality_estimator = QualityEstimator(mqot_dim)
            self.mqot = MQOTLayer(
                lambda_time=config.get('mqot', {}).get('lambda_time', 0.5),
                lambda_qual=config.get('mqot', {}).get('lambda_qual', 5.0)
            )
            self.guided_attention = GuidedAttention(
                mqot_dim,
                num_heads=8,
                dropout=dropout
            )
            
            # Downsample back to d_model for Conformer
            self.downsample = nn.Linear(mqot_dim, d_model)
        
        # ============================================================
        # STAGE 3: CONFORMER ENCODER
        # ============================================================
        self.encoder = nn.ModuleList([
            ConformerBlock(d_model, num_heads, conv_kernel=31, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # ============================================================
        # STAGE 4: HYBRID DECODER
        # ============================================================
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
        """
        # ========================================
        # STAGE 1: Coarse Fusion (QualityGate)
        # ========================================
        audio_feat = self.audio_proj(audio)     # [B, Ta, D]
        visual_feat = self.visual_proj(visual)  # [B, Tv, D]
        
        gate_out = self.quality_gate(audio_feat, visual_feat)
        fused_coarse = gate_out['fused']        # [B, Ta, D]
        
        # ========================================
        # STAGE 2: Refinement (M-QOT)
        # ========================================
        transport_map = None
        if self.use_mqot:
            # 1. Prepare rich features
            audio_rich = self.audio_upsample(fused_coarse)  # [B, Ta, 768]
            visual_rich = self.visual_adapter(visual)       # [B, Tv, 768]
            
            # 2. Estimate quality & Compute Transport
            quality = self.quality_estimator(visual_rich)   # [B, Tv]
            transport_map = self.mqot(audio_rich, visual_rich, quality) # [B, Ta, Tv]
            
            # 3. Guided Attention
            fused_fine = self.guided_attention(
                q=audio_rich,
                k=visual_rich,
                v=visual_rich,
                guide_map=transport_map
            ) # [B, Ta, 768]
            
            # 4. Residual Connection (Coarse + Fine)
            fused = fused_coarse + self.downsample(fused_fine)
        else:
            fused = fused_coarse
            
        # ========================================
        # STAGE 3: Conformer Encoding
        # ========================================
        encoded = fused
        for layer in self.encoder:
            encoded = layer(encoded)
            
        # ========================================
        # STAGE 4: Hybrid Decoding
        # ========================================
        decoder_out = self.decoder(encoded, target)
        
        outputs = {
            'ctc_logits': decoder_out['ctc_logits'],
            'ar_logits': decoder_out['ar_logits'],
            'gate_weights': gate_out['gate_weights'],
            'q_audio': gate_out['q_audio'],
            'q_visual': gate_out['q_visual']
        }
        
        if transport_map is not None:
            outputs['transport_map'] = transport_map
            
        return outputs

# Factory function
def create_model(config: Dict) -> MOTA:
    """Create MOTA model from config"""
    return MOTA(config)

# Test Block
if __name__ == "__main__":
    print("="*60)
    print("Testing Updated MOTA (Modular + MQOT Ready)")
    print("="*60)
    
    # Test Config Phase 1 (No MQOT)
    config_p1 = {
        'audio_dim': 768,
        'visual_dim': 512,
        'd_model': 256,
        'use_mqot': False,
        'num_encoder_layers': 2,
        'num_decoder_layers': 2
    }
    
    model = create_model(config_p1)
    print("Model Phase 1 initialized. Params:", sum(p.numel() for p in model.parameters()))
    
    # Test Forward Phase 1
    B, Ta, Tv = 2, 100, 75
    a = torch.randn(B, Ta, 768)
    v = torch.randn(B, Tv, 512)
    with torch.no_grad():
        out = model(a, v)
    print(f"Phase 1 Output: CTC size {out['ctc_logits'].shape}")
    
    # Test Config Phase 2 (With MQOT)
    config_p2 = config_p1.copy()
    config_p2['use_mqot'] = True
    
    model_mqot = create_model(config_p2)
    print("\nModel Phase 2 (MQOT) initialized. Params:", sum(p.numel() for p in model_mqot.parameters()))
    
    with torch.no_grad():
        out_mqot = model_mqot(a, v)
    print(f"Phase 2 Output: CTC size {out_mqot['ctc_logits'].shape}")
    print(f"Transport Map: {out_mqot['transport_map'].shape}")
