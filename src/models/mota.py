"""
Architecture:
- Audio: Whisper encoder (frozen/finetune) -> 768
- Visual: ResNet18 (frozen/finetune) -> 512
- Fusion Stage 1: Quality gating (Coarse)
- Fusion Stage 2: M-QOT + Guided Attention (Fine/Optional)
- Encoder: Conformer
- Decoder: Hybrid CTC + Attention
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

# External Backbones (for E2E)
try:
    from transformers import WhisperModel
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError:
    pass # Managed by use_backbones check

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
    3. E2E: Raw Audio/Video -> Backbones -> Fusion
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Dimensions
        self.audio_dim = config.get('audio_dim', 768)
        self.visual_dim = config.get('visual_dim', 512)
        d_model = config.get('d_model', 256)
        
        # Architecture Params
        num_encoder_layers = config.get('num_encoder_layers', 6)
        num_decoder_layers = config.get('num_decoder_layers', 4)
        num_heads = config.get('num_heads', 4)
        vocab_size = config.get('vocab_size', 220)
        dropout = config.get('dropout', 0.1)
        
        # Toggle Flags
        self.use_mqot = config.get('use_mqot', False)
        self.use_backbones = config.get('use_backbones', False)
        
        # ============================================================
        # STAGE 0: OPTIONAL BACKBONES (E2E Readiness)
        # ============================================================
        if self.use_backbones:
            # Audio: Whisper
            self.whisper = WhisperModel.from_pretrained("openai/whisper-tiny")
            self.whisper.encoder.requires_grad_(False) # Default Frozen
            
            # Visual: ResNet18
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            # Remove FC and AvgPool to get spatial/temporal features
            self.visual_backbone = nn.Sequential(*list(resnet.children())[:-2]) 
            self.visual_backbone.requires_grad_(False) # Default Frozen
            
        # ============================================================
        # STAGE 1: COARSE FUSION (QualityGate - Baseline)
        # ============================================================
        self.audio_proj = nn.Linear(self.audio_dim, d_model)
        self.visual_proj = nn.Linear(self.visual_dim, d_model)
        self.quality_gate = QualityGate(d_model)
        
        # ============================================================
        # STAGE 2: FINE-GRAINED ALIGNMENT (M-QOT - Refinement)
        # ============================================================
        if self.use_mqot:
            # Configurable MQOT dimension (default to audio_dim)
            mqot_dim = config.get('mqot_dim', self.audio_dim)
            
            # Adapters to upscale/transform for MQOT space
            # Refinement (0.9.5): Parallelize audio_upsample (Direct 768->MQOT) to remove bottleneck
            self.audio_upsample = nn.Linear(self.audio_dim, mqot_dim)
            self.visual_adapter = VisualAdapter(self.visual_dim, mqot_dim)
            
            # MQOT Components
            self.quality_estimator = QualityEstimator(mqot_dim)
            self.mqot = MQOTLayer(
                dim=mqot_dim,
                lambda_time=config.get('mqot', {}).get('lambda_time', 0.5),
                lambda_qual=config.get('mqot', {}).get('lambda_qual', 5.0),
                epsilon_init=config.get('mqot', {}).get('epsilon', 0.15),
                n_iters=config.get('mqot', {}).get('n_iters', 20),
                use_unbalanced=config.get('mqot', {}).get('use_unbalanced', True),
                kl_penalty=config.get('mqot', {}).get('kl_penalty', 0.1),
                num_heads=config.get('mqot', {}).get('num_heads', 1),
            )
            self.guided_attention = GuidedAttention(
                mqot_dim,
                num_heads=8,
                dropout=dropout
            )
            
            # Downsample back to d_model for Conformer
            self.downsample = nn.Linear(mqot_dim, d_model)
            
            # Learnable Gate for Residual Connection (Step 0.7.1)
            # Init at 0.1 to allow gradient flow (Tuned 0.9.5: 0.01 -> 0.1)
            self.fine_align_gate = nn.Parameter(torch.tensor(0.1))
        
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
    
    def forward_backbones(self, audio, visual):
        """Helper to run backbones if inputs are raw"""
        B = visual.shape[0]
        
        # Visual: [B, T, C, H, W] -> [B, T, D]
        if self.use_backbones and visual.ndim == 5:
            T, C, H, W = visual.shape[1], visual.shape[2], visual.shape[3], visual.shape[4]
            
            # Flatten time: [B*T, C, H, W]
            visual_flat = visual.view(B * T, C, H, W)
            
            # Forward ResNet (Frozen)
            with torch.no_grad():
                # self.visual_backbone outputs [B*T, 512, H', W']
                feat_map = self.visual_backbone(visual_flat)
                
                # Global Average Pool -> [B*T, 512, 1, 1]
                # Note: We need to define pooling if not in backbone.
                # In __init__, we stripped last 2 layers (AvgPool, FC).
                # So we apply AdaptiveAvgPool here.
                feat = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1))
                feat = feat.flatten(1) # [B*T, 512]
            
            # Reshape back: [B, T, 512]
            visual = feat.view(B, T, -1)

        # Audio: [B, Samples] or [B, T, 768]
        # Current GridDataset returns [B, T, 768] even in raw mode (AudioFeatureExtractor)
        # So we pass through.
        
        return audio, visual

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
        
        # 0. Backbones (E2E Bridge - Fixed 0.9.6)
        # Handles Raw Video [B, T, C, H, W] -> Features [B, T, D]
        audio, visual = self.forward_backbones(audio, visual)

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
            # Critical Fix (0.8.2): Use audio_feat (clean) instead of fused_coarse (leakage)
            # Refinement (0.9.5): Use raw 'audio' input (768) instead of projected 'audio_feat' (256)
            audio_rich = self.audio_upsample(audio)    # [B, Ta, 768]
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
            
            # 4. Residual Connection with Learnable Gate
            fused = fused_coarse + self.fine_align_gate * self.downsample(fused_fine)
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
        
        if self.use_mqot and 'quality' in locals():
            outputs['mqot_quality'] = quality
            
        return outputs

# Factory function
def create_model(config: Dict) -> MOTA:
    """Create MOTA model from config"""
    return MOTA(config)
