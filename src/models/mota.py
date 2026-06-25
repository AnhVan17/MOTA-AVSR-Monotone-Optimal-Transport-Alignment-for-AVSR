"""
Architecture:
- Audio: precomputed Whisper features -> 768 (raw-audio Whisper backbone optional, OFF by default)
- Visual: raw mouth-crop frames -> ResNet18 (use_backbones) -> 512, or precomputed features
- Fusion Stage 1: Quality gating (Coarse)
- Fusion Stage 2: M-QOT + Guided Attention (Fine/Optional)
- Encoder: Conformer
- Decoder: Hybrid CTC + Attention
"""

import logging

import torch
import torch.nn as nn
from typing import Dict, Optional

# External visual backbone (legacy 2D path). Whisper is imported lazily only when the raw-audio
# backbone is explicitly enabled; importing it at module load pulls transformers generation/sklearn,
# which is unnecessary for the normal precomputed-feature path and brittle in lightweight test envs.
try:
    from torchvision.models import resnet18, ResNet18_Weights
except Exception:  # pragma: no cover - handled when the legacy visual backbone is requested
    resnet18 = None
    ResNet18_Weights = None

# Import Modular Components
from .layers.conformer import ConformerBlock
from .layers.decoders import HybridDecoder
from .layers.adapters import VisualAdapter
from .fusion.quality_gate import QualityGate
from .fusion.router_gate import RouterGatedFusion
from .fusion.mqot import MQOTLayer, QualityEstimator, GuidedAttention
from .visual.lipreading_frontend import LipReadingFrontend
from src.utils.warn_once import warn_once

logger = logging.getLogger(__name__)


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
        self.use_rgf = config.get('use_rgf', False)  # Stage-1 = Router-Gated Fusion (Section B) vs QualityGate
        self.use_visual_ctc_aux = config.get('use_visual_ctc_aux', False)
        # use_backbones → run the VISUAL ResNet on raw frames at train time (frame-shard pipeline).
        # use_audio_backbone → run Whisper on RAW audio; OFF by default since the shards already
        # store precomputed Whisper features (the model reads them directly).
        self.use_backbones = config.get('use_backbones', False)
        self.use_audio_backbone = config.get('use_audio_backbone', False)
        # Visual frontend: 'resnet2d' (legacy ImageNet 2D) | 'lipreading_tcn' (Conv3D stem + 2D ResNet, pretrained)
        self.visual_frontend = config.get('visual_frontend', 'resnet2d')
        
        # ============================================================
        # STAGE 0: OPTIONAL BACKBONES (raw → features at train time)
        # ============================================================
        # Visual ResNet: needed when the dataset yields raw frames [B,T,C,H,W] (WebDataset frame
        # shards). Frozen by default (Section A); per-epoch frame augmentation is upstream in the loader.
        if self.use_backbones:
            if self.visual_frontend == 'lipreading_tcn':
                # Pretrained lip-reading encoder (Conv3D stem + 2D ResNet) → 512-D/frame. Motion-aware,
                # frozen by default; trainer may gradual-unfreeze its last block. Replaces the useless
                # 2D-per-frame ImageNet backbone (carried no lip motion → visual contributed ~0).
                self.visual_backbone = LipReadingFrontend(
                    weights=config.get('visual_frontend_weights'),
                    relu_type=config.get('visual_frontend_relu', 'prelu'),
                )
            else:
                # Legacy 2D ResNet18. visual_pretrained=False → no ImageNet download (offline/from-scratch).
                if resnet18 is None or ResNet18_Weights is None:
                    raise ImportError("torchvision ResNet18 is required for visual_frontend='resnet2d'")
                visual_weights = ResNet18_Weights.DEFAULT if config.get('visual_pretrained', True) else None
                resnet = resnet18(weights=visual_weights)
                # Strip FC + AvgPool → spatial feature map; pooled per-frame in forward_backbones.
                self.visual_backbone = nn.Sequential(*list(resnet.children())[:-2])
                self.visual_backbone.requires_grad_(False)  # frozen

        # Audio Whisper backbone: ONLY for true raw-audio E2E. Off by default — the shards store
        # precomputed Whisper features, so loading Whisper here otherwise = dead weight.
        if self.use_audio_backbone:
            from transformers import WhisperModel

            self.whisper = WhisperModel.from_pretrained("openai/whisper-small")
            self.whisper.encoder.requires_grad_(False)
            
        # ============================================================
        # STAGE 1: COARSE FUSION (QualityGate - Baseline)
        # ============================================================
        self.audio_proj = nn.Linear(self.audio_dim, d_model)
        self.visual_proj = nn.Linear(self.visual_dim, d_model)
        self.quality_gate = QualityGate(d_model)

        # Visual bootstrap head: direct CTC supervision on the visual timeline before fusion.
        # It is opt-in and ignored by normal inference unless the trainer/verify script reads it.
        if self.use_visual_ctc_aux:
            self.visual_ctc_norm = nn.LayerNorm(d_model)
            self.visual_ctc_head = nn.Linear(d_model, vocab_size)

        # RGF (Section B): hard routing over {audio-only, visual-only, fusion}. Reuses the
        # QualityGate above as the 'fusion' expert so a Phase-2 warm-start keeps its weights.
        if self.use_rgf:
            rgf_cfg = config.get('rgf', {})
            self.rgf = RouterGatedFusion(
                d_model,
                chunk_size=rgf_cfg.get('chunk_size', 15),
                tau0=rgf_cfg.get('tau0', 2.0),
                tau_min=rgf_cfg.get('tau_min', 0.5),
                gamma=rgf_cfg.get('gamma', 0.99995),
                num_heads=num_heads,
                dropout=dropout,
                quality_gate=self.quality_gate,
            )
        
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
            self.fine_align_gate = nn.Parameter(torch.logit(torch.tensor(0.1)))
        
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
        """Run the visual backbone if inputs are raw frames [B,T,C,H,W] -> features [B,T,D].

        Audio is already precomputed Whisper features [B,T_a,768] (passed through).
        """
        if self.use_backbones and visual.ndim == 5:
            if self.visual_frontend == 'lipreading_tcn':
                # Lip-reading frontend does grayscale+normalize and manages its own grad (frozen
                # stem; last block optionally trainable). Takes [B,T,C,H,W] -> [B,T,512] directly.
                visual = self.visual_backbone(visual)
            else:
                # Legacy 2D path: flatten time, run frozen ResNet18 per frame, global-avg-pool.
                B, T, C, H, W = visual.shape
                visual_flat = visual.view(B * T, C, H, W)
                with torch.no_grad():
                    feat_map = self.visual_backbone(visual_flat)              # [B*T,512,H',W']
                    feat = torch.nn.functional.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)
                visual = feat.view(B, T, -1)
        return audio, visual

    def forward(
        self,
        audio: torch.Tensor,
        visual: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
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

        # Padding mask (F3): True = padding. None → attention KHÔNG mask padding (cảnh báo 1 lần).
        if audio_mask is None:
            warn_once(logger, "mota_no_audio_mask",
                      "audio_mask=None → self/cross-attention KHÔNG mask padding (frame thật bị "
                      "pha loãng; trainer nên truyền audio_mask/visual_mask).")
        pad_mask = (~audio_mask) if audio_mask is not None else None  # [B, Ta], True = pad

        # ========================================
        # STAGE 1: Coarse Fusion (QualityGate)
        # ========================================
        audio_feat = self.audio_proj(audio)     # [B, Ta, D]
        visual_feat = self.visual_proj(visual)  # [B, Tv, D]
        visual_ctc_logits = None
        if self.use_visual_ctc_aux:
            visual_ctc_logits = self.visual_ctc_head(self.visual_ctc_norm(visual_feat))
        
        # Stage-1 fusion: RGF hard-routing (Section B) if enabled, else QualityGate soft fusion.
        gate_out = (
            self.rgf(audio_feat, visual_feat) if self.use_rgf
            else self.quality_gate(audio_feat, visual_feat, audio_mask, visual_mask)
        )
        fused_coarse = gate_out['fused']        # [B, Ta, D]
        
        # ========================================
        # STAGE 2: Refinement (M-QOT)
        # ========================================
        transport_map = None
        if self.use_mqot:
            if audio_mask is not None:
                warn_once(logger, "mqot_no_mask",
                          "MQOT chưa mask padding (Sinkhorn Ta×Tv) → defer run-2; transport có thể "
                          "lệch do padding.")
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
            fused = fused_coarse + torch.sigmoid(self.fine_align_gate) * self.downsample(fused_fine)
        else:
            fused = fused_coarse
            
        # ========================================
        # STAGE 3: Conformer Encoding
        # ========================================
        encoded = fused
        for layer in self.encoder:
            encoded = layer(encoded, pad_mask=pad_mask)
            
        # ========================================
        # STAGE 4: Hybrid Decoding
        # ========================================
        decoder_out = self.decoder(encoded, target, memory_key_padding_mask=pad_mask)
        
        outputs = {
            'ctc_logits': decoder_out['ctc_logits'],
            'ar_logits': decoder_out['ar_logits'],
            'gate_weights': gate_out['gate_weights'],
            'q_audio': gate_out['q_audio'],
            'q_visual': gate_out['q_visual']
        }

        if visual_ctc_logits is not None:
            outputs['visual_ctc_logits'] = visual_ctc_logits

        if self.use_rgf and 'router_probs' in gate_out:
            outputs['router_probs'] = gate_out['router_probs']  # [B, n, 3] for load-balancing loss
        
        if transport_map is not None:
            outputs['transport_map'] = transport_map
        
        if self.use_mqot and 'quality' in locals():
            outputs['mqot_quality'] = quality
            
        return outputs

# Factory function
def create_model(config: Dict) -> MOTA:
    """Create MOTA model from config"""
    return MOTA(config)
