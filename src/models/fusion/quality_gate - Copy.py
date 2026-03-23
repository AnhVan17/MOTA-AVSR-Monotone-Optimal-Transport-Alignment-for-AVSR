import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

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