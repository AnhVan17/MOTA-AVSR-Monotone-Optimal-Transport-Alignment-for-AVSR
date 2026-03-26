import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class QualityGate(nn.Module):
    """
    QualityGate v2 — Learnable cross-modal alignment + quality-aware fusion.

    Thay đổi so với v1:
    - F.interpolate: giả định uniform motion → SAI với speech rate thay đổi
    - Cross-attention: audio frame tự quyết attend visual frame nào → LEARNED alignment
    - ReLU → GELU + LayerNorm: stable gradient flow
    - Residual gate zero-init: stable ở epoch đầu

    Forward:
      1. Cross-attention alignment (visual → audio timeline)
      2. Quality scoring (GELU + LayerNorm)
      3. Gated fusion (audio + aligned_visual)
      4. Residual connection
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # --- Bước 1: Cross-attention alignment ---
        # Q = audio (T_audio frames), K = V = visual (T_visual frames)
        # Output: aligned_visual [B, T_audio, D] — visual đã align theo audio timeline
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_scale = nn.Parameter(torch.tensor(d_model ** -0.5))

        # --- Bước 2: Quality scoring (GELU + LayerNorm thay vì ReLU) ---
        self.audio_quality = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),          # thay vì ReLU
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        self.visual_quality = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),          # thay vì ReLU
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # --- Bước 3: Fusion gate ---
        # Input: [audio, aligned_visual, q_audio, q_visual] = D + D + 1 + 1 = 2D + 2
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2 + 2, d_model),
            nn.GELU(),          # thay vì ReLU
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
            nn.Softmax(dim=-1)
        )

        # --- Bước 4: Residual gate (zero-init — identity at start) ---
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def _create_causal_mask(self, Ta: int, Tv: int, device: torch.device) -> torch.Tensor:
        """
        Audio frame i chỉ attend visual frames <= proportional time.
        Đảm bảo monotonic alignment: audio frame 0 → visual frame 0.
        """
        mask = torch.zeros(Ta, Tv, device=device)
        for i in range(Ta):
            max_j = int((i / Ta) * Tv) + 1
            mask[i, max_j:] = float('-inf')
        return mask

    def _cross_attention_align(
        self,
        audio_feat: torch.Tensor,
        visual_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Learnable alignment: audio query attends visual key/value.
        Output shape: [B, Ta, D] — visual aligned to audio timeline.
        """
        B, Ta, D = audio_feat.shape
        Tv = visual_feat.shape[1]

        Q = self.q_proj(audio_feat)
        K = self.k_proj(visual_feat)
        V = self.v_proj(visual_feat)

        # Multi-head: [B, H, Ta, hd]
        H = self.num_heads
        hd = D // H
        Q = Q.view(B, Ta, H, hd).transpose(1, 2)
        K = K.view(B, Tv, H, hd).transpose(1, 2)
        V = V.view(B, Tv, H, hd).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.attn_scale

        # Causal mask: audio i chỉ attend visual frames <= i*Tv/Ta
        if Ta > 1 and Tv > 1:
            causal_mask = self._create_causal_mask(Ta, Tv, audio_feat.device)
            # Broadcast: [Ta, Tv] → [B, H, Ta, Tv]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
            scores = scores + causal_mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Aggregate: [B, H, Ta, hd] → [B, Ta, D]
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).reshape(B, Ta, D)

        return context

    def forward(
        self,
        audio_feat: torch.Tensor,
        visual_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_feat: [B, T_a, D] — audio features sau projection
            visual_feat: [B, T_v, D] — visual features sau projection

        Returns:
            dict:
              fused [B, T_a, D]: audio-visual fusion
              gate_weights [B, T_a, 2]: audio vs visual weight per frame
              q_audio [B]: audio quality score (pooled)
              q_visual [B]: visual quality score (pooled)
              alignment_weights [B, T_a, T_v]: cross-attention alignment map
        """
        B, Ta, D = audio_feat.shape
        Tv = visual_feat.shape[1]

        # --- Bước 1: Alignment ---
        # Fast path: T_a == T_v → skip attention overhead
        if Ta == Tv:
            aligned_visual = visual_feat
            alignment_weights = None
        else:
            aligned_visual = self._cross_attention_align(audio_feat, visual_feat)
            # Store alignment weights for visualization/debugging
            Q = F.normalize(self.q_proj(audio_feat), p=2, dim=-1)
            K = F.normalize(self.k_proj(visual_feat), p=2, dim=-1)
            sim = torch.matmul(Q, K.transpose(1, 2))  # [B, Ta, Tv]
            alignment_weights = F.softmax(sim, dim=-1)

        # --- Bước 2: Quality scoring ---
        q_audio = self.audio_quality(audio_feat)          # [B, T_a, 1]
        q_visual = self.visual_quality(aligned_visual)    # [B, T_a, 1]

        # --- Bước 3: Gated fusion ---
        combined = torch.cat([
            audio_feat,
            aligned_visual,
            q_audio,
            q_visual
        ], dim=-1)                                        # [B, T_a, 2D + 2]

        gate_weights = self.gate(combined)                # [B, T_a, 2]
        fused = (
            gate_weights[..., 0:1] * audio_feat +
            gate_weights[..., 1:2] * aligned_visual
        )

        # --- Bước 4: Residual ---
        fused = fused + F.sigmoid(self.residual_gate) * fused

        return {
            'fused': fused,                               # [B, T_a, D]
            'gate_weights': gate_weights,                 # [B, T_a, 2]
            'q_audio': q_audio.mean(dim=1).squeeze(-1),  # [B]
            'q_visual': q_visual.mean(dim=1).squeeze(-1),# [B]
            'alignment_weights': alignment_weights,       # [B, T_a, T_v] or None
        }
