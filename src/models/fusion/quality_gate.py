import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class QualityGate(nn.Module):
    """
    QualityGate v2 — Learnable cross-modal alignment + quality-aware fusion.

    Thay đổi so với v1:
    - F.interpolate: giả định uniform motion → SAI với speech rate thay đổi
    - Cross-attention: audio frame tự quyết attend visual frame nào → LEARNED alignment
    - ReLU → GELU + LayerNorm: stable gradient flow
    - Residual gate zero-init: stable ở epoch đầu

    Padding-aware (F3): khi truyền audio_mask/visual_mask, cross-attention chỉ attend visual frame
    THẬT, causal mask tính theo ĐỘ DÀI THẬT (không phải chiều đã pad), và quality score pool có mask.

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

        # --- Bước 4: Residual gate — convex giữa audio thuần và bản fusion ---
        # Lưu raw = logit(0.1) vì forward bọc sigmoid → g ∈ (0,1), khởi đầu g≈0.1
        # (tin audio ~90% lúc đầu), model tự học tăng dần độ tin fusion.
        self.residual_gate = nn.Parameter(torch.logit(torch.tensor(0.1)))

    def _create_causal_mask(self, Ta: int, Tv: int, device: torch.device) -> torch.Tensor:
        """Fallback (KHÔNG có độ dài thật): causal theo chiều đã pad. [Ta, Tv]."""
        mask = torch.zeros(Ta, Tv, device=device)
        for i in range(Ta):
            max_j = int((i / Ta) * Tv) + 1
            mask[i, max_j:] = float('-inf')
        return mask

    def _align_mask(
        self, Ta: int, Tv: int, audio_len: torch.Tensor, visual_len: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Additive mask [B, Ta, Tv]: monotonic causal theo ĐỘ DÀI THẬT + bỏ visual padding.

        audio frame i (trong la frame thật) chỉ attend visual frame j <= (i/la)*lv, và j < lv.
        Không phụ thuộc lượng padding → đầu ra tại frame thật bất biến với padding.
        """
        i = torch.arange(Ta, device=device).view(1, Ta, 1).float()
        j = torch.arange(Tv, device=device).view(1, 1, Tv)
        la = audio_len.view(-1, 1, 1).clamp(min=1).float()
        lv = visual_len.view(-1, 1, 1).clamp(min=1)
        max_j = (i / la) * lv.float() + 1.0
        allow = (j.float() < max_j) & (j < lv)            # [B, Ta, Tv] causal + visual-valid
        add = torch.zeros(allow.shape, device=device)
        return add.masked_fill(~allow, float('-inf'))

    def _cross_attention_align(
        self,
        audio_feat: torch.Tensor,
        visual_feat: torch.Tensor,
        audio_len: Optional[torch.Tensor] = None,
        visual_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Learnable alignment: audio query attends visual key/value → [B, Ta, D]."""
        B, Ta, D = audio_feat.shape
        Tv = visual_feat.shape[1]

        Q = self.q_proj(audio_feat)
        K = self.k_proj(visual_feat)
        V = self.v_proj(visual_feat)

        H = self.num_heads
        hd = D // H
        Q = Q.view(B, Ta, H, hd).transpose(1, 2)
        K = K.view(B, Tv, H, hd).transpose(1, 2)
        V = V.view(B, Tv, H, hd).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.attn_scale  # [B, H, Ta, Tv]

        if Ta > 1 and Tv > 1:
            if audio_len is not None and visual_len is not None:
                add_mask = self._align_mask(Ta, Tv, audio_len, visual_len, audio_feat.device)  # [B,Ta,Tv]
            else:
                add_mask = self._create_causal_mask(Ta, Tv, audio_feat.device).unsqueeze(0)     # [1,Ta,Tv]
            scores = scores + add_mask.unsqueeze(1)  # broadcast over heads → [B,H,Ta,Tv]

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).reshape(B, Ta, D)
        return context

    def forward(
        self,
        audio_feat: torch.Tensor,
        visual_feat: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_feat: [B, T_a, D] — audio features sau projection
            visual_feat: [B, T_v, D] — visual features sau projection
            audio_mask: [B, T_a] bool, True = frame thật (None → không mask, fallback v1)
            visual_mask: [B, T_v] bool, True = frame thật

        Returns:
            dict: fused [B,Ta,D], aligned_visual [B,Ta,D], gate_weights [B,Ta,2],
                  q_audio [B], q_visual [B], alignment_weights [B,Ta,Tv] or None
        """
        B, Ta, D = audio_feat.shape
        Tv = visual_feat.shape[1]
        audio_len = audio_mask.sum(1) if audio_mask is not None else None
        visual_len = visual_mask.sum(1) if visual_mask is not None else None

        # --- Bước 1: Alignment ---
        if Ta == Tv:
            aligned_visual = visual_feat
            alignment_weights = None
        else:
            aligned_visual = self._cross_attention_align(audio_feat, visual_feat, audio_len, visual_len)
            Q = F.normalize(self.q_proj(audio_feat), p=2, dim=-1)
            K = F.normalize(self.k_proj(visual_feat), p=2, dim=-1)
            sim = torch.matmul(Q, K.transpose(1, 2))  # [B, Ta, Tv]
            alignment_weights = F.softmax(sim, dim=-1)

        # --- Bước 2: Quality scoring ---
        q_audio = self.audio_quality(audio_feat)          # [B, T_a, 1]
        q_visual = self.visual_quality(aligned_visual)    # [B, T_a, 1]

        # --- Bước 3: Gated fusion ---
        combined = torch.cat([audio_feat, aligned_visual, q_audio, q_visual], dim=-1)  # [B, T_a, 2D+2]
        gate_weights = self.gate(combined)                # [B, T_a, 2]
        fused = (
            gate_weights[..., 0:1] * audio_feat +
            gate_weights[..., 1:2] * aligned_visual
        )

        # --- Bước 4: Residual gate ---
        g = torch.sigmoid(self.residual_gate)
        fused = (1.0 - g) * audio_feat + g * fused

        # --- Pooled quality (masked theo audio timeline; bỏ frame padding) ---
        if audio_mask is not None:
            am = audio_mask.float()
            denom = am.sum(1).clamp(min=1)
            q_audio_p = (q_audio.squeeze(-1) * am).sum(1) / denom
            q_visual_p = (q_visual.squeeze(-1) * am).sum(1) / denom
        else:
            q_audio_p = q_audio.mean(dim=1).squeeze(-1)
            q_visual_p = q_visual.mean(dim=1).squeeze(-1)

        return {
            'fused': fused,                               # [B, T_a, D]
            'aligned_visual': aligned_visual,             # [B, T_a, D]
            'gate_weights': gate_weights,                 # [B, T_a, 2]
            'q_audio': q_audio_p,                          # [B]
            'q_visual': q_visual_p,                        # [B]
            'alignment_weights': alignment_weights,       # [B, T_a, T_v] or None
        }
