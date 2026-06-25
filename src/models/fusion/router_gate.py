"""RGF — Router-Gated Fusion (Section B.3).

Hard *routing* over 3 experts (vs QualityGate's soft weighted-sum), to fight double
degradation by letting the model DROP a corrupted stream entirely instead of always
blending in garbage:

  expert 0 = audio-only   (audio features)
  expert 1 = visual-only  (visual aligned to the audio timeline, from QualityGate)
  expert 2 = fusion       (QualityGate's full quality-gated fusion)

A per-chunk router MLP reads [pooled audio, pooled visual, q_a, q_v] (q from NRQE) and emits
3 logits → Gumbel-Softmax(τ). Training mixes softly (gradient reaches all 3 experts, the
router and NRQE); τ is annealed high→low (soft→near-one-hot). Inference takes argmax
(one expert). `router_probs` is exposed for the Switch-Transformer load-balancing loss
(computed in losses.py) that prevents mode-collapse.

When use_rgf=true, RGF REPLACES QualityGate as Stage-1; QualityGate is reused as the
fusion expert (so a Phase-2 warm-start keeps its trained weights). Refs: Jang et al. 2017
(Gumbel-Softmax), Fedus et al. 2021 (load balancing), arXiv 2508.18734 (router-gated AVSR).
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nrqe import NRQE
from .quality_gate import QualityGate


class RouterGatedFusion(nn.Module):
    N_EXPERTS = 3  # audio-only, visual-only, fusion

    def __init__(
        self,
        d_model: int = 256,
        chunk_size: int = 15,
        tau0: float = 2.0,
        tau_min: float = 0.5,
        gamma: float = 0.99995,
        hidden: Optional[int] = None,
        num_heads: int = 4,
        dropout: float = 0.1,
        quality_gate: Optional[QualityGate] = None,
        nrqe: Optional[NRQE] = None,
    ):
        super().__init__()
        hidden = hidden or d_model
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.tau0, self.tau_min, self.gamma = tau0, tau_min, gamma

        # Reused as the "fusion" expert — injectable so a Phase-2 warm-start keeps trained weights.
        self.quality_gate = quality_gate or QualityGate(d_model, num_heads=num_heads, dropout=dropout)
        self.nrqe = nrqe or NRQE(d_model, chunk_size=chunk_size)
        self.router = nn.Sequential(
            nn.Linear(2 * d_model + 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.N_EXPERTS),
        )
        # annealing step counter (buffer → saved/restored on resume so τ schedule continues).
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def current_tau(self) -> float:
        return max(self.tau_min, self.tau0 * (self.gamma ** float(self._step)))

    def _chunk_pool(self, x: torch.Tensor, n: int, K: int, T: int) -> torch.Tensor:
        """Mean-pool [B, T, C] into [B, n, C] over fixed-size chunks (pad tail to n*K)."""
        B, _, C = x.shape
        pad = n * K - T
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        return x.view(B, n, K, C).mean(dim=2)

    def forward(self, audio_feat: torch.Tensor, visual_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, Ta, D = audio_feat.shape
        K = self.chunk_size
        n = (Ta + K - 1) // K

        # --- Experts ---
        qg = self.quality_gate(audio_feat, visual_feat)
        audio_expert = audio_feat           # expert 0
        visual_expert = qg["aligned_visual"]  # expert 1 — visual on the audio timeline
        fused_expert = qg["fused"]          # expert 2 — QualityGate fusion

        # --- Reliability (NRQE) on the common timeline ---
        nrqe_out = self.nrqe(audio_feat, visual_expert)
        q_a, q_v = nrqe_out["q_a"], nrqe_out["q_v"]  # [B, Ta]

        # --- Router (per chunk) ---
        a_ctx = self._chunk_pool(audio_feat, n, K, Ta)          # [B, n, D]
        v_ctx = self._chunk_pool(visual_expert, n, K, Ta)       # [B, n, D]
        qa_ctx = self._chunk_pool(q_a.unsqueeze(-1), n, K, Ta)  # [B, n, 1]
        qv_ctx = self._chunk_pool(q_v.unsqueeze(-1), n, K, Ta)  # [B, n, 1]
        logits = self.router(torch.cat([a_ctx, v_ctx, qa_ctx, qv_ctx], dim=-1))  # [B, n, 3]
        router_probs = F.softmax(logits, dim=-1)  # for load-balancing loss

        if self.training:
            self._step += 1
            weights = F.gumbel_softmax(logits, tau=self.current_tau(), hard=False, dim=-1)
        else:
            weights = F.one_hot(logits.argmax(dim=-1), self.N_EXPERTS).to(logits.dtype)

        # broadcast per-chunk weights to per-frame: [B, n, 3] → [B, Ta, 3]
        w = (
            weights.unsqueeze(2)
            .expand(B, n, K, self.N_EXPERTS)
            .reshape(B, n * K, self.N_EXPERTS)[:, :Ta, :]
        )

        experts = torch.stack([audio_expert, visual_expert, fused_expert], dim=-1)  # [B, Ta, D, 3]
        out = (experts * w.unsqueeze(2)).sum(dim=-1)  # [B, Ta, D]

        return {
            "fused": out,
            "router_probs": router_probs,        # [B, n, 3] — for load-balancing loss
            "router_weights": w,                  # [B, Ta, 3]
            "gate_weights": qg["gate_weights"],   # passthrough (output compat)
            "q_audio": q_a.mean(dim=1),           # [B]
            "q_visual": q_v.mean(dim=1),          # [B]
        }
