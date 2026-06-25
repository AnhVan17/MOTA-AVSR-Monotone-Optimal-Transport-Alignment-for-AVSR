"""NRQE — Noise-Robust Quality Estimator (Section B.2).

Per-frame reliability scores q_a, q_v ∈ [0,1] for the audio and visual streams, computed
on a COMMON timeline (visual already aligned to the audio timeline upstream). Two signals:

  (i)  a per-modality quality head (small MLP → sigmoid), and
  (ii) cross-modal cosine consistency over time chunks — high agreement between the two
       streams in a chunk ⇒ both reliable; disagreement ⇒ at least one is corrupted.

Consistency modulates the per-modality scores. MC-dropout (proposal Eq.565, K=5 forwards)
is intentionally DROPPED: 5× forward conflicts with the ~2 GiB budget and measures the
wrong (epistemic, not aleatoric) uncertainty. A learnable heteroscedastic log-variance is
available behind `learn_logvar` (default off) per Kendall & Gal (2017).
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class NRQE(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        chunk_size: int = 15,
        hidden: Optional[int] = None,
        learn_logvar: bool = False,
    ):
        super().__init__()
        hidden = hidden or max(d_model // 2, 1)
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.learn_logvar = learn_logvar

        self.audio_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        self.visual_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        # cross-modal consistency (scalar per frame) → per-modality bias on the quality logit.
        self.cons_gate = nn.Linear(1, 2)
        if learn_logvar:
            self.logvar_head = nn.Linear(d_model * 2, 2)

    def _chunk_consistency(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Per-frame cross-modal cosine consistency ∈ [0,1], computed chunk-wise then
        broadcast back to frames. a, v: [B, T, D] on a common timeline → [B, T, 1]."""
        B, T, D = a.shape
        K = self.chunk_size
        n = (T + K - 1) // K
        pad = n * K - T
        if pad:
            a = F.pad(a, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        a_ch = a.view(B, n, K, D).mean(dim=2)  # [B, n, D]
        v_ch = v.view(B, n, K, D).mean(dim=2)
        cos = F.cosine_similarity(a_ch, v_ch, dim=-1)  # [B, n] ∈ [-1, 1]
        c = ((cos + 1.0) * 0.5).clamp(0.0, 1.0)  # [B, n] ∈ [0, 1]
        c = c.unsqueeze(-1).expand(B, n, K).reshape(B, n * K, 1)[:, :T, :]
        return c  # [B, T, 1]

    def forward(self, audio: torch.Tensor, visual_aligned: torch.Tensor) -> Dict[str, torch.Tensor]:
        """audio, visual_aligned: [B, T, D] (same timeline). Returns q_a, q_v, consistency [B, T]."""
        c = self._chunk_consistency(audio, visual_aligned)  # [B, T, 1]
        bias = self.cons_gate(c)  # [B, T, 2]
        q_a = torch.sigmoid(self.audio_head(audio) + bias[..., 0:1]).squeeze(-1)  # [B, T]
        q_v = torch.sigmoid(self.visual_head(visual_aligned) + bias[..., 1:2]).squeeze(-1)
        out = {"q_a": q_a, "q_v": q_v, "consistency": c.squeeze(-1)}
        if self.learn_logvar:
            out["logvar"] = self.logvar_head(torch.cat([audio, visual_aligned], dim=-1))
        return out
