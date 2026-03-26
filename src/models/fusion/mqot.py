import torch
import torch.nn as nn
import torch.nn.functional as F


class MQOTLayer(nn.Module):
    """
    MQOT v2 — Multi-modal Quality-aware Optimal Transport.

    Thay đổi so với v1:
    - Learnable epsilon (Softplus → luôn dương, tự giảm dần qua training)
    - Unbalanced Sinkhorn (Chizat et al., 2018): cho phép Ta ≠ Tv bằng KL penalty
    - Multi-head OT: mỗi head có transport plan riêng (AlignMamba style)
    - Cost function learnable

    Reference: PROGOT (NeurIPS 2025), AlignMamba (CVPR 2025)
    """

    def __init__(
        self,
        dim: int = 768,
        lambda_time: float = 0.5,
        lambda_qual: float = 5.0,
        epsilon_init: float = 0.15,
        epsilon_min: float = 0.01,
        n_iters: int = 20,
        use_unbalanced: bool = True,
        kl_penalty: float = 0.1,
        num_heads: int = 1,
    ):
        super().__init__()
        self.n_iters = n_iters
        self.use_unbalanced = use_unbalanced
        self.kl_penalty = kl_penalty
        self.num_heads = num_heads

        # Learnable epsilon (Softplus → always positive)
        # Annealing: epsilon giảm dần → transport plan rõ hơn
        self.log_epsilon = nn.Parameter(torch.tensor(epsilon_init).log())

        # Cost weights (learnable)
        self.lambda_time = nn.Parameter(torch.tensor(lambda_time))
        self.lambda_qual = nn.Parameter(torch.tensor(lambda_qual))

        # Learnable temporal bias network
        self.time_bias_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, num_heads)
        )

    @property
    def epsilon(self) -> float:
        """Annealing: epsilon giảm theo training (torch.clamp prevents NaN)."""
        return F.softplus(self.log_epsilon).clamp(min=0.005).item()

    def compute_cost(
        self,
        audio_emb: torch.Tensor,
        video_emb: torch.Tensor,
        quality: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cost matrix C[Ta, Tv] cho mỗi batch.

        C = cosine_distance + λ_time * temporal_smoothness + λ_qual * quality_penalty
        """
        B, Ta, D = audio_emb.shape
        Tv = video_emb.shape[1]
        H = max(1, self.num_heads)
        device = audio_emb.device

        # Cosine distance
        a_norm = F.normalize(audio_emb, p=2, dim=-1)
        v_norm = F.normalize(video_emb, p=2, dim=-1)
        C_feat = (1.0 - torch.bmm(a_norm, v_norm.transpose(1, 2))).unsqueeze(1)  # [B, 1, Ta, Tv]

        # Temporal cost: learnable bias per head
        i = torch.arange(Ta, device=device).float() / Ta
        j = torch.arange(Tv, device=device).float() / Tv
        time_coord = torch.stack([
            i.unsqueeze(1).expand(-1, Tv),
            j.unsqueeze(0).expand(Ta, -1)
        ], dim=-1)  # [Ta, Tv, 2]
        time_bias = self.time_bias_net(time_coord).permute(2, 0, 1)  # [H, Ta, Tv]

        # Quality penalty
        Q_penalty = (1.0 - quality).unsqueeze(1).expand(-1, Ta, -1).unsqueeze(1)  # [B, 1, Ta, Tv]

        # Combine: broadcast C_feat[1,Ta,Tv] → [B,H,Ta,Tv]
        total_cost = (
            C_feat +
            self.lambda_time * time_bias.unsqueeze(0) +
            self.lambda_qual * Q_penalty
        )

        # Expand for multi-head: [B, H, Ta, Tv]
        total_cost = total_cost.expand(-1, H, -1, -1)

        return total_cost  # [B, H, Ta, Tv]

    def sinkhorn_unbalanced(
        self,
        cost: torch.Tensor,
        Ta: int,
        Tv: int,
    ) -> torch.Tensor:
        """
        Unbalanced Sinkhorn với KL penalty (Chizat et al., 2018).

        Cho phép Ta ≠ Tv bằng cách thay hard marginal constraints bằng soft KL penalty.

        Minimize: <P, C> + ε * KL(P | a⊗b)
        với a, b là marginal distributions (uniform default).

        Update rule:
          u_t+1 = logsumexp(K + v_t) / ε  *  (1/(1+α))  +  u_t * (α/(1+α))
          v_t+1 = logsumexp(K^T + u_t) / ε * (1/(1+α))  +  v_t * (α/(1+α))
        """
        B, H = cost.shape[0], cost.shape[1]
        eps = F.softplus(self.log_epsilon).clamp(min=0.005)
        alpha = self.kl_penalty
        denom = 1.0 + alpha

        u = torch.zeros(B, H, Ta, device=cost.device)
        v = torch.zeros(B, H, Tv, device=cost.device)

        eps_tensor = F.softplus(self.log_epsilon).clamp(min=0.005)

        for _ in range(self.n_iters):
            u_prev, v_prev = u.clone(), v.clone()

            # u update: cost[B,H,Ta,Tv] + v[B,H,Tv] → v.unsqueeze(2) = [B,H,1,Tv]
            K_plus_v = (-cost + v.unsqueeze(2)) / eps_tensor
            u = -torch.logsumexp(K_plus_v, dim=3) * (1.0 / denom) + u_prev * (alpha / denom)

            # v update: cost^T[B,H,Tv,Ta] + u[B,H,Ta] → u.unsqueeze(2) = [B,H,1,Ta]
            Kt_plus_u = (-cost.transpose(2, 3) + u.unsqueeze(2)) / eps_tensor
            v = -torch.logsumexp(Kt_plus_u, dim=3) * (1.0 / denom) + v_prev * (alpha / denom)

            if torch.max(torch.abs(u - u_prev)) < 1e-3:
                break

        # log_P: u[B,H,Ta] + v[B,H,Tv]
        log_P = (-cost + u.unsqueeze(3) + v.unsqueeze(2)) / eps_tensor
        return torch.exp(log_P)  # [B, H, Ta, Tv]

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        quality: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            audio: [B, Ta, D]
            video: [B, Tv, D]
            quality: [B, Tv]

        Returns:
            transport_map: [B, H, Ta, Tv] — multi-head OT plans
                            H=1 → backward compatible với GuidedAttention
        """
        B, Ta, D = audio.shape
        Tv = video.shape[1]
        H = max(1, self.num_heads)

        cost = self.compute_cost(audio, video, quality)  # [B, H, Ta, Tv]
        P = self.sinkhorn_unbalanced(cost, Ta, Tv)      # [B, H, Ta, Tv]

        # Row-normalize: P[i, :, j].sum() = 1
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)

        return P
    
    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        quality: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            audio: [B, Ta, D]
            video: [B, Tv, D]
            quality: [B, Tv]

        Returns:
            transport_map: [B, H, Ta, Tv] — multi-head OT plans (H=1 if num_heads=1)
        """
        Ta = audio.shape[1]
        Tv = video.shape[1]

        # Compute cost: [B, H, Ta, Tv]
        cost = self.compute_cost(audio, video, quality)

        # Solve unbalanced OT
        P = self.sinkhorn_unbalanced(cost, Ta, Tv)  # [B, H, Ta, Tv]

        # Row-normalize
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)

        return P  # [B, H, Ta, Tv]


class QualityEstimator(nn.Module):
    """
    Estimate per-frame visual quality
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        self.estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output ∈ [0, 1]
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, T, D] Visual features
            
        Returns:
            quality: [B, T] Quality scores
        """
        return self.estimator(features).squeeze(-1)


class GuidedAttention(nn.Module):
    """
    Manual multi-head cross-attention với transport map bias trực tiếp.

    Fix P0-3: PyTorch nn.MultiheadAttention mong đợi attn_mask shape [B, H, L, S].
    Code cũ truyền [B*H, L, S] → shape mismatch → transport plan KHÔNG ảnh hưởng attention.

    Giải pháp: Tự implement attention, cộng transport bias trực tiếp vào attention scores.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        # Gated residual (zero-init → identity initially → stable)
        self.gate = nn.Parameter(torch.zeros(1))

        # Learnable transport bias scale (replaces fixed self.scale)
        self.bias_scale = nn.Parameter(torch.tensor(10.0))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        guide_map: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            q: [B, Ta, D] — audio query
            k: [B, Tv, D] — visual key
            v: [B, Tv, D] — visual value
            guide_map: [B, Ta, Tv] OR [B, H, Ta, Tv] — transport plan(s) từ MQOT

        Returns:
            output: [B, Ta, D] — attended features
        """
        B, Ta, D = q.shape
        Tv = k.shape[1]

        # --- Bước 1: QKV projection ---
        Q = self.W_q(q).view(B, Ta, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Ta, hd]
        K = self.W_k(k).view(B, Tv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Tv, hd]
        V = self.W_v(v).view(B, Tv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, Tv, hd]

        # --- Bước 2: Attention scores + transport bias ---
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, Ta, Tv]

        # guide_map: [B, Ta, Tv] (H=1) or [B, H_mqot, Ta, Tv] (multi-head)
        if guide_map is not None:
            if guide_map.dim() == 3:
                # [B, Ta, Tv] → expand → [B, H, Ta, Tv]
                guide_map = guide_map.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            else:
                # [B, H_mqot, Ta, Tv] — broadcast to [B, H_attn, Ta, Tv]
                # repeat_interleave if H_attn % H_mqot == 0, else mean
                H_mqot = guide_map.size(1)
                if H_mqot == self.num_heads:
                    pass  # Already matching
                elif self.num_heads % H_mqot == 0:
                    guide_map = guide_map.repeat_interleave(
                        self.num_heads // H_mqot, dim=1
                    )
                else:
                    guide_map = guide_map.mean(dim=1, keepdim=True).expand(-1, self.num_heads, -1, -1)
            transport_bias = torch.log(guide_map + 1e-8) * F.softplus(self.bias_scale)
            scores = scores + transport_bias

        # --- Bước 3: Softmax + Dropout ---
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        # --- Bước 4: Aggregate values ---
        context = torch.matmul(attn_weights, V)  # [B, H, Ta, hd]
        context = context.transpose(1, 2).reshape(B, Ta, D)  # [B, Ta, D]
        output = self.W_o(context)

        # --- Bước 5: Gated residual ---
        output = self.norm(q + F.sigmoid(self.gate) * output)

        return output
