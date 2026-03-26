import torch
import torch.nn as nn
import torch.nn.functional as F

class MQOTLayer(nn.Module):
    """
    Multi-modal Quality-aware Optimal Transport
    
    Computes soft alignment between audio and visual features
    considering:
    1. Feature similarity (Cosine distance)
    2. Temporal smoothness (L1 distance in time)
    3. Quality penalty (Down-weight low-quality frames)
    """
    
    def __init__(
        self,
        lambda_time: float = 0.5,
        lambda_qual: float = 5.0,
        epsilon: float = 0.15,
        n_iters: int = 20
    ):
        super().__init__()
        
        # Learnable parameters (allow fine-tuning)
        self.lambda_time = nn.Parameter(torch.tensor(lambda_time))
        self.lambda_qual = nn.Parameter(torch.tensor(lambda_qual))
        self.epsilon = epsilon
        self.n_iters = n_iters
    
    def compute_cost(
        self,
        audio_emb: torch.Tensor,
        video_emb: torch.Tensor,
        quality: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cost matrix for optimal transport
        """
        B, Ta, D = audio_emb.shape
        Tv = video_emb.shape[1]
        device = audio_emb.device
        
        # 1. Feature cost (Cosine distance)
        # Normalize embeddings
        a_norm = F.normalize(audio_emb, p=2, dim=-1)  # [B, Ta, D]
        v_norm = F.normalize(video_emb, p=2, dim=-1)  # [B, Tv, D]
        
        # Compute similarity
        similarity = torch.bmm(a_norm, v_norm.transpose(1, 2))  # [B, Ta, Tv]
        
        # Convert to distance
        C_feat = 1.0 - similarity  # [B, Ta, Tv]
        
        # 2. Temporal cost (Smooth alignment)
        # Encourage nearby frames to align
        i = torch.arange(Ta, device=device).float().unsqueeze(1)  # [Ta, 1]
        j = torch.arange(Tv, device=device).float().unsqueeze(0)  # [1, Tv]
        
        # Normalized temporal distance
        time_dist = torch.abs(i / Ta - j / Tv)  # [Ta, Tv]
        C_time = time_dist.unsqueeze(0).expand(B, -1, -1).clone()  # [B, Ta, Tv]
        
        # 3. Quality penalty
        # Low quality → High cost
        Q_penalty = (1.0 - quality).unsqueeze(1).expand(-1, Ta, -1)  # [B, Ta, Tv]
        
        # Combine costs
        total_cost = (
            C_feat +
            self.lambda_time * C_time +
            self.lambda_qual * Q_penalty
        )
        
        return total_cost
    
    def sinkhorn(self, cost: torch.Tensor) -> torch.Tensor:
        """
        Sinkhorn algorithm in log-space (numerically stable)
        """
        # Log-space formulation
        B, Ta, Tv = cost.shape
        K = -cost / self.epsilon  # [B, Ta, Tv]

        u = torch.zeros(B, Ta, device=cost.device)
        v = torch.zeros(B, Tv, device=cost.device)

        for _ in range(self.n_iters):
            u_prev, v_prev = u.clone(), v.clone()

            # Fix u using current v (row normalization)
            u = -torch.logsumexp(K + v.unsqueeze(1), dim=2)
            # Fix v using updated u (column normalization)
            v = -torch.logsumexp(K.transpose(1, 2) + u.unsqueeze(1), dim=2)

            if torch.max(torch.abs(u - u_prev)) < 1e-3:
                break

        # Row-stochastic transport plan: P.sum(dim=-1) = 1 for each row
        # NOTE: Log-space Sinkhorn only guarantees both margins converge to 1
        # when Ta == Tv. With Ta != Tv, explicit row-normalization is needed
        # to ensure P[i, :].sum() == 1 for CTC/GuidedAttention weighting.
        transport = torch.exp(K + u.unsqueeze(2) + v.unsqueeze(1))
        transport = transport / (transport.sum(dim=-1, keepdim=True) + 1e-8)

        return transport
    
    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
        quality: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            audio: [B, Ta, D] Audio features
            video: [B, Tv, D] Visual features
            quality: [B, Tv] Quality scores ∈ [0, 1]
            
        Returns:
            transport_map: [B, Ta, Tv] Optimal transport plan
        """
        # Compute cost matrix
        cost = self.compute_cost(audio, video, quality)
        
        # Solve optimal transport
        transport_map = self.sinkhorn(cost)
        
        return transport_map


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
        guide_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            q: [B, Ta, D] — audio query
            k: [B, Tv, D] — visual key
            v: [B, Tv, D] — visual value
            guide_map: [B, Ta, Tv] — transport plan P (từ Sinkhorn)

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
        # Base scores: [B, H, Ta, Tv]
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Transport bias: log(P_ij) × scale
        # guide_map [B, Ta, Tv] → [B, 1, Ta, Tv] → broadcast sang [B, H, Ta, Tv]
        if guide_map is not None:
            transport_bias = torch.log(guide_map + 1e-8).unsqueeze(1) * F.softplus(self.bias_scale)
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
