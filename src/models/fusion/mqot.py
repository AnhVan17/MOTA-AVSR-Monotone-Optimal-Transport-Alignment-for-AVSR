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
        C_time = time_dist.unsqueeze(0).expand(B, -1, -1)  # [B, Ta, Tv]
        
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
        K = -cost / self.epsilon  # [B, Ta, Tv]
        
        # Initialize dual variables
        u = torch.zeros_like(K[:, :, 0])  # [B, Ta]
        v = torch.zeros_like(K[:, 0, :])  # [B, Tv]
        
        # Sinkhorn iterations
        for i in range(self.n_iters):
            u_prev = u.clone()
            
            # Update u
            u = -torch.logsumexp(K + v.unsqueeze(1), dim=2)
            
            # Update v
            v = -torch.logsumexp(K + u.unsqueeze(2), dim=1)
            
            # Check convergence
            err = torch.max(torch.abs(u - u_prev))
            if err < 1e-3:
                break
        
        # Compute transport plan
        transport = torch.exp(K + u.unsqueeze(2) + v.unsqueeze(1))
        
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
    Cross-attention with transport map guidance
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Gated residual (zero-init for stability)
        self.gate = nn.Parameter(torch.zeros(1))
        
        # Attention bias scale (learnable)
        self.scale = nn.Parameter(torch.tensor(10.0))
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        guide_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        """
        B, Ta, D = q.shape
        Tv = k.shape[1]
        
        # Optimize (Fix 0.9.3): Log first, then expand (Avoids huge log computation and extra copy)
        # 1. Compute Base Bias [B, Ta, Tv]
        base_bias = torch.log(guide_map + 1e-8) * self.scale
        
        # 2. Expand and Reshape [B, Ta, Tv] -> [B, H, Ta, Tv] -> [B*H, Ta, Tv]
        attn_bias = base_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(B * self.num_heads, Ta, Tv)
        
        # Cross-attention with bias
        context, _ = self.attn(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_bias,
            need_weights=False
        )
        
        # Gated residual
        output = self.norm(q + self.gate * context)
        
        return output
