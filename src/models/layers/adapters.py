import torch
import torch.nn as nn

class VisualAdapter(nn.Module):
    """
    Visual feature adapter
    
    Transform ResNet features (512) to Whisper-compatible space (768)
    inspired by Q-Former logic but preserving temporal dimension.
    
    Args:
        input_dim: Input feature dimension (default: 512)
        output_dim: Output feature dimension (default: 768)
        hidden_dim: Hidden layer dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Projection
        self.proj = nn.Linear(input_dim, output_dim)
        
        # Adapter layers (residual)
        self.adapter = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim] Visual features from ResNet
            
        Returns:
            out: [B, T, output_dim] Adapted features
        """
        # Project to target dimension
        x = self.proj(x)
        
        # Apply adapter with residual
        out = x + self.adapter(x)
        
        # Normalize
        out = self.norm(out)
        
        return out