import torch
import torch.nn as nn

class ConformerBlock(nn.Module):
    """
    Conformer block: Conv + Attention
    
    Best for speech: captures acoustic (conv) + context (attention)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Feed-forward 1
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        # 2. Multi-head attention
        self.norm_mha = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            d_model, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout_mha = nn.Dropout(dropout)
        
        # 3. Depthwise convolution (KEY!)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(
                d_model, d_model,
                kernel_size=conv_kernel,
                padding=conv_kernel // 2,
                groups=d_model  # Depthwise
            ),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 4. Feed-forward 2
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            x: [B, T, D]
        """
        # FFN 1
        x = x + 0.5 * self.ffn1(x)
        
        # Multi-head attention
        x_norm = self.norm_mha(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.dropout_mha(attn_out)
        
        # Convolution
        x_norm = self.norm_conv(x)
        x_conv = x_norm.transpose(1, 2)  # [B, D, T]
        x_conv = self.conv(x_conv)
        x = x + x_conv.transpose(1, 2)
        
        # FFN 2
        x = x + 0.5 * self.ffn2(x)
        
        return x
