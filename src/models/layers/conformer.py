import torch
import torch.nn as nn


class ConformerBlock(nn.Module):
    """
    Conformer block: FFN + MHSA + Depthwise Conv + FFN

    Thay đổi so với v1:
    - Configurable residual scaling (Gulati et al., 2020 spec: all 0.5)
    - Final LayerNorm (stable output)

    Residual scaling theo Gulati et al. (2020):
      x = x + 0.5 * FFN1(x) + 0.5 * MHSA(x) + 0.5 * Conv(x) + 0.5 * FFN2(x)

    Mặc định giữ nguyên behavior cũ (FFN=0.5, MHA=1, Conv=1).
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        ffn_scale: float = 0.5,
        mha_scale: float = 1.0,
        conv_scale: float = 1.0,
    ):
        super().__init__()
        self.ffn_scale = ffn_scale
        self.mha_scale = mha_scale
        self.conv_scale = conv_scale

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

        # 3. Depthwise convolution
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

        # 5. Final LayerNorm
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            x: [B, T, D]
        """
        # FFN 1
        x = x + self.ffn_scale * self.ffn1(x)

        # Multi-head attention
        x_norm = self.norm_mha(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        x = x + self.mha_scale * self.dropout_mha(attn_out)

        # Convolution
        x_norm = self.norm_conv(x)
        x_conv = x_norm.transpose(1, 2)  # [B, D, T]
        x_conv = self.conv(x_conv)
        x = x + self.conv_scale * x_conv.transpose(1, 2)

        # FFN 2
        x = x + self.ffn_scale * self.ffn2(x)

        # Final LayerNorm
        x = self.final_norm(x)

        return x
