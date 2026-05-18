"""Attention Components for OlmoEarth Inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-head attention module for Inference."""
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_norm: bool = False, norm_layer: nn.Module = nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # 必须保留这三个单独的 Linear 以兼容预训练权重 (不能合并成 qkv)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        # 映射并调整维度以便 Multi-head 注意力计算
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        # PyTorch SDPA 支持直接将布尔型一维掩码自动广播，极低内存消耗
        if attn_mask is not None:
            # attn_mask 形状是 [B, L]，扩展为 [B, 1, 1, L] 即可
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    """Standard MLP for Vision Transformer."""
    def __init__(self, in_features: int, hidden_features: int, act_layer: nn.Module = nn.GELU, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    """Learnable scaling layer."""
    def __init__(self, dim: int, init_values: float | None = None):
        super().__init__()
        # 只有明确给了数值，才去串联这个衰减器
        if init_values is not None:
            self.gamma = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma = None # 不创建参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 如果有 gamma，就乘上去；如果没有，就当透明导线直接输出 x
        return x * self.gamma if self.gamma is not None else x


class Block(nn.Module):
    """Transformer block with self attention and MLP (Inference Only)."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, qk_norm: bool = False, init_values: float | None = None, norm_layer: nn.Module = nn.LayerNorm, **kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, norm_layer=norm_layer)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x