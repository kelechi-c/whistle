"""Small Qwen-style decoder primitives shared by both token generators."""

import torch
from torch import nn
from torch.nn import functional as F

from nero.model.types import ModelConfig


class RMSNorm(nn.Module):
    """Normalizes in float32 and restores the input dtype for stable inference."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        variance = values.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(values.dtype)


class RotaryEmbedding(nn.Module):
    """Applies temporal RoPE to query and key heads."""

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(query.shape[-2], device=query.device)
        angles = torch.outer(positions.float(), self.frequencies)
        cos = angles.cos()[None, None, :, :].to(query.dtype)
        sin = angles.sin()[None, None, :, :].to(query.dtype)
        return self._rotate(query, cos, sin), self._rotate(key, cos, sin)

    @staticmethod
    def _rotate(
        values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        even, odd = values[..., 0::2], values[..., 1::2]
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)


class Attention(nn.Module):
    """Implements causal grouped-query self-attention with Q/K normalization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.attention_heads
        self.kv_heads = config.key_value_heads
        self.head_dim = config.hidden_size // self.heads
        if self.head_dim % 2 or self.heads % self.kv_heads:
            raise ValueError("attention dimensions must support RoPE and GQA")
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.rope = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, _ = values.shape
        query = self.q_proj(values).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(values).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(values).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        query, key = self.rope(self.q_norm(query), self.k_norm(key))
        repeats = self.heads // self.kv_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        merged = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(merged)


class MLP(nn.Module):
    """Applies the Qwen SwiGLU feed-forward transformation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(values)) * self.up_proj(values))


class DecoderLayer(nn.Module):
    """Combines pre-normalized attention and MLP residual blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.self_attn(self.input_layernorm(values))
        return values + self.mlp(self.post_attention_layernorm(values))


class Decoder(nn.Module):
    """Runs a stack of decoder layers followed by RMS normalization."""

    def __init__(self, config: ModelConfig, layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(DecoderLayer(config) for _ in range(layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            values = layer(values)
        return self.norm(values)
