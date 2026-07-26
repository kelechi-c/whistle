"""Weight-compatible Qwen3-TTS decoder primitives with explicit KV state."""

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F


class DecoderConfig(Protocol):
    """Structural type shared by talker, predictor, and codec transformers."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    attention_bias: bool
    attention_dropout: float


@dataclass(frozen=True, slots=True)
class LayerKV:
    """Stores one attention layer's immutable key/value history."""

    key: torch.Tensor
    value: torch.Tensor


KVCache = tuple[LayerKV, ...]


class RMSNorm(nn.Module):
    """Matches Qwen3-TTS float32 RMS normalization and parameter naming."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.variance_epsilon = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        dtype = values.dtype
        normalized = values.float()
        variance = normalized.square().mean(dim=-1, keepdim=True)
        return self.weight * (normalized * torch.rsqrt(variance + self.variance_epsilon)).to(dtype)


class RotaryEmbedding(nn.Module):
    """Builds standard or three-axis Qwen rotary embeddings without parameters."""

    def __init__(self, head_dim: int, theta: float, axes: int = 1) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.register_buffer(
            "inv_freq", self._frequencies(torch.device("cpu")), persistent=False
        )
        self.axes = axes

    def _frequencies(self, device: torch.device) -> torch.Tensor:
        """Creates the parameter-free RoPE frequencies on the requested device."""
        indices = torch.arange(
            0, self.head_dim, 2, dtype=torch.float32, device=device
        )
        return 1.0 / (self.theta ** (indices / self.head_dim))

    def materialize(self, device: torch.device) -> None:
        """Recreates the non-persistent buffer after meta-device construction."""
        self.inv_freq = self._frequencies(device)

    def forward(
        self, values: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.axes == 1:
            position_ids = position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids
            angles = position_ids.float().unsqueeze(-1) * self.inv_freq
        else:
            if position_ids.ndim == 2:
                position_ids = position_ids.unsqueeze(0).expand(self.axes, -1, -1)
            angles = position_ids.float().unsqueeze(-1) * self.inv_freq
        embedding = torch.cat((angles, angles), dim=-1)
        return embedding.cos().to(values.dtype), embedding.sin().to(values.dtype)


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    """Rotates the two contiguous halves used by the official implementation."""
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _multimodal_frequencies(
    values: torch.Tensor,
    sections: tuple[int, int, int],
    interleaved: bool,
) -> torch.Tensor:
    """Selects temporal/height/width frequencies exactly like Qwen mRoPE."""
    half = values.shape[-1] // 2
    axes = values[..., :half]
    if interleaved:
        selected = axes[0].clone()
        modalities = len(sections)
        for axis, size in enumerate(sections[1:], start=1):
            selected[..., axis : size * modalities : modalities] = axes[
                axis, ..., axis : size * modalities : modalities
            ]
    else:
        split = axes.split(sections, dim=-1)
        selected = torch.cat(tuple(part[index] for index, part in enumerate(split)))
    return torch.cat((selected, selected), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    sections: tuple[int, int, int] | None,
    interleaved: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies standard RoPE or the talker's three-axis variant."""
    if sections is not None:
        cos = _multimodal_frequencies(cos, sections, interleaved)
        sin = _multimodal_frequencies(sin, sections, interleaved)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin


def _repeat_kv(values: torch.Tensor, repeats: int) -> torch.Tensor:
    """Expands grouped key/value heads to the query-head count."""
    return values if repeats == 1 else values.repeat_interleave(repeats, dim=1)


class Attention(nn.Module):
    """Implements official-shaped GQA and returns new immutable cache state."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        multimodal_sections: tuple[int, int, int] | None = None,
        multimodal_interleaved: bool = False,
        normalize_qk: bool = True,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.groups = self.heads // self.kv_heads
        if self.heads % self.kv_heads:
            raise ValueError("num_attention_heads must divide num_key_value_heads")
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if normalize_qk else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if normalize_qk else nn.Identity()
        self.rotary = RotaryEmbedding(
            self.head_dim,
            config.rope_theta,
            axes=3 if multimodal_sections is not None else 1,
        )
        self.sections = multimodal_sections
        self.interleaved = multimodal_interleaved
        self.sliding_window = sliding_window

    def forward(
        self,
        values: torch.Tensor,
        position_ids: torch.Tensor,
        cache: LayerKV | None = None,
    ) -> tuple[torch.Tensor, LayerKV]:
        batch, length, _ = values.shape
        query = self.q_norm(self.q_proj(values).view(batch, length, self.heads, self.head_dim))
        key = self.k_norm(self.k_proj(values).view(batch, length, self.kv_heads, self.head_dim))
        value = self.v_proj(values).view(batch, length, self.kv_heads, self.head_dim)
        query, key = query.transpose(1, 2), key.transpose(1, 2)
        value = value.transpose(1, 2)
        cos, sin = self.rotary(values, position_ids)
        query, key = _apply_rope(
            query,
            key,
            cos,
            sin,
            self.sections,
            self.interleaved,
        )
        if cache is not None:
            key = torch.cat((cache.key, key), dim=-2)
            value = torch.cat((cache.value, value), dim=-2)
        new_cache = LayerKV(key, value)
        repeated_key = _repeat_kv(key, self.groups)
        repeated_value = _repeat_kv(value, self.groups)
        mask = self._attention_mask(length, key.shape[-2], values.device)
        attended = F.scaled_dot_product_attention(
            query,
            repeated_key,
            repeated_value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=mask is None and cache is None and length > 1,
        )
        merged = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(merged), new_cache

    def _attention_mask(
        self, query_length: int, key_length: int, device: torch.device
    ) -> torch.Tensor | None:
        """Builds only the masks not covered by SDPA's fast causal mode."""
        if query_length == 1 and self.sliding_window is None:
            return None
        if key_length == query_length and self.sliding_window is None:
            return None
        query_positions = torch.arange(
            key_length - query_length, key_length, device=device
        ).unsqueeze(-1)
        key_positions = torch.arange(key_length, device=device).unsqueeze(0)
        allowed = key_positions <= query_positions
        if self.sliding_window is not None:
            allowed &= key_positions > query_positions - self.sliding_window
        return allowed


class MLP(nn.Module):
    """Matches the official bias-free SwiGLU parameter layout."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(values)) * self.up_proj(values))


class DecoderLayer(nn.Module):
    """Combines official pre-norm attention and SwiGLU residual blocks."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        multimodal_sections: tuple[int, int, int] | None = None,
        multimodal_interleaved: bool = False,
        normalize_qk: bool = True,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(
            config,
            multimodal_sections=multimodal_sections,
            multimodal_interleaved=multimodal_interleaved,
            normalize_qk=normalize_qk,
            sliding_window=sliding_window,
        )
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        values: torch.Tensor,
        position_ids: torch.Tensor,
        cache: LayerKV | None = None,
    ) -> tuple[torch.Tensor, LayerKV]:
        attended, new_cache = self.self_attn(
            self.input_layernorm(values), position_ids, cache
        )
        values = values + attended
        return values + self.mlp(self.post_attention_layernorm(values)), new_cache


class Decoder(nn.Module):
    """Runs a weight-compatible layer stack and exposes its KV cache."""

    def __init__(
        self,
        config: DecoderConfig,
        *,
        multimodal_sections: tuple[int, int, int] | None = None,
        multimodal_interleaved: bool = False,
        normalize_qk: bool = True,
        sliding_window: int | None = None,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(
                config,
                multimodal_sections=multimodal_sections,
                multimodal_interleaved=multimodal_interleaved,
                normalize_qk=normalize_qk,
                sliding_window=sliding_window,
            )
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        values: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("cache layer count does not match the decoder")
        next_cache: list[LayerKV] = []
        for index, layer in enumerate(self.layers):
            previous = None if cache is None else cache[index]
            values, current = layer(values, position_ids, previous)
            next_cache.append(current)
        return self.norm(values), tuple(next_cache)
