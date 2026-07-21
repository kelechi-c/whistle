import torch
import torch.nn as nn
import torch.nn.functional as F

from mini_qwen3.config import Qwen3Config


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            variance = x.float().pow(2).mean(-1, keepdim=True)
            return self.weight * (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        dim = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv_freq @ pos).transpose(1, 2)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, H, T, D = x.shape
    if n_rep == 1:
        return x
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


def _eager_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, scale: float) -> torch.Tensor:
    attn = torch.matmul(q, k.transpose(2, 3)) * scale
    if mask is not None:
        attn = attn + mask
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return attn @ v


def _sdpa_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None, scale: float) -> torch.Tensor:
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


ATTN_IMPLS = {"eager": _eager_attn, "sdpa": _sdpa_attn}


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self.scale = self.head_dim ** -0.5
        self.sliding_window = config.sliding_window if config.use_sliding_window else None

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_fn: str = "eager",
        mask: torch.Tensor | None = None,
        cache_k: torch.Tensor | None = None,
        cache_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = apply_rotary(q, k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        is_decode = cache_k is not None
        if is_decode:
            k = torch.cat([cache_k, k], dim=2)
            v = torch.cat([cache_v, v], dim=2)

        if self.sliding_window is not None and k.shape[2] > self.sliding_window:
            if is_decode:
                k = k[:, :, -self.sliding_window:]
                v = v[:, :, -self.sliding_window:]
                if mask is not None:
                    mask = mask[..., -self.sliding_window:]

        kv_k, kv_v = k.detach(), v.detach()

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        out = ATTN_IMPLS[attn_fn](q, k, v, mask, self.scale)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o_proj(out)

        return out, kv_k, kv_v


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_fn: str = "eager",
        mask: torch.Tensor | None = None,
        cache_k: torch.Tensor | None = None,
        cache_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        x = self.input_layernorm(x)
        x, new_k, new_v = self.self_attn(x, cos, sin, attn_fn, mask, cache_k, cache_v)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x, new_k, new_v


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attn_fn: str = "eager",
        mask: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        if input_ids is not None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        B, T = x.shape[:2]

        kv_cache = kv_cache or [None] * len(self.layers)

        if position_ids is None:
            offset = 0
            if kv_cache[0] is not None:
                offset = kv_cache[0][0].shape[2]
            position_ids = torch.arange(T, device=x.device).unsqueeze(0) + offset

        cos, sin = self.rotary_emb(x, position_ids)

        new_kv = []
        for i, layer in enumerate(self.layers):
            cache_k, cache_v = kv_cache[i] if kv_cache[i] is not None else (None, None)
            x, new_k, new_v = layer(x, cos[:, -T:], sin[:, -T:], attn_fn, mask, cache_k, cache_v)
            new_kv.append((new_k, new_v))

        x = self.norm(x)
        return x, new_kv


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attn_fn: str = "eager",
        mask: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        hidden_states, new_kv = self.model(input_ids, inputs_embeds, position_ids, attn_fn, mask, kv_cache)
        logits = self.lm_head(hidden_states)
        return logits, new_kv

    @classmethod
    def from_pretrained(cls, model_name: str, device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        from transformers import AutoConfig, AutoModelForCausalLM

        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config = Qwen3Config(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=getattr(hf_config, 'num_key_value_heads', hf_config.num_attention_heads),
            head_dim=getattr(hf_config, 'head_dim', hf_config.hidden_size // hf_config.num_attention_heads),
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=getattr(hf_config, 'rope_theta', 1_000_000.0),
            sliding_window=getattr(hf_config, 'sliding_window', None),
            use_sliding_window=getattr(hf_config, 'use_sliding_window', False),
            tie_word_embeddings=getattr(hf_config, 'tie_word_embeddings', True),
            bos_token_id=hf_config.bos_token_id,
            eos_token_id=hf_config.eos_token_id,
            pad_token_id=getattr(hf_config, 'pad_token_id', None),
        )

        model = cls(config)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device, trust_remote_code=True
        )
        state = hf_model.state_dict()
        del hf_model

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"missing keys: {missing}")
        if unexpected:
            print(f"unexpected keys: {unexpected}")
        return model.to(device=device, dtype=dtype)

    @classmethod
    def tiny_test(cls, device: str = "cpu"):
        from mini_qwen3.config import TINY_CONFIG
        return cls(TINY_CONFIG).to(device=device)
