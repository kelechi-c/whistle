import torch

from mini_qwen3.config import Qwen3Config
from mini_qwen3.model import Qwen3ForCausalLM
from mini_qwen3.latency import latency, LatencyTracker


def build_causal_mask(T: int, device: torch.device, sliding_window: int | None = None) -> torch.Tensor:
    mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
    if sliding_window is not None:
        for i in range(T):
            mask[i, :max(0, i - sliding_window + 1)] = float('-inf')
    return mask


class KVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, static_len: int = 0):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.static_len = static_len
        self._dynamic: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers
        self._static: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers

    def reset(self):
        self._dynamic = [None] * self.num_layers
        self._static = [None] * self.num_layers

    def get(self) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        if self.static_len:
            return self._static
        return self._dynamic

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        if self.static_len:
            pass
        else:
            self._dynamic[layer_idx] = (k, v)

    def init_static(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        for i in range(self.num_layers):
            self._static[i] = (
                torch.zeros(batch_size, self.num_kv_heads, self.static_len, self.head_dim, device=device, dtype=dtype),
                torch.zeros(batch_size, self.num_kv_heads, self.static_len, self.head_dim, device=device, dtype=dtype),
            )


class InferenceEngine:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        attn_impl: str = "eager",
        sliding_window: int | None = None,
        static_cache_len: int = 0,
    ):
        self.model = model
        self.config: Qwen3Config = model.config
        self.attn_impl = attn_impl
        self.sliding_window = sliding_window
        self.static_cache_len = static_cache_len
        self.device = next(model.parameters()).device
        self._kv_cache: KVCache = KVCache(
            model.config.num_hidden_layers,
            model.config.num_key_value_heads,
            model.config.head_dim,
            static_cache_len,
        )
        model.eval()

    def _get_kv_list(self) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        return self._kv_cache.get()

    def reset(self):
        self._kv_cache.reset()

    @latency("prefill")
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.shape[1]
        mask = build_causal_mask(T, input_ids.device, self.sliding_window)

        logits, new_kv = self.model(input_ids=input_ids, attn_fn=self.attn_impl, mask=mask)

        for i, (k, v) in enumerate(new_kv):
            self._kv_cache.update(i, k, v)

        return logits

    @latency("decode_step")
    def decode_step(self, token: torch.Tensor) -> torch.Tensor:
        logits, new_kv = self.model(input_ids=token, attn_fn=self.attn_impl, kv_cache=self._get_kv_list())

        for i, (k, v) in enumerate(new_kv):
            self._kv_cache.update(i, k, v)

        return logits

    @latency("generate")
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        self.reset()
        eos_token_id = eos_token_id or self.config.eos_token_id

        with torch.no_grad():
            logits = self.prefill(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = [input_ids, next_token]

            for _ in range(max_new_tokens - 1):
                logits = self.decode_step(next_token)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_token)

                if next_token.item() == eos_token_id:
                    break

        return torch.cat(generated, dim=1)

    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        draft_model: Qwen3ForCausalLM,
        max_new_tokens: int = 128,
        gamma: int = 4,
    ) -> torch.Tensor:
        draft_engine = InferenceEngine(draft_model, self.attn_impl, self.sliding_window)
        self.reset()
        draft_engine.reset()

        with torch.no_grad():
            self.prefill(input_ids)
            draft_engine.prefill(input_ids)

            generated = [input_ids]
            cur = input_ids[:, -1:]
            steps = 0

            while steps < max_new_tokens:
                draft_tokens: list[torch.Tensor] = []
                for _ in range(gamma):
                    logits = draft_engine.decode_step(cur)
                    cur = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    draft_tokens.append(cur)

                merged = torch.cat(draft_tokens, dim=1)
                logits, _ = self.model(input_ids=merged)

                accepted = 0
                for i, tok in enumerate(draft_tokens):
                    if logits[:, i, :].argmax(dim=-1).item() == tok.item():
                        accepted += 1
                        self.decode_step(tok)
                        draft_engine.decode_step(tok)
                        generated.append(tok)
                        cur = tok
                    else:
                        target_tok = logits[:, i, :].argmax(dim=-1, keepdim=True)
                        self.decode_step(target_tok)
                        draft_engine.decode_step(target_tok)
                        generated.append(target_tok)
                        cur = target_tok
                        break
                else:
                    logits = self.decode_step(cur)
                    bonus = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    draft_engine.decode_step(bonus)
                    generated.append(bonus)
                    cur = bonus

                steps += accepted + 1
                if cur.item() == self.config.eos_token_id:
                    break

        return torch.cat(generated, dim=1)
