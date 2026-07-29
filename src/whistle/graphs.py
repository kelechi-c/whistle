"""Static-cache eager/CUDA-graph decode blocks for Whistle Qwen3-TTS."""

from functools import cache
from typing import Any, TypeAlias

import torch
from transformers import DynamicCache, StaticCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

Masks: TypeAlias = dict[str, torch.Tensor]


def _masks(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    position: torch.Tensor,
    cache: StaticCache,
) -> Masks:
    """Builds the official causal masks once, outside graph capture."""
    kwargs = {  
        "config": model.config,
        "input_embeds": inputs,
        "attention_mask": None,
        "cache_position": position,
        "past_key_values": cache,
    }
    masks = {"full_attention": create_causal_mask(**kwargs)}
    if getattr(model.config, "sliding_window", None) is not None or (
        "sliding_attention" in getattr(model.config, "layer_types", ())
    ):
        masks["sliding_attention"] = create_sliding_window_causal_mask(**kwargs)
    if any(mask is None for mask in masks.values()):
        raise RuntimeError("static-cache decode requires explicit causal masks")
    return masks  # type: ignore[return-value]


@cache
def predictor_embedding_weights(predictor: torch.nn.Module) -> torch.Tensor:
    """Stacks predictor embedding tables once per loaded module."""
    return torch.stack(tuple(layer.weight for layer in predictor.get_input_embeddings()))


def predictor_loop(
    predictor: Any,
    inputs: torch.Tensor,
    cache: StaticCache,
    weights: torch.Tensor,
    positions: tuple[torch.Tensor, ...],
    masks: tuple[Masks, ...],
    output: torch.Tensor,
) -> torch.Tensor:
    """Runs the complete greedy residual-code sequence into a static output."""
    hidden = predictor.model(
        inputs_embeds=predictor.small_to_mtp_projection(inputs),
        attention_mask=masks[0],
        past_key_values=cache,
        cache_position=positions[0],
        use_cache=True,
        return_dict=True,
    ).last_hidden_state
    token = predictor.lm_head[0](hidden[:, -1]).argmax(dim=-1)
    output[:, 0].copy_(token)
    for index in range(1, output.shape[1]):
        embedding = weights[index - 1, token].unsqueeze(1)
        hidden = predictor.model(
            inputs_embeds=predictor.small_to_mtp_projection(embedding),
            attention_mask=masks[index],
            past_key_values=cache,
            cache_position=positions[index],
            use_cache=True,
            return_dict=True,
        ).last_hidden_state
        token = predictor.lm_head[index](hidden[:, -1]).argmax(dim=-1)
        output[:, index].copy_(token)
    return output


compiled_predictor_loop = torch.compile(predictor_loop, mode="reduce-overhead")


class PredictorGraph:
    """Owns buffers for one Inductor-managed predictor-loop graph."""

    def __init__(
        self,
        predictor: Any,
        talker_hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.predictor = predictor
        self.device = device
        groups = predictor.config.num_code_groups - 1
        self.inputs = torch.zeros(
            (1, 2, talker_hidden_size), device=device, dtype=dtype
        )
        self.output = torch.zeros((1, groups), device=device, dtype=torch.long)
        self.weights = predictor_embedding_weights(predictor)
        self.cache = StaticCache(config=predictor.model.config, max_cache_len=groups + 1)
        config = predictor.model.config
        self.cache.early_initialization(
            1, config.num_key_value_heads, config.head_dim, dtype, device
        )
        positions = [torch.arange(2, device=device)]
        positions += [torch.tensor([index], device=device) for index in range(2, groups + 1)]
        self.positions = tuple(positions)
        seed = torch.zeros(
            (1, 2, config.hidden_size), device=device, dtype=dtype
        )
        token = torch.zeros(
            (1, 1, config.hidden_size), device=device, dtype=dtype
        )
        self.masks = tuple(
            _masks(predictor.model, seed if index == 0 else token, position, self.cache)
            for index, position in enumerate(self.positions)
        )
        self.loop = compiled_predictor_loop if device.type == "cuda" else predictor_loop
        self.ready = False

    def _step(self) -> None:
        self.loop(
            self.predictor,
            self.inputs,
            self.cache,
            self.weights,
            self.positions,
            self.masks,
            self.output,
        )

    def capture(self) -> None:
        """Warms the compiled loop and lets Inductor own its cudagraph tree."""
        if self.device.type != "cuda" or self.ready:
            return
        for _ in range(3):
            self.cache.reset()
            self._step()
        torch.cuda.synchronize(self.device)
        self.cache.reset()
        self.ready = True

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        """Copies one frame input and returns the reusable residual-code buffer."""
        self.inputs.copy_(inputs)
        self.cache.reset()
        self._step()
        return self.output


class TalkerGraph:
    """Dynamic-cache eager one-token talker decode.

    Uses DynamicCache so KV holds only populated slots and SDPA takes the
    mask-free flash path (q_len=1, no padding, is_compileable=False). This is
    the correct eager baseline. A StaticCache here needs an explicit mask to
    zero the un-populated slots; passing attention_mask=None on a StaticCache
    makes SDPA attend over zero-padded KV and degenerates to silence. No CUDA
    graph: DynamicCache.update concatenates per step, so KV addresses change
    every frame and cannot be captured.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        max_cache_len: int,
    ) -> None:
        self.model = model
        self.device = device
        self.max_cache_len = max_cache_len
        self.cache = DynamicCache(config=model.config)
        self.rope_deltas = torch.zeros((1, 1), device=device, dtype=torch.float32)

    def capture(self) -> None:
        """No-op: the talker runs eager with a growing dynamic cache."""

    def reset(self, prompt_length: int, rope_deltas: torch.Tensor | None = None) -> None:
        """Rebuilds the dynamic cache for a new request and validates the budget."""
        if prompt_length >= self.max_cache_len:
            raise ValueError("prompt exceeds the fixed talker cache capacity")
        self.cache = DynamicCache(config=self.model.config)
        self.rope_deltas.zero_()
        if rope_deltas is not None:
            self.rope_deltas.copy_(rope_deltas)

    def set_rope_deltas(self, rope_deltas: torch.Tensor) -> None:
        """Copies request mRoPE state without clearing the filled prefill cache."""
        self.rope_deltas.copy_(rope_deltas)

    def run(self, inputs: torch.Tensor, position: int) -> torch.Tensor:
        """Runs one eager talker forward, growing the cache; returns [1, 1, H]."""
        if position >= self.max_cache_len:
            raise ValueError("decode exceeds the fixed talker cache capacity")
        cache_position = torch.tensor([position], device=self.device, dtype=torch.long)
        position_ids = self.rope_deltas + cache_position.to(self.rope_deltas.dtype)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        return self.model(
            inputs_embeds=inputs,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        ).last_hidden_state


class DecodeGraphs:
    """Groups the two graph boundaries so a full-frame graph can replace them."""

    def __init__(
        self, talker: Any, device: torch.device, dtype: torch.dtype, max_cache_len: int
    ) -> None:
        self.predictor = PredictorGraph(
            talker.code_predictor, talker.config.hidden_size, device, dtype
        )
        self.talker = TalkerGraph(talker.model, device, dtype, max_cache_len)

    def capture(self) -> None:
        """Captures both reusable decode blocks once per loaded talker."""
        self.predictor.capture()
        self.talker.capture()


@cache
def decode_graphs(talker: torch.nn.Module, max_cache_len: int) -> DecodeGraphs:
    """Creates persistent graph objects and allocations once per talker module."""
    parameter = next(talker.parameters())
    graphs = DecodeGraphs(talker, parameter.device, parameter.dtype, max_cache_len)
    graphs.capture()
    return graphs
