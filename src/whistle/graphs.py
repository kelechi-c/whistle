"""Static-cache eager/CUDA-graph decode blocks for Whistle Qwen3-TTS."""

from functools import cache
from typing import Any, Literal, TypeAlias

import torch
from transformers import DynamicCache, StaticCache
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)

Masks: TypeAlias = dict[str, torch.Tensor]
TalkerMode: TypeAlias = Literal[
    "official-eager",
    "predictor-ffn-graphs",
    "compile",
    "cuda-graph",
]


class PrefixStaticLayer(CacheLayerMixin):
    """Preallocates predictor KV storage while exposing only valid positions."""

    def __init__(self, max_cache_len: int) -> None:
        super().__init__()
        self.max_cache_len = max_cache_len
        self.cumulative_length = 0

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        """Allocates a fixed backing buffer from the first update shape."""
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device
        shape = (
            self.max_batch_size,
            self.num_heads,
            self.max_cache_len,
            self.head_dim,
        )
        self.keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        self.values = torch.empty(shape, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Writes new states and returns views with dynamic-cache shapes."""
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        position = (
            cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        )
        if position is None:
            position = torch.arange(
                self.cumulative_length,
                self.cumulative_length + key_states.shape[-2],
                device=key_states.device,
            )
        self.keys.index_copy_(2, position, key_states)
        self.values.index_copy_(2, position, value_states)
        self.cumulative_length += key_states.shape[-2]
        return (
            self.keys[..., : self.cumulative_length, :],
            self.values[..., : self.cumulative_length, :],
        )

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        """Reports the same active attention length as DynamicCache."""
        return self.cumulative_length + cache_position.shape[0], 0

    def get_seq_length(self) -> int:
        """Returns the populated prefix length without scanning GPU memory."""
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Retains dynamic-cache mask semantics despite bounded storage."""
        return -1

    def reset(self) -> None:
        """Invalidates old slots without clearing the backing tensors."""
        self.cumulative_length = 0


def prefix_cache(config: Any, max_cache_len: int) -> Cache:
    """Creates one prefix-visible cache layer per predictor decoder layer."""
    return Cache(
        layers=[
            PrefixStaticLayer(max_cache_len)
            for _ in range(config.num_hidden_layers)
        ]
    )


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


def talker_step(
    model: Any,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    position_ids: torch.Tensor,
    cache: StaticCache,
    cache_position: torch.Tensor,
) -> torch.Tensor:
    """Runs one inner-talker token with explicit fixed-cache state."""
    return model(
        inputs_embeds=inputs,
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    ).last_hidden_state


compiled_talker_step = torch.compile(talker_step, mode="reduce-overhead")
graphable_talker_step = torch.compile(
    talker_step, mode="max-autotune-no-cudagraphs"
)


class OfficialPredictor:
    """Runs the residual codebooks through the official dynamic-cache generator."""

    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor
        self.groups = predictor.config.num_code_groups - 1

    def capture(self) -> None:
        """Leaves the correctness reference eager and allocation-compatible."""

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        """Returns the official greedy residual sequence for one talker token."""
        return self.predictor.generate(
            inputs_embeds=inputs,
            max_new_tokens=self.groups,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        ).sequences


class PredictorGraphs:
    """Captures one exact eager predictor step for each residual codebook."""

    def __init__(
        self,
        predictor: Any,
        talker_hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.predictor = predictor
        self.device = device
        self.groups = predictor.config.num_code_groups - 1
        self.inputs = torch.zeros(
            (1, 2, talker_hidden_size), device=device, dtype=dtype
        )
        self.tokens = torch.zeros((1, self.groups), device=device, dtype=torch.long)
        self.cache = prefix_cache(predictor.model.config, self.groups + 1)
        self.graphs: list[torch.cuda.CUDAGraph] = []

    def _step(self, index: int) -> None:
        """Runs one original eager predictor step into the token buffer."""
        if index == 0:
            inputs = self.inputs
        else:
            embedding = self.predictor.get_input_embeddings()[index - 1]
            inputs = embedding(self.tokens[:, index - 1]).unsqueeze(1)
        hidden = self.predictor.model(
            inputs_embeds=self.predictor.small_to_mtp_projection(inputs),
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
        ).last_hidden_state
        token = self.predictor.lm_head[index](hidden[:, -1]).argmax(dim=-1)
        self.tokens[:, index].copy_(token)

    def _sequence(self) -> None:
        """Runs all residual positions with fresh logical cache state."""
        self.cache.reset()
        for index in range(self.groups):
            self._step(index)

    def capture(self) -> None:
        """Captures fixed eager kernels in a shared CUDA graph memory pool."""
        if self.device.type != "cuda" or self.graphs:
            return
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._sequence()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)

        self.cache.reset()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.stream(stream):
            for index in range(self.groups):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self._step(index)
                self.graphs.append(graph)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.cache.reset()

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        """Replays the sequence or uses the identical eager CPU fallback."""
        self.inputs.copy_(inputs)
        if not self.graphs:
            self._sequence()
            return self.tokens
        for graph in self.graphs:
            graph.replay()
        return self.tokens


class DecoderFfnGraph(torch.nn.Module):
    """Keeps talker attention eager and graphs its fixed residual FFN."""

    def __init__(
        self,
        layer: torch.nn.Module,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.inputs = torch.zeros((1, 1, hidden_size), device=device, dtype=dtype)
        self.output = torch.zeros_like(self.inputs)
        self.graph: torch.cuda.CUDAGraph | None = None

    def _ffn(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs the official post-attention norm, MLP, and residual order."""
        return inputs + self.layer.mlp(self.layer.post_attention_layernorm(inputs))

    def capture(
        self,
        stream: torch.cuda.Stream,
        pool: tuple[int, int],
    ) -> None:
        """Captures the one-token FFN while leaving prefill eager."""
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.output.copy_(self._ffn(self.inputs))
        torch.cuda.current_stream(self.inputs.device).wait_stream(stream)
        torch.cuda.synchronize(self.inputs.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(self.graph, pool=pool):
                self.output.copy_(self._ffn(self.inputs))
        torch.cuda.current_stream(self.inputs.device).wait_stream(stream)

    def forward(self, hidden_states: torch.Tensor, **kwargs: Any) -> tuple[Any, ...]:
        """Reproduces the layer and replays only its decode-shape FFN half."""
        if hidden_states.shape[1] != 1 or self.graph is None:
            return self.layer(hidden_states, **kwargs)
        residual = hidden_states
        attention_input = self.layer.input_layernorm(hidden_states)
        attention, weights = self.layer.self_attn(
            hidden_states=attention_input,
            attention_mask=kwargs.get("attention_mask"),
            position_ids=kwargs.get("position_ids"),
            past_key_values=kwargs.get("past_key_values"),
            output_attentions=kwargs.get("output_attentions", False),
            use_cache=kwargs.get("use_cache", False),
            cache_position=kwargs.get("cache_position"),
            position_embeddings=kwargs.get("position_embeddings"),
        )
        self.inputs.copy_(residual + attention)
        self.graph.replay()
        if kwargs.get("output_attentions", False):
            return self.output, weights
        return (self.output,)


class OfficialTalker:
    """Runs one inner-talker token with the official growing dynamic cache."""

    mode: TalkerMode = "official-eager"

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_cache_len: int,
    ) -> None:
        self.model = model
        self.device = device
        self.max_cache_len = max_cache_len
        self.cache = DynamicCache(config=model.config)
        self.rope_deltas = torch.zeros((1, 1), device=device, dtype=torch.float32)

    def capture(self) -> None:
        """Leaves the correctness reference eager because its cache grows."""

    def reset(self, prompt_length: int, rope_deltas: torch.Tensor | None = None) -> None:
        """Creates fresh request state and validates its maximum frame budget."""
        if prompt_length >= self.max_cache_len:
            raise ValueError("prompt exceeds the talker cache capacity")
        self.cache = DynamicCache(config=self.model.config)
        self.rope_deltas.zero_()
        if rope_deltas is not None:
            self.rope_deltas.copy_(rope_deltas)

    def set_rope_deltas(self, rope_deltas: torch.Tensor) -> None:
        """Copies the prefill mRoPE delta used by later one-token forwards."""
        self.rope_deltas.copy_(rope_deltas)

    def run(self, inputs: torch.Tensor, position: int) -> torch.Tensor:
        """Runs the official mask-free one-token dynamic-cache forward."""
        if position >= self.max_cache_len:
            raise ValueError("decode exceeds the talker cache capacity")
        cache_position = torch.tensor([position], device=self.device, dtype=torch.long)
        position_ids = self.rope_deltas + cache_position.to(self.rope_deltas.dtype)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        attention_mask = torch.ones(
            (1, position + 1), device=self.device, dtype=torch.long
        )
        return self.model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        ).last_hidden_state


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
    """Owns the fixed buffers for compiled or manually graphed talker decode."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        max_cache_len: int,
        mode: TalkerMode,
    ) -> None:
        self.model = model
        self.device = device
        self.max_cache_len = max_cache_len
        self.mode = mode
        config = model.config
        self.cache = StaticCache(config=config, max_cache_len=max_cache_len)
        self.cache.early_initialization(
            1, config.num_key_value_heads, config.head_dim, dtype, device
        )
        self.inputs = torch.zeros((1, 1, config.hidden_size), device=device, dtype=dtype)
        self.output = torch.zeros_like(self.inputs)
        self.cache_position = torch.zeros(1, device=device, dtype=torch.long)
        self.position_ids = torch.zeros((3, 1, 1), device=device, dtype=torch.float32)
        self.rope_deltas = torch.zeros((1, 1), device=device, dtype=torch.float32)
        dummy = torch.zeros_like(self.inputs)
        mask_name = (
            "sliding_attention"
            if getattr(config, "sliding_window", None) is not None
            else "full_attention"
        )
        self.mask_table = tuple(
            _masks(model, dummy, torch.tensor([index], device=device), self.cache)[
                mask_name
            ]
            for index in range(max_cache_len)
        )
        self.mask = self.mask_table[0].clone()
        if device.type != "cuda":
            self.step = talker_step
        elif mode == "compile":
            self.step = compiled_talker_step
        else:
            self.step = graphable_talker_step
        self.graph: torch.cuda.CUDAGraph | None = None

    def _step(self) -> None:
        """Runs the selected callable into the stable output buffer."""
        hidden = self.step(
            self.model,
            self.inputs,
            self.mask,
            self.position_ids,
            self.cache,
            self.cache_position,
        )
        self.output.copy_(hidden)

    def capture(self) -> None:
        """Warms compilation and optionally wraps the callable in a CUDA graph."""
        if self.device.type != "cuda":
            return
        for _ in range(3):
            self.cache.reset()
            self._step()
        torch.cuda.synchronize(self.device)
        self.cache.reset()
        if self.mode == "compile":
            return
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(self.graph):
                self._step()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        self.cache.reset()

    def reset(self, prompt_length: int, rope_deltas: torch.Tensor | None = None) -> None:
        """Clears request KV state and validates the fixed cache budget."""
        if prompt_length >= self.max_cache_len:
            raise ValueError("prompt exceeds the fixed talker cache capacity")
        self.cache.reset()
        self.rope_deltas.zero_()
        if rope_deltas is not None:
            self.rope_deltas.copy_(rope_deltas)

    def set_rope_deltas(self, rope_deltas: torch.Tensor) -> None:
        """Copies request mRoPE state without clearing the filled prefill cache."""
        self.rope_deltas.copy_(rope_deltas)

    def run(self, inputs: torch.Tensor, position: int) -> torch.Tensor:
        """Updates fixed inputs and runs one compiled or graphed talker token."""
        if position >= self.max_cache_len:
            raise ValueError("decode exceeds the fixed talker cache capacity")
        self.inputs.copy_(inputs)
        self.cache_position.fill_(position)
        position_ids = self.rope_deltas + self.cache_position.to(self.rope_deltas.dtype)
        self.position_ids.copy_(position_ids.unsqueeze(0).expand(3, -1, -1))
        self.mask.copy_(self.mask_table[position])
        if self.graph is None:
            self._step()
        else:
            self.graph.replay()
        return self.output


class DecodeGraphs:
    """Groups the two graph boundaries so a full-frame graph can replace them."""

    def __init__(
        self,
        talker: Any,
        device: torch.device,
        dtype: torch.dtype,
        max_cache_len: int,
        talker_mode: TalkerMode,
    ) -> None:
        self.ffn_graphs: tuple[DecoderFfnGraph, ...] = ()
        if talker_mode == "official-eager":
            self.predictor = OfficialPredictor(talker.code_predictor)
            self.talker = OfficialTalker(talker.model, device, max_cache_len)
        elif talker_mode == "predictor-ffn-graphs":
            self.predictor = PredictorGraphs(
                talker.code_predictor,
                talker.config.hidden_size,
                device,
                dtype,
            )
            self.talker = OfficialTalker(talker.model, device, max_cache_len)
            if device.type == "cuda":
                wrappers = []
                for index, layer in enumerate(talker.model.layers):
                    wrapper = DecoderFfnGraph(
                        layer,
                        talker.config.hidden_size,
                        device,
                        dtype,
                    )
                    talker.model.layers[index] = wrapper
                    wrappers.append(wrapper)
                self.ffn_graphs = tuple(wrappers)
        else:
            self.predictor = PredictorGraph(
                talker.code_predictor, talker.config.hidden_size, device, dtype
            )
            self.talker = TalkerGraph(
                talker.model, device, dtype, max_cache_len, talker_mode
            )

    def capture(self) -> None:
        """Captures both reusable decode blocks once per loaded talker."""
        self.predictor.capture()
        if self.ffn_graphs:
            stream = torch.cuda.Stream(device=self.talker.device)
            stream.wait_stream(torch.cuda.current_stream(self.talker.device))
            pool = torch.cuda.graph_pool_handle()
            for graph in self.ffn_graphs:
                graph.capture(stream, pool)
            torch.cuda.synchronize(self.talker.device)
        self.talker.capture()


@cache
def decode_graphs(
    talker: torch.nn.Module,
    max_cache_len: int,
    talker_mode: TalkerMode = "predictor-ffn-graphs",
) -> DecodeGraphs:
    """Creates persistent graph objects and allocations once per talker module."""
    parameter = next(talker.parameters())
    graphs = DecodeGraphs(
        talker, parameter.device, parameter.dtype, max_cache_len, talker_mode
    )
    graphs.capture()
    return graphs
