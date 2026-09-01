"""Static-cache eager/CUDA-graph decode blocks for Whistle Qwen3-TTS.

Track A additions: ``sample_token`` plus a second lazily captured predictor
graph set per ``(temperature, top_k)`` so official-style do_sample decoding
runs inside CUDA graphs instead of the eager residual loop. Random ops are
capturable: each replay advances the generator's philox offset, so replays
produce fresh multinomial draws.
"""

import os
from functools import cache
from typing import Any

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache, CacheLayerMixin

class PrefixStaticLayer(CacheLayerMixin):
    """Preallocates predictor KV storage while exposing only valid positions.

    Mirrors DynamicCache semantics with bounded storage: ``get_max_cache_shape``
    reports ``-1`` so mask helpers keep their dynamic-cache behavior, and the
    visible length grows with ``cumulative_length`` while the backing tensors
    stay fixed-size for CUDA graph capture.
    """

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


def sample_token(
    scores: torch.Tensor,
    temperature: float,
    top_k: int | None = None,
) -> torch.Tensor:
    """F32 score row -> one sampled token via top-k + softmax/temperature.

    Mirrors the official do_sample recipe (generation_config: temperature 0.9,
    top_k 50, top_p 1.0 — top_p at 1.0 is a no-op so it is not implemented).
    All ops are CUDA-graph capturable; multinomial draws stay fresh per replay
    because replay advances the captured philox offset.
    """
    if top_k:
        threshold = torch.topk(scores, top_k, dim=-1).values[:, -1:]
        scores = torch.where(scores < threshold, -torch.inf, scores)
    probs = torch.softmax(scores / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(-1)


class PredictorGraphs:
    """Captures one exact eager predictor step for each residual codebook.

    ``graphs`` holds the greedy argmax steps captured at startup;
    ``sample_graphs`` lazily captures the same steps with top-k sampling baked
    in, keyed by ``(temperature, top_k)`` (one set per config, re-captured on
    config change). Set ``WHISTLE_SAMPLE_GRAPHS=0`` to force the eager sampled
    fallback, which is the A/B baseline for the graph-sampled path.
    """

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
        self.sample_graphs: dict[tuple[float, int | None], list[torch.cuda.CUDAGraph]] = {}

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

    def _sample_step(self, index: int, temperature: float, top_k: int | None) -> None:
        """Runs one sampled predictor step into the token buffer."""
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
        logits = self.predictor.lm_head[index](hidden[:, -1])
        token = sample_token(logits.to(dtype=torch.float32, copy=True), temperature, top_k)
        self.tokens[:, index].copy_(token)

    def _sample_sequence(self, temperature: float, top_k: int | None) -> None:
        """Eager residual sequence with per-book temperature sampling."""
        self.cache.reset()
        for index in range(self.groups):
            self._sample_step(index, temperature, top_k)

    def _capture_sampling(self, temperature: float, top_k: int | None) -> None:
        """Captures the sampled step sequence once per (temperature, top_k)."""
        key = (temperature, top_k)
        if self.device.type != "cuda" or key in self.sample_graphs:
            return
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._sample_sequence(temperature, top_k)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)

        self.cache.reset()
        graphs: list[torch.cuda.CUDAGraph] = []
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.stream(stream):
            for index in range(self.groups):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self._sample_step(index, temperature, top_k)
                graphs.append(graph)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.cache.reset()
        self.sample_graphs[key] = graphs

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

    def run(
        self,
        inputs: torch.Tensor,
        sampling: dict[str, float] | None = None,
    ) -> torch.Tensor:
        """Replays the greedy or sampled graph set for one talker frame.

        The sampled set is captured lazily on first use of a new
        ``(temperature, top_k)`` pair; ``WHISTLE_SAMPLE_GRAPHS=0`` keeps the
        eager sampled loop for A/B benchmarking.
        """
        self.inputs.copy_(inputs)
        if sampling:
            temperature = float(sampling["temperature"])
            top_k = sampling.get("top_k")
            if os.environ.get("WHISTLE_SAMPLE_GRAPHS", "1") != "0":
                self._capture_sampling(temperature, top_k)
                for graph in self.sample_graphs[(temperature, top_k)]:
                    graph.replay()
                return self.tokens
            self._sample_sequence(temperature, top_k)
            return self.tokens
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
        attention = self.layer.self_attn(
            hidden_states=attention_input,
            attention_mask=kwargs.get("attention_mask"),
            position_ids=kwargs.get("position_ids"),
            past_key_values=kwargs.get("past_key_values"),
            output_attentions=kwargs.get("output_attentions", False),
            use_cache=kwargs.get("use_cache", False),
            cache_position=kwargs.get("cache_position"),
            position_embeddings=kwargs.get("position_embeddings"),
        )[0]
        self.inputs.copy_(residual + attention)
        self.graph.replay()
        return (self.output,)


class OfficialTalker:
    """Runs one inner-talker token with the official growing dynamic cache."""

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


class DecodeGraphs:
    """Groups the two graph boundaries so a full-frame graph can replace them."""

    def __init__(
        self,
        talker: Any,
        device: torch.device,
        dtype: torch.dtype,
        max_cache_len: int,
    ) -> None:
        self.ffn_graphs: tuple[DecoderFfnGraph, ...] = ()
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

    def capture(self) -> None:
        """Captures both reusable decode blocks once per loaded talker.

        FFN graphs share one side stream and memory pool; the predictor keeps
        its own pool so its cache-reset sequencing stays independent.
        """
        self.predictor.capture()
        if self.ffn_graphs:
            stream = torch.cuda.Stream(device=self.talker.device)
            stream.wait_stream(torch.cuda.current_stream(self.talker.device))
            pool = torch.cuda.graph_pool_handle()
            for graph in self.ffn_graphs:
                graph.capture(stream, pool)
            torch.cuda.current_stream(self.talker.device).wait_stream(stream)
            torch.cuda.synchronize(self.talker.device)
        self.talker.capture()


@cache
def decode_graphs(
    talker: torch.nn.Module,
    max_cache_len: int,
) -> DecodeGraphs:
    """Creates persistent graph objects and allocations once per talker module."""
    parameter = next(talker.parameters())
    graphs = DecodeGraphs(
        talker, parameter.device, parameter.dtype, max_cache_len
    )
    graphs.capture()
    return graphs
