# Official Qwen3-TTS vs faster-qwen3-tts

## Scope and source snapshots

This report compares the local reference checkouts:

- `refs/Qwen3-TTS` at `022e286` (2026-03-17)
- `refs/faster-qwen3-tts` at `7cdef7e` (2026-04-22), package version
  `0.2.6`

The analysis is based on their Python source, tests, and benchmark
documentation. No full checkpoint or GPU benchmark was run on this CPU-only
machine. Performance numbers below are the faster repository's reported
measurements, not locally reproduced results.

## Executive conclusion

The official repository owns the model definition, checkpoint loading, prompt
construction, Hugging Face generation integration, and speech tokenizer.
`faster-qwen3-tts` keeps all of those learned modules and weights. It imports
`Qwen3TTSModel`, loads the official model, extracts the talker and code
predictor, then replaces the repeated decode scheduling with a custom,
single-batch CUDA path.

The fast path is:

```text
official prompt construction
  -> dynamic, variable-length talker prefill
  -> copy prefill KV into a fixed StaticCache
  -> repeat:
       replay one-token talker CUDA graph
       replay one full 15-codebook predictor CUDA graph
       sample and assemble the next frame
  -> official speech-tokenizer decode
```

This is an execution-engine optimization rather than a new TTS architecture.
The main gain is removal of hundreds of small Python-dispatched CUDA launches
per 12 Hz audio frame.

## Component comparison

| Area | Official Qwen3-TTS | faster-qwen3-tts |
|---|---|---|
| Model classes | Defines talker, code predictor, prompt logic, and codec | Imports and retains the official model |
| Weight format | Native released checkpoint layout | Same official checkpoints; no conversion |
| Prompt preparation | Canonical implementation | Calls official helpers and carries a local copy of the talker-input builder |
| Talker prefill | Hugging Face forward with `DynamicCache` | Same dynamic forward, then copies KV into `StaticCache` |
| Talker decode | `GenerationMixin.generate`, one model call per step | One-token model forward captured in `torch.cuda.CUDAGraph` |
| Residual codebooks | Nested `code_predictor.generate` for 15 tokens per frame | All 15 predictor steps unrolled into one CUDA graph |
| KV storage | Grows dynamically | Preallocated, fixed maximum lengths |
| Attention mask | Constructed for the active sequence | Prebuilt mask table for every static-cache position |
| Sampling | Hugging Face logits processors and sampling | Small local suppress/top-k/top-p/temperature sampler |
| Codec | Official 12 Hz speech tokenizer | The same official speech tokenizer |
| Output streaming | Public wrapper decodes after token generation | Generator yields code chunks and incrementally decodes audio |
| Batch inference | Public APIs accept lists and pad batches | Graph buffers are hard-coded to batch size one |
| Hardware | CPU or accelerator depending on Torch support | NVIDIA CUDA is mandatory |
| Attention backends | SDPA or optional FlashAttention 2 | Uses the selected official attention backend inside the graph; defaults to SDPA |

## Official implementation

### Generation hierarchy

The top-level official wrapper tokenizes assistant and optional instruction
prompts, validates language and speaker choices, and calls
`Qwen3TTSForConditionalGeneration.generate`. That method builds the aligned
text/codec prompt and delegates the outer sequence to
`Qwen3TTSTalkerForConditionalGeneration.generate`.

For every outer talker token, the talker's `forward` invokes
`code_predictor.generate` to autoregressively create the remaining 15 codec
codebooks. One 12 Hz frame therefore contains:

1. one token from the large talker;
2. 15 residual tokens from the smaller code predictor;
3. 16 summed codebook embeddings as the next talker input.

Both transformer backbones create `DynamicCache` objects when caching is
enabled. Hugging Face `GenerationMixin` manages stopping, logits processors,
sampling, masks, cache positions, and model keyword updates.

This approach is general and easy to extend, but a single audio frame crosses
Python and the Transformers generation stack many times. The predictor alone
performs a prefill plus 14 single-token decoder calls for every talker frame.

### Prompting and text modes

The official code builds combined text and codec embeddings for Base,
CustomVoice, and VoiceDesign variants. `non_streaming_mode` describes how the
text conditioning is supplied to the talker:

- `True` puts the full text-aligned conditioning in the prefill;
- `False` retains trailing text hidden states and adds them during decode.

This name does not by itself mean that the public Python API yields waveform
chunks. In the inspected wrapper, generated codes are passed to the speech
tokenizer and the completed waveform list is returned.

### Codec decoding

The official 12 Hz tokenizer transposes `[batch, frames, codebooks]` codes into
the decoder layout and calls `decoder.chunked_decode`. Internally it uses
300-frame chunks with 25 frames of left context, removes the context audio from
each decoded chunk, concatenates the result, and trims it to the expected
length.

It is already a chunk-safe implementation, but the top-level generation API
does not expose each chunk while the talker is still running.

## faster-qwen3-tts implementation

### It wraps the official package

`FasterQwen3TTS.from_pretrained` imports `Qwen3TTSModel` from `qwen_tts` and
loads it normally. The faster package declares `qwen-tts>=0.1.1` as a runtime
dependency. Its graph objects hold references to:

- the official talker transformer;
- the official codec embedding and output head;
- the official code-predictor transformer, embeddings, projections, and heads;
- the official speech tokenizer.

Consequently, it is checkpoint compatible by construction, but it cannot run
without the official Python package and is sensitive to upstream internal API
changes.

### Split prefill and decode

Variable-length prefill is intentionally left outside CUDA graphs. The faster
loop calls the official talker `forward` once with `use_cache=True`, obtaining
the first logits, last talker hidden state, generation step, RoPE delta, and a
dynamic KV cache.

It then copies every prefill K/V tensor into a preallocated Transformers
`StaticCache`. This creates a clean optimization boundary:

```text
variable shapes: official eager prefill
fixed shapes:    custom CUDA-graph decode
```

This is the same boundary Nero should preserve for future independent kernels.

### Talker CUDA graph

`TalkerGraph` allocates stable `[1, 1, hidden_size]` input/output tensors, a
scalar cache-position tensor, position IDs, RoPE deltas, and a fixed-length
`StaticCache`. It captures the official inner talker model's one-token forward
pass.

Before each replay, Python copies the new embedding into the stable input
buffer, updates cache position, selects the already-built causal mask, updates
position IDs, and calls `graph.replay()`. The returned tensor is a shared static
output buffer and must be consumed immediately or cloned.

Padding-aware causal masks are precomputed for every position up to
`max_seq_len`. This keeps mask shapes and addresses stable during replay and
preserves the prefill padding semantics.

### Predictor CUDA graph

`PredictorGraph` is the more important fusion boundary. It allocates a
17-position static cache and captures the entire residual-codebook process:

1. project the two-token input consisting of talker hidden state and codebook-0
   embedding;
2. run the predictor prefill and sample residual codebook 1;
3. embed the previously sampled residual;
4. run 14 one-token predictor decode calls and their separate LM heads;
5. write all 15 residual IDs into a stable output tensor.

All predictor transformer calls, codebook-specific embeddings and heads, and
predictor sampling execute inside one graph replay. This collapses the most
fragmented nested generation loop into a single host launch.

The residual embedding index matches the official dependency: predictor step
`i` consumes the preceding residual through
`codec_embedding[i - 1]`.

### Static allocation and warmup

Both graphs force lazy cache initialization, run three warmup iterations, and
capture on a dedicated CUDA stream. Capture happens lazily on the first
generation request and is reused afterward.

Static allocation removes cache growth, most temporary allocation, and Python
launch scheduling from the transformer bodies. Its costs are:

- a fixed `max_seq_len` memory reservation;
- an initial warmup/capture latency;
- recapture or separate engines for incompatible shapes/settings;
- a hard stop before the static sequence capacity is exceeded.

### Local sampling

The outer talker token is sampled outside its CUDA graph by a compact helper
that applies token suppression, temperature, top-k, top-p, and multinomial
sampling. Repetition penalty is also applied locally.

Predictor sampling is captured inside `PredictorGraph`. In the inspected
implementation, the predictor graph is constructed with fixed defaults
(`do_sample=True`, `top_k=50`, `top_p=1.0`, `temperature=0.9`). Per-call
`subtalker_*` overrides are used by the parity path but are not propagated into
an already captured fast predictor graph. This is an API/semantic limitation,
not merely an implementation detail.

### Audio streaming

The streaming loop uses the same talker and predictor graph replays and yields
a tensor of codec frames every `chunk_size` steps. The wrapper:

1. initially decodes accumulated codes;
2. estimates samples per codec frame;
3. subsequently decodes the new chunk plus up to 25 left-context frames;
4. removes context samples and yields only new waveform data.

This matches the context principle of the official codec's `chunked_decode`
while making audio available before token generation completes. Smaller chunks
lower time-to-first-audio but invoke the codec more often. The generator is
pull-based, so playback must run independently if generation and playback are
to overlap.

The repository also includes a deliberately slow dynamic-cache streaming loop
for parity testing.

### Prompt caching and service features

Reference-audio voice-clone prompts are cached by reference inputs, avoiding
repeated tokenizer/encoder work. It also supports precomputed x-vector prompts,
so serving can skip reference audio encoding entirely.

Around the core engine, the repository adds a Click CLI, a streaming demo,
benchmark scripts, a hot HTTP server, and an OpenAI-compatible speech endpoint.
These are deployment features, not model-kernel optimizations.

## What is actually optimized

The concrete hot-path optimizations are:

1. **manual CUDA graph capture** for the talker's fixed one-token decoder;
2. **whole-loop CUDA graph capture** for all 15 residual predictor tokens;
3. **static KV caches** with fixed tensor addresses and no per-step growth;
4. **precomputed attention masks** indexed by decode position;
5. **explicit split prefill/decode**, keeping variable shapes out of capture;
6. **small local generation loop**, bypassing nested `GenerationMixin` calls;
7. **stable reusable I/O buffers** populated with `copy_`;
8. **incremental codec output** with left-context decoding;
9. **cached/precomputed speaker prompts** for repeated voice-clone requests.

The repository explicitly does not attribute its main gains to Triton, vLLM,
FlashAttention, or `torch.compile`. It can select FlashAttention through the
official loader, but SDPA is the default and the manual graph scheduler is the
core optimization.

## Reported performance

The faster README reports the following end-to-end results, including
tokenization:

| Model/GPU | Baseline RTF | Graph RTF | Baseline TTFA | Graph TTFA |
|---|---:|---:|---:|---:|
| 0.6B / RTX 4090 | 0.82 | 4.78 | 800 ms | 156 ms |
| 0.6B / RTX 4060 | 0.23 | 2.26 | 2,697 ms | 413 ms |
| 0.6B / H100 | 0.435 | 3.884 | 1,474 ms | 228 ms |
| 1.7B / RTX 4090 | 0.82 | 4.22 | 850 ms | 174 ms |

The stated RTX 4090 gains are 5.8x throughput and 5.1x TTFA for 0.6B, and 5.1x
throughput and 4.9x TTFA for 1.7B. These comparisons need two qualifications:

- the official wrapper has no equivalent incremental-output measurement, so
  baseline TTFA comes from a community streaming fork or the faster repo's
  dynamic-cache parity loop;
- CUDA graph results can vary substantially by GPU clocks, architecture,
  sequence length, chunk size, and warm/cold state.

The README's Jetson 0.6B chunk study shows the expected latency/throughput
tradeoff: chunk size 1 reports 240 ms TTFA and 0.750 RTF, while chunk size 12
reports 753 ms TTFA and 1.449 RTF. Non-streaming reports 1.57 RTF.

## Parity and correctness

The faster project separates two meanings of parity:

- a dynamic-cache parity mode calls official `talker.generate`, providing
  token-level equality for tests;
- the static-cache graph path aims for behavioral and prefix parity, but is not
  promised to be bit-exact.

A fixed-length masked SDPA operation may choose a different kernel than a
shorter dynamic-cache causal operation. BF16/TF32 reduction order can therefore
slightly change logits and sampled continuations even if the mathematical
attention is equivalent.

Its end-to-end tests cover CustomVoice, VoiceDesign, x-vector and ICL cloning,
and streaming/non-streaming comparisons. Those tests require real checkpoints
and CUDA and were not run here.

## Remaining bottlenecks and limitations

The fast repository is effective but not a finished inference engine:

- graph buffers are batch size one, while the official wrapper supports batch
  lists;
- outer generation still executes a Python loop;
- `token.item()` synchronizes the host for EOS checking every frame;
- codec embedding assembly uses Python lists and multiple small operations;
- repetition history is repeatedly stacked and uniqued;
- chunk boundaries call `torch.cuda.synchronize()` for timing/yielding;
- variable-length prefill and codec decode are not graphed or fused;
- the talker and predictor remain official Transformers modules rather than
  specialized kernels;
- attention-mask tables scale with `max_seq_len`;
- predictor sampling settings are fixed at graph construction;
- the local copy of official prompt-building logic can drift as upstream
  changes;
- only NVIDIA CUDA is supported.

These limitations identify useful next steps: device-side stop handling,
preallocated history, fused codebook embedding reduction, static batching,
separate graph variants for sampling policies, optimized prefill, and
specialized attention/MLP kernels.

## Implications for Nero

Nero's goal differs from faster-qwen3-tts: Nero is an independent,
weights-compatible model implementation intended to expose and modify every
inference boundary. The most useful faster-repo ideas to adopt are:

1. preserve separate prompt preparation, prefill, talker decode, predictor, and
   codec calls;
2. make static KV state an explicit input/output rather than hiding it in
   Hugging Face generation;
3. treat the complete residual predictor loop as one fusion/capture unit;
4. keep variable prefill separate from fixed one-token decode;
5. preallocate stable buffers for tokens, positions, masks, history, and
   codebook embeddings;
6. build streaming above codec-frame chunks with the official decoder's
   25-frame context rule;
7. maintain a slow, structurally clear reference loop for parity tests.

Nero should not copy the faster wrapper's dependency strategy because the
independent talker/code-predictor implementation is what enables deeper
changes. Reusing only the installed official
`Qwen3TTSTokenizerV2Decoder` remains a sensible boundary: it avoids
reimplementing the vocoder while leaving token generation fully controllable.

For CPU tests, the shrunk official-layout checkpoint validates tensor names,
shapes, state flow, generation, and codec plumbing. It cannot validate real
speech quality or CUDA graph performance.
