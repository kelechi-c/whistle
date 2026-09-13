# project notes

Whistle is a low-latency inference runtime for **Qwen3-TTS CustomVoice** (0.6B and
1.7B). It leaves the official `qwen-tts` modules and weights untouched and
replaces only the scheduling around them: the batch decode loop is written out
explicitly, and the two fixed-shape blocks inside it are captured as CUDA graphs.
The target is minimum time-to-audio on a small GPU, with exact codec-token and
waveform parity against the official greedy runtime.

This file is the architecture map for the released code. Benchmark numbers and
their caveats live in `latency_report.md` and `evidence/`. The written article
and its full figure set are published separately (Sciel) and kept out of this
repository under the ignored `article/` directory. Pre-release history was
archived under `stash/release-cleanup/`.

## repository layout

| Path | Role |
| --- | --- |
| `src/whistle/` | the installed runtime (cli, config, graphs, inference, streaming, server) |
| `tests/` | cpu structural tests; tiny random weights, no checkpoint needed |
| `tools/` | benchmark harnesses, the release gate, and the figure generator (not installed) |
| `alicia.txt` | long-form benchmark input used by every latency number |
| `evidence/` | compact benchmark records cited by the report and figures |
| `latency_report.md` | consolidated timing tables and measurement caveats |

## module map

```
cli.py ──> config.py, inference.py ──> graphs.py ──> qwen_tts / transformers
streaming.py ──> inference.py (shared helpers) + graphs.py
server.py ──> streaming.py, config.py
tools/profile_tts.py ──> inference.py     (split vs official measurement)
tools/bench_streaming.py ──> streaming.py (first-audio latency)
```

- `config.py` — shared choices, the single source for all of them:
  `CHECKPOINT`, `SPEAKER` (`ryan`), `LANGUAGE`, `MAX_CACHE_LEN = 2048`, `OUTPUT`,
  `SEED`. `__init__.py` re-exports the four public constants.
- `cli.py` — the packaged `whistle` console script: refuses to run without CUDA,
  loads the checkpoint in bf16/sdpa, calls `tts_infer`, writes a WAV.
- `graphs.py` — all CUDA-graph machinery and the talker cache state.
- `inference.py` — prompt construction, prefill, the batch decode loop, codec
  decode. `tts_infer` is the public entry point.
- `streaming.py` — `stream_tts`, the same decode loop but yielding audio chunks.
- `server.py` — optional FastAPI wrapper that streams WAV over HTTP.

Nothing in the package schedules work with Hugging Face `generate`; the runtime
drives the official modules forward directly.

## the batch path (`inference.py`)

`tts_infer(model, text, ...)` takes an already-loaded `Qwen3TTSModel`, so
checkpoint loading never falls inside a measurement. It runs in four phases:

1. **prepare** — `_prepare` calls `build_prompt`, which reproduces the official
   CustomVoice prompt from the wrapper's own helpers (role header, codec
   prefix/suffix ids, speaker embedding, projected text with TTS bos/eos/pad,
   codec padding). It then asserts `prefill_length + max_new_tokens - 1 <=
   MAX_CACHE_LEN`, fetches the persistent decode graphs, resets the talker cache
   and mRoPE delta, and allocates the token history. Returns a frozen `Prompt`.
2. **prefill** — `_prefill` runs one eager talker forward over the whole prompt
   into the talker's `DynamicCache`, builds the logits processors (repetition
   penalty when it differs from 1.0, plus suppression of the top 1024 vocab ids
   except the EOS id), selects the first token, and stores the prefill mRoPE
   delta on the graph talker. Returns a frozen `Prefill`.
3. **decode** — one iteration per frame: embed the previous primary token, fill
   the predictor input buffer with `(past_hidden, token_embedding)`, replay the
   predictor graphs for every residual codebook, record the frame and token, then
   rebuild the next talker input by summing the primary and residual codec
   embeddings and adding the TTS pad embedding, run one talker step, and select
   the next token from `codec_head`.
4. **codec** — trim the code tensor to the frames actually emitted and decode
   once with the official speech-tokenizer model. Nothing is copied to the CPU
   unless the caller asks.

Every selection goes through `_select_token`: logits are cast to float32, passed
through the official `LogitsProcessorList`, then argmaxed or sampled with top-k +
temperature. EOS is never allowed on the first frame.

`_maybe_eos_row` is the shared chunked EOS scan (`EOS_CHECK_EVERY = 8`): an
on-device scan at a fixed cadence instead of a per-frame `token.eq(eos).item()`
host sync. Callers pass `force=True` wherever `codes` becomes observable (each
streaming chunk boundary, and the final frame). Timings come from five CUDA
events on CUDA (non-overlapping `prepare`/`prefill`/`decode`/`codec` spans, one
synchronisation at the end) and from wall clock on CPU.

## decode graphs (`graphs.py`)

The decode state is deliberately split so that only fixed-shape work is graphed
and every variable-length step stays eager:

- **`PrefixStaticLayer`** — a `CacheLayerMixin` that preallocates a fixed
  `[batch, heads, max_cache_len, head_dim]` KV buffer on first update and writes
  into it with `index_copy_`. It exposes only the populated prefix through
  `cumulative_length` and answers `get_seq_length()` without scanning GPU memory.
  `prefix_cache` builds one per predictor decoder layer; because addresses never
  move, capture stays legal. `get_max_cache_shape()` returns `-1`, so mask
  helpers keep dynamic-cache semantics while the storage stays bounded.
- **`PredictorGraphs`** — owns the residual-codebook predictor. `groups` is
  `num_code_groups - 1`, one per residual codebook. `_step` runs one predictor
  forward and writes the argmax token into a fixed buffer; `capture` warms the
  sequence up three times, then records one graph per codebook into a shared
  pool. `run` copies its inputs in and replays the set, so a frame of residual
  codes is one launch sequence.
- **Sampled predictor set** — `_sample_step`/`_sample_sequence` mirror the above
  using `sample_token` (top-k mask, softmax over temperature, multinomial), and
  `_capture_sampling` lazily captures a second graph set per `(temperature,
  top_k)`. The multinomial runs inside the captured graph; replays advance the
  generator's offset, so successive frames keep drawing fresh tokens.
- **`DecoderFfnGraph`** — wraps one talker decoder layer. `forward` delegates to
  the original layer unless the input is a single decode token *and* a graph was
  captured. For that shape it runs input layernorm and self-attention eagerly
  (attention is cache-aware and variable-length), stores `residual + attention` in
  a fixed buffer, and replays a graph of the post-attention norm + MLP + residual.
  It returns a 1-tuple, since only the hidden state is consumed.
- **`Talker`** — the eager talker backbone: a `DynamicCache` plus an mRoPE delta
  buffer. `reset(prompt_length)` validates capacity and installs a fresh cache;
  `run(inputs, position)` is the mask-free one-token forward with `position_ids =
  rope_deltas + position` expanded over the three mRoPE sections.
- **`DecodeGraphs`** — builds the predictor graphs and talker state, and on CUDA
  swaps each entry of `talker.model.layers` for a `DecoderFfnGraph`. `capture()`
  records the predictor graphs, then the per-layer FFN graphs, on one side stream
  and one pool. The talker backbone is never captured.
- **`decode_graphs(talker, max_cache_len)`** — `@functools.cache`d: graphs and
  buffers are built and captured once per loaded talker module and reused by every
  later request. `_prepare` resets the cache between requests; the buffers
  themselves are shared on purpose.

Wrapping the layer list once means the official talker forward runs through these
wrappers, which keeps the fast path and the official path on the same hidden
states.

## the streaming path (`streaming.py`)

`stream_tts` reuses `_prepare`, `_prefill`, `_select_token`, and `_maybe_eos_row`
from `inference.py`, so batch and streaming cannot drift apart. It differs in how
often it stops to hand work back:

- **Chunk schedule** — boundaries fire at the `ramp_frames` counts first
  (default `(2, 4, 8)`), then every `chunk_size` frames (default 12). Small early
  chunks are what buy the latency win; `ramp_frames=()` restores fixed cadence.
- **Incremental codec** — `_streaming_decoder` keeps a rolling `left_context`
  (25 frames) and trims the context warmup samples (`context * total_upsample`)
  from each chunk, so chunks concatenate without duplication.
- **First-chunk silence trim** — `_trim_leading_silence` drops samples before the
  first 10 ms RMS window above `threshold` (0.002), keeping a short lead-in so the
  first phoneme is not clipped. One host sync, first chunk only.
- **EOS** — the scan is forced at every chunk boundary (a yielded chunk can never
  contain stale EOS frames), at the cadence in between, and at the final frame.

Each yielded dict carries `codes` (the frames in this chunk), `audio`,
`sample_rate`, `chunk_frames`, `prefill_ms`, `cumulative_ms`, `chunk_ms`, and
`final`. `prefill_ms` ends before any audio exists. `cumulative_ms` and
`chunk_ms` are host timestamps taken just after that chunk's codec decode is
enqueued; the first chunk is additionally host-synchronized by the silence trim,
so the first `cumulative_ms` is a true audio-ready milestone while later ones are
enqueue boundaries. Tensors stay on-device, so a consumer copying them measures
its own true first-audio time.

## the server (`server.py`)

`GET /health` and `GET /synthesize?text=...`, the latter returning `audio/wav`
built from a streaming WAV header (unknown-size RIFF/data fields) plus int16 PCM
chunks. The model loads lazily on first use. Generation is serialised behind a
module-level `threading.Lock`: decode mutates shared per-model state (cache, mRoPE
delta, graph input buffers), so concurrent requests would interleave writes into
the same buffers. `python -m whistle.server` warms the model before serving.

## tests (`tests/test_inference.py`)

Seven CPU-only structural tests on a tiny official-shaped model and a tensor-only
stub codec: repeatability across two identical requests, talker cache reset,
capacity/budget validation, the eager CPU sampling fallback, sampling-policy
forwarding through prefill, streaming ramp/final-chunk boundaries, and EOS scan
cadence. They validate plumbing and loop contracts, not speech quality or
checkpoint speed.

## reproducing the results

```bash
# one-command gate: tests, exact parity, sampled capture, streaming spot
bash tools/run_promo_check.sh

# structural tests (cpu, no checkpoint, any machine)
uv run --no-sync python -m unittest discover -s tests -v

# cli synthesis (cuda required)
uv run --no-sync whistle "text to synthesize" --out out.wav

# batch timing, one backend per process
uv run --no-sync python tools/profile_tts.py --text-file alicia.txt --backend split --iterations 5 --warmup 2 --json-out /tmp/split.json

# streaming first-audio latency and chunk cadence
uv run --no-sync python tools/bench_streaming.py --text-file alicia.txt --iterations 3 --warmup 1 --json-out /tmp/streaming.json

# exact codec-id and waveform parity: same flags, same --parity-dir, run both
uv run --no-sync python tools/profile_tts.py --text-file alicia.txt --backend split --parity-dir /tmp/whistle-parity
uv run --no-sync python tools/profile_tts.py --text-file alicia.txt --backend official --parity-dir /tmp/whistle-parity

# regenerate the article figures (matplotlib is not a runtime dependency)
uv run --no-project --with matplotlib python tools/make_figures.py [--only STEM] [--variant both]
```

Parity needs two processes: the reference must be a pristine official model, and a
6 GB card cannot hold both at once. The first run saves its ids, waveform and
settings, the second compares and exits non-zero on any difference; both must use
identical flags, and a reference saved under different settings is rejected rather
than diffed. Compare at natural EOS: a run that stops at its frame cap is not
directly comparable, because the official entrypoint reports one frame fewer for
the same cap and the codec decoder's lookahead then shifts the last frame's audio,
so align the caps (official `--max-new-tokens N+1` against split `N`) when a
cap-bound comparison is needed. `profile_tts.py` also takes `--fixed-tokens`
(ignore EOS and emit the full budget), `--temperature`/`--top-k` for sampled
decoding, and `--trace-out` for a chrome trace; `bench_streaming.py` takes
`--chunk-size`, `--ramp`, `--no-trim`.

## where the numbers live

- `latency_report.md` — consolidated comparison tables per GPU, with their
  caveats (warmup, power state, sampling policy, timing boundaries).
- `evidence/` — the raw records behind those tables and figures, indexed by
  `evidence/README.md`. `tools/make_figures.py` reads that directory directly, so
  no chart value is transcribed by hand.
- `article/` — local-only reference (ignored by git): the published write-up port
  and its figures, including the mobile variants. `tools/make_figures.py` writes
  there.

Add new benchmark output there only when a published claim cites it.
