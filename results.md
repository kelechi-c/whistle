# Benchmark results

## Version summary

All versions use the complete [Alicia input](alicia.txt), Ryan, English,
bfloat16, SDPA, one excluded warmup, three measured runs, and 1,279 complete
codec frames (102.320 seconds of audio).

| Version | Tag | p50 latency | p50 RTF | Throughput | Change from prior |
|---|---|---:|---:|---:|---:|
| Official baseline | Official `qwen-tts` | 91.722 s | 0.896 | 1.116× | — |
| `faster_decode` v1 | Explicit decode scheduler | 71.382 s | 0.698 | 1.433× | −22.18% vs official |
| `faster_decode` v2 | Reduced Python/CPU sync overhead | 67.974 s | 0.664 | 1.505× | −4.78% vs v1 |
| **`faster_decode` v3** | **Static cache + torch-compiled predictor pass** | **57.477 s** | **0.562** | **1.780×** | **−15.44% vs v2** |

The official baseline is retained for future version comparisons rather than
rerun after every fast-path change.

## Official Qwen3-TTS 0.6B baseline

This baseline uses the official `qwen-tts` runtime and
`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`.

### Environment

| Component | Value |
|---|---|
| GPU | RTX 3050 6 GB Laptop GPU |
| PyTorch | 2.13.0 |
| CUDA | 13.0 |
| Dtype | bfloat16 |
| Attention | SDPA |

### Method

Every run uses the complete contents of [alicia.txt](alicia.txt). The profiler
forces `min_new_tokens == max_new_tokens == 1280`, preventing EOS from making
one implementation finish early. The official generator produces 1,279
complete codec frames from those 1,280 selected talker tokens, or 102.320
seconds of 24 kHz audio.

One warmup is excluded. The reported p50 values are the medians of three
measured runs.

```bash
uv run --no-sync python profile_tts.py \
  --text-file alicia.txt \
  --backend official \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --lang English \
  --max-new-tokens 1280 \
  --fixed-tokens \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/official_tts_0.6b_alicia.json
```

### Per-run results

| Run | Generation latency | Audio | RTF | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 91.722 s | 102.320 s | 0.896 | 1.116× |
| 2 | 91.837 s | 102.320 s | 0.898 | 1.114× |
| 3 | 91.631 s | 102.320 s | 0.896 | 1.117× |
| **p50** | **91.722 s** | **102.320 s** | **0.896** | **1.116×** |

### Aggregate results

| Metric | Result |
|---|---:|
| Cached model load | 12.418 s |
| Mean generation latency | 91.730 s |
| Generation latency range | 91.631–91.837 s |
| Mean RTF | 0.896 |
| Peak allocated GPU memory | 2,836.4 MiB |

### Phase breakdown

Phase measurements are non-overlapping: code-predictor time is subtracted from
the inclusive talker measurement.

| Generation phase | Mean latency | Share of wall time |
|---|---:|---:|
| Code predictor | 65.791 s | 71.72% |
| Talker excluding code predictor | 24.054 s | 26.22% |
| Speech codec | 1.838 s | 2.00% |
| Wrapper overhead | 46.65 ms | 0.05% |

The complete machine-readable report is
[`benchmarks/official_tts_0.6b_alicia.json`](benchmarks/official_tts_0.6b_alicia.json).

## Explicit `faster_decode` baseline

This path uses the same official model modules and weights but replaces the
official nested generation scheduler with the explicit prefill and greedy
decode loop in `faster_decode.py`.

### Method

The environment and Alicia input are the same as the official baseline. The
shared budget is 1,280 selected talker tokens. Because the explicit loop's
internal limit counts completed frames, its profiler adapter requests 1,279
frames to match the official generator's token-to-frame convention exactly.
Both paths therefore produce 1,279 frames and 102.320 seconds of audio. One
warmup is excluded, followed by three measured runs.

```bash
uv run --no-sync python profile_tts.py \
  --text-file alicia.txt \
  --backend split \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --lang English \
  --max-new-tokens 1280 \
  --fixed-tokens \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/faster_decode_0.6b_alicia.json
```

### Per-run results

| Run | Generation latency | Audio | RTF | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 71.382 s | 102.320 s | 0.698 | 1.433× |
| 2 | 71.210 s | 102.320 s | 0.696 | 1.437× |
| 3 | 71.423 s | 102.320 s | 0.698 | 1.433× |
| **p50** | **71.382 s** | **102.320 s** | **0.698** | **1.433×** |

### Aggregate results

| Metric | Result |
|---|---:|
| Cached model load | 14.395 s |
| Mean generation latency | 71.339 s |
| Generation latency range | 71.210–71.423 s |
| Mean RTF | 0.697 |
| Peak allocated GPU memory | 3,013.2 MiB |

### Phase breakdown

| Generation phase | Mean latency | Share of wall time |
|---|---:|---:|
| Decode | 69.450 s | 97.35% |
| Codec | 1.835 s | 2.57% |
| Prefill | 50.16 ms | 0.07% |
| Preparation | 1.96 ms | <0.01% |

### Decode breakdown

This diagnostic run adds asynchronous CUDA events around the three repeated
decode stages. It does not synchronize between forwards. The uninstrumented
71.382-second p50 above remains the headline latency; the separate instrumented
run's p50 was 71.266 seconds.

| Decode stage | Calls | p50 latency | Share of decode | Average |
|---|---:|---:|---:|---:|
| Predictor seed stage | 1,279 | 3.627 s | 5.23% | 2.836 ms/frame |
| Predictor residual stage | 1,279 × 14 | 45.777 s | 66.00% | 35.792 ms/frame |
| Talker step | 1,278 | 19.901 s | 28.69% | 15.572 ms/step |
| Other and host overhead | — | 58.02 ms | 0.08% | — |
| **Total decode** | — | **69.363 s** | **100%** | — |

The two predictor stages total 49.404 seconds, or 71.23% of decode and 38.627
ms per generated frame. The residual stage includes its embedding, forward,
argmax, and cache-update sequence for all 14 residual codebooks. The talker
stage includes frame-embedding assembly, mask and position preparation, the
cached backbone forward, and next-token selection. “Other” is the decode wall
time left after subtracting the three CUDA-event regions.

The machine-readable diagnostic report is
[`benchmarks/faster_decode_0.6b_alicia_breakdown.json`](benchmarks/faster_decode_0.6b_alicia_breakdown.json).

The complete machine-readable report is
[`benchmarks/faster_decode_0.6b_alicia.json`](benchmarks/faster_decode_0.6b_alicia.json).

## `faster_decode` v2 — reduced Python/CPU sync overhead

V2 shrinks the hot path and removes Python/CPU synchronization and bookkeeping
from repeated decode work. Fixed codec buffers, cache positions, position IDs,
and predictor embedding weights are prepared once and remain on-device. The
codec writes chunk outputs directly into its final GPU waveform buffer.

The split API now counts completed codec frames directly, so this command uses
1,279 frames to match the retained official baseline's 1,280-token output.

```bash
uv run --no-sync python profile_tts.py \
  --text-file alicia.txt \
  --backend split \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --lang English \
  --max-new-tokens 1279 \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/v2_faster_decode_0.6b_alicia.json
```

### Per-run results

| Run | Generation latency | Audio | RTF | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 67.606 s | 102.320 s | 0.661 | 1.513× |
| 2 | 67.974 s | 102.320 s | 0.664 | 1.505× |
| 3 | 68.229 s | 102.320 s | 0.667 | 1.500× |
| **p50** | **67.974 s** | **102.320 s** | **0.664** | **1.505×** |

### Aggregate results

| Metric | Result |
|---|---:|
| Cached model load | 19.631 s |
| Mean generation latency | 67.937 s |
| Generation latency range | 67.606–68.229 s |
| Mean RTF | 0.664 |
| Peak allocated GPU memory | 3,079.4 MiB |

### Phase breakdown

The current implementation records only broad CUDA-event boundaries to avoid
reintroducing fine-grained synchronization into the optimized loop.

| Generation phase | Mean latency | Share of wall time |
|---|---:|---:|
| Decode | 66.052 s | 97.23% |
| Codec | 1.831 s | 2.70% |
| Prefill | 50.07 ms | 0.07% |
| Preparation | 1.99 ms | <0.01% |

The complete v2 report is
[`benchmarks/v2_faster_decode_0.6b_alicia.json`](benchmarks/v2_faster_decode_0.6b_alicia.json).

## `faster_decode` v3 — static cache + torch-compiled predictor pass

V3 replaces the growing talker and per-frame predictor caches with
preallocated `StaticCache` instances. The predictor transformer is compiled
once with `torch.compile(mode="reduce-overhead")` and reused across frames.
The excluded warmup absorbs predictor compilation before measurements begin.

```bash
uv run --no-sync python profile_tts.py \
  --text-file alicia.txt \
  --backend split \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --lang English \
  --max-new-tokens 1279 \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/v3_faster_decode_0.6b_alicia.json
```

### Per-run results

| Run | Generation latency | Audio | RTF | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 57.554 s | 102.320 s | 0.562 | 1.778× |
| 2 | 57.455 s | 102.320 s | 0.562 | 1.781× |
| 3 | 57.477 s | 102.320 s | 0.562 | 1.780× |
| **p50** | **57.477 s** | **102.320 s** | **0.562** | **1.780×** |

### Aggregate results

| Metric | Result |
|---|---:|
| Cached model load | 20.211 s |
| Mean generation latency | 57.495 s |
| Generation latency range | 57.455–57.554 s |
| Mean RTF | 0.562 |
| Peak allocated GPU memory | 3,080.6 MiB |

### Phase breakdown

| Generation phase | Mean latency | Share of wall time |
|---|---:|---:|
| Decode | 55.548 s | 96.61% |
| Codec | 1.843 s | 3.20% |
| Prefill | 79.69 ms | 0.14% |
| Preparation | 21.82 ms | 0.04% |

V3 is 15.44% lower latency than v2 and 37.33% lower than the retained official
baseline, equivalent to a 1.596× speedup over official inference.

The complete v3 report is
[`benchmarks/v3_faster_decode_0.6b_alicia.json`](benchmarks/v3_faster_decode_0.6b_alicia.json).

### Modular decode comparison

The separate `profile_tts_modular.py` harness applies checked CUDA-event
instrumentation to an in-memory copy of the current hot path. It does not edit
`faster_decode.py`, synchronize between forwards, or include compilation in
the measured runs.

```bash
uv run --no-sync python profile_tts_modular.py \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/v3_faster_decode_0.6b_alicia_breakdown.json
```

| Decode stage | Calls | v1 p50 | v3 p50 | Change | V3 average |
|---|---:|---:|---:|---:|---:|
| Predictor seed | 1,279 | 3.627 s | 2.004 s | −44.74% | 1.567 ms/frame |
| Predictor residuals | 1,279 × 14 | 45.777 s | 27.409 s | −40.12% | 1.531 ms/pass |
| **Combined predictor** | — | **49.404 s** | **29.413 s** | **−40.46%** | **22.997 ms/frame** |
| Talker step | 1,278 | 19.901 s | 26.106 s | +31.18% | 20.427 ms/step |
| Other decode overhead | — | 58.02 ms | 11.52 ms | −80.14% | — |
| **Total decode** | — | **69.363 s** | **55.518 s** | **−19.96%** | — |

The compiled static-cache predictor is improving: its combined p50 cost falls
by 19.991 seconds. The talker path regresses by 6.205 seconds and now consumes
47.02% of decode, versus 28.69% in v1. This comparison cannot independently
attribute the regression to static cache or another intervening talker change;
an A/B run with only the talker cache type changed is required for that.

The instrumented v3 run measured 57.426 seconds p50 wall latency, close to the
57.477-second uninstrumented headline. The machine-readable breakdown is
[`benchmarks/v3_faster_decode_0.6b_alicia_breakdown.json`](benchmarks/v3_faster_decode_0.6b_alicia_breakdown.json).

### Talker cache A/B

This experiment retains the compiled predictor and its static cache and changes
only the talker cache implementation.

| Metric | Static talker cache | Dynamic talker cache | Change |
|---|---:|---:|---:|
| p50 wall latency | 57.426 s | 50.011 s | −12.91% |
| p50 RTF | 0.561 | 0.489 | −12.91% |
| Total decode | 55.518 s | 48.127 s | −13.31% |
| Talker step | 26.106 s | 18.685 s | −28.43% |
| Combined predictor | 29.413 s | 29.430 s | +0.06% |

The recommended dynamic-talker-cache configuration has this p50 decode
composition:

| Decode module | Calls | p50 latency | Share of decode | Average |
|---|---:|---:|---:|---:|
| Predictor seed | 1,279 | 2.019 s | 4.19% | 1.578 ms/frame |
| Predictor residuals | 1,279 × 14 | 27.411 s | 56.95% | 1.531 ms/pass |
| **Combined predictor** | — | **29.430 s** | **61.15%** | **23.010 ms/frame** |
| Talker step | 1,278 | 18.685 s | 38.82% | 14.620 ms/step |
| Other decode overhead | — | 11.53 ms | 0.02% | — |
| **Total decode** | — | **48.127 s** | **100%** | — |

The predictor's 0.06% difference is measurement noise, confirming that the
talker cache alone causes the regression. In the installed Transformers cache
and masking implementation, `StaticCache` returns its full maximum-length KV
buffers and reports that maximum as the mask length. Its compileable flag also
disables SDPA's causal-mask skip. Every eager one-token talker forward
therefore materializes a mask and attends over the full 1,432-slot allocation.
`DynamicCache` returns only populated KV entries and permits the mask-free SDPA
single-token path.

Static talker cache is counterproductive until the talker forward is compiled
or captured to exploit its fixed addresses and shapes. The immediate
recommendation is to retain the compiled static-cache predictor but use a
dynamic talker cache. The A/B report is
[`benchmarks/v3_dynamic_talker_cache_ab.json`](benchmarks/v3_dynamic_talker_cache_ab.json).

## `faster_decode` v4 — predictor and talker CUDA graphs

V4 captures the complete 15-token predictor loop as one CUDA graph and the
one-token talker pass as a second graph. Both use persistent fixed buffers,
precomputed masks, and static caches. Graph construction and eager capture
warmups occur in the excluded first inference.

The code moved under `src/whistle`, so the current uninstalled source tree
requires `PYTHONPATH=src` with `uv run --no-sync`.

```bash
PYTHONPATH=src uv run --no-sync python profile_tts.py \
  --text-file alicia.txt \
  --backend split \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --lang English \
  --max-new-tokens 1279 \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/v4_faster_decode_0.6b_alicia.json
```

### Per-run results

| Run | Generation latency | Audio | RTF | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 64.432 s | 102.320 s | 0.630 | 1.588× |
| 2 | 64.431 s | 102.320 s | 0.630 | 1.588× |
| 3 | 64.430 s | 102.320 s | 0.630 | 1.588× |
| **p50** | **64.431 s** | **102.320 s** | **0.630** | **1.588×** |

### Aggregate results

| Metric | Result |
|---|---:|
| Cached model load | 16.422 s |
| Mean generation latency | 64.431 s |
| Generation latency range | 64.430–64.432 s |
| Mean RTF | 0.630 |
| Peak allocated GPU memory | 3,158.6 MiB |

V4 remains 29.75% lower latency than official, but regresses 12.10% from v3
and 28.83% from the mixed-cache A/B candidate.

### Modular breakdown

Asynchronous CUDA events wrap the two graph replay boundaries. They do not
modify or synchronize inside the captured graphs.

```bash
PYTHONPATH=src uv run --no-sync python profile_tts_modular.py \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/v4_faster_decode_0.6b_alicia_breakdown.json
```

| Decode module | Calls | p50 latency | Share of decode | Average |
|---|---:|---:|---:|---:|
| Predictor-loop graph | 1,279 | 36.240 s | 57.95% | 28.334 ms/frame |
| Talker graph | 1,278 | 26.180 s | 41.87% | 20.485 ms/step |
| Other decode overhead | — | 113.49 ms | 0.18% | 0.089 ms/frame |
| **Total decode** | — | **62.533 s** | **100%** | — |

The predictor graph is 23.21% slower than v3's 29.413-second compiled
predictor. V4 captures and replays the eager predictor kernels, so it removes
CPU launch overhead but loses the fusion and kernel improvements supplied by
`torch.compile`. Capturing a compiled full-frame predictor is the required A/B.

The talker graph is effectively unchanged from v3 static-cache talker
latency—26.180 versus 26.106 seconds—and remains 40.11% slower than the dynamic
talker-cache control. Graph replay does not offset full-capacity static-cache
SDPA work. The headline and modular reports are
[`benchmarks/v4_faster_decode_0.6b_alicia.json`](benchmarks/v4_faster_decode_0.6b_alicia.json)
and
[`benchmarks/v4_faster_decode_0.6b_alicia_breakdown.json`](benchmarks/v4_faster_decode_0.6b_alicia_breakdown.json).

## `faster_decode` v5 — compiled predictor loop + talker CUDA graph

V5 disables explicit CUDA graph capture for the predictor and compiles the
complete fixed 15-token predictor loop with `torch.compile(mode="reduce-overhead")`.
The one-token talker CUDA graph remains enabled. Predictor compilation and its
Inductor graph-tree warmups occur before the excluded first full inference.

```bash
PYTHONPATH=src uv run --no-sync python profile_tts.py \
  --text-file alicia.txt \
  --backend split \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --lang English \
  --max-new-tokens 1279 \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/v5_faster_decode_0.6b_alicia.json
```

### Per-run results

| Run | Generation latency | Audio | RTF | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 48.099 s | 102.320 s | 0.470 | 2.127× |
| 2 | 48.113 s | 102.320 s | 0.470 | 2.127× |
| 3 | 48.106 s | 102.320 s | 0.470 | 2.127× |
| **p50** | **48.106 s** | **102.320 s** | **0.470** | **2.127×** |

### Aggregate results

| Metric | Result |
|---|---:|
| Cached model load | 10.685 s |
| Mean generation latency | 48.106 s |
| Generation latency range | 48.099–48.113 s |
| Mean RTF | 0.470 |
| Peak allocated GPU memory | 3,146.5 MiB |

V5 is 25.33% lower latency than v4, 3.80% lower than the mixed-cache A/B
candidate, and 47.55% lower than official inference (1.906× speedup).

> **Invalidated (correctness).** The v5 talker graph set
> `StaticCache.is_compileable = False` per layer and passed `attention_mask=None`,
> which re-enables SDPA's mask-free causal skip on a zero-padded static cache.
> SDPA then attends over all 1,432 KV slots (mostly zeros) with `is_causal=False`,
> the attention output attenuates toward zero, and generated codes degenerate to
> silence after ~2 seconds of audio. The 15.452-second talker time and 48.106-second
> headline measure broken output, not a real win. The compiled-predictor half is
> correct and retained. The trustworthy correct best remains the A/B candidate
> (50.011 s); the talker has been reverted to dynamic-eager `DynamicCache`.

### Modular breakdown

| Decode module | Calls | p50 latency | Share of decode | Average |
|---|---:|---:|---:|---:|
| Compiled predictor loop | 1,279 | 30.647 s | 66.32% | 23.962 ms/frame |
| Talker graph | 1,278 | 15.452 s | 33.44% | 12.091 ms/step |
| Other decode overhead | — | 112.40 ms | 0.24% | 0.088 ms/frame |
| **Total decode** | — | **46.211 s** | **100%** | — |

The compiled predictor loop is 15.43% faster than v4's eager predictor graph
(30.647 versus 36.240 seconds). The measured talker graph also falls 40.98%
(15.452 versus 26.180 seconds), though this version changes predictor execution
as well, so that talker improvement should not be attributed to a single
isolated talker change without another A/B. The reports are
[`benchmarks/v5_faster_decode_0.6b_alicia.json`](benchmarks/v5_faster_decode_0.6b_alicia.json)
and
[`benchmarks/v5_faster_decode_0.6b_alicia_breakdown.json`](benchmarks/v5_faster_decode_0.6b_alicia_breakdown.json).

## `faster_decode` v5.1 — explicit-mask talker graph

V5.1 retains V5's compiled predictor loop and talker CUDA graph, with an
explicit per-position causal mask copied into the talker graph input. One
warmup left a late setup pass in the first measured iteration, so the canonical
result uses three excluded full-inference warmups before three measurements.

```bash
PYTHONPATH=src uv run --no-sync python profile_tts.py \
  --text-file alicia.txt \
  --backend split \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --lang English \
  --max-new-tokens 1279 \
  --warmup 3 \
  --iterations 3 \
  --json-out benchmarks/v5_1_faster_decode_0.6b_alicia_warm3.json
```

### Per-run results

| Run | Generation latency | Audio | RTF | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 64.200 s | 102.320 s | 0.627 | 1.594× |
| 2 | 64.188 s | 102.320 s | 0.627 | 1.594× |
| 3 | 64.165 s | 102.320 s | 0.627 | 1.595× |
| **p50** | **64.188 s** | **102.320 s** | **0.627** | **1.594×** |

### Aggregate results

| Metric | Result |
|---|---:|
| Cached model load | 21.855 s |
| Mean generation latency | 64.184 s |
| Generation latency range | 64.165–64.200 s |
| Mean RTF | 0.627 |
| Peak allocated GPU memory | 3,150.5 MiB |

V5.1 is 33.43% slower than V5 but remains 30.02% lower latency than official
inference.

### Modular breakdown

The modular run uses the same three excluded warmups.

| Decode module | Calls | p50 latency | Share of decode | Average |
|---|---:|---:|---:|---:|
| Compiled predictor loop | 1,279 | 33.516 s | 53.98% | 26.205 ms/frame |
| Explicit-mask talker graph | 1,278 | 28.459 s | 45.84% | 22.269 ms/step |
| Other decode overhead | — | 122.46 ms | 0.20% | 0.096 ms/frame |
| **Total decode** | — | **62.090 s** | **100%** | — |

Compared with V5, the predictor loop rises 9.36%, while the explicit-mask
talker graph rises 84.18% (15.452 to 28.459 seconds). Since V5 is invalidated
(silence from the un-masked static-cache talker), V5.1 is the correct
graphed-talker configuration — but its explicit-mask SDPA path is slower than
the dynamic-eager A/B talker (18.685 s). The talker has therefore been reverted
to dynamic-eager `DynamicCache` (no graph); the current correct baseline is the
A/B candidate (50.011 s) with the compiled predictor loop retained. The reports are
[`benchmarks/v5_1_faster_decode_0.6b_alicia_warm3.json`](benchmarks/v5_1_faster_decode_0.6b_alicia_warm3.json)
and
[`benchmarks/v5_1_faster_decode_0.6b_alicia_breakdown_warm3.json`](benchmarks/v5_1_faster_decode_0.6b_alicia_breakdown_warm3.json).
