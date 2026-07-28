# Benchmark results

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
forces `min_new_tokens == max_new_tokens == 512`, preventing EOS from making
one implementation finish early. The official generator produces 511 complete
codec frames from those 512 selected talker tokens, or 40.880 seconds of
24 kHz audio.

One warmup is excluded. The reported p50 values are the medians of three
measured runs.

```bash
uv run --no-sync python profile_tts.py \
  --text-file alicia.txt \
  --backend official \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --lang English \
  --max-new-tokens 512 \
  --fixed-tokens \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/official_tts_0.6b_alicia.json
```

### Per-run results

| Run | Generation latency | Audio | RTF | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 35.267 s | 40.880 s | 0.863 | 1.159× |
| 2 | 35.098 s | 40.880 s | 0.859 | 1.165× |
| 3 | 35.111 s | 40.880 s | 0.859 | 1.164× |
| **p50** | **35.111 s** | **40.880 s** | **0.859** | **1.164×** |

### Aggregate results

| Metric | Result |
|---|---:|
| Cached model load | 8.487 s |
| Mean generation latency | 35.159 s |
| Generation latency range | 35.098–35.267 s |
| Mean RTF | 0.860 |
| Peak allocated GPU memory | 2,772.7 MiB |

### Phase breakdown

Phase measurements are non-overlapping: code-predictor time is subtracted from
the inclusive talker measurement.

| Generation phase | Mean latency | Share of wall time |
|---|---:|---:|
| Code predictor | 25.199 s | 71.67% |
| Talker excluding code predictor | 9.189 s | 26.14% |
| Speech codec | 749.68 ms | 2.13% |
| Wrapper overhead | 21.35 ms | 0.06% |

The complete machine-readable report is
[`benchmarks/official_tts_0.6b_alicia.json`](benchmarks/official_tts_0.6b_alicia.json).

## Explicit `faster_decode` baseline

This path uses the same official model modules and weights but replaces the
official nested generation scheduler with the explicit prefill and greedy
decode loop in `faster_decode.py`.

### Method

The environment and Alicia input are the same as the official baseline. The
split path forces `min_new_tokens == max_new_tokens == 512`, producing 512
codec frames and 40.960 seconds of audio. One warmup is excluded, followed by
three measured runs.

The official path's 512-token budget emits 511 complete frames, while the
explicit loop's public limit counts completed frames directly. The 80 ms
output-duration difference is recorded here rather than silently treating the
two output shapes as identical.

```bash
uv run --no-sync python profile_tts.py \
  --text-file alicia.txt \
  --backend split \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --lang English \
  --max-new-tokens 512 \
  --fixed-tokens \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/faster_decode_0.6b_alicia.json
```

### Per-run results

| Run | Generation latency | Audio | RTF | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 27.902 s | 40.960 s | 0.681 | 1.468× |
| 2 | 27.857 s | 40.960 s | 0.680 | 1.470× |
| 3 | 28.057 s | 40.960 s | 0.685 | 1.460× |
| **p50** | **27.902 s** | **40.960 s** | **0.681** | **1.468×** |

### Aggregate results

| Metric | Result |
|---|---:|
| Cached model load | 15.914 s |
| Mean generation latency | 27.939 s |
| Generation latency range | 27.857–28.057 s |
| Mean RTF | 0.682 |
| Peak allocated GPU memory | 2,869.2 MiB |

### Phase breakdown

| Generation phase | Mean latency | Share of wall time |
|---|---:|---:|
| Decode | 27.138 s | 97.13% |
| Codec | 747.64 ms | 2.68% |
| Prefill | 50.21 ms | 0.18% |
| Preparation | 1.98 ms | 0.01% |

### Decode breakdown

This diagnostic run adds asynchronous CUDA events around the three repeated
decode stages. It does not synchronize between forwards. The uninstrumented
27.902-second p50 above remains the headline latency; the instrumented run's
p50 was 28.292 seconds.

| Decode stage | Calls | p50 latency | Share of decode | Average |
|---|---:|---:|---:|---:|
| Predictor seed stage | 512 | 1.445 s | 5.26% | 2.823 ms/frame |
| Predictor residual stage | 512 × 14 | 18.482 s | 67.23% | 36.098 ms/frame |
| Talker step | 511 | 7.540 s | 27.43% | 14.756 ms/step |
| Other and host overhead | — | 22.97 ms | 0.08% | — |
| **Total decode** | — | **27.491 s** | **100%** | — |

The two predictor stages total 19.928 seconds, or 72.49% of decode and 38.921
ms per generated frame. The residual stage includes its embedding, forward,
argmax, and cache-update sequence for all 14 residual codebooks. The talker
stage includes frame-embedding assembly, mask and position preparation, the
cached backbone forward, and next-token selection. “Other” is the decode wall
time left after subtracting the three CUDA-event regions.

The machine-readable diagnostic report is
[`benchmarks/faster_decode_0.6b_alicia_breakdown.json`](benchmarks/faster_decode_0.6b_alicia_breakdown.json).

The complete machine-readable report is
[`benchmarks/faster_decode_0.6b_alicia.json`](benchmarks/faster_decode_0.6b_alicia.json).
