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
