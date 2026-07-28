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
