# Whistle

Minimal Qwen3-TTS inference experiments focused on reducing generation
latency.

## Headline TTS benchmarks

Official `qwen-tts` runtime, `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, and the
canonical [Alicia input](alicia.txt):

| Version | Tag | p50 latency | p50 RTF | Throughput |
|---|---|---:|---:|---:|
| Official baseline | Official `qwen-tts` | 91.722 s | 0.896 | 1.116× |
| v1 | Explicit prefill/decode | 71.382 s | 0.698 | 1.433× |
| v2 | Reduced Python/CPU sync overhead | 67.974 s | 0.664 | 1.505× |
| **v3** | **Static cache + torch-compiled predictor pass** | **57.477 s** | **0.562** | **1.780×** |
| v4 | CUDA graphs for predictor loop + talker pass | 64.431 s | 0.630 | 1.588× |
| **v5** | **Torch-compiled predictor loop + talker graph** | **48.106 s** | **0.470** | **2.127×** |
| v5.1 | Explicit-mask talker graph | 64.188 s | 0.627 | 1.594× |
| v6 *(invalid)* | Compiled explicit-mask talker | 56.577 s | 0.553 | 1.808× |
| **v7 *(exact)*** | **Predictor + talker FFN eager CUDA graphs** | **56.858 s** | **0.556** | **1.800×** |

Every version emits 1,279 complete codec frames, or 102.320 seconds of audio.
Lower latency and RTF are better. Measurements use an RTX 3050 6 GB Laptop GPU
with PyTorch 2.13.0, CUDA 13.0, bfloat16, and SDPA. V2 reduces latency by 4.78%
from v1. V3 reduces latency by another 15.44%, or 37.33% from the retained
official baseline. V3's modular profile shows 40.46% lower predictor latency
than v1, partly offset by 31.18% higher talker-step latency. V4 is deterministic
but regresses 12.10% from v3 because it replays an uncompiled predictor loop
and retains full-capacity static-cache talker attention. V5 compiles the full
predictor loop instead of explicitly graph-capturing it, reducing p50 latency
by 25.33% from v4 and 47.55% from official.
V5.1 restores an explicit talker mask and regresses 33.43% from v5; the
fully warmed modular profile shows the talker graph rising 84.18%.
V7 replaces numerically divergent compilation with exact eager-kernel CUDA
graphs: one graph per residual-predictor codebook position and one graph for
each talker layer's fixed-shape residual FFN. It is 38.01% faster than official
and only 0.50% slower than the invalid V6 result.

Correctness status: V5 and V6 are performance diagnostics with invalid codec
output. V7 passed an independent full-sequence validation against a reference
generated before its graph wrappers were installed: all 20,480 codec IDs and
2,457,600 waveform samples matched exactly.

## Head-to-head vs `faster-qwen3-tts`

Same GPU, identical protocol (RTX 3050 6 GB Laptop GPU, 0.6B CustomVoice,
alicia input, greedy decode, repetition penalty 1.2, fixed 1,279 frames):

| Engine | p50 wall | RTF | Throughput | Parity |
|---|---:|---:|---:|---|
| `faster-qwen3-tts` 0.3.2 | 66.6 s | 0.651 | 1.54× | not bit-exact |
| **whistle v7** | **56.9 s** | **0.556** | **1.80×** | exact |

Whistle is ~14.5% faster than `faster-qwen3-tts` on identical hardware while
also holding exact codec-ID + waveform parity. The gap comes from keeping the
talker's growing-KV attention on the cheap dynamic-cache path (their static
full-capacity talker attends over padding and was measured ~28% slower) plus a
leaner on-device hot loop (preallocated buffers, no per-step list/stack/clone).

`docs/qwen3_tts_official_vs_faster.md` analyzes the two implementations in
detail.

See [results.md](results.md) for the benchmark method, command, individual
runs, and phase breakdown. See [report.md](report.md) for a concise optimization
history and the talker-cache A/B findings. For the complete codebase map and
an in-depth explanation of every optimization with diagrams, see
[docs/technical_deep_dive.md](docs/technical_deep_dive.md).
