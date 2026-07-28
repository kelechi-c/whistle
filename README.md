# Whistle

Minimal Qwen3-TTS and Qwen3-ASR inference experiments focused on reducing
generation latency.

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

Every version emits 1,279 complete codec frames, or 102.320 seconds of audio.
Lower latency and RTF are better. Measurements use an RTX 3050 6 GB Laptop GPU
with PyTorch 2.13.0, CUDA 13.0, bfloat16, and SDPA. V2 reduces latency by 4.78%
from v1. V3 reduces latency by another 15.44%, or 37.33% from the retained
official baseline. V3's modular profile shows 40.46% lower predictor latency
than v1, partly offset by 31.18% higher talker-step latency. V4 is deterministic
but regresses 12.10% from v3 because it replays an uncompiled predictor loop
and retains full-capacity static-cache talker attention.

See [results.md](results.md) for the benchmark method, command, individual
runs, and phase breakdown. See [report.md](report.md) for a concise optimization
history and the talker-cache A/B findings.
