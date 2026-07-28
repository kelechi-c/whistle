# Whistle

Minimal Qwen3-TTS and Qwen3-ASR inference experiments focused on reducing
generation latency.

## Headline TTS benchmarks

Official `qwen-tts` runtime, `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, and the
canonical [Alicia input](alicia.txt):

| Version | Tag | p50 latency | p50 RTF | Throughput |
|---|---|---:|---:|---:|
| Official baseline | Official `qwen-tts` | 91.722 s | 0.896 | 1.116× |
| `faster_decode` v1 | Explicit decode scheduler | 71.382 s | 0.698 | 1.433× |
| **`faster_decode` v2** | **Reduced Python/CPU sync overhead** | **67.974 s** | **0.664** | **1.505×** |

Every version emits 1,279 complete codec frames, or 102.320 seconds of audio.
Lower latency and RTF are better. Measurements use an RTX 3050 6 GB Laptop GPU
with PyTorch 2.13.0, CUDA 13.0, bfloat16, and SDPA. V2 reduces latency by 4.78%
from v1 and by 25.89% from the retained official baseline.

See [results.md](results.md) for the benchmark method, command, individual
runs, and phase breakdown.
