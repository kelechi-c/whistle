# Whistle

Minimal Qwen3-TTS and Qwen3-ASR inference experiments focused on reducing
generation latency.

## Headline TTS baseline

Official `qwen-tts` runtime, `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, and the
canonical [Alicia input](alicia.txt):

| Metric | Official runtime | Explicit `faster_decode` |
|---|---:|---:|
| **p50 generation latency** | **91.722 s** | **71.382 s** |
| **p50 real-time factor (RTF)** | **0.896** | **0.698** |
| p50 throughput | 1.116× | 1.433× |
| Output duration | 102.320 s | 102.320 s |
| Fixed talker-token budget | 1,280 | 1,280 |
| Complete codec frames | 1,279 | 1,279 |

Lower latency and RTF are better. The result was measured on an RTX 3050 6 GB
Laptop GPU with PyTorch 2.13.0, CUDA 13.0, bfloat16, and SDPA. Under this
shared budget, `faster_decode` reduces p50 latency by 22.18% (1.285× speedup).

See [results.md](results.md) for the benchmark method, command, individual
runs, and phase breakdown.
