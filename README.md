# Whistle

Minimal Qwen3-TTS and Qwen3-ASR inference experiments focused on reducing
generation latency.

## Headline TTS baseline

Official `qwen-tts` runtime, `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`, and the
canonical [Alicia input](alicia.txt):

| Metric | Official runtime | Explicit `faster_decode` |
|---|---:|---:|
| **p50 generation latency** | **35.111 s** | **27.902 s** |
| **p50 real-time factor (RTF)** | **0.859** | **0.681** |
| p50 throughput | 1.164×  | 1.468× |
| Output duration | 40.880 s | 40.960 s |
| Fixed generation budget | 512 talker tokens | 512 codec frames |

Lower latency and RTF are better. The result was measured on an RTX 3050 6 GB
Laptop GPU with PyTorch 2.13.0, CUDA 13.0, bfloat16, and SDPA.

See [results.md](results.md) for the benchmark method, command, individual
runs, and phase breakdown.
