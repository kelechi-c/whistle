# Qwen3-TTS inference optimization report

## Scope

This project optimizes batch-one Qwen3-TTS 0.6B CustomVoice inference while
retaining the official model weights and codec decoder. Benchmarks use the full
`alicia.txt` input, Ryan, English, bfloat16 SDPA, one excluded warmup, three
measured runs, and a fixed 1,279-frame output (102.320 seconds of audio) on an
RTX 3050 6 GB Laptop GPU.

## Optimization progression

| Version | Main change | p50 latency | RTF | Gain from prior |
|---|---|---:|---:|---:|
| Official | Official nested generation runtime | 91.722 s | 0.896 | — |
| V1 | Explicit prefill and decode scheduler | 71.382 s | 0.698 | 22.18% |
| V2 | Reduced Python and CPU synchronization overhead | 67.974 s | 0.664 | 4.78% |
| V3 | Static caches and torch-compiled predictor | 57.477 s | 0.562 | 15.44% |
| A/B candidate | V3 with dynamic talker cache | 50.011 s | 0.489 | 12.91% |

V1 exposed the predictor and talker forwards instead of relying on nested
Hugging Face generation. V2 kept buffers, cache positions, codec IDs, and
waveform assembly on-device and removed repeated Python/CPU synchronization.
V3 compiled the predictor transformer in reduce-overhead mode and gave it a
fixed static KV cache.

## Findings

The compiled predictor optimization worked. Combined predictor latency fell
from 49.404 seconds in V1 to 29.413 seconds in V3, a 40.46% reduction. Residual
predictor passes now average 1.531 ms each.

The initial V3 static talker cache regressed talker latency from 19.901 to
26.106 seconds. A controlled A/B changed only that cache back to dynamic:
talker latency fell to 18.685 seconds while predictor latency changed by just
0.06%. End-to-end p50 fell from 57.426 to 50.011 seconds.

The cause is the interaction between eager SDPA and Transformers
`StaticCache`. Static cache returns the full 1,432-slot KV allocation, reports
that maximum as mask length, and disables SDPA's mask-free causal shortcut.
Each one-token eager talker pass therefore materializes a causal mask and
attends across the full allocation. Dynamic cache exposes only populated KV
entries and uses the cheaper mask-free single-token path.

## Conclusion

Static cache is beneficial for the compiled predictor but currently harmful
for the eager talker. The best measured configuration is a compiled
static-cache predictor with a dynamic talker cache: 50.011 seconds p50,
1.834× faster than the official baseline. Static talker cache should return
only when the talker forward is compiled or CUDA-graph captured so fixed
addresses and shapes can offset its larger attention extent.
