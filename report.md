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
| V4 | CUDA graphs for predictor loop and talker pass | 64.431 s | 0.630 | −28.83% vs candidate |
| V5 | Compiled predictor loop and talker graph | 48.106 s | 0.470 | 25.33% vs V4 |
| V5.1 | Explicit-mask talker graph | 64.188 s | 0.627 | −33.43% vs V5 |

V1 exposed the predictor and talker forwards instead of relying on nested
Hugging Face generation. V2 kept buffers, cache positions, codec IDs, and
waveform assembly on-device and removed repeated Python/CPU synchronization.
V3 compiled the predictor transformer in reduce-overhead mode and gave it a
fixed static KV cache.

V4 captured the complete eager predictor loop and one talker token as separate
CUDA graphs. Its extremely stable 64.431-second result shows deterministic
replay, but latency regressed.

V5 disabled explicit predictor graph capture and compiled the complete
predictor loop with `torch.compile(mode="reduce-overhead")`; the talker graph
remained captured. It is the fastest measured configuration at 48.106 seconds
p50, 1.906× faster than official.

V5.1 adds an explicit causal mask to the talker graph. It requires three full
excluded warmups to remove a late setup pass, then stabilizes at 64.188 seconds
p50. The talker graph rises from 15.452 to 28.459 seconds, an 84.18%
regression.

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

V4's modular profile explains its regression. The captured predictor loop
takes 36.240 seconds, 23.21% slower than the torch-compiled V3 predictor,
because graph capture replays the unfused eager kernels instead of retaining
the compiled predictor kernels. The captured talker remains at 26.180 seconds:
graph replay reduces launch overhead but does not reduce full-capacity static
attention work.

V5 restores the compiled predictor advantage: predictor-loop time falls from
36.240 seconds in V4 to 30.647 seconds. Talker-graph time also falls from
26.180 to 15.452 seconds in the measured configuration. Total decode reaches
46.211 seconds: 66.32% predictor, 33.44% talker, and 0.24% remaining overhead.

V5.1 shifts the decode mix to 53.98% predictor and 45.84% talker. The explicit
mask increases total decode from 46.211 to 62.090 seconds; predictor rises
9.36% while the talker increase is the dominant cost at 84.18%.

## Conclusion

Static cache is beneficial for the compiled predictor but currently harmful
for the eager talker. The best measured configuration is a compiled
static-cache predictor with a dynamic talker cache: 50.011 seconds p50,
1.834× faster than the official baseline. Static talker cache should return
only when the talker forward is compiled or CUDA-graph captured so fixed
addresses and shapes can offset its larger attention extent. V4 shows that
capture alone is insufficient: the next experiment should capture a compiled
full-frame predictor and avoid full-capacity talker attention. V5 effectively
validates the first half of that direction: preserve the compiler-managed
predictor graph tree rather than replacing it with an eager CUDA graph.
The explicit-mask talker path should remain disabled for this workload.
