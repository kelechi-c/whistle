# Qwen3-TTS inference optimization report

## Scope

This project optimizes batch-one Qwen3-TTS 0.6B CustomVoice inference while
retaining the official model weights and codec decoder. Benchmarks use the full
`alicia.txt` input, Ryan, English, bfloat16 SDPA, three measured runs, and a
fixed 1,279-frame output (102.320 seconds of audio) on an RTX 3050 6 GB Laptop
GPU. V5.1 and V6 use three excluded full warmups.

## Optimization progression

| Version | Main change | p50 latency | RTF | Gain from vprior |
|---|---|---:|---:|---:|
| Official | Official nested generation runtime | 91.722 s | 0.896 | — |
| V1 | Explicit prefill and decode scheduler | 71.382 s | 0.698 | 22.18% |
| V2 | Reduced Python and CPU synchronization overhead | 67.974 s | 0.664 | 4.78% |
| V3 | Static caches and torch-compiled predictor | 57.477 s | 0.562 | 15.44% |
| A/B candidate | V3 with dynamic talker cache | 50.011 s | 0.489 | 12.91% |
| V4 | CUDA graphs for predictor loop and talker pass | 64.431 s | 0.630 | −28.83% vs candidate |
| V5 | Compiled predictor loop and talker graph | 48.106 s | 0.470 | 25.33% vs V4 (invalid) |
| V5.1 | Explicit-mask talker graph | 64.188 s | 0.627 | −33.43% vs V5 |
| V6 | Compiled explicit-mask talker | 56.577 s | 0.553 | 11.86% vs V5.1 (invalid) |

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
remained captured. **V5 is invalidated.** Its talker graph set
`StaticCache.is_compileable = False` per layer and passed `attention_mask=None`,
which re-enables SDPA's mask-free causal skip on a zero-padded static cache:
SDPA then attends over all 1,432 KV slots (mostly zeros) with `is_causal=False`,
the attention output attenuates toward zero, and the generated codes degenerate
to silence after ~2 seconds. The 15.452-second talker time and 48.106-second
headline measure broken output, not a real win. The compiled predictor half is
correct and retained.

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

V6 compiles that explicit-mask talker. It reduces total decode to 54.720
seconds: 30.721 seconds (56.14%) predictor, 23.887 seconds (43.66%) talker,
and 112.18 ms (0.20%) remaining overhead. Its 56.577-second headline p50 is
38.32% below official. However, exact official codec-token parity fails at
frame 1/codebook 13 after 1,181 of 20,448 shared IDs match. The first frame
and the first thirteen codebooks of frame 1 match, so a small numerical change
in the compiled talker hidden state crosses a later greedy predictor argmax
boundary; autoregression then makes the sequence diverge. V6 is therefore a
performance diagnostic, not a valid quality result.

## Correctness failure and recovery

### Codec-token divergence

The optimized runtime was initially judged by whether it produced plausible
audio. Exact comparison showed that this was insufficient: static eager
talker/predictor execution first diverged from the official runtime at frame 3,
codebook 15. Compiling the static predictor moved the first mismatch to frame
1, codebook 13, while compiling only the static talker produced its first
mismatch at frame 5, codebook 15.

Several differences contributed to the failure:

- Static-cache attention and its explicit mask changed the bf16 reduction path
  relative to the official `DynamicCache` implementation. Tiny logit changes
  eventually changed an `argmax`, after which autoregression amplified the
  mismatch.
- The compiled predictor mutated static cache storage through a different
  execution path, making divergence occur earlier.
- The primary logits were incorrectly sliced to the 2,048 codec-token
  vocabulary, excluding talker EOS token 2,150.
- Residual embeddings were combined in a different bf16 reduction order.
- Repetition penalty and token suppression were applied to bf16 logits, while
  the official generation path processes logits in float32.
- The parity profiler compared differently aligned frame sets, and the decoder
  consumed the full preallocated token buffer instead of stopping at EOS.

### Silent-tail failure

The audio was not physically truncated at eight seconds. Under greedy
generation, repetition penalties 1.05 and 1.1 caused the model to enter a
low-energy repetitive state after roughly 16 seconds, so the remainder sounded
cut off despite still containing samples. This generation-policy collapse was
separate from the codec-token parity failure.

Raising the greedy repetition penalty to 1.2 prevented that collapse for the
test input: signal energy remained healthy throughout the 97.28-second output,
and the model reached EOS naturally.

### Fix

The correctness reference now follows the official numerical path while
retaining greedy `argmax` selection:

- use `DynamicCache` and the official outer talker forward one token at a time;
- use the official greedy residual predictor;
- preserve the official bf16 embedding reduction order;
- apply repetition penalty, suppression rules, and `argmax` to float32 logits;
- keep the full talker vocabulary so EOS remains reachable;
- stop and trim the generated sequence at EOS;
- align complete frames correctly in the parity profiler; and
- decode with the official codec tokenizer rather than a duplicate decoder.

This is a correctness baseline, not the final optimized design. Manual CUDA
graph capture is incompatible with changing `DynamicCache` storage addresses.
Any future static-cache implementation must reproduce the official
populated-key attention math before graph capture or compilation is considered
valid.

### Validation

With repetition penalty 1.2, the repaired greedy path reached natural EOS after
1,216 frames and matched the official runtime exactly:

| Check | Result |
| --- | ---: |
| Codec IDs | 19,456 / 19,456 exact |
| Waveform samples | 2,334,720 / 2,334,720 exact |
| Audio duration | 97.28 s |
| RMS, 0–8 s | 0.02960 |
| RMS, 8–16 s | 0.03217 |
| RMS, 16–32 s | 0.02785 |
| RMS, 32–64 s | 0.02482 |
| RMS, 64–97.28 s | 0.02298 |

## Conclusion

V5 and V6 latency results remain invalid because those paths fail exact token
parity. The official-eager greedy path is now the correctness reference.
Optimized variants must match its complete codec-token and waveform output
before their latency results are treated as valid.
