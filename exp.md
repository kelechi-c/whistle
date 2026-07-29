# Qwen3-TTS optimization experiments

## Goal

The objective was to reduce batch-one Qwen3-TTS 0.6B CustomVoice inference
latency on the local RTX 3050 Laptop GPU without changing the official weights,
greedy token policy, codec IDs, or decoded waveform.

All risky implementations were developed under `sandbox/latency_lab/` before
the successful subset was promoted into `src/whistle/`.

## Experimental controls

The main benchmark configuration was:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| GPU | RTX 3050 Laptop GPU, 6 GB |
| Dtype | bfloat16 |
| Attention | PyTorch SDPA |
| Input | Complete `alicia.txt` |
| Speaker | Ryan for the historical fixed-budget benchmark |
| Language | English |
| Selection | Greedy `argmax` |
| Repetition penalty | 1.2 |
| Budget | 1,280 selected talker tokens |
| Complete codec frames | 1,279 |
| Audio duration | 102.320 seconds |
| Timing policy | One excluded warmup, then three measured runs |

The fixed-token benchmark ignores EOS so every historical version performs the
same amount of autoregressive work. Natural-EOS runs were kept separate for
correctness and listening diagnostics.

## Correctness gate

Plausible audio was not accepted as proof of correctness. Every candidate had
to be checked at progressively stronger levels:

1. Codec tensor shape.
2. Location of the first differing codec ID.
3. Exact equality of every codec ID.
4. Exact waveform shape and sample equality.
5. Full-sequence validation where the eager reference was generated before
   installing any candidate wrappers.

This distinction invalidated several earlier fast results. Small bf16
differences can flip one greedy `argmax`; autoregression then magnifies that
single mismatch across the remaining sequence.

The final independent validation matched:

| Validation | Result |
|---|---:|
| Codec IDs | 20,480 / 20,480 exact |
| Waveform samples | 2,457,600 / 2,457,600 exact |
| Candidate peak allocated VRAM | approximately 3.22 GB |

## Baseline investigation

The official eager path performs two nested autoregressive processes:

- The talker emits the primary codec code for each audio frame.
- The residual predictor emits the remaining 15 codebook values for that
  frame.

Initial modular profiling showed that the predictor dominated decode:

| Exact dynamic path, 128 frames | p50 | Decode share |
|---|---:|---:|
| Residual predictor | 4.914 s | 71.87% |
| Talker | 1.821 s | 26.64% |
| Scheduling and token work | 0.102 s | 1.49% |

This changed the optimization priority. Python cleanup had a small remaining
ceiling; the predictor's many tiny launches were the primary target.

Kernel profiling confirmed that batch-one bf16 matrix-vector work dominated.
Two GEMV kernel families accounted for about 55.6% of measured CUDA time, with
ordinary matrix multiplication adding about 9.7%. SDPA itself was only a few
percent, so attention-only tuning could not produce the next large gain.

## Experiments

### 1. Remove nested predictor generation scheduling

The first candidate reproduced the official 15 predictor steps directly:

- Create fresh dynamic predictor KV state.
- Run the two-token predictor seed.
- Select each residual code with greedy `argmax`.
- Feed the selected codebook embedding into the next position.

This removed Hugging Face `GenerationMixin` scheduling while retaining the
same model forwards and operation order.

At 128 frames:

| Path | p50 |
|---|---:|
| Official outer eager | 7.838 s |
| Direct dynamic scheduler | 7.007 s |

Result: 10.6% lower wall latency with exact codec and waveform parity.

### 2. Prefix-visible preallocated KV cache

`DynamicCache` repeatedly concatenates keys and values. I tested a cache with
fixed backing storage that returned only the populated prefix:

- Writes use `index_copy_`.
- Attention receives the same active KV shape as `DynamicCache`.
- Reset changes only the logical length.
- Unused capacity is never exposed to SDPA.

This preserved exact parity but measured 7.004 seconds p50 at 128 frames,
effectively identical to the 7.007-second direct dynamic path.

Conclusion: predictor KV allocation was not a meaningful bottleneck. The cache
was still useful as stable storage for graph capture.

### 3. One CUDA graph per predictor codebook

The predictor always performs the same 15 residual positions. Each position
has a fixed query and active-cache shape, even though the shapes differ between
positions. Instead of forcing all positions into one incorrect full-capacity
static graph, I captured 15 eager CUDA graphs:

- Graph 0 consumes the two-token predictor seed.
- Graphs 1–14 consume the previous residual codebook token.
- All graphs share one prefix-visible cache allocation and graph memory pool.
- The original eager kernels and per-position attention shapes are retained.

At 128 frames:

| Path | p50 |
|---|---:|
| Direct dynamic scheduler | 7.007 s |
| Predictor graphs | 5.682 s |

Result: another 18.9% reduction from the direct path, with exact codec and
waveform parity.

### 4. Talker MLP-only graphs

Each talker MLP was wrapped in a fixed one-token CUDA graph while prefill fell
back to eager execution.

This was slower. The input copy and replay boundary did not capture enough of
the layer to offset their overhead.

Conclusion: rejected.

### 5. Talker residual-FFN graphs

The graph boundary was expanded to include:

1. Post-attention RMS normalization.
2. Gate/up/down MLP.
3. Residual addition.

Talker attention remained eager because its KV length grows every frame.
Prefill also remained eager because its sequence length is request-dependent.

At 128 frames:

| Path | p50 |
|---|---:|
| Predictor graphs | 5.682 s |
| Predictor + residual-FFN graphs | 5.525 s |

Result: a further 2.8% reduction with exact output.

The updated modular split was:

| V7 decode component | p50 | Decode share |
|---|---:|---:|
| Residual predictor | 3.510 s | 65.94% |
| Talker | 1.743 s | 32.75% |
| Scheduling and token work | 0.069 s | 1.30% |

### 6. Gate/up projection fusion

The MLP gate and up projections were concatenated into one larger matrix
multiplication, then split before activation and multiplication.

It matched the first 64 frames exactly but improved the 128-frame result by
only about 0.4%. It did not receive the independent full-sequence parity run.

Conclusion: left experimental and not promoted.

### 7. QKV projection fusion

Q, K, and V projections were combined into one matrix multiplication. Although
the equations were equivalent, the larger GEMV selected a different numerical
reduction path.

The first codec mismatch occurred at frame 6, codebook 1. The candidate was
rejected immediately rather than evaluated by audio plausibility.

### 8. Earlier static and compiled paths

Earlier experiments used full-capacity `StaticCache`, explicit masks,
`torch.compile`, and manual talker graphs. They were useful performance
diagnostics but failed exact parity:

| Talker | Predictor | First mismatch |
|---|---|---|
| Static eager | Static eager | Frame 3, codebook 15 |
| Compiled static | Static eager | Frame 5, codebook 15 |
| Static eager | Compiled static | Frame 1, codebook 13 |

The full-capacity static talker also made SDPA attend over unused cache slots or
required a large explicit mask. Some configurations produced plausible but
incorrect audio; one degenerated toward silence.

Conclusion: these paths remain diagnostic modes, not valid optimized output.

## Final V7 design

The promoted `predictor-ffn-graphs` path consists of:

- `PrefixStaticLayer`: stable predictor KV allocation with dynamic-prefix
  attention semantics.
- `PredictorGraphs`: the 15 official eager predictor positions captured as
  separate CUDA graphs.
- `DecoderFfnGraph`: fixed-shape talker residual FFN capture.
- `OfficialTalker`: growing dynamic talker cache and eager SDPA.
- Official bf16 embedding reduction order.
- Float32 repetition penalty and token suppression.
- Greedy `argmax`.
- Reachable EOS and natural termination by default.
- Official codec decoder.

On CPU, `PredictorGraphs` runs the same sequence eagerly because CUDA graphs
are unavailable.

## Final benchmark

The historical fixed-budget V7 command is:

```bash
uv run python profile_tts.py \
  --text-file alicia.txt \
  --backend split \
  --talker-mode predictor-ffn-graphs \
  --speaker Ryan \
  --lang English \
  --max-new-tokens 1279 \
  --fixed-tokens \
  --repetition-penalty 1.2 \
  --warmup 1 \
  --iterations 3 \
  --json-out benchmarks/v7_exact_graphs_0.6b_alicia.json
```

The profiler's explicit loop counts completed frames, so 1,279 frames match
the official benchmark's 1,280 selected-token convention.

| Run | Wall latency | RTF | Throughput |
|---:|---:|---:|---:|
| 1 | 56.829 s | 0.555 | 1.800× |
| 2 | 56.858 s | 0.556 | 1.800× |
| 3 | 56.881 s | 0.556 | 1.799× |
| **p50** | **56.858 s** | **0.556** | **1.800×** |

Mean phase times:

| Phase | Mean | Wall share |
|---|---:|---:|
| Decode | 54.981 s | 96.70% |
| Codec | 1.821 s | 3.20% |
| Prefill | 50.85 ms | 0.09% |
| Preparation | 2.10 ms | less than 0.01% |

V7 is 38.01% lower latency than the retained 91.722-second official baseline.
It is 0.50% slower than the 56.577-second V6 diagnostic, but V6 fails codec
parity. V7 is therefore the fastest promoted path with full exact-output
validation.

## Why this conclusion

V7 was selected because it targets the measured bottleneck—thousands of tiny
predictor and talker FFN launches—without changing the operations whose
numerical differences had already caused autoregressive divergence.

The final design deliberately leaves some theoretical speed on the table:

- Talker attention remains dynamic and eager.
- GEMV weights and reduction algorithms remain official.
- The codec decoder is unchanged.
- No quantization or approximate attention is used.

Those constraints make the result defensible: the latency reduction comes from
scheduling and graph replay, not from silently changing generated speech.

## Next optimization targets

The remaining opportunities, in priority order, are:

1. A persistent fused predictor kernel or device-side 15-position loop that
   preserves the official GEMV accumulation order.
2. Larger talker graph regions around shape-invariant projections and residual
   work without graphing variable-length attention.
3. Output-length-bucketed codec decoder compilation or capture; its maximum
   current benefit is roughly 3.2% of total wall time.
4. Device-side token processing and frame scheduling; its present ceiling is
   about 1.3% of decode.
5. Separately labeled quality experiments with weight-only int8 GEMV,
   quantization, or speculative decoding.

Any new candidate should pass the same independent full codec-token and
waveform validation before its latency is added to the valid headline table.
