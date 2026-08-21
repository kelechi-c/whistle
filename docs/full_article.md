# Whistle — The Full Technical Article

*How to make Qwen3-TTS 1.8× real-time on a 6 GB laptop GPU without changing a
single output token — and why the two fastest versions had to be thrown away.*

All measurements: RTX 3050 Laptop GPU (6 GB), PyTorch 2.13.0, CUDA 13.0,
bfloat16, SDPA attention, greedy decoding, batch one. Every latency figure is a
p50 over three measured runs after excluded warmups, on the same fixed input,
speaker, and output budget.

---

## 1. The thesis

Whistle is an execution engine wrapped around the official Qwen3-TTS model. It
does not change the weights, the tokenizer, the codec decoder, the sampling
policy, or the math. It changes only *how the forward passes are scheduled on
the GPU*. The result: batch-one generation of `Qwen3-TTS-12Hz-0.6B-CustomVoice`
drops from **91.722 s to 56.858 s** for 102.320 s of audio (RTF 0.896 → 0.556,
**1.800× real-time**, −38.01%), while every codec token and every waveform
sample stays bit-exact against the official runtime.

Two claims sit underneath that number:

1. **The official runtime is launch-bound, not compute-bound.** Each 80 ms
   frame of audio triggers roughly 95 transformer layer forwards plus all of
   Hugging Face's generation machinery — about 20,000+ tiny Python-dispatched
   CUDA launches per utterance. The kernels themselves are trivial; the cost is
   getting them onto the GPU. So the win comes from scheduling and graph
   capture, not from faster kernels.

2. **A fast number is not a correct number.** Autoregressive decoding turns one
   flipped bf16 `argmax` into a permanently diverged sequence, and plausible
   audio proves nothing. Two configurations that measured *faster* than the
   final result — V5 at 48.1 s and V6 at 56.6 s — were invalidated by an exact
   codec-token + waveform parity gate: V5 produced silence, V6 diverged at
   frame 1. What survives is the fastest version that is *exactly* right.

The rest of this article walks through the model, the bottleneck, the full
optimization ladder V1→V7 with its failures, and the correctness engineering
that made the final number trustworthy.

---

## 2. What one frame of speech actually costs

### 2.1 Two nested autoregressive transformers

Qwen3-TTS generates speech as discrete codec tokens at 12.5 Hz — one frame per
80 ms of audio. Each frame is a row of **16 codebook tokens**: one *primary*
token predicted by a large transformer (the **talker**) and fifteen *residual*
tokens predicted by a small one (the **code predictor**), Multi-Token-Prediction
style. A neural codec then decodes the `[frames × 16]` ID matrix into 24 kHz
waveform.

```text
talker (~0.6B): 20 layers · hidden 1024 · GQA 16Q/2KV · head_dim 64
                3D mRoPE · Q/K-norm · SwiGLU
    └─ per frame: ONE primary codec token + hidden state
codec_head:     Linear(1024 → 3072) → primary codebook logits

code predictor: 5 layers · hidden 1024 · GQA 16Q/8KV · head_dim 128
                31 codebook embeddings + 31 lm_heads
    └─ per frame: FIFTEEN autoregressive steps
       step i consumes the embedding of residual codebook i−1,
       each step reads out through its own lm_head[i]

speech_tokenizer: official codec decoder, codec IDs → waveform
```

The predictor is genuinely sequential: step *i* needs token *i−1*'s embedding.
And both models are fed by history — the talker's KV cache grows by one
position per frame, and within a frame the predictor attends over a growing
prefix of up to 16 positions. Everything depends on everything before it.

### 2.2 Frame arithmetic, or why this workload hates the GPU

| Quantity | Value |
|---|---:|
| Codec frame | 12.5 Hz → 80 ms audio |
| Codebooks per frame | 16 = 1 primary + 15 residual |
| Transformer layer forwards per frame | 20 (talker) + 5 × 15 (predictor) = **95** |
| Layer forwards per benchmark run | 1,279 frames × 95 ≈ **121,500** |
| Audio per benchmark run | 102.320 s |

At batch one, every one of those forwards is a pile of GEMV-shaped work —
matrix-vector multiplies with tiny arithmetic intensity. Kernel profiling of
the baseline confirmed where CUDA time actually goes:

- two GEMV kernel families ≈ **55.6%** of measured CUDA time;
- ordinary matmul ≈ 9.7%;
- **SDPA attention — the thing people usually optimize — only a few percent.**

That single measurement set the strategy for the whole project: do not write
attention kernels, do not quantize (yet), do not touch the math. Attack the
~20,000 launches and the Python between them.

### 2.3 Where the official runtime spends the time

The official `qwen-tts` implementation nests three layers of generality inside
each other:

```text
GenerationMixin.generate (talker)
 └─ for each of ~1,279 outer tokens:
     └─ talker.forward
         └─ code_predictor.generate (nested GenerationMixin!)
             ├─ 1 prefill forward (2-token seed)
             └─ 14 single-token decode forwards
     └─ embed & sum 16 codebook vectors → next talker input
```

Every forward crosses Python, Transformers scheduling, logits processors, mask
construction, and cache bookkeeping. Modular profiling put the baseline decode
time at roughly **71.9% residual predictor, 26.6% talker, 1.5% scheduling**.
The predictor — five small layers run fifteen times per frame through the full
generation stack — was unambiguously the first target.

---

## 3. The contract: fixed budget, exact parity

Before optimizing anything, the project fixed two rules that every later result
had to obey.

**Fixed-budget benchmarking.** All versions generate the complete `alicia.txt`
input with speaker Ryan in English, bfloat16, SDPA, forced to exactly 1,280
selected talker tokens = 1,279 complete codec frames = 102.320 s of audio.
EOS is ignored so every version does identical work; natural-EOS runs are a
separate protocol used only for correctness and listening tests. One warmup
(more, when warmup proved insufficient) is excluded, three runs are measured,
p50 is reported, and phase breakdowns via asynchronous CUDA events with a
single terminal synchronization are kept alongside every headline number.

**Exact parity.** "Sounds fine" is not evidence. A candidate replaces the
reference only if it matches the official greedy runtime on:

1. codec tensor shape,
2. the location of the first differing codec ID,
3. every codec ID exactly,
4. waveform shape,
5. every waveform sample exactly,

with the strongest form being an independent full-sequence validation where
the reference is generated *before* the candidate's wrappers are installed.
This gate is what killed V5 and V6, and it is the reason the final claim can be
made without hedging.

---

## 4. The optimization ladder

| Version | Change | p50 (s) | RTF | × real-time | Verdict |
|---|---|---:|---:|---:|---|
| Official | nested HF generation | 91.722 | 0.896 | 1.116× | reference |
| V1 | explicit prefill/decode scheduler | 71.382 | 0.698 | 1.433× | valid |
| V2 | on-device buffers, sync removal | 67.974 | 0.664 | 1.505× | valid |
| V3 | static cache + compiled predictor | 57.477 | 0.562 | 1.780× | valid |
| A/B | V3 but dynamic talker cache | 50.011 | 0.489 | 2.046× | valid, diagnostic |
| V4 | manual CUDA graphs, naive boundaries | 64.431 | 0.630 | 1.588× | regression |
| V5 | compiled loop + static talker graph | 48.106 | 0.470 | 2.127× | **invalid: silence** |
| V5.1 | V5 + explicit causal mask | 64.188 | 0.627 | 1.594× | valid, slow |
| V6 | V5.1 + compiled talker | 56.577 | 0.553 | 1.808× | **invalid: parity fails** |
| **V7** | per-position + FFN eager graphs | **56.858** | **0.556** | **1.800×** | **valid, promoted** |

```text
official  91.7  ██████████████████████████████████████████████
V1        71.4  ███████████████████████████████████████
V2        68.0  ██████████████████████████████████████
V3        57.5  ████████████████████████████████
A/B       50.0  ██████████████████████████████
V4        64.4  ████████████████████████████████████
V5        48.1  ██████████████████████████████  ! silent audio
V5.1      64.2  ████████████████████████████████████
V6        56.6  ███████████████████████████████  ! parity fail @ f1/cb13
V7*       56.9  ███████████████████████████████  ← promoted, bit-exact
```

### 4.1 V1 — delete the scheduler, keep the math (91.7 → 71.4 s)

V1 replaces nested `GenerationMixin` scheduling with a hand-written loop that
calls the official modules directly: explicit prefill over the full prompt,
one talker token forward per frame, the 15 predictor steps unrolled. Same
forwards, same operation order — only the per-token Python tax (mask building,
cache-position bookkeeping, argument plumbing, logits-processor dispatch) goes
away. At 128 frames the direct scheduler ran 7.007 s vs the official outer path
at 7.838 s: **−10.6% with exact parity**, from deleting nothing but
bookkeeping.

### 4.2 V2 — stop ping-ponging with the host (71.4 → 68.0 s)

V2 keeps every intermediate on-device and preallocated: output tensors written
with `copy_`, cache positions and position IDs precomputed, predictor embedding
weights stacked into one tensor once, no per-frame `.item()` EOS sync in the
fixed-budget path, codec chunks decoded straight into one GPU waveform, CUDA
events instead of intermediate synchronizations. Every host↔device round trip
or allocation inside the inner loop costs more than the kernel it schedules;
V2 removes most of them. Modest percentage, real lesson: *the hot loop must
never talk to the CPU.*

### 4.3 V3 — compile the predictor (68.0 → 57.5 s)

V3 gives the predictor a preallocated `StaticCache` and compiles the pass with
`torch.compile(mode="reduce-overhead")`, letting Inductor fuse the small GEMVs
and own a cudagraph tree. Combined predictor time fell **49.404 s → 29.413 s
(−40.46%)**; individual residual passes dropped to ~1.53 ms. This worked
because the predictor's shapes are regular and its cache addresses became
fixed — exactly the conditions compilation needs.

But the same commit also gave the **talker** a static cache, and talker time
rose 19.901 s → 26.106 s (+31%). That regression hid behind the bigger
predictor win until a controlled A/B changed *only* the talker cache back to
dynamic: talker time fell to 18.685 s, predictor time moved 0.06% (noise),
and end-to-end p50 dropped to 50.011 s.

The explanation is the most transferable finding in this article. Transformers'
`StaticCache` exposes its **full maximum allocation** — 1,432 zero-padded slots
— and reports that length to masking code. Eager SDPA therefore materializes an
explicit mask and attends across every slot, every step. `DynamicCache`
exposes only populated entries, which lets SDPA take its cheap mask-free
single-token causal path. Static storage is only faster when something
exploits the fixed shapes (graphs, compilation); for eager attention it is a
pure loss. This A/B redirected the whole project toward keeping talker
attention dynamic and eager.

### 4.4 V4 — CUDA graphs done naively (64.4 s, regression)

V4 captured the *complete eager predictor loop* as one manual CUDA graph, plus
a second graph for the talker pass. The replay was beautifully deterministic
(64.430–64.432 s across runs) and 12% slower than V3. Why: the graph preserved
whatever kernels it captured — here, unfused eager ones (36.240 s vs 29.413 s
compiled). A CUDA graph removes launch overhead; it does not create fusion.
Wrapping an already-fast region buys nothing, and wrapping the wrong variant
freezes the wrong variant in place. Lesson: capture the fast kernel sequence,
not merely the current one.

### 4.5 V5 — the fastest number in the repo is wrong (48.1 s, invalid)

V5 compiled the full predictor loop and captured a static-cache talker graph.
It measured 48.106 s — 2.13× real-time, the best number ever recorded in the
project. It also produced **silence after ~2 seconds**.

The root cause chain is worth reading slowly:

1. the talker graph uses a `StaticCache`;
2. to dodge compile errors, each cache layer is marked non-compileable;
3. to recover SDPA's fast path, the code passes `attention_mask=None`;
4. `attention_mask=None` re-enables SDPA's mask-free causal skip;
5. but the cache is zero-padded to 1,432 slots;
6. SDPA attends over all slots with `is_causal=False` — softmax over mostly
   zero keys;
7. attention outputs attenuate toward zero, greedy codes degenerate, the audio
   goes quiet.

`attention_mask=None` is *correct* for a dynamic cache (no padding exists) and
*catastrophic* for a static one. The shortcut that made it fast is precisely
what made it broken — a perfect trap, because the fast path and the correct
path differ by a single `None`.

### 4.6 V5.1 — correctness restored, speed gone (64.2 s)

Restoring an explicit per-position causal mask fixed the output and exploded
the talker graph time from 15.452 s to 28.459 s (+84%). Explicit-mask,
full-capacity SDPA is simply more work than the dynamic populated-prefix path.
Conclusion, firmly: at batch one on this GPU, a graphed static-cache talker is
a dead end. Keep talker attention dynamic and eager; find the speedup
elsewhere. (V5.1 also needed three excluded warmups — one left a late setup
pass inside the first measured run.)

### 4.7 V6 — nearly as fast as V7, still invalid (56.6 s)

Compiling the explicit-mask static talker recovered most of the loss: 56.577 s.
Exact parity failed at **frame 1, codebook 13**, after 1,181 of 20,448 shared
IDs matched — frame 0 and thirteen codebooks of frame 1 agreed, localizing the
first flip to the very first compiled talker hidden state. bf16 arithmetic is
reduction-order sensitive; compilation reordered reductions; one logit crossed
an `argmax` boundary; autoregression amplified the flip forever after. Note
what this means: **mathematically equivalent code is not numerically identical
code, and for greedy AR decoding only identical counts.**

### 4.8 V7 — exact eager graphs at the right boundaries (56.9 s, promoted)

V7 is the design that survived every failure, built from four constraints
derived directly from V4–V6:

1. **No numerical change anywhere.** Every captured region is the original
   eager kernel sequence — no fusion, no compiled variants, no full-capacity
   attention, official bf16 embedding-sum order, float32 logits processing.
2. **Graph only shape-invariant work.** Variable-length prompt prefill stays
   eager. Talker attention, whose KV grows every frame, stays eager on a
   dynamic cache. What remains fixed-shaped: the predictor's 15 residual
   positions (each attends over a known prefix length) and the talker layers'
   post-attention FFN block (`[1,1,1024]`).
3. **Stable storage without changed semantics.** A custom prefix-visible cache
   holds the predictor KV in a fixed buffer written with `index_copy_` but
   returns views of *only the populated prefix*, reproducing DynamicCache's
   attention shapes exactly — fixed addresses for graph capture, dynamic
   numerics for parity.
4. **One capture, one memory pool, cached per model.** Shared CUDA graph pool,
   three eager warmups per boundary on a side stream, all outside measured
   iterations.

Concretely, V7 installs **15 per-position eager CUDA graphs** for the predictor
(one per residual codebook position — capturing the whole loop at padded
capacity would have reproduced the static-cache numerics that broke parity)
and **20 FFN graphs** wrapping each talker layer's `inputs +
mlp(post_attention_layernorm(inputs))` region, leaving each layer's attention
eager. Per frame, the entire decode collapses to ~2 Python-side graph
boundaries replacing ~16 nested forward calls.

Result: **56.858 s p50, RTF 0.556, 1.800× real-time** — 38.01% below official
and within 0.5% of invalid V6, but unlike V5/V6 it passed independent
full-sequence validation: **20,480/20,480 codec IDs and 2,457,600/2,457,600
waveform samples matched exactly** against a reference generated before the
wrappers were installed. Decode is 96.7% of wall (codec 3.2%, prefill 0.09%);
the short-run modular split attributes decode to ~66% predictor / ~33% talker /
~1.3% scheduling.

---

## 5. Correctness engineering — the part that made any of this trustworthy

### 5.1 How a one-bit difference becomes a different sentence

bf16 has ~8 bits of mantissa. Reordering a reduction — a fused projection, a
masked attention over padded slots, a reordered embedding sum — changes the
last ulp of some logit. If that logit sits near an `argmax` boundary, the
greedy token flips. The flipped residual token becomes next step's *input
embedding*, so the predictor's hidden state shifts, the talker's next-frame
input shifts, and the divergence compounds across 1,279 frames. Observed first
mismatches under various optimizations: frame 3/codebook 15, frame 5/codebook
15, frame 1/codebook 13 — and in one case 1,181 matching IDs before the first
difference. Plausible audio throughout. Listening tests detect none of this.

Hence the escalation ladder: shapes → first-mismatch location → exact IDs →
exact samples → independent full-sequence validation with the reference
generated before installation. And hence a list of bugs the gate caught that
"it sounds right" never would have:

- primary logits truncated to the 2,048-codec vocabulary, making talker EOS
  (token 2150) unreachable;
- repetition penalty and suppression applied to bf16 logits where the official
  path uses float32;
- a parity harness comparing misaligned budgets (1,279 explicit frames vs
  1,278 official complete frames — the official generator counts *selected
  tokens*, the explicit loop counts *completed frames*);
- the codec decoding the full preallocated buffer instead of stopping at EOS,
  manufacturing false trailing audio.

### 5.2 Silence has a second cause: policy collapse

Separately from V5's masking bug, greedy decoding with repetition penalties of
1.05 or 1.1 drove generation into a low-energy repetitive tail after ~16 s —
audio that *looked* truncated but contained samples. Penalty 1.2 keeps energy
healthy and reaches natural EOS (1,216 frames / 97.28 s, itself bit-exact:
19,456 IDs, 2,334,720 samples). Two failure modes, superficially identical
quiet audio, completely different causes — one numerical, one behavioral.

### 5.3 Measurement discipline earned the hard way

Phase timing via asynchronous CUDA events with one terminal sync — host timers
measure queueing, not execution. Profiler traces live in separate diagnostic
runs; they distort the loop they observe. Warmups absorb compilation and
capture, but verify how many (V5.1 needed three). Report p50 *with phase
breakdowns* — V3's headline concealed a +31% talker regression behind its −40%
predictor win, and only the A/B exposed it. Keep input, speaker, dtype,
attention backend, and budget frozen across versions; save timing JSON before
running optional parity checks so a failed check doesn't erase performance
evidence.

---

## 6. Context: what `faster-qwen3-tts` does differently

An upstream study compared whistle's approach with `faster-qwen3-tts`, which
also wraps the official weights. Its engine follows the same instincts — eager
prefill, copy KV state into static buffers, one talker-token graph, one
whole-predictor graph, official codec with chunked decoding — and reports much
larger wins on an RTX 4090 (RTF 0.82 → 4.78, TTFA 800 ms → 156 ms),
repository-reported and not locally reproduced. Notable differences: it accepts
fixed sampling settings inside the captured predictor (whistle requires greedy
exactness), it is CUDA-only and batch-one-only, and its outer loop retains
Python work, EOS synchronization, and codec-chunk syncs. The projects converge
on the same thesis from opposite directions: the model is fine; the scheduling
is the product.

---

## 7. What's left on the table

Decode is 96.7% of wall time, and the predictor is ~66% of decode. In priority
order: fuse the predictor into a persistent device-side loop that preserves the
official GEMV accumulation order; grow talker graph regions around
shape-invariant projections without touching variable-length attention; bucket
the codec decode (ceiling ≈ 3.2%); move stopping/scheduling fully on-device
(ceiling ≈ 1.3%); then, as separately labeled quality experiments, weight-only
int8 GEMV and speculative decoding — never mixed into the exact-output
benchmark. A companion Triton lab (fused RMSNorm+residual, fused SwiGLU, eager
parity references) exists for exactly this pipeline.

The standing rule for every future version: no latency claim enters the
headline table until it passes the independent full codec-token and waveform
validation.

---

## 8. Takeaways

1. **Profile before believing folklore.** Attention wasn't the bottleneck;
   launch overhead was. The optimization that won was scheduling, not kernels.
2. **Graphs preserve; they don't improve.** A CUDA graph replays whatever
   kernels you captured. Capture the fast variant, at the largest
   shape-invariant boundary that doesn't change numerics.
3. **Static caches aren't automatically faster.** They pay off only when
   fixed shapes are exploited; for eager attention they're a measured loss,
   and mishandled masks make them silent.
4. **Equivalence isn't identity.** In bf16, at greedy, under autoregression,
   "mathematically equivalent" loses to "bit-identical." Keep an official
   reference path and compare everything.
5. **Fastest ≠ best.** The two fastest measurements in the project's history
   were deleted. The parity gate is the reason the surviving number means
   anything.
