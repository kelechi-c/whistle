# Whistle — The Full Technical Article

*How to make Qwen3-TTS run 1.8× real-time on a 6 GB laptop GPU without changing a single output token — and a block-by-block walkthrough of the runtime code, written so you could rewrite it from scratch.*

Whistle is an execution engine wrapped around the official `Qwen3-TTS-12Hz-0.6B-CustomVoice` checkpoint. It does not change the weights, the tokenizer, the codec, or the math. It changes only *how the forward passes are scheduled on the GPU*. On an RTX 3050 Laptop (6 GB), batch-one synthesis of a 1,216-frame / 97.3 s script drops from **82.91 s (official runtime) to 54.08 s** — RTF 0.852 → 0.556, **1.80× real-time** — while every codec token and every waveform sample stays *bit-exact* against the official output.

This article has two halves. Sections 1–6 are the story: the model, the bottleneck, the optimization ladder V1→V7 with its two invalidated failures, and the correctness engineering that makes the final number mean anything. Section 7 is the entire runtime code, block by block, with real snippets from `src/whistle` and explanations thorough enough to reimplement the whole thing. Section 8 compresses that into a checklist of invariants. Sections 9–11 are the 2026-08-29 postscript: where the surrounding ecosystem landed, the parity-safe Track A wins now shipped in the main runtime (streaming TTFA at ~100 ms, graph-captured sampling), and a numerics probe that answers the one question the parity wall left open — whether a custom kernel could ever be safe.

> **Numbers at a glance** (RTX 3050 Laptop 6 GB, PyTorch 2.13 / CUDA 13, bfloat16, SDPA, greedy, speaker Ryan, natural EOS, 1,216 frames = 97.28 s audio): official `qwen-tts` 82.91 s wall / RTF 0.852; whistle V7 54.08 s / RTF 0.556; codec + waveform parity **exact** (19,456 codec IDs, 2,334,720 samples); Qwen3-ASR WER 3.02%. Historical fixed-budget protocol (1,280 selected tokens = 1,279 frames): official 91.722 s → V7 56.858 s.

------------------------------------------------------------------------

## 1. The thesis

Two claims sit underneath every number in this project:

1.  **The official runtime is launch-bound, not compute-bound.** Each 80 ms frame of audio triggers roughly 95 transformer-layer forwards plus all of Hugging Face's generation machinery — about 20,000+ tiny Python-dispatched CUDA launches per utterance. The kernels themselves are trivial; the cost is getting them onto the GPU. So the win comes from scheduling and graph capture, not from faster kernels.

2.  **A fast number is not a correct number.** Autoregressive decoding turns one flipped bf16 `argmax` into a permanently diverged sequence, and plausible audio proves nothing. Two configurations that measured *faster* than the final result were invalidated by an exact codec-token + waveform parity gate: one produced silence, one diverged at frame 1. What survives is the fastest version that is *exactly* right.

------------------------------------------------------------------------

## 2. What one frame of speech actually costs

### 2.1 Two nested autoregressive transformers

Qwen3-TTS generates speech as discrete codec tokens at 12.5 Hz — one frame per 80 ms of audio. Each frame is a row of **16 codebook tokens**: one *primary* token predicted by a large transformer (the **talker**) and fifteen *residual* tokens predicted by a small one (the **code predictor**), Multi-Token-Prediction style. A neural codec then decodes the `[frames × 16]` ID matrix into 24 kHz waveform.

```text
talker (~0.6B): 20 layers · hidden 1024 · GQA 16Q/2KV · head_dim 64
                3D mRoPE · Q/K-norm · SwiGLU
    └─ per frame: ONE primary codec token + a carried hidden state
codec_head:     Linear(1024 → 3072) → primary codebook logits
                (0–2047 ordinary tokens, 2048–3071 specials, EOS = 2150)

code predictor: 5 layers · hidden 1024 · GQA 16Q/8KV · head_dim 128
                31 codebook embeddings + 31 lm_heads
    └─ per frame: FIFTEEN autoregressive steps
       step i consumes the embedding of residual codebook i−1,
       each step reads out through its own lm_head[i]

speech_tokenizer: official codec decoder, codec IDs → waveform
```

The predictor is genuinely sequential: step *i* needs token *i−1*'s embedding. And both models are fed by history — the talker's KV cache grows by one position per frame, and within a frame the predictor attends over a growing prefix of up to 16 positions. Everything depends on everything before it.

### 2.2 Frame arithmetic, or why this workload hates the GPU

| Quantity | Value |
|----|---:|
| Codec frame | 12.5 Hz → 80 ms audio |
| Codebooks per frame | 16 = 1 primary + 15 residual |
| Transformer layer forwards per frame | 20 (talker) + 5 × 15 (predictor) = **95** |
| Layer forwards per 1,279-frame run | ≈ **121,500** |

At batch one, every one of those forwards is a pile of GEMV-shaped work — matrix-vector multiplies with tiny arithmetic intensity. Kernel profiling of the baseline confirmed where CUDA time actually goes:

- two GEMV kernel families ≈ **55.6%** of measured CUDA time;
- ordinary matmul ≈ 9.7%;
- **SDPA attention — the thing people usually optimize — only a few percent.**

That single measurement set the strategy for the whole project: do not write attention kernels, do not quantize (yet), do not touch the math. Attack the ~20,000 launches and the Python between them.

### 2.3 Where the official runtime spends the time

The official `qwen-tts` implementation nests three layers of generality inside each other:

```text
GenerationMixin.generate (talker)
 └─ for each of ~1,279 outer tokens:
     └─ talker.forward
         └─ code_predictor.generate (nested GenerationMixin!)
             ├─ 1 prefill forward (2-token seed)
             └─ 14 single-token decode forwards
     └─ embed & sum 16 codebook vectors → next talker input
```

Every forward crosses Python, Transformers scheduling, logits processors, mask construction, and cache bookkeeping. Modular profiling put the baseline decode time at roughly **71.9% residual predictor, 26.6% talker, 1.5% scheduling**. The predictor — five small layers run fifteen times per frame through the full generation stack — was unambiguously the first target.

------------------------------------------------------------------------

## 3. The contract: fixed budget, exact parity

Before optimizing anything, the project fixed two rules that every later result had to obey.

**Fixed-budget benchmarking.** All versions generate the complete `alicia.txt` input with speaker Ryan in English, bfloat16, SDPA. The historical ladder used a forced budget of exactly 1,280 selected talker tokens = 1,279 complete codec frames = 102.320 s of audio, with EOS suppressed so every version does identical work. The current canonical comparison runs *both sides at natural EOS* (the official API always stops at codec EOS, so fixed-token runs are not parity-comparable against it). Warmups are excluded, p50 of measured runs is reported, and phase breakdowns via asynchronous CUDA events with a single terminal synchronization are kept alongside every headline number.

**Exact parity.** "Sounds fine" is not evidence. A candidate replaces the reference only if it matches the official greedy runtime on:

1.  codec tensor shape,
2.  the location of the first differing codec ID,
3.  every codec ID exactly,
4.  waveform shape,
5.  every waveform sample exactly,

with the strongest form being an independent full-sequence validation where the reference is generated *after* the measurements, untimed. This gate killed the two fastest configurations in project history, and it is the reason the final claim can be made without hedging.

------------------------------------------------------------------------

## 4. The optimization ladder

| Version | Change | p50 (s) | RTF | × real-time | Verdict |
|----|----|---:|---:|---:|----|
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

V1 replaces nested `GenerationMixin` scheduling with a hand-written loop that calls the official modules directly: explicit prefill over the full prompt, one talker token forward per frame, the 15 predictor steps unrolled. Same forwards, same operation order — only the per-token Python tax (mask building, cache-position bookkeeping, argument plumbing, logits-processor dispatch) goes away. At 128 frames the direct scheduler ran 7.007 s vs the official outer path at 7.838 s: **−10.6% with exact parity**, from deleting nothing but bookkeeping.

### 4.2 V2 — stop ping-ponging with the host (71.4 → 68.0 s)

V2 keeps every intermediate on-device and preallocated: output tensors written with `copy_`, cache positions and position IDs precomputed, predictor embedding weights stacked once, no per-frame `.item()` EOS sync in the fixed-budget path, codec chunks decoded straight into one GPU waveform, CUDA events instead of intermediate synchronizations. Every host↔device round trip or allocation inside the inner loop costs more than the kernel it schedules. Modest percentage, real lesson: *the hot loop must never talk to the CPU.*

### 4.3 V3 — compile the predictor, and learn what static costs (68.0 → 57.5 s)

V3 gives the predictor a preallocated `StaticCache` and compiles the pass with `torch.compile(mode="reduce-overhead")`, letting Inductor fuse the small GEMVs and own a cudagraph tree. Combined predictor time fell **49.404 s → 29.413 s (−40.46%)**. This worked because the predictor's shapes are regular and its cache addresses became fixed — exactly the conditions compilation needs.

But the same commit also gave the **talker** a static cache, and talker time rose 19.901 s → 26.106 s (+31%). That regression hid behind the bigger predictor win until a controlled A/B changed *only* the talker cache back to dynamic: talker time fell to 18.685 s, predictor time moved 0.06% (noise), and end-to-end p50 dropped to 50.011 s.

The explanation is the most transferable finding in this article. Transformers' `StaticCache` exposes its **full maximum allocation** — 1,432 zero-padded slots — and reports that length to masking code. Eager SDPA therefore materializes an explicit mask and attends across every slot, every step. `DynamicCache` exposes only populated entries, which lets SDPA take its cheap mask-free single-token causal path. Static storage is only faster when something exploits the fixed shapes (graphs, compilation); for eager attention it is a pure loss. This A/B redirected the whole project toward keeping talker attention dynamic and eager.

### 4.4 V4 — CUDA graphs done naively (64.4 s, regression)

V4 captured the *complete eager predictor loop* as one manual CUDA graph, plus a second graph for the talker pass. The replay was beautifully deterministic (64.430–64.432 s across runs) and 12% slower than V3. Why: the graph preserved whatever kernels it captured — here, unfused eager ones (36.240 s vs 29.413 s compiled). A CUDA graph removes launch overhead; it does not create fusion. Wrapping an already-fast region buys nothing, and wrapping the wrong variant freezes the wrong variant in place. Lesson: **capture the fast kernel sequence, not merely the current one.**

### 4.5 V5 — the fastest number in the repo is wrong (48.1 s, invalid)

V5 compiled the full predictor loop and captured a static-cache talker graph. It measured 48.106 s — 2.13× real-time, the best number ever recorded in the project. It also produced **silence after ~2 seconds**.

The root cause chain is worth reading slowly:

1.  the talker graph uses a `StaticCache`;
2.  to dodge compile errors, each cache layer is marked non-compileable;
3.  to recover SDPA's fast path, the code passes `attention_mask=None`;
4.  `attention_mask=None` re-enables SDPA's mask-free causal skip;
5.  but the cache is zero-padded to 1,432 slots;
6.  SDPA attends over all slots with `is_causal=False` — softmax over mostly zero keys;
7.  attention outputs attenuate toward zero, greedy codes degenerate, the audio goes quiet.

`attention_mask=None` is *correct* for a dynamic cache (no padding exists) and *catastrophic* for a static one. The shortcut that made it fast is precisely what made it broken — a perfect trap, because the fast path and the correct path differ by a single `None`.

**The greedy vs. sampling empirical proof:** A direct empirical probe of this V5 configuration (`sandbox/probe_v5.py`) isolates how decoding policy interacts with this numerical bug. Under greedy decoding (`argmax`), the RMS energy plummets by 98% between seconds 2 and 4 (`0.0503 → 0.0013 → 0.0011`), collapsing into an infinite repetitive loop of silence/pad tokens (Token 117 emitted 43×, Token 1368 emitted 41×). But under *sampled decoding* (`temperature=0.9, top_k=50`), top-50 multinomial sampling provides sufficient entropy to escape the deterministic silence attractor: audio energy remains healthy (RMS `0.077 → 0.072 → 0.071`), and Qwen3-ASR scores **0.00% CER (100% phonetic accuracy)**. The silence failure is the fatal collision of attention attenuation with greedy argmax selection.

### 4.6 V6 — nearly as fast as V7, still invalid (56.6 s)

Compiling the explicit-mask static talker recovered most of V5.1's loss: 56.577 s. Exact parity failed at **frame 1, codebook 13**, after 1,181 of 20,448 shared IDs matched — frame 0 and thirteen codebooks of frame 1 agreed, localizing the first flip to the very first compiled talker hidden state. bf16 arithmetic is reduction-order sensitive; compilation reordered reductions; one logit crossed an `argmax` boundary; autoregression amplified the flip forever after. Note what this means: **mathematically equivalent code is not numerically identical code, and for greedy AR decoding only identical counts.**

### 4.7 V7 — exact eager graphs at the right boundaries (56.9 s, promoted)

The primary optimization outside the baseline memory plumbing is **CUDA graph replay**. But the secret to making CUDA graphs work without breaking correctness is **surgical boundary placement** around invariant operation shapes.

Why does this matter so much? At batch size 1, Qwen3-TTS 0.6B is heavily **kernel launch-bound, not compute-bound**. Each 80 ms audio frame dispatches 20 talker layers plus 15 residual predictor steps — over 20,000 Python-to-CUDA driver launches per 100-second utterance. CUDA graphs eliminate that CPU dispatch stall by recording the kernel sequence once and replaying it directly on the GPU command processor with near-zero dispatch overhead.

V7 is the design that survived every failure, built from four constraints derived directly from V4–V6:

1.  **No numerical change anywhere.** Every captured region is the original eager kernel sequence — no fusion, no compiled variants, no full-capacity attention, official bf16 embedding-sum order, float32 logits processing.
2.  **Graph only shape-invariant work.** Variable-length prompt prefill stays eager. Talker self-attention, whose KV grows every frame, stays eager on a dynamic cache. What remains strictly fixed-shaped: the predictor's 15 residual positions (each attends over a known prefix length) and the talker layers' post-attention SwiGLU FFN block (`[1, 1, 1024]`).
3.  **Stable storage without changed semantics (`PrefixStaticLayer`).** Standard `StaticCache` zero-pads up to capacity, poisoning SDPA attention with padding (which caused the V5 silence bug) or requiring an explicit mask rebuilt every step. Standard `DynamicCache` reallocates memory pointers at every step, breaking CUDA graph capture. `PrefixStaticLayer` reconciles both: it holds the predictor KV in a fixed backing buffer written with `index_copy_` (stable pointers for graphs), but returns views of *only the active prefix* (`[:cumulative_length]`), reproducing `DynamicCache` attention shapes and unmasked SDPA numerics bit-for-bit.
4.  **One capture, one memory pool, cached per model.** Shared CUDA graph pool (`torch.cuda.graph_pool_handle()`), three eager warmups per boundary on a side stream, all cached once per model instance.

Concretely, V7 installs **15 per-position eager CUDA graphs** for the predictor (one per residual codebook position — capturing the whole loop at padded capacity would have reproduced the static-cache numerics that broke parity) and **20 FFN graphs** wrapping each talker layer's `inputs + mlp(post_attention_layernorm(inputs))` region (where the SwiGLU MLP accounts for ~65% of layer FLOPs), leaving each layer's self-attention eager. Per frame, the entire decode collapses to ~2 Python-side graph boundaries replacing ~16 nested forward calls.

**Results:** At the historical fixed-budget benchmark (1,279 complete frames / 102.32 s audio), generation drops from **91.722 s (official) → 71.382 s (V1) → 67.974 s (V2) → 56.858 s p50 (V7)** — an **RTF of 0.556 (1.800× real-time)**, a **38.01% latency cut** while passing independent full-sequence validation: **20,480/20,480 codec IDs and 2,457,600/2,457,600 waveform samples matched exactly**.

On the natural-EOS protocol (Alicia natural stop at frame 1,216 / 97.28 s audio, verified live on Victoria's RTX 3050), generation drops from **83.834 s (official baseline) to 54.298 s p50 (Whistle V7)** — an RTF of **0.558 (1.792× real-time)**, cutting **29.54 seconds of wall time per utterance** with 100% bitwise codec and audio sample parity.

### 4.8 Scaling to 1.7B — crossing the real-time threshold

Does the launch-bound scheduling thesis hold when the model scales nearly 3×, from 0.6B to 1.7B (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`)?

We executed a head-to-head benchmark on Victoria (40W RTX 3050 6GB Laptop GPU) across the canonical 1,280 target tokens (~102.4 s audio) using the Alicia input (Ryan voice, English, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` across 1 warmup + 3 timed benchmark iterations). Peak VRAM allocated was **4,964 MiB** — fitting comfortably within the 6 GB physical hardware ceiling.

<div class="table-wrap">

| Metric | Official Runtime (1.7B) | Whistle V7 (1.7B) | Delta / Improvement |
|----|---:|---:|---:|
| Iteration 1 | 107.758 s (RTF 1.053, 0.950×) | 77.440 s (RTF 0.756, 1.322×) | −30.318 s |
| Iteration 2 | 108.645 s (RTF 1.062, 0.942×) | 77.448 s (RTF 0.756, 1.322×) | −31.197 s |
| Iteration 3 | 108.415 s (RTF 1.060, 0.944×) | 77.471 s (RTF 0.757, 1.322×) | −30.944 s |
| **p50 Wall Latency** | **108.415 s** | **77.448 s** | **−30.967 s (−28.56% latency cut)** |
| **p50 Real-Time Factor (RTF)** | **1.060** *(slower than RT)* | **0.756** *(faster than RT)* | **−28.68% lower RTF** |
| **p50 Throughput (xRT)** | 0.944× realtime | 1.322× realtime | **+40.04% throughput increase** |

</div>

**Phase breakdown for Whistle V7 (1.7B):** Inside Whistle V7, the 77.448 s p50 generation breaks down as: preparation 2.32 ms, prefill forward 119.47 ms, autoregressive decode loop **75.495 s** (97.5% of wall time), and speech tokenizer codec decode 1.832 s (2.4% of wall time). Artifacts: `benchmarks/v7_1.7b_3runs.json` and `benchmarks/official_1.7b_3runs.json`.

**The critical milestone:** On an entry-level 40W laptop GPU, the official HuggingFace runtime **cannot achieve real-time streaming** on the 1.7B model (RTF 1.060 = 0.944× realtime). In interactive playback, this means audio buffer underruns, stuttering, and an inability to maintain interactive conversational latency. Whistle V7 **crosses the real-time threshold**, generating 102.4 seconds of 1.7B audio in 77.45 seconds (0.756 RTF = 1.322× realtime) with zero quality degradation, cutting over **30 seconds of wall-clock latency**.

------------------------------------------------------------------------

## 5. Correctness engineering

### 5.1 How a one-bit difference becomes a different sentence

bf16 has ~8 bits of mantissa. Reordering a reduction — a fused projection, a masked attention over padded slots, a reordered embedding sum — changes the last ulp of some logit. If that logit sits near an `argmax` boundary, the greedy token flips. The flipped residual token becomes next step's *input embedding*, so the predictor's hidden state shifts, the talker's next-frame input shifts, and the divergence compounds across 1,279 frames. Observed first mismatches under various optimizations: frame 3/codebook 15, frame 5/codebook 15, frame 1/codebook 13 — and in one case 1,181 matching IDs before the first difference. Plausible audio throughout. Listening tests detect none of this.

Every perturbation tried — compiled talker, static-cache attention, QKV fusion, int8 quantization — diverged *first at a residual codebook argmax*, never at the primary token. The mechanism: book k's token is the argmax of its own head and is embedded into book k+1's input (a sequential chain cascading through all 15 books within one frame); all 16 codes sum into the next frame's talker input (one flip perturbs the whole trajectory); residual codebooks encode the hard-to-predict remainder with thin logit margins, so any numeric noise crosses the argmax boundary easily where primary tokens sit on a stable plateau; and greedy selection has zero hedging — a flipped argmax is permanent.

Hence the escalation ladder: shapes → first-mismatch location → exact IDs → exact samples → independent full-sequence validation. And hence a list of bugs the gate caught that "it sounds right" never would have:

- primary logits truncated to the 2,048-codec vocabulary, making talker EOS (token 2150) unreachable;
- repetition penalty and suppression applied to bf16 logits where the official path uses float32;
- a parity harness comparing misaligned budgets (1,279 explicit frames vs 1,278 official complete frames — the official generator counts *selected tokens*, the explicit loop counts *completed frames*);
- the codec decoding the full preallocated buffer instead of stopping at EOS, manufacturing false trailing audio;
- a Triton fused embedding-sum that used the talker's codec table for the residuals — but the residuals use 15 *separate* codebook tables (9,200% WER garbage until fixed).

### 5.2 Silence has two causes

Silence is the signature failure of this model, and it has two independent causes that are easy to conflate. The first is *numerical*: V5's mask bug attenuated attention outputs until codes degenerated. The second is *behavioral*: greedy decoding with repetition penalties of 1.05 or 1.1 drives generation into a low-energy repetitive tail after ~16 s — audio that *looks* truncated but contains samples. Penalty 1.2 keeps energy healthy and reaches natural EOS (1,216 frames / 97.28 s, itself bit-exact). Two failure modes, superficially identical quiet audio, completely different causes — one numerical, one behavioral. Any trajectory that diverges (from a fused kernel, a quantized weight, a frame-graph mask) reliably slides into the same collapse.

### 5.3 Measurement discipline earned the hard way

- Phase timing via asynchronous CUDA events with one terminal sync — host timers measure queueing, not execution.
- Profiler traces live in separate diagnostic runs; they distort the loop they observe.
- Warmups absorb compilation and capture, but verify how many (one version needed three because a late setup pass leaked into the first measured run).
- Report p50 *with phase breakdowns* — V3's headline concealed a +31% talker regression behind its −40% predictor win, and only the A/B exposed it.
- Keep input, speaker, dtype, attention backend, and budget frozen across versions; save timing JSON before running optional parity checks so a failed check doesn't erase performance evidence.
- Fresh processes and interleaved A/B: this 40 W mobile GPU throttles after consecutive runs, so unfair back-to-back comparisons are easy to manufacture by accident.

------------------------------------------------------------------------

## 6. After V7: the parity wall

The post-V7 sessions attacked everything left on the table. Nearly everything met the same wall: bit-identity under greedy autoregression.

### 6.1 The one change that survived: chunked EOS

The serving path still drained the GPU once per frame checking EOS (`tensor.eq(eos).item()` is a host sync). The promoted replacement scans `codes` on-device every 8 frames (`EOS_CHECK_EVERY = 8`), plus a forced scan whenever the codes become observable (streaming chunk boundaries, final frame). The GPU runs ahead; the trim is identical by construction: the EOS row is written, detected, and sliced away. This lives in `inference._maybe_eos_row` (section 7.4) and is verified bit-exact.

### 6.2 Whole-frame graphs, fusion, and the bitwise verdict

**Whole-frame CUDA graphs: measured slower, not faster.** Both topologies — one combined graph per frame, and the faster-qwen3-tts-style split (predictor graph + talker graph) — were ~8% *slower* than V7 on the full benchmark. The entire penalty is the static 2,048-slot masked attention replacing the dynamic prefix; topology (1 vs 2 graphs) is timing-neutral. Parity failed at frame 1 (the known static-cache divergence class); short audio stayed perfect; the diverged long-form trajectory hit the greedy low-energy collapse — digital silence after ~30 s.

| Mode | wall | RTF | peak | parity |
|----|---:|---:|---:|----|
| V7 default (promoted) | 58.28 s | 0.569 | 3037 MB | exact |
| frame-graph (1 combined graph) | 63.16 s | 0.616 | 3122 MB | fail, frame 1/cb 8 |
| dual-graph (predictor + talker) | 63.07 s | 0.616 | 3122 MB | fail, frame 1/cb 1 |
| faster-qwen3-tts 0.3.2 | 66.52 s | 0.650 | 3084 MB | not bit-exact by design |

**Fused kernels: speed-neutral and wrong.** A Triton RMSNorm→QKV kernel (one launch replacing norm + three GEMVs + two head-norms) measured 224 µs vs 259 µs eager — the path is bandwidth-bound on the weight loads, which fusion does not reduce — and its accumulation-order drift (0–26% of elements bit-exact) compounded across layers and frames into **100% WER** end-to-end. Inside a CUDA graph, launch-fusion is worthless anyway: launches are already free. Conclusion: in this decoder only *bitwise-exact* fusion survives; scheduling and graph replay remain the better investment.

**Same-GPU verdict vs faster-qwen3-tts.** The installed 0.3.2, same protocol (alicia, Ryan, fixed 1,279 frames, fresh process): whistle **56.71 s / RTF 0.554** vs faster **66.52 s / RTF 0.650** — whistle ~14.8% faster while holding exact parity (faster does not). Both engines converge on the same thesis from opposite directions — the model is fine; the scheduling is the product.

**Why faster-qwen3-tts wraps the whole predictor loop at once, and why we cannot:** In `faster-qwen3-tts/predictor_graph.py`, all 15 residual predictor steps are inlined into a single monolithic CUDA graph over a 17-slot `StaticCache` with 14 precomputed causal mask tensors. While intuitive, wrapping the entire predictor loop at once forces SDPA to compute attention across padded slots using \$-\infty\$ masks on every step. In bfloat16, masked softmax over padding produces different reduction rounding than unmasked SDPA over the exact active prefix (\$2, 3, \dots, 16\$). Under greedy decoding, this single-bit ULP rounding difference flips the argmax at **frame 1, codebook 1** (1642 vs 957), permanently breaking exact parity. Whistle's `PrefixStaticLayer` instead exposes only the active prefix slice, which requires capturing 15 separate per-position CUDA graphs (each with its exact unmasked shape). Because launching 15 graph replays takes only ~2 µs each (hidden behind the GPU's 25–35 µs kernel execution), Whistle avoids both launch bubbles and the static padding penalty, running ~15–22% faster than faster-qwen3-tts's monolithic graph while preserving 100% bit-exact parity.

### 6.3 Temperature sampling and voice choice

The checkpoint's `generation_config.json` ships `do_sample: true` — temperature 0.9, top_k 50, repetition penalty 1.05. Greedy was the benchmark contract, not the model's default. Adding sampling to the split path regresses on both axes:

| Mode                                        |    wall |   RTF |   WER |   CER |
|---------------------------------------------|--------:|------:|------:|------:|
| greedy (rp 1.2)                             | 56.72 s | 0.554 | 3.02% | 0.82% |
| sampled t0.9/k50 (rp 1.2)                   | 74.16 s | 0.724 | 6.47% | 3.40% |
| sampled t0.9/k50 (rp 1.05, official recipe) | 73.55 s | 0.718 | 4.74% | 1.75% |

Speed regresses ~30% because the sampled predictor must fall back to its eager 15-step loop — the captured graphs bake the greedy `argmax` — plus the per-frame sampling kernels. WER regresses because sampling occasionally draws non-canonical tokens. One side-finding matters more than the table: the same greedy path scores 38.79% WER on speaker serena but 3.02% on Ryan — the voice embedding dominates this metric, which is why `ryan` is the hard-coded default everywhere. Sampling remains a labeled naturalness experiment, not a latency or quality path. (Section 10.2 later closes the speed half of this regression by capturing a second graph set with the sampling recipe baked in: +38% wall becomes +3.2%.)

### 6.4 Stress, streaming, and quantization

**Stress.** The V7 path was driven on a fresh corpus — seven texts from 7 words to 1,560, including punctuation-heavy and deliberately repetitive inputs. RTF held at 0.54–0.56 on every text, and bit-exactness survived on a new long story (549 frames, 1,054,080 samples) against the official API; on the stories the official runtime was 1.34× and 1.60× slower — the speedup is not a one-text artifact. Caveats: symbol-heavy text (URLs, numbers) is spoken verbatim, so raw-text WER on it is an eval artifact; and a 200-word "la la la" input triggers the known greedy low-energy collapse (412% WER) — model behavior, not a V7 regression.

**Streaming.** `stream_tts` (section 7.5) delivered first audio in ~0.9 s at a fixed 12-frame chunk size, decoding chunks at 1.71–1.78× realtime. Section 10.1 cuts that first-audio latency by 5× with a ramped chunk schedule.

**w8a16 quantization: latency-neutral, quality-catastrophic.** Per-channel int8 weights on the talker's MLP linears only (everything upstream of the residual argmaxes untouched) cut MLP weight storage 49.8% and changed wall latency by *nothing* (56.79 s vs 56.78 s — at batch one the path is memory-bound and dequant-fp16 GEMMs match bf16), but int8 rounding flipped a predictor argmax at frame 0/codebook 1 and dragged greedy into collapse (WER 100–199%). Quantization of anything feeding the residual argmaxes needs bit-exact-grade numerics or sampling; this model accepts neither cheaply.

------------------------------------------------------------------------

## 7. The code, block by block

Everything above is the *why*. This section is the *how*: the whole runtime, file by file, with the real code (lightly trimmed, `…` marking elisions) and enough explanation to reimplement it. Snippets reflect `src/whistle` as of 2026-08-27, after the concurrency/code-review fixes.

### 7.1 How the repo is shaped

```text
whistle/
├── src/whistle/
│   ├── config.py       constants + RuntimeConfig (single source of defaults)
│   ├── inference.py    tts_infer: prompt build, prefill, V7 decode loop, EOS trim, codec
│   ├── graphs.py       PrefixStaticLayer, PredictorGraphs, DecoderFfnGraph,
│   │                   OfficialTalker, DecodeGraphs, decode_graphs (V7 machinery)
│   ├── streaming.py    stream_tts: chunked streaming + incremental codec decode
│   └── server.py       FastAPI: /health + /synthesize (streaming WAV)
├── infer.py            CLI: load model, synthesize TEXT, write WAV
├── profile_tts.py      benchmark harness: split vs official, parity gate, JSON
├── eval_asr_wer.py     optional Qwen3-ASR WER/CER evaluation
├── tests/              unit test with tiny official-shaped weights (no GPU audio)
└── sandbox/            rejected experiments: frame/dual graphs, Triton fusion, int8
```

And the request flow. The two entry points (`tts_infer` batch, `stream_tts` streaming) share `_prepare` and `_prefill` so they cannot drift:

```text
server.py / infer.py / profile_tts.py
  └─ stream_tts() or tts_infer()
       ├─ _prepare()   build_prompt → capacity check → decode_graphs(talker) → reset state
       ├─ _prefill()   official talker forward over the whole prompt → first token,
       │               logits processors, predictor embedding tables, rope deltas
       ├─ decode loop  (per frame):
       │     predictor graphs  → 15 residual tokens        [15 CUDA graph replays]
       │     embedding sum     → next talker input          [eager, official order]
       │     talker step       → attention eager + 20 FFN graph replays
       │     codec_head + fp32 processors → argmax          [eager]
       │     chunked EOS scan  (every 8 frames, device-side)
       ├─ official codec decode → waveform (on device)
       └─ timings via 5 CUDA events, one terminal sync
```

The dependency direction is strict: `config` ← `graphs` ← `inference` ← `streaming` ← `server`. Nothing in `src/whistle` touches the checkpoint's internals beyond the official `qwen_tts.Qwen3TTSModel` API; every weight, table, and decoder is the official module.

### 7.2 config.py — the shared runtime choices

```python
"""Central runtime choices for Whistle inference and profiling."""

from dataclasses import dataclass
import pathlib as pl
from typing import Literal

import torch

CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
SPEAKER = "ryan"
LANGUAGE = "english"
MAX_CACHE_LEN = 2_048

DeviceChoice = Literal["auto", "cpu", "cuda"]
DTypeChoice = Literal["float32", "float16", "bfloat16"]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Holds every mutable runtime choice outside model checkpoint structure."""

    checkpoint: pl.Path = pl.Path(CHECKPOINT)
    output: pl.Path = pl.Path("whistle.wav")
    device: DeviceChoice = "auto"
    dtype: DTypeChoice = "bfloat16"
    seed: int = 0
    max_frames: int = 1_280

    def resolved_device(self) -> torch.device:
        """Selects CUDA only when requested or available under auto mode."""
        use_cuda = self.device == "cuda" or (
            self.device == "auto" and torch.cuda.is_available()
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda was requested but is unavailable")
        return torch.device("cuda" if use_cuda else "cpu")

    def resolved_dtype(self, device: torch.device) -> torch.dtype:
        """Maps the configured dtype and keeps CPU inference broadly supported."""
        dtype = getattr(torch, self.dtype)
        if device.type == "cpu" and dtype == torch.float16:
            raise RuntimeError("float16 cpu inference is unsupported; use float32")
        return dtype


RUNTIME = RuntimeConfig()
```

What to notice:

- **Four constants are the single source of truth** — checkpoint id, speaker, language, and `MAX_CACHE_LEN = 2_048` (the talker KV capacity). Every entry point imports them. Before the review that introduced this file's current role, the speaker default was duplicated in four places and had drifted to `serena` in two of them — which silently changed WER from 3% to 38.8%. Speaker choice dominates quality; it is now a constant, not an option.
- **`RuntimeConfig` is frozen with `slots=True`** — immutable, cheap, and IDE-completable. CLIs that need different values use `dataclasses.replace(RUNTIME, ...)` instead of mutating globals, so state changes flow through returns, not side effects.
- **`MAX_CACHE_LEN` is a hard ceiling, not a heuristic**: `_prepare` refuses to start when `prefill_length + max_new_tokens − 1` would exceed it, and `OfficialTalker.run` re-checks every decode position. Overflow would corrupt cache addressing, so both layers guard.

### 7.3 graphs.py — the decode machinery

This file is the heart of V7. It contains four cooperating pieces: a cache that behaves dynamically but stores statically, the predictor's 15 captured graphs, the talker's 20 captured FFN graphs, and the eager talker step that keeps the numerics official.

#### The problem every piece solves

A CUDA graph is a recorded DAG of kernels replayed with a single launch. Capture freezes three things: *which kernels* run, *their shapes*, and *the addresses of every buffer they touch*. So anything under a graph must (a) be shape-invariant across replays, and (b) read/write only preallocated fixed buffers. The decoder's shape-invariant parts are the predictor's 15 steps and the talker's FFN half; its shape-varying part is talker attention (KV grows every frame) and the whole prefill (variable prompt length). V7 graphs the former and leaves the latter eager.

#### `PrefixStaticLayer` — static storage, dynamic semantics

```python
class PrefixStaticLayer(CacheLayerMixin):
    """Preallocates predictor KV storage while exposing only valid positions."""

    def __init__(self, max_cache_len: int) -> None:
        super().__init__()
        self.max_cache_len = max_cache_len
        self.cumulative_length = 0

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        """Allocates a fixed backing buffer from the first update shape."""
        self.max_batch_size, self.num_heads, _, self.head_dim = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device
        shape = (self.max_batch_size, self.num_heads, self.max_cache_len, self.head_dim)
        self.keys = torch.empty(shape, dtype=self.dtype, device=self.device)
        self.values = torch.empty(shape, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, cache_kwargs=None):
        """Writes new states and returns views with dynamic-cache shapes."""
        if not self.is_initialized:
            self.lazy_initialization(key_states)
        position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None
        if position is None:
            position = torch.arange(
                self.cumulative_length,
                self.cumulative_length + key_states.shape[-2],
                device=key_states.device,
            )
        self.keys.index_copy_(2, position, key_states)
        self.values.index_copy_(2, position, value_states)
        self.cumulative_length += key_states.shape[-2]
        return (self.keys[..., : self.cumulative_length, :],
                self.values[..., : self.cumulative_length, :])

    def get_mask_sizes(self, cache_position) -> tuple[int, int]:
        """Reports the same active attention length as DynamicCache."""
        return self.cumulative_length + cache_position.shape[0], 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        """Retains dynamic-cache mask semantics despite bounded storage."""
        return -1

    def reset(self) -> None:
        """Invalidates old slots without clearing the backing tensors."""
        self.cumulative_length = 0
```

This class is the entire resolution of the V5/V6 crisis, so read it against that history:

- **Storage is static**: one fixed `[batch, heads, max_cache_len, head_dim]` pair of tensors, allocated once. Addresses never move — the precondition for graph capture.
- **Visibility is dynamic**: `update()` writes the new K/V at the incoming `cache_position` with `index_copy_`, bumps `cumulative_length`, and returns *views* of only the populated prefix `[..., :cumulative_length, :]`. Attention therefore sees exactly what a `DynamicCache` would show — 2 entries after step 0, 3 after step 1, … — never zero padding. SDPA takes its mask-free causal fast path, the one whose numerics match the official runtime.
- **`get_max_cache_shape() → -1` is the load-bearing line.** Transformers' masking helpers branch on this: a real `StaticCache` reports its capacity, which triggers explicit full-capacity mask materialization (the V5/V6 numerics poison and a measured slowdown); returning −1 mimics `DynamicCache` and keeps the cheap path. `get_mask_sizes` mirrors the dynamic (active, 0-padding) pair for the same reason.
- **`reset()` is O(1)**: it just zeroes `cumulative_length`. Stale K/V beyond the prefix are invisible (views and mask sizes never extend past it), so there is nothing to clear. Per-request state restoration costs nothing and never recaptures graphs.
- **The replay subtlety worth understanding**: during capture, every shape and position decision is made by Python from `cumulative_length`, then baked into the graph. Replay re-executes the recorded kernels against the same buffers — including the `index_copy_` at the recorded positions. Because the replay order is deterministic (graph 0, then 1, … then 14) and the Python state at capture time matches it, replay reproduces the eager sequence bit-for-bit. The prefix grows in software between replays; each graph was captured *for* its exact prefix length.

`prefix_cache(config, max_cache_len)` simply wraps one such layer per predictor decoder layer in a Transformers `Cache` container, so the official predictor model can consume it through the normal `past_key_values` interface.

#### `PredictorGraphs` — the 15-step inner loop as 15 graphs

```python
class PredictorGraphs:
    """Captures one exact eager predictor step for each residual codebook."""

    def __init__(self, predictor, talker_hidden_size, device, dtype) -> None:
        self.predictor = predictor
        self.device = device
        self.groups = predictor.config.num_code_groups - 1        # 15
        self.inputs = torch.zeros((1, 2, talker_hidden_size), device=device, dtype=dtype)
        self.tokens = torch.zeros((1, self.groups), device=device, dtype=torch.long)
        self.cache = prefix_cache(predictor.model.config, self.groups + 1)  # 16 slots
        self.graphs: list[torch.cuda.CUDAGraph] = []

    def _step(self, index: int) -> None:
        """Runs one original eager predictor step into the token buffer."""
        if index == 0:
            inputs = self.inputs                                   # [past_hidden, last_id_emb]
        else:
            embedding = self.predictor.get_input_embeddings()[index - 1]
            inputs = embedding(self.tokens[:, index - 1]).unsqueeze(1)
        hidden = self.predictor.model(
            inputs_embeds=self.predictor.small_to_mtp_projection(inputs),
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
        ).last_hidden_state
        token = self.predictor.lm_head[index](hidden[:, -1]).argmax(dim=-1)
        self.tokens[:, index].copy_(token)

    def _sequence(self) -> None:
        """Runs all residual positions with fresh logical cache state."""
        self.cache.reset()
        for index in range(self.groups):
            self._step(index)

    def capture(self) -> None:
        """Captures fixed eager kernels in a shared CUDA graph memory pool."""
        if self.device.type != "cuda" or self.graphs:
            return
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._sequence()                                   # eager warmups
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)

        self.cache.reset()
        pool = torch.cuda.graph_pool_handle()
        with torch.cuda.stream(stream):
            for index in range(self.groups):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    self._step(index)
                self.graphs.append(graph)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.cache.reset()

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        """Replays the sequence or uses the identical eager CPU fallback."""
        self.inputs.copy_(inputs)
        if not self.graphs:
            self._sequence()
            return self.tokens
        for graph in self.graphs:
            graph.replay()
        return self.tokens
```

Walkthrough:

- **The buffers are the contract.** `self.inputs` `[1, 2, hidden]` holds the two-token seed every frame: the talker's carried hidden state plus the primary token's embedding. `self.tokens` `[1, 15]` receives the residual outputs. Both are allocated once and written in place — replay only ever touches these addresses.
- **`_step(index)` is the official step, verbatim.** Step 0 consumes the 2-token seed; step *i* embeds book *i−1*'s token through the predictor's *own* codebook table (`get_input_embeddings()[index-1]` — 15 separate tables, not the talker's codec table; using the wrong table was a real fusion bug that produced garbage audio). The input goes through `small_to_mtp_projection` (the official module's internal projection), the 5-layer transformer runs against the prefix cache, and `lm_head[index]` — each residual position has its *own* output head — produces the argmax, written into `tokens[:, index]`.
- **Why 15 graphs, not one?** Step *i* attends over a prefix of length *i+2* — every step has a different, but *fixed and known*, shape. One graph for the whole loop would require padding the cache to capacity, i.e. static-cache numerics, i.e. the V6 divergence class. Fifteen graphs, each captured for its exact prefix length, cover the whole loop while every individual replay sees exactly the dynamic shapes. This is the central trick of V7.
- **The capture protocol is the standard CUDA-graph dance:** (1) skip if already captured or not on CUDA; (2) warm up *eagerly* on a side stream — 3 full sequences trigger lazy cache allocation, cuBLAS workspace setup, and allocator settling, all of which must not be frozen inside a graph; (3) synchronize; (4) capture each step on the side stream into one *shared memory pool* (`graph_pool_handle()`), so 15 graphs share allocations instead of each owning private memory; (5) sync and reset. The side stream matters because capture must not interleave with other GPU work, and the default stream is off-limits for capture.
- **`run()` is a two-line state machine.** Copy the caller's seed into the fixed buffer; if graphs exist, replay 0…14 in order and return `tokens`; otherwise run the identical eager `_sequence()` (the CPU fallback and the reference for debugging). Note the double buffer: the decode loop copies into `predictor_input`, and `run` copies into `self.inputs` — graphs read only their baked addresses, so inputs must live where capture put them.

#### `DecoderFfnGraph` — the talker's fixed half

```python
class DecoderFfnGraph(torch.nn.Module):
    """Keeps talker attention eager and graphs its fixed residual FFN."""

    def __init__(self, layer, hidden_size, device, dtype) -> None:
        super().__init__()
        self.layer = layer
        self.inputs = torch.zeros((1, 1, hidden_size), device=device, dtype=dtype)
        self.output = torch.zeros_like(self.inputs)
        self.graph: torch.cuda.CUDAGraph | None = None

    def _ffn(self, inputs: torch.Tensor) -> torch.Tensor:
        """Runs the official post-attention norm, MLP, and residual order."""
        return inputs + self.layer.mlp(self.layer.post_attention_layernorm(inputs))

    def capture(self, stream: torch.cuda.Stream, pool: tuple[int, int]) -> None:
        """Captures the one-token FFN while leaving prefill eager."""
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.output.copy_(self._ffn(self.inputs))
        torch.cuda.current_stream(self.inputs.device).wait_stream(stream)
        torch.cuda.synchronize(self.inputs.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(self.graph, pool=pool):
                self.output.copy_(self._ffn(self.inputs))
        torch.cuda.current_stream(self.inputs.device).wait_stream(stream)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple[Any, ...]:
        """Reproduces the layer and replays only its decode-shape FFN half."""
        if hidden_states.shape[1] != 1 or self.graph is None:
            return self.layer(hidden_states, **kwargs)
        residual = hidden_states
        attention_input = self.layer.input_layernorm(hidden_states)
        attention = self.layer.self_attn(
            hidden_states=attention_input,
            attention_mask=kwargs.get("attention_mask"),
            position_ids=kwargs.get("position_ids"),
            past_key_values=kwargs.get("past_key_values"),
            output_attentions=kwargs.get("output_attentions", False),
            use_cache=kwargs.get("use_cache", False),
            cache_position=kwargs.get("cache_position"),
            position_embeddings=kwargs.get("position_embeddings"),
        )[0]
        self.inputs.copy_(residual + attention)
        self.graph.replay()
        return (self.output,)
```

Why this shape of wrapper exists:

- **The talker layer splits into a variable half and a fixed half.** Attention reads a KV cache that grows every frame — shapes change, so it cannot be captured without the static-cache numerics poison. Everything after attention, `inputs + mlp(post_attention_layernorm(inputs))`, is always `[1, 1, 1024]` during decode — perfectly capturable. The wrapper runs attention eagerly through the official module (same masks, same rope, same growing `DynamicCache`) and replays the FFN half.
- **The wrapper is installed *into* the official model**: `talker.model.layers[i] = DecoderFfnGraph(layer, …)`. The official talker forward then calls these objects transparently — no forked forward code, so prefill and decode share one code path and cannot drift.
- **`forward` dispatches on shape**: anything that isn't the one-token decode shape (i.e. the whole prefill) delegates to the untouched official layer. Decode runs norm → official attention (`[0]` takes the hidden state positionally; attentions/hidden states are never requested) → copy residual+attention into the fixed `self.inputs` → `graph.replay()` → return the fixed `self.output` wrapped in the tuple shape the official layer would return.
- **The `copy_` pattern**: capture records `output.copy_(_ffn(inputs))`, so replay re-reads `self.inputs` (just written) and re-writes `self.output`. The graph result lands in a known buffer instead of a fresh allocation — fresh allocations inside capture would be frozen anyway; `copy_` makes the hand-off explicit.
- **One stream, one pool for all 20 layers** (orchestrated by `DecodeGraphs.capture` below): the layers are captured sequentially on the same side stream into a shared pool, so the 20 graphs cost one pool's worth of memory, not twenty.

#### `OfficialTalker` — the eager, dynamic, correctness anchor

```python
class OfficialTalker:
    """Runs one inner-talker token with the official growing dynamic cache."""

    def __init__(self, model, device, max_cache_len) -> None:
        self.model = model
        self.device = device
        self.max_cache_len = max_cache_len
        self.cache = DynamicCache(config=model.config)
        self.rope_deltas = torch.zeros((1, 1), device=device, dtype=torch.float32)

    def capture(self) -> None:
        """Leaves the correctness reference eager because its cache grows."""

    def reset(self, prompt_length: int, rope_deltas: torch.Tensor | None = None) -> None:
        """Creates fresh request state and validates its maximum frame budget."""
        if prompt_length >= self.max_cache_len:
            raise ValueError("prompt exceeds the talker cache capacity")
        self.cache = DynamicCache(config=self.model.config)
        self.rope_deltas.zero_()
        if rope_deltas is not None:
            self.rope_deltas.copy_(rope_deltas)

    def set_rope_deltas(self, rope_deltas: torch.Tensor) -> None:
        """Copies the prefill mRoPE delta used by later one-token forwards."""
        self.rope_deltas.copy_(rope_deltas)

    def run(self, inputs: torch.Tensor, position: int) -> torch.Tensor:
        """Runs the official mask-free one-token dynamic-cache forward."""
        if position >= self.max_cache_len:
            raise ValueError("decode exceeds the talker cache capacity")
        cache_position = torch.tensor([position], device=self.device, dtype=torch.long)
        position_ids = self.rope_deltas + cache_position.to(self.rope_deltas.dtype)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        attention_mask = torch.ones((1, position + 1), device=self.device, dtype=torch.long)
        return self.model(
            inputs_embeds=inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        ).last_hidden_state
```

Points worth internalizing:

- **This is the piece that must never be "improved".** The talker's attention is the one place where every failed optimization bit: static caches (V3 regression, V5 silence, V6 divergence, frame-graph +8%). V7 keeps it exactly official: a fresh `DynamicCache` per request, growing by one position per frame, with a full-population ones mask. Eager, mask-consistent, bit-exact.
- **`rope_deltas` is the 3D-mRoPE bookkeeping.** The talker uses 3D multi-resolution rope over a mixed text/audio stream; the offset between sequence position and cache position is computed once during prefill and must be reused for every one-token decode forward. `position_ids = rope_deltas + cache_position`, expanded to the 3 rope axes. Losing or recomputing this per step is a silent numerics change — it is copied from prefill, never re-derived.
- **Position bookkeeping is explicit.** The caller passes `position = prefill_length + frame_index`; the class re-checks capacity on every call. The mask is `ones(1, position+1)` — "attend to everything populated" — which is exactly what the official one-token decode does with a fully-populated dynamic cache.

#### `DecodeGraphs` and the `decode_graphs` cache

```python
class DecodeGraphs:
    """Groups the two graph boundaries so a full-frame graph can replace them."""

    def __init__(self, talker, device, dtype, max_cache_len) -> None:
        self.ffn_graphs: tuple[DecoderFfnGraph, ...] = ()
        self.predictor = PredictorGraphs(
            talker.code_predictor, talker.config.hidden_size, device, dtype,
        )
        self.talker = OfficialTalker(talker.model, device, max_cache_len)
        if device.type == "cuda":
            wrappers = []
            for index, layer in enumerate(talker.model.layers):
                wrapper = DecoderFfnGraph(layer, talker.config.hidden_size, device, dtype)
                talker.model.layers[index] = wrapper
                wrappers.append(wrapper)
            self.ffn_graphs = tuple(wrappers)

    def capture(self) -> None:
        """Captures both reusable decode blocks once per loaded talker.

        FFN graphs share one side stream and memory pool; the predictor keeps
        its own pool so its cache-reset sequencing stays independent.
        """
        self.predictor.capture()
        if self.ffn_graphs:
            stream = torch.cuda.Stream(device=self.talker.device)
            stream.wait_stream(torch.cuda.current_stream(self.talker.device))
            pool = torch.cuda.graph_pool_handle()
            for graph in self.ffn_graphs:
                graph.capture(stream, pool)
            torch.cuda.current_stream(self.talker.device).wait_stream(stream)
            torch.cuda.synchronize(self.talker.device)
        self.talker.capture()          # no-op: eager on purpose


@cache
def decode_graphs(talker: torch.nn.Module, max_cache_len: int) -> DecodeGraphs:
    """Creates persistent graph objects and allocations once per talker module."""
    parameter = next(talker.parameters())
    graphs = DecodeGraphs(talker, parameter.device, parameter.dtype, max_cache_len)
    graphs.capture()
    return graphs
```

- **`decode_graphs` is `functools.cache`-memoized on the talker module identity**: the first request constructs and captures everything (seconds of setup, which is why servers warm up); every later request gets the same objects and pays only `reset()`. Capture once, replay forever.
- **State separation is what makes reuse safe.** Per-request mutable state — the talker's `DynamicCache`, `rope_deltas`, the predictor's `cumulative_length` — is reset at the start of each request; the graphs themselves are stateless replay templates. Nothing is re-captured per request.
- **`DecodeGraphs` is also the *seam* for future work**: it groups the per-frame boundaries behind `predictor.run()` and `talker.run()`, so a candidate replacement (a fused kernel, a better cache) can swap in at exactly one place — after passing the parity gate.

### 7.4 inference.py — scheduling and the decode loop

This file owns everything the graphs don't: prompt construction, prefill, token selection policy, EOS handling, the frame loop itself, and phase timing.

#### Chunked EOS — `_maybe_eos_row`

```python
EOS_CHECK_EVERY = 8


def _maybe_eos_row(codes, upto, eos_token_id, *, stop_at_eos, force) -> int | None:
    """Returns the first EOS frame index below ``upto``, at the check cadence.

    Replaces the per-frame ``token.eq(eos).item()`` host sync with a device
    scan every ``EOS_CHECK_EVERY`` frames (plus any forced frame), so the GPU
    never drains per frame and the emitted trim stays identical. Callers must
    force a check before any point where ``codes`` becomes observable, such
    as a streaming chunk boundary.
    """
    if not stop_at_eos or (upto % EOS_CHECK_EVERY != 0 and not force):
        return None
    hit = (codes[:upto, 0] == eos_token_id).nonzero()
    return int(hit[0, 0]) if hit.numel() else None
```

The problem it solves: checking "did we emit EOS?" naively requires `.item()` — a device→host synchronization that drains the entire GPU pipeline every frame. Instead, the scan runs *on device* every 8 frames: `nonzero()` stays a GPU tensor; only an actual hit forces a value across. The `force` flag exists for the one rule this design imposes: **the scan must be forced before the codes become observable to anyone** — a streaming chunk boundary (a listener would hear stale frames past EOS) and the final frame (the last row must still be checked). The trim semantics are identical to a per-frame check by construction: the EOS row is written, detected, and sliced away.

#### Token selection — `_select_token`

```python
def _select_token(logits, history, *, eos_token_id, processors, allow_eos) -> torch.Tensor:
    """Applies official processors and returns one greedy token."""
    scores = processors(history, logits[:, -1].to(dtype=torch.float32, copy=True))
    if not allow_eos or history.shape[1] < 2:
        scores[:, eos_token_id] = -torch.inf
    return scores.argmax(dim=-1)
```

Three policy decisions hide in four lines:

- **float32 logits.** The official generation path applies repetition penalty and suppression to float32 scores. Doing it in bf16 (an early whistle bug) flips argmaxes near boundaries — parity fails. `.to(dtype=torch.float32, copy=True)` guarantees the official numerics.
- **The suppression list is built once in `_prefill`**: the codec head emits 3,072 logits — 0–2047 are ordinary codec tokens, 2048–3071 are specials, and codec EOS is 2150. Everything in the special range *except* EOS is suppressed. Truncating to the 2,048 ordinary tokens instead (another early bug) made talker EOS unreachable, so generation never stopped naturally.
- **The early-EOS guard** (`history.shape[1] < 2`) mirrors the official runtime: EOS only becomes selectable once a couple of primary tokens exist. `allow_eos=stop_at_eos` connects the fixed-budget benchmark mode (EOS force-masked forever, full 1,280 frames) to the same selector used in natural mode.

#### The shared request state — `Prompt` and `Prefill`

```python
@dataclass(frozen=True, slots=True)
class Prompt:
    """Named prompt-build outputs plus the reset shared decode graphs."""

    talker_input: torch.Tensor            # [1, prefill_len, hidden] embeddings
    attention_mask: torch.Tensor          # ones([1, prefill_len])
    tts_pad: torch.Tensor                 # [1, 1, hidden] projected pad embedding
    primary_history: torch.Tensor         # [1, max_new_tokens] token id buffer
    codec_embeddings: torch.nn.Module     # talker input-embedding table (primary tokens)
    graphs: DecodeGraphs
    prefill_length: int


@dataclass(frozen=True, slots=True)
class Prefill:
    """First-frame decode state shared by the batch and streaming loops."""

    token: torch.Tensor                   # first selected primary token
    past_hidden: torch.Tensor             # talker's carried hidden state
    processors: LogitsProcessorList       # repetition penalty + suppression
    residual_embeddings: tuple[torch.nn.Module, ...]   # 15 predictor codebook tables
    eos_token_id: int
    num_code_groups: int                  # 16
    hidden_size: int
```

These two frozen dataclasses are the interface between setup and the two consumers. `_prepare` produces a `Prompt`; `_prefill` consumes it and produces a `Prefill`; both `tts_infer` (batch) and `stream_tts` (streaming) are just different decode loops over the same `Prefill`. Frozen + slots means no accidental mutation and no drift between the two paths — this structure came directly out of a code review that found the streaming path reimplementing setup and diverging (it still used the per-frame EOS sync the batch path had removed).

#### `_prepare` and `_prefill`

```python
def _prepare(tts, text, *, speaker, language, device, max_new_tokens) -> Prompt:
    """Builds the prompt tensors, checks cache capacity, and resets decode state."""
    talker = tts.model.talker
    talker_input, attention_mask, tts_pad, prefill_length, codec_embeddings = build_prompt(
        tts, text, language=language, speaker=speaker, device=device
    )
    if prefill_length + max_new_tokens - 1 > MAX_CACHE_LEN:
        raise ValueError("prompt and frames exceed the fixed talker cache capacity")
    graphs = decode_graphs(talker, MAX_CACHE_LEN)   # memoized: capture happens once
    graphs.talker.reset(prefill_length)
    talker.rope_deltas = None
    return Prompt(...)


def _prefill(tts, prompt, *, repetition_penalty, stop_at_eos) -> Prefill:
    """Runs the prefill forward and selects the first primary token."""
    model, talker = tts.model, tts.model.talker
    talker_config = model.config.talker_config
    talker_output = talker(
        inputs_embeds=prompt.talker_input,
        attention_mask=prompt.attention_mask,
        past_key_values=prompt.graphs.talker.cache,
        past_hidden=None,                    # no previous frame yet
        trailing_text_hidden=prompt.tts_pad,
        tts_pad_embed=prompt.tts_pad,
        generation_step=None,
        use_cache=True,
        return_dict=True,
    )
    eos_token_id = talker_config.codec_eos_token_id          # 2150
    suppress_from = talker_config.vocab_size - 1_024         # 2048
    suppress_tokens = [t for t in range(suppress_from, talker_config.vocab_size)
                       if t != eos_token_id]
    processors = LogitsProcessorList()
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    processors.append(SuppressTokensLogitsProcessor(suppress_tokens, device=…))
    token = _select_token(
        talker_output.logits, prompt.primary_history[:, :0],
        eos_token_id=eos_token_id, processors=processors, allow_eos=stop_at_eos,
    )
    prompt.graphs.talker.set_rope_deltas(talker.rope_deltas)
    return Prefill(
        token=token,
        past_hidden=talker_output.past_hidden,
        processors=processors,
        residual_embeddings=tuple(talker.code_predictor.get_input_embeddings()),
        eos_token_id=eos_token_id,
        num_code_groups=talker_config.num_code_groups,
        hidden_size=talker_config.hidden_size,
    )
```

Notes:

- **Prefill is one eager forward over the whole prompt** into the talker's `DynamicCache`. It runs through the FFN-wrapped layers, but those delegate to the official layer for any shape ≠ 1 — so prefill numerics are untouched by the graphs machinery.
- **The official module's extra arguments are part of its contract**: `past_hidden=None` (no prior frame), `trailing_text_hidden`/`tts_pad_embed` (the TTS pad embedding that joins the attention stream and gets added to every later frame input), `generation_step=None` (prefill has no step counter).
- **`past_hidden` is a second carried state**, alongside the KV cache: the hidden state of the last processed token, which the next forward consumes. That's why the predictor seed is *two* tokens: `[past_hidden, last_token_embedding]`.
- **`residual_embeddings` are the predictor's 15 codebook tables**, fetched once. In the decode loop they embed the residual tokens for the next frame's talker input — using these exact tables in this exact order is required for bit-exact embedding sums.
- **`set_rope_deltas` snapshots the prefill-computed mRoPE delta** onto the `OfficialTalker`, closing the position bookkeeping loop described in 7.3.

#### `build_prompt` — the CustomVoice prompt anatomy

```python
def build_prompt(tts, text, *, language, speaker, device):
    """Builds the official CustomVoice prompt tensors (verbatim from tts_infer)."""
    …
    input_ids = tts._tokenize_texts([tts._build_assistant_text(text)])[0]
    language_id = …                          # codec_language_id[language] (dialect-aware)
    speaker_id = talker_config.spk_id[speaker_key]

    text_embeddings = talker.get_text_embeddings()      # text-space table
    codec_embeddings = talker.get_input_embeddings()    # codec-space table
    project_text = talker.text_projection

    speaker_embed = codec_embeddings(torch.tensor(speaker_id, device=device)).view(1, 1, -1)
    special_text = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]], …)
    tts_bos, tts_eos, tts_pad = project_text(text_embeddings(special_text)).chunk(3, dim=1)

    codec_prefix = [codec_think_id, codec_think_bos_id, language_id, codec_think_eos_id]
    codec_prompt = torch.cat(
        [codec_embeddings(prefix_ids), speaker_embed, codec_embeddings(suffix_ids)], dim=1)
    role = project_text(text_embeddings(input_ids[:, :3]))
    codec_header = torch.cat(
        [tts_pad.expand(-1, codec_prompt.shape[1] - 2, -1), tts_bos], dim=1
    ) + codec_prompt[:, :-1]
    spoken_text = torch.cat([project_text(text_embeddings(input_ids[:, 3:-5])), tts_eos], dim=1)
    codec_pad = codec_embeddings(torch.full((1, spoken_text.shape[1]), codec_pad_id, …))
    codec_bos = codec_embeddings(torch.tensor([[codec_bos_id]], device=device))
    talker_input = torch.cat(
        [role, codec_header, spoken_text + codec_pad, tts_pad + codec_bos], dim=1)
    attention_mask = torch.ones(talker_input.shape[:2], device=device, dtype=torch.long)
    return talker_input, attention_mask, tts_pad, talker_input.shape[1], codec_embeddings
```

This is a verbatim copy of the official prompt builder, because the prompt *is* part of the model contract: any difference here changes the first hidden state and therefore everything after. The anatomy, in stream order:

- **`role`** — the first 3 text tokens (the chat-role header), text-embedded and projected.
- **`codec_header`** — codec-space control tokens (think BOS/EOS framing, the language ID, the speaker embedding) with the projected TTS pad/bos specials *added* elementwise onto all but the last position. The talker was trained on streams where text-space and codec-space embeddings are summed at certain positions; the additions reproduce that training format. (The snippet shows the language-known framing; the `auto`-language branch uses a `codec_nothink_id` variant of the same prefix.)
- **`spoken_text + codec_pad`** — the actual input text (text tokens 3..−5, skipping the role header and trailing specials), each position summed with a codec `pad` embedding, terminated by `tts_eos`.
- **`tts_pad + codec_bos`** — the final position that hands off to generation; and `tts_pad` is *also returned separately*, because it is added to every decoded frame's input embedding during decode (the persistent voice/stream marker).
- **Two distinct embedding tables** matter here: `get_text_embeddings()` for text tokens and `get_input_embeddings()` for codec tokens — the same distinction that the Triton fusion experiment got wrong for the residual tables.

#### `tts_infer` — the frame loop

```python
@torch.inference_mode()
def tts_infer(tts, text, *, speaker="ryan", language="english",
              max_new_tokens=1_280, stop_at_eos=True, repetition_penalty=1.2):
    """Runs batch-one CustomVoice inference through explicit forward passes."""
    …
    prompt = _prepare(tts, text, speaker=speaker, language=language,
                      device=device, max_new_tokens=max_new_tokens)
    first = _prefill(tts, prompt, repetition_penalty=repetition_penalty,
                     stop_at_eos=stop_at_eos)

    codes = torch.empty((max_new_tokens, first.num_code_groups), device=device, dtype=torch.long)
    predictor_input = torch.empty((1, 2, first.hidden_size), device=device,
                                  dtype=first.past_hidden.dtype)
    token = first.token
    past_hidden = first.past_hidden
    frame_count = 0

    for frame_index in range(max_new_tokens):
        last_id_hidden = prompt.codec_embeddings(token.view(1, 1))
        predictor_input[:, :1].copy_(past_hidden)          # seed = [past_hidden, token_emb]
        predictor_input[:, 1:].copy_(last_id_hidden)
        residual_codes = prompt.graphs.predictor.run(predictor_input)   # 15 graph replays
        codes[frame_index, 0].copy_(token[0])
        codes[frame_index, 1:].copy_(residual_codes[0])
        prompt.primary_history[:, frame_index].copy_(token)
        frame_count = frame_index + 1
        hit = _maybe_eos_row(codes, frame_count, first.eos_token_id,
                             stop_at_eos=stop_at_eos,
                             force=frame_index + 1 == max_new_tokens)
        if hit is not None:
            frame_count = hit                              # trim to before EOS
            break
        if frame_index + 1 == max_new_tokens:
            break

        codec_hiddens = torch.cat(                          # official embedding reduction
            [last_id_hidden]
            + [embedding(residual_codes[:, index : index + 1])
               for index, embedding in enumerate(first.residual_embeddings)],
            dim=1,
        )
        talker_input = codec_hiddens.sum(dim=1, keepdim=True) + prompt.tts_pad
        past_hidden = prompt.graphs.talker.run(talker_input, prompt.prefill_length + frame_index)
        token = _select_token(
            talker.codec_head(past_hidden),
            prompt.primary_history[:, :frame_count],
            eos_token_id=first.eos_token_id,
            processors=first.processors,
            allow_eos=stop_at_eos,
        )

    # === codec: official decoder implementation, kept on-device ===
    speech_model = tts.model.speech_tokenizer.model
    codes = codes[:frame_count]
    if frame_count:
        decoded = speech_model.decode(codes.unsqueeze(0), return_dict=False)[0]
        waveform = decoded[0].unsqueeze(0)
    else:
        waveform = torch.empty((1, 0), device=device, dtype=prompt.tts_pad.dtype)
    …
    return waveform, codes, int(speech_model.get_output_sample_rate()), timings
```

The loop, step by step — this is the entire engine:

1.  **Embed the current primary token** through the talker's codec table (`last_id_hidden`), and stage the predictor seed: slot 0 gets the carried `past_hidden`, slot 1 gets the token embedding. Two copies into a preallocated buffer — no allocation, no host traffic.
2.  **Run the predictor**: one call, 15 graph replays inside, 15 residual tokens out into the fixed `tokens` buffer. This replaces the official nested `generate()` — 15 forwards through the full HF generation stack — with 15 single launches. It is the single biggest win in the project.
3.  **Publish the frame**: primary token into `codes[:, 0]`, residuals into `codes[:, 1:]`, primary token into `primary_history` (the repetition-penalty processor reads this later).
4.  **Chunked EOS check** (forced on the final frame so the last row is always verified): a hit trims `frame_count` to the pre-EOS index and breaks; rows after the trim are uninitialized but never observed.
5.  **Build the next talker input** exactly the official way: embed each residual token through its own codebook table, concatenate all 16 embeddings *in codebook order*, `sum(dim=1)`, add `tts_pad`. bf16 addition is order-sensitive — the `cat + sum` order is part of the parity contract.
6.  **Talker step** at absolute position `prefill_length + frame_index`: eager attention over the growing dynamic cache + 20 FFN graph replays. Returns the new `past_hidden`.
7.  **Primary selection**: `codec_head` (Linear 1024→3072) → float32 processors → argmax. Loop.

Then the codec: the official `speech_tokenizer` decodes `codes[:frame_count]` on-device — whistle does not reimplement or chunk it in the batch path (the only CPU transfer in the whole program happens in the CLI, for WAV writing).

#### Phase timing with CUDA events

```text
cuda_timing = device.type == "cuda"
phase_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if cuda_timing else None
if phase_events is not None:
    phase_events[0].record()
… phase_events[1].record()   # after prepare
… phase_events[2].record()   # after prefill
… phase_events[3].record()   # after decode
… phase_events[4].record()   # after codec
phase_events[4].synchronize()                     # ONE terminal sync
prepare_seconds = phase_events[0].elapsed_time(phase_events[1]) / 1000
…
```

Events are recorded into the GPU's own timeline; a single synchronization at the very end converts them to seconds. Host timers inserted mid-loop would force syncs (poisoning the pipeline they measure) or measure queueing depth instead of execution. The CPU fallback path (non-CUDA) uses `perf_counter` deltas instead. The returned dict carries `prepare / prefill / decode / codec / total / frames` — the phase breakdown discipline from section 5.3, built into the engine itself.

### 7.5 streaming.py — chunked streaming

Streaming reuses the batch path's brain (`_prepare`, `_prefill`, `_maybe_eos_row`, `_select_token`) and changes only the loop's *output cadence* and the codec decode.

#### Incremental codec decode with left context

```python
def _streaming_decoder(speech_model, left_context: int):
    """Returns a callable that decodes the next codec chunk with left context."""
    decoder = speech_model.decoder
    upsample = int(decoder.total_upsample)        # 1920 samples per frame @ 24 kHz / 12.5 Hz
    decoded_frames = 0

    def decode_next(codes_transposed: torch.Tensor) -> torch.Tensor:
        """Decodes codes[..., start-ctx:end] and trims the context warmup samples."""
        nonlocal decoded_frames
        start = decoded_frames
        end = codes_transposed.shape[-1]
        context = min(left_context, start)
        wav = decoder(codes_transposed[..., start - context : end])
        decoded_frames = end
        return wav[..., context * upsample :]

    return decode_next
```

The codec decoder is convolutional: decoding chunk *i* in isolation would sound different at its edges than the same frames inside the full-sequence decode. The fix is a sliding window — feed the last `left_context = 25` frames (2 seconds) of history plus the new frames, then *trim* the first `context × 1920` samples (the context's own "warmup" audio) from the output. Each yielded chunk's audio is therefore identical to the corresponding slice of the full-sequence decode, while never re-decoding audio the listener already has.

#### The streaming loop

```text
codes_transposed = torch.empty((1, first.num_code_groups, max_new_tokens), …)  # codec wants [B, C, T]

for frame_index in range(max_new_tokens):
    … identical 4 lines: embed token, stage seed, predictor.run, publish codes row …
    frame_count = frame_index + 1
    boundary = frame_count % chunk_size == 0           # chunk_size = 12 frames = 960 ms
    is_last = frame_index + 1 == max_new_tokens
    hit = _maybe_eos_row(codes, frame_count, first.eos_token_id,
                         stop_at_eos=stop_at_eos, force=boundary or is_last)
    if hit is not None:
        frame_count = hit                              # trim to before EOS
        is_last = True
    elif not is_last:
        … identical talker step + token selection as tts_infer …

    if not (is_last or boundary):
        continue
    if frame_count > chunk_start_frame:
        codes_transposed[..., :frame_count].copy_(codes[:frame_count].t().unsqueeze(0))
        now = time.perf_counter()
        yield chunk_dict(is_last, now)
    if is_last:
        break
```

Differences from the batch loop, all of them small and deliberate:

- **Forced EOS scan at every chunk boundary** — a yielded chunk must never contain frames past EOS, so observability forces the check (between boundaries the 8-frame cadence holds; there is still no per-frame host sync).
- **The talker step is skipped once EOS hits** (`elif not is_last`) — no point computing a next frame that will never be yielded.
- **A second, transposed codes buffer** feeds the codec decoder, which wants `[batch, codebooks, time]`; the row-major `codes` buffer stays as the canonical output.
- **Yield cadence**: only at boundaries or at the end, and only if the chunk still contains frames after EOS trimming. Each payload is a dict:

```text
{"codes": [chunk, 16] int64, "audio": [1, samples] float32,
 "sample_rate": int, "chunk_frames": int, "ttft_ms": float,
 "cumulative_ms": float, "chunk_ms": float, "final": bool}
```

Everything stays on-device; the caller transfers what it needs. TTFT is measured after `torch.cuda.synchronize()` following prefill — an honest "first audio possible" number rather than a queueing artifact. Measured behavior: first audio in ~0.9 s, chunks decoding at 1.71–1.78× realtime. The module's own CLI (`python -m whistle.streaming "text"`) warms up once, then prints per-chunk milestones.

### 7.6 server.py — a one-GPU API

```python
app = FastAPI(title="whistle-tts", docs_url=None, redoc_url=None)
_model: Qwen3TTSModel | None = None
_sample_rate = 24_000
# Decoding mutates shared per-model state (rope deltas, graph input buffers),
# so concurrent requests must not interleave; they queue here instead.
_generate_lock = threading.Lock()


def _wav_chunks(text, speaker, language, chunk_size) -> Generator[bytes, None, None]:
    """Yields WAV header + int16 PCM chunks as they are decoded."""
    tts = get_model()
    with _generate_lock:
        yield _wav_header(_sample_rate)
        for chunk in stream_tts(tts, text, speaker=speaker, language=language,
                                chunk_size=chunk_size):
            pcm = (chunk["audio"].float().cpu().numpy() * 32767.0).astype("<i2").tobytes()
            yield pcm


@app.get("/synthesize")
def synthesize(text: str, speaker: str = SPEAKER, language: str = LANGUAGE,
               chunk_size: int = 12) -> StreamingResponse:
    """Streams synthesized speech as audio/wav, chunk by chunk."""
    return StreamingResponse(_wav_chunks(text, speaker, language, chunk_size),
                             media_type="audio/wav",
                             headers={"Cache-Control": "no-cache"})
```

Three design points:

- **The lock is a correctness requirement, not a performance nicety.** Decode mutates singleton state — `rope_deltas`, the graph input buffers (`self.inputs`, `self.tokens`, FFN `inputs`/`output`), the talker's `DynamicCache`. Two concurrent requests would interleave writes into the same CUDA graph buffers and produce interleaved garbage audio. Serialization is the honest policy for a batch-one engine; requests queue on the lock.
- **The WAV header lies about sizes on purpose**: `0xFFFFFFFF` in the RIFF/data size fields, because chunked streaming doesn't know the length up front. Players handle streaming WAV with unknown sizes; the header is emitted *inside* the lock so the first byte arrives only when generation is actually starting.
- **The model loads lazily** on first request (get_model), bf16 on CUDA; `/health` reports load state. Test with `curl -N 'http://127.0.0.1:8000/synthesize?text=Hello%20world' -o stream.wav`.

### 7.7 profile_tts.py and infer.py — measurement and parity

`infer.py` is deliberately boring: `replace(RUNTIME, …)` from CLI flags, load the official model (bf16, SDPA), call `tts_infer`, `sf.write` the waveform (the program's single host transfer), print frame counts and phase milliseconds.

`profile_tts.py` is the measurement instrument. Its two halves:

**The benchmark half** normalizes both backends into a `Sample` (audio, sample rate, phases, codec IDs), runs excluded warmups then measured iterations (`reset_peak_memory_stats` before each), and reports mean and p50 wall/RTF plus per-phase p50s. The official side is driven through the wrapper's own API with `do_sample=False` and `subtalker_dosample=False` — the exact greedy configuration.

**The parity half** is the gate:

```python
def _check_codec_parity(split, official) -> None:
    """Requires identical codec shapes and reports the first differing token."""
    …
    mismatch = shared_split.ne(shared_official).nonzero()
    if mismatch.numel() != 0:
        frame, codebook = mismatch[0].tolist()
        matches = shared_split.eq(shared_official).sum().item()
        raise click.ClickException(
            f"codec id mismatch at frame {frame}, codebook {codebook}: "
            f"split={split[frame, codebook].item()}, "
            f"official={official[frame, codebook].item()} "
            f"({matches}/{shared_split.numel()} shared ids match)")
    …
    print(f"codec id parity: exact match ({split.shape[0]} frames)")


def _check_audio_parity(split, official) -> None:
    """Reports exact waveform equality after codec-token parity succeeds."""
    …
    maximum_error = float(np.max(np.abs(split_audio - official_audio), initial=0.0))
    if maximum_error != 0.0:
        raise click.ClickException(f"audio mismatch: max absolute error={maximum_error:.8g}")
```

Details that make the gate trustworthy:

- **First-mismatch localization** (`nonzero()[0]` → frame + codebook) — this is what traced V6's failure to "the first compiled talker hidden state" and taught the residual-argmax sensitivity lesson.
- **The selected-token offset**: the official generator counts *selected* tokens, the explicit loop counts *completed frames*; the parity check therefore runs the official side with `max_new_tokens + 1` and compares completed frames. Getting this wrong once produced a false parity failure (1,279 vs 1,278 frames).
- **Ordering**: parity runs untimed, *after* all measurements, and timing JSON is written before it — a failed parity check must never erase the performance evidence.
- **Fixed-token runs skip parity by design**: the official API always stops at codec EOS, so a forced 1,280-frame run has no official counterpart. The canonical comparison is natural-EOS on both sides (the README's two commands).

------------------------------------------------------------------------

## 8. A reimplementation checklist

If you rebuild this from scratch — or port it to another AR-audio model — these are the invariants. Break any one of them and you will not get a slower result; you will get a *different* result.

1.  **Copy the reference's numerics, not just its math.** Float32 logits processing, the exact suppression list (specials suppressed, EOS reachable), the exact embedding tables per codebook, the exact `cat + sum` order of the 16 codebook embeddings, the exact prompt-construction order. bf16 forgives nothing.
2.  **Never compile, fuse, or quantize anything on the talker→predictor path.** Compilation reorders reductions; kernels change accumulation order; int8 flips weights. Every one of these diverged at a residual argmax and collapsed the audio. "Mathematically equivalent" is not a passing grade.
3.  **Keep attention dynamic where history grows.** Talker attention must see only populated KV with dynamic-cache mask semantics. If you introduce static storage anywhere, its cache class must report dynamic-style mask sizes (`get_max_cache_shape() → -1`) and return populated-prefix views — or you have silently changed the attention math.
4.  **Graph only shape-invariant regions, into fixed buffers.** Everything a graph reads or writes must be a preallocated tensor written with `copy_`/`index_copy_`. Capture the original eager kernels (warm up first, side stream, shared pool). Per-position graphs (15 small ones) beat one padded loop graph.
5.  **Separate capture-time state from request-time state.** Capture once per model; reset logical state per request (cache length counters, rope deltas, fresh KV). Never re-capture per request; never let request data flow through Python-side graph logic.
6.  **No host synchronization in the hot loop.** No `.item()`, no per-frame CPU reads, no mid-loop allocations. Device-side EOS scans on a cadence, forced exactly when outputs become observable. Phase-time with CUDA events and one terminal sync.
7.  **Carry the model's extra state faithfully.** Here: the talker's `past_hidden` and the prefill-computed mRoPE `rope_deltas`. Any runtime that drops or recomputes carried state has changed the model.
8.  **Measure both sides under one protocol, then gate on exact parity.** Natural EOS on both, same speaker/text/dtype/backend, p50 with phase breakdowns, untimed reference generated after measurement. The gate must be able to reject your fastest idea — it will, and it will be right.
9.  **Serialize the engine.** Graph buffers and caches are singletons; one GPU serves one synthesis. Put the lock in before the second request, not after the first corrupted one.

------------------------------------------------------------------------

## 9. The landscape: where whistle sits

Everything above was built before reading a single competing runtime. This section, written 2026-08-29, maps the field and checks the work against it (`docs/literature_landscape.md` is the full map).

**The official anchor.** The Qwen3-TTS technical report (arXiv 2601.15621) publishes vLLM-based serving numbers on datacenter hardware: 97 ms first-packet latency for the 0.6B 12 Hz model, RTF 0.288, with the talker predicting codebook 0, an MTP/code-predictor stage emitting the 15 residual codebooks, and a causal ConvNet codec that streams with left context. Every claimed speedup should be read against that anchor, not against folk benchmarks.

**The serving tier.** nari-labs (the Dia team) published the current serving state of the art: sub-50 ms p95 audible TTFA at 10 RPS on one H100, by scheduling the talker, predictor, and codec as three tasks under one urgency-aware scheduler. Their per-module techniques read like this article's checklist: the predictor's fixed 15-step loop captured as *one* CUDA graph with preallocated KV (our V7, section 4.7), deferred EOS checks while EOS is suppressed (our chunked EOS, section 6.1), and state-cached incremental codec decoding (our streaming left-context, section 7.5). Three of their six headline techniques, arrived at independently. Below them, M\* (arXiv 2606.12688) generalizes composite-model serving and reports 2.9× lower RTF on Qwen3-Omni TTS, vLLM-Omni ships a dedicated optimization design doc, and qwen-tts-turbo runs fused CUDA megakernels — 4 ms time-to-first-packet on an RTX 5090 — with its talker megakernel *disabled in production* over deadlocks, falling back to a CUDA-graph talker: the megakernel authors converged on our architecture under production pressure.

**The local-runtime tier.** faster-qwen3-tts (CUDA graphs, now with a GGML backend), a Triton-fusion + batched-AR project claiming 5× single-clip and 14× per-sample at batch 16, Jetson real-time ports, C++/GGUF ports. None of them maintains bit-exactness; all validate with WER or listening tests. That is the niche whistle owns uncontested: a runtime whose every number is provably identical to the official one, on a 6 GB consumer GPU.

**The research thread.** Speculative decoding for speech tokens is the active algorithmic frontier — strict verification rejects acoustically equivalent tokens, so the field moved to relaxed (SSD, Interspeech 2025, 1.4×) and principled group-level acceptance (Apple's PCG, ICASSP 2026, 1.4× at exact sampling over acoustic similarity groups). Distillation attempts without pretraining-scale resources fail. Both directions change the output distribution; both are outside the parity contract by construction.

**Where that leaves this project.** The remaining frontier is scheduler-level work (multi-request urgency batching) and kernel-level work (megakernels) — the first needs a serving workload, the second breaks parity. What whistle adds to the conversation is not the fastest number; it is the only controlled, same-GPU, voice-controlled characterization of what bit-exactness costs and where it ends — including the negative results the ecosystem keeps rediscovering anecdotally (quantization collapse, chunk-boundary numerics, drift sensitivity in section 11).

------------------------------------------------------------------------

## 10. Track A: the parity-safe wins

The landscape reading defined a Track A: everything left that improves latency *without* leaving the parity contract. Three candidates were built and measured in `sandbox/track_a`; two promoted to the main runtime, one rejected on this GPU. All gates re-run green: parity32 exact, alicia natural-EOS parity exact (1,216 frames, 2,334,720 samples), regression RTF 0.555, unit test pass.

### 10.1 TTFA: from 507 ms to 97 ms

Interactive TTS latency is time-to-first-audio, not RTF. The old streaming path held the first chunk until 12 frames existed (960 ms of audio buffered before the first sample shipped). Three changes: a *ramp schedule* that ships the first chunk after 2 frames and grows boundaries (2, 4, 8, then steady 12) so later chunks keep playback headroom; a leading-silence trim on the first chunk (10 ms RMS windows, 20 ms lead-in kept — the nari-style dynamic trim, essentially free at ~0.2 ms); and an incremental transposed-buffer fill that removes an O(n²) per-chunk copy.

| text | TTFA fixed-12 (old) | TTFA ramp+trim (new) | codes vs batch |
|----|---:|---:|----|
| short (1 sentence) | 507.0 ms | 97.1 ms | identical |
| medium (3 sentences) | 509.4 ms | 100.4 ms | identical |
| alicia (97.3 s audio) | — | 146.2 ms | identical |

Steady-state cadence is unchanged (530–555 ms per 12-frame chunk, 1.73–1.81× realtime), emitted codec ids are bit-identical to the batch path, and Qwen3-ASR WER on the ramped+trimmed alicia stream scores CER 0.82% — the same as the batch canonical. For scale: official vLLM reports 97 ms first-packet on datacenter hardware; a 6 GB laptop GPU now lands at ~100 ms.

### 10.2 Sampling inside the graphs

Section 6.3's sampling regression was structural: the 15 captured predictor graphs bake greedy `argmax`, so sampled decoding fell back to the eager 15-step loop — every frame re-launching ~75 kernels — and paid +38% wall (78.4 s on the alicia budget). The fix is a second graph set. Each predictor step's selection becomes top-k mask → softmax/temperature → multinomial, and the whole step is captured lazily per `(temperature, top_k)` pair, reusing the same capture protocol as the greedy set (warmup on a side stream, shared pool, fixed buffers).

The mechanism that makes this legal is that PyTorch's CUDA RNG is graph-capturable: capture records the philox offset consumption, and each replay advances the generator by the captured increment. Fresh multinomial draws every frame, no host involvement, deterministically re-seedable. Verified empirically: two sampled runs produce different codes; the eager fallback remains available under `WHISTLE_SAMPLE_GRAPHS=0` for A/B.

| mode (fixed 1280, alicia)      |        wall |         RTF | vs greedy |
|--------------------------------|------------:|------------:|----------:|
| greedy graphs                  | 56.8–57.0 s | 0.555–0.556 |         — |
| sampled, eager fallback (old)  | 78.3–78.5 s | 0.765–0.766 |      +38% |
| sampled, captured graphs (new) | 58.7–59.0 s | 0.573–0.576 |     +3.2% |

Peak memory grows 8 MB (3,035 → 3,043 MB). Sampling is now a usable path at essentially greedy speed, which also matters for section 11: the kernel-feasible regime is sampled decoding.

### 10.3 Codec overlap: a flag for bigger GPUs

The codec phase costs 1.8 s per alicia run, strictly serialized after decode. Overlapping it into the decode loop on a side stream works mechanically — and reading the codec source settled the exactness question first: the official full decode *is* chunked decode (300-frame chunks, 25-frame left context), and the codec's transformer is fully causal (`sliding_window: None`), so any other chunk cadence sees different attention context and deviates bitwise (measured: max abs 4.4e-2 on a fixed-64 run; stream-vs-batch audio deviation 0.126 at 12-frame cadence). Overlapped audio therefore fails the waveform-parity gate by construction — codec tokens stay identical, samples do not.

It was measured anyway, and on the 3050 it is a *net loss*: 64.3 s vs 56.9 s. The side-stream conv/transformer kernels contend for the same 20 SMs as the decode loop, and the contention costs more wall than the 1.8 s it hides. The implementation ships behind `--overlap-codec/--no-overlap-codec` (default off) as a documented A/B lever for GPUs with SM headroom, and as the boundary marker of the parity contract: greedy default keeps bit-exact waveforms; streaming already lives on the other side of that line by design.

One candidate was rejected by analysis alone: per-speaker KV-prefix caching. The fixed prompt prefix is 8–9 tokens out of 20–200+, so the ceiling is ~10–50 ms on short requests — not worth split-prefill correctness risk around mRoPE deltas.

------------------------------------------------------------------------

## 11. Stage 0: the kernel question, answered without kernels

The parity wall leaves one question open: section 6.2's failed Triton kernels had real bugs (wrong embedding tables, misused `tl.dot`), producing 1e-2-level errors. Would a *correct* kernel — fp32 accumulation, merely a different reduction order, ~1e-3-relative drift — also collapse long-form greedy? That question decides whether a whistle v2 with custom kernels is possible, and it can be answered without writing one.

**The instrument** (`sandbox/stage0`, PyTorch-only): wrap selected `nn.Linear` ops *before* graph capture so the probe is baked into the captured FFN graphs. Two regimes: an **fp32-compute swap** (`x.float() @ W.float().T` cast back to bf16 — cuBLAS bf16 GEMV already accumulates fp32 in its own order, so the swap reproduces exactly what a careful kernel does: same products, different reduction path, boundary last-bit flips), and a **noise sweep** adding fresh `eps·randn` per frame to all 247 decode linears (RNG inside the graphs advances per replay, so noise is fresh each frame). Calibration on live activations: the fp32 swap flips 40–50% of output elements by one bf16 ULP (o_proj max abs 2.0e-3, mean 1.4e-5); the output head is nearly drift-free (3e-7), which doubles as a control.

**That control matters:** the head-scope run (drift ≈ 0) stayed *bit-identical* to baseline for all 1,216 frames — the harness measures what it claims. Then the escalation, one alicia natural-EOS run each:

| run | frames | first divergence | tail RMS | verdict |
|----|---:|----|----|----|
| baseline | 1,216 | reference | −34.4 dB | healthy |
| drift head (zero-drift control) | 1,216 | identical | −34.4 dB | harness clean |
| drift attention (132 linears) | 1,280 | frame 3 | −68.3 dB | COLLAPSE |
| drift FFN (99 linears) | 1,229 | frame 3 | −30.7 dB | healthy |
| drift talker+predictor (247 linears) | 1,280 | frame 1 | −56.7 dB | COLLAPSE |
| noise 1e-4 | 1,280 | frame 5 | −60.5 dB | COLLAPSE |
| noise 1e-3 | 1,144 | frame 3 | −35.0 dB | healthy |
| noise 1e-2 | 1,280 | frame 0 | −94.0 dB | total collapse |

Three findings. First, **any** ULP-level drift diverges the greedy trajectory within 1–5 frames — bit-exactness with recompiled numerics is not achievable, full stop. Second, and more interesting: divergence is not the danger, the *silence attractor* is. Collapse is stochastic across drift realizations, not monotone in magnitude — noise 1e-4 collapsed while 1e-3 survived, and roughly half of the drifted trajectories fell in. There is no safe drift threshold to engineer against; each diverged trajectory is a lottery draw. Third, bf16 output rounding makes sub-ULP drift literally invisible — the drift floor of any future kernel is one ULP per element, and the fp32 swap shows what that floor does at scale.

**Verdict: red for greedy.** A v2 kernel path is only viable for sampled decoding — which has no deterministic attractor to fall into, and which the checkpoint ships as its default anyway. The roadmap (`docs/kernel_plan.md`) scopes a sampled-only persistent predictor/talker kernel at a measured ceiling of ~1.5–1.75× on this GPU (the predictor re-reads its weights 15× per frame; the sequential dependency makes that traffic compulsory — no persistent kernel can remove it, only the launch overhead around it), gated by WER rather than parity, shipping alongside the parity-exact greedy edition. For now, by project direction, everything stays PyTorch: the graphs edition is the product, and this probe is the documented reason why.

------------------------------------------------------------------------

## 12. Takeaways

1.  **Profile before believing folklore.** Attention wasn't the bottleneck; launch overhead was. The optimization that won was scheduling, not kernels.
2.  **Graphs preserve; they don't improve.** A CUDA graph replays whatever kernels you captured. Capture the fast variant, at the largest shape-invariant boundary that doesn't change numerics — here, 15 per-position predictor graphs and 20 talker-FFN graphs, with attention left eager and dynamic.
3.  **Static caches aren't automatically faster.** They pay off only when fixed shapes are exploited; for eager attention they're a measured loss, and mishandled masks make them silent.
4.  **Equivalence isn't identity.** In bf16, at greedy, under autoregression, "mathematically equivalent" loses to "bit-identical." Keep an official reference path and compare everything; let the first-mismatch location teach you where your numerics actually changed.
5.  **Fastest ≠ best.** The two fastest measurements in the project's history were deleted. The parity gate is the reason the surviving number means anything.
6.  **The product is the scheduler.** Weights, tokenizer, codec, and math all stayed official; a ~1,000-line runtime around them bought 1.8× real-time with zero quality change — and a streaming API whose first audio arrives in about a tenth of a second.
7.  **Independent convergence is validation.** When the serving state of the art was published, three of its six headline techniques were already here — the 15-graph predictor, the deferred EOS check, the incremental codec. Convergence across independent teams is the closest thing this discipline has to peer review.
8.  **Drift doesn't just diverge — it lotteries.** One ULP of per-op drift diverges greedy decoding within frames, and whether the diverged trajectory survives 100 seconds of speech is a coin flip against the silence attractor, not a function of drift size. That is why the parity gate exists, and why any future kernel ships for sampled decoding only.

------------------------------------------------------------------------

## Appendix: The Graveyard of Failures and Trials

> **The Golden Rule of Autoregressive TTS:** A small bf16 reduction difference flips one greedy `argmax`; autoregression then magnifies that single flipped token across the rest of the sequence until it collapses into gibberish or digital silence. Plausible audio is never proof of correctness.

### A.1 The Invalidated Speedup Hall of Fame

The fastest numbers recorded in the project's history were completely invalid:

<div class="table-wrap">

| Version / Mode | Claimed Time | Why It Failed | Root Cause |
|----|---:|----|----|
| **V5 (Static Talker Graph)** | **48.1 s** *(fastest in repo history)* | **Digital silence** on long input. | Used full-capacity `StaticCache` with `attention_mask=None`. While valid for `DynamicCache`, in `StaticCache` SDPA attended over all 1,432 empty zero-padded slots. Attention diluted to zero, killing energy. |
| **V6 (Compiled Static Talker)** | **56.6 s** | **Parity divergence** at frame 1, codebook 13. | `torch.compile` altered the bf16 reduction/accumulation order. One `argmax` flipped (344 vs 1484), fed the next residual embedding, and permanently diverged the sequence. |
| **Full Frame-Graph (1 or 2 graphs)** | **63.1 s** | **~8% slower** than default + collapsed to silence after 30s. | Capturing the entire frame into monolithic graphs forced static 2048-slot masked attention. Attending over padding added a +8% compute penalty, and numerical drift triggered a greedy silence collapse at ~30s (WER 99%). |

</div>

### A.2 The two distinct causes of silent audio

One of the most insidious traps was that two completely different bugs produced the exact same symptom: the model stopped speaking and output digital silence.

1.  **The Masking Bug (Numerical Collapse):** Occurred in V5 and Full Frame-Graph. Missing masks on zero-padded static caches or bf16 accumulation drift flipped a token, causing hidden states to drift into an invalid latent space where the model output zero energy.
2.  **The Greedy Penalty Trap (Behavioral Collapse):** Occurred in early repetition penalty tuning (1.05 or 1.1). Under greedy decoding, low repetition penalty caused Alicia to collapse into a low-energy repetitive loop after ~16s. Raising greedy repetition penalty to **1.2** matched official greedy behavior bit-for-bit and reached natural EOS at frame 1,216.

### A.3 Kernel fusion and quantization trials

- **Triton Fused RMSNorm→QKV (`rmsnorm_qkv`):** Fused LayerNorm, Q/K/V projections, and head-norms into one launch. Microbenchmark was **speed-neutral** (224 µs fused vs 259 µs unfused, ±13%) because batch-1 is bandwidth-bound on weight loads. End-to-end result was **100% WER**: max absolute error of 0.004–0.06 against cuBLAS compounded across 20 layers × 64 frames, destroying greedy decoding. Inside CUDA graphs, kernel launches are already free.
- **W8A16 Int8 Talker MLP Quantization (`quant.py`):** Quantized Talker SwiGLU MLP weights to per-channel int8 (623 MB → 312 MB). Latency was **completely unchanged** (56.78s → 56.79s) because dequant-to-fp16 GEMVs save no time at batch-1 on tensor cores. Quality was **instantly destroyed (WER 198%)**: weight rounding flipped the very first residual codebook at frame 0, codebook 1.
- **The Codebook Embedding Fusion Bug:** Batched all 16 embedding lookups into one call, but accidentally looked up residuals 1–15 in the *codec* embedding table instead of their 15 distinct residual codebook tables (`nn.ModuleList`). Emitted white noise (WER 9,200%). Reverting to the official tables restored 100% bitwise parity.

### A.4 Serving and architectural dead ends

- **`overlap_codec` (Side-Stream Codec Decoding):** Ran incremental codec decoding on a second CUDA stream concurrently with Talker decode. Resulted in a **net slowdown (64.3s vs 56.9s)** on the RTX 3050 because the neural vocoder and Talker fought for SM execution units. Furthermore, the official codec transformer is full-causal without sliding windows, so any chunk boundary other than official 300/25 breaks bitwise waveform parity.
- **Stage 0 Sensitivity Probe:** Tested injecting 1 ULP of arithmetic noise (~10⁻³ relative error) into PyTorch GEMVs to simulate custom C++ megakernels. Verdict was **RED**: even 10⁻⁴ noise stochastically triggered silence collapse in ~50% of runs, proving custom non-cuBLAS kernels are mathematically incapable of guaranteeing greedy parity.
- **KV-Prefix Caching:** Fixed prompt prefix is only 8–9 tokens out of 50–200+ prompt tokens. Maximum theoretical savings was ~10–30 ms per utterance, not worth cache invalidation risks.

### A.5 Measurement, voice, and tooling traps

- **The Serena vs. Ryan Voice Effect:** The exact same bit-exact engine produced **38.8% WER on voice `serena`** but **3.02% WER on voice `ryan`** because Qwen3-ASR degraded on Serena's formal cadence. Ryan was made project default.
- **The Positional CLI Trap:** Invoking `profile_tts.py short.txt` passed the literal string `"short . txt"` as text instead of reading the file, producing 150% WER and triggering hours of debugging for a nonexistent regression.
- **Thermal Throttling on Victoria (RTX 3050 Mobile 40W):** After two consecutive benchmark runs, GPU clock speeds throttled, causing runtimes to drift from 56.8s to 75s+. Accurate benchmarks required fresh processes, cool-down periods, and monitoring clocks.
- **Host Timer Illusion:** Python `time.perf_counter()` inside the decode loop measured CPU kernel queueing, not GPU execution. Accurate phase breakdowns required asynchronous CUDA events with a single terminal synchronization.

### A.6 The predictor brittleness thesis

Across all failed trials (V6 compile, full frame-graphs, QKV fusion, Int8 quantization), **divergence never started at the primary Talker token. It ALWAYS blew up first at a residual codebook argmax:**

1.  **Sequential Dependency:** Book k depends on Book k−1's argmax. One flip cascades through all 15 books within the frame.
2.  **Summation Mixing:** All 16 codebook embeddings sum into the next frame's Talker input, immediately shifting the next trajectory.
3.  **Razor-Thin Margins:** Residual codebooks encode subtle acoustic details with tiny logit margins. Any arithmetic noise crosses the argmax threshold easily.

Conclusion: in greedy mode, exact CUDA graph replay of the official eager kernels (V7) is the only path that stays on the rails.
