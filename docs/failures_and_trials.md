# Whistle — Failures & Trials (research notes)

Living log of every failure mode, dead end, and finding across the whistle
experiments, including the historical `refs/exp.md` record. Two gates decide
what counts as a result:

1. **Exact parity** — the candidate must reproduce every official codec ID and
   waveform sample. Plausible audio is never accepted as proof.
2. **Audio plausibility** (newer, for experiments that intentionally give up
   bit-exactness) — Qwen3-ASR WER/CER on short samples.

A small bf16 difference flips one greedy `argmax`; autoregression then magnifies
that single mismatch across the whole sequence. This single fact invalidated
more results than any other cause.

---

## 1. The historical ladder (V1–V7) and what failed

| Version | Idea | p50 wall | Verdict |
|---|---|---|---|
| V1 | Explicit prefill/decode scheduler, official modules | 71.382 s | valid, −22.2% |
| V2 | Kill host/GPU ping-pong: on-device buffers, sync removal | 67.974 s | valid, −4.8% |
| V3 | Static cache + `torch.compile` predictor | 57.477 s | valid, −15.4% (headline concealed a +31% talker regression — only phase A/B exposed it) |
| V4 | Dynamic-cache eager talker diagnostic | 91.148 s | noncompetitive; dynamic KV growth changes buffer addresses + per-step concat/reallocation |
| V4.1 | Compiled/eager talker variants | 100.541 s (modular) | noncompetitive |
| V5 | Fastest number in repo history | 48.1 s | **invalid** — masking bug made audio silent on long input |
| V5.1 | Correctness restored | 64.2 s | valid, slower |
| V6 | Compiled static-cache talker + static predictor | 56.577 s | **invalid** — first codec divergence at frame 1 / codebook 13 (`refs/exp.md` says 128-frame diagnostics diverged at frame 1/cb 13) |
| V7 | Predictor graphs (15 separate) + talker residual-FFN graphs, eager dynamic attention | 56.858 s | **promoted** — exact: 20,480/20,480 codec IDs, 2,457,600/2,457,600 samples |

### The silent static-cache paths (from `exp.md`)

Full-capacity `StaticCache` variants failed exact parity in every combination:

| Talker | Predictor | First mismatch |
|---|---|---|
| Static eager | Static eager | Frame 3, codebook 15 |
| Compiled static | Static eager | Frame 5, codebook 15 |
| Static eager | Compiled static | Frame 1, codebook 13 |

Some produced plausible-but-wrong audio; **one degenerated toward silence**.
The static talker either made SDPA attend over unused cache slots or required a
large explicit mask, and both paths wandered off the official trajectory.

### Two different causes of "silent audio"

- **Masking bug (V5, numerical):** a one-bit/arithmetic difference in the
  compiled talker flipped a greedy `argmax`; the decoder left the valid
  trajectory and the audio died. Cause is numerical, symptom is silence.
- **Greedy policy collapse (behavioral):** repetition penalties of 1.05/1.1
  drove generation into a low-energy repetitive tail after ~16 s — audio that
  *looked* truncated but contained samples. Penalty 1.2 keeps energy healthy
  and reaches natural EOS (alicia: 1,216 frames / 97.28 s, itself bit-exact:
  19,456 IDs, 2,334,720 samples).

Same symptom, completely different causes. This is why "it sounds fine" is not
a correctness gate.

### Measurement discipline (earned the hard way)

- Phase timing via async CUDA events with **one terminal sync** — host timers
  measure queueing, not execution.
- Profiler traces distort the loop they observe — run separately.
- Warmup counts matter (V5.1 needed three before measurements stabilized).
- Report p50 **with phase breakdowns** — V3's headline hid a talker regression.
- Freeze input/speaker/dtype/attention/budget across versions; save the timing
  JSON *before* running parity checks so a failed check can't erase evidence.

---

## 2. Session 2026-08-21: chunked EOS, frame/dual graphs, fusion trials

### A — Chunked EOS (promoted to main, commit `801e03d`)

- *Change:* the decode loop synced to the host every frame via
  `token.eq(eos).item()`. Replaced with a device-side scan of the emitted
  `codes` rows every `EOS_CHECK_EVERY = 8` frames (plus the final frame),
  trimming at the first EOS row exactly as the per-frame check did.
- *Why it's safe:* the EOS row is written then identified; `codes[:first_eos]`
  is byte-identical to the eager emitter, so trimmed output and waveform are
  unchanged. The GPU can run up to 8 frames ahead instead of draining per
  frame.
- *Verdict:* verified on victoria with the main package parity harness —
  **codec + audio parity exact** (32 frames, 61,440 samples). The one change
  promoted from this session.

### B — Full-frame CUDA graphs: `frame-graph` (1 graph) and `dual-graph` (2 graphs)

- *Design:* the entire decode frame as one captured graph — predictor 15-step
  loop inlined, static-cache talker with fixed 2048-slot masked attention,
  codec head, f32 processors, greedy argmax; loop-carried buffers
  (token/past_hidden/position/frame/allow_eos/penalty/history) and a mask row
  gathered from a prebuilt tril table. `dual-graph` splits predictor and
  talker into two graphs (the faster-qwen3-tts structure).
- *Speed:* **neither beats the eager default — both are ~8% slower.**

| Mode | alicia 1280 (102.4 s audio) | RTF | peak | verdict |
|---|---:|---:|---:|---|
| default (promoted) | 58.28 s | 0.569 | 3037 MB | parity **exact** |
| frame-graph (1 graph) | 63.09–63.16 s | 0.616 | 3122 MB | parity fail, frame 1/cb 8 (1137 vs 564) |
| dual-graph (2 graphs) | 63.07 s | 0.616 | 3122 MB | parity fail, frame 1/cb 1 (1642 vs 957) |
| faster-qwen3-tts 0.3.2 | 66.52 s | 0.650 | 3084 MB | not bit-exact by design |

  The +8% is entirely the static 2048-slot attention (masked SDPA over the
  padded cache) vs the dynamic prefix — confirming the deep-dive prediction
  ("dynamic-cache attention ~28% faster than static full-capacity attention
  over padding" at the talker level; ~8% end-to-end). Graph topology (1 vs 2)
  is timing-neutral.
- *Output:* short samples (5.4 s) stay perfect despite the parity failure —
  Qwen3-ASR WER **0%**. Long form (1,280 frames) **collapses to digital
  silence after ~30 s** (per-quarter RMS 0.005 / 0.0 / 0.0 / 0.0) — the greedy
  low-energy collapse triggered by the diverged trajectory (section 1).
- *Verdict:* structural experiments only; the dynamic-eager default remains the
  fastest and only exact path.

**Bugs found and fixed while building B (all sandbox-only):**
- EOS (2150) leaked into fixed-token codes because the `allow_eos` flip ran
  unconditionally; the codec `chunked_decode` then hit a scatter-gather OOB.
  Fixed by gating the flip on `stop_at_eos` + clamping emitted codes to ≤2047.
- `TalkerStaticLayer` was missing the abstract `get_mask_sizes` (`CacheLayerMixin`).
- The attention mask table must be built in the model dtype (bf16), not f32 —
  SDPA rejects a bias dtype mismatch.
- `DynamicCache` in transformers 4.57.3 exposes `layers[].keys/.values`, not
  the older `key_cache` lists — `bind()` prefix copies were updated.
- Host syncs (`.item()`) are forbidden inside `torch.cuda.graph` capture.

### C — Fusion trials: torch-native and Triton

- **Embed-sum fusion (torch-native, eager default):** replaced 16 separate
  embedding lookups + `cat` + `sum` with one batched call. **Bug:** residuals
  were embedded with the *codec* table instead of their 15 own codebook tables
  → parity broke at frame 1/cb 0 (210 vs 215) and audio became garbage (WER
  9200%). The 15 residual tables are a separate `ModuleList` of `nn.Embedding`.
  Reverted the eager path to its exact original ops (parity exact again) and
  wrote a correct multi-table Triton kernel for the experimental graph modes.
- **Triton fused RMSNorm→QKV** (`rmsnorm_qkv`, one launch vs norm + 3 GEMVs +
  2 head-norms): parity max abs err 0.0039–0.0625 on real shapes (only
  0–26% of elements bit-exact); microbench **speed-neutral** (224 µs vs
  259 µs, ±13% — the M=1 path is bandwidth-bound on the weight loads, which
  fusion doesn't reduce); end-to-end in the graph mode **WER 100%** —
  accumulation-order drift compounds across 20 layers × ~64 frames and
  destroys greedy generation.
- **The verdict that matters:** in a greedy TTS decoder, kernel fusion is only
  viable if **bitwise-exact** (like torch-op-preserving reorganization).
  Approximate Triton GEMVs look fine on isolated tensors and fail end-to-end.
  In-graph launch fusion is pointless (launches are already free inside a CUDA
  graph); the costs are weight-memory traffic and the static-attention penalty.
- The companion Triton lab lives in `decode-lab` (`rmsnorm`, `swiglu` kernels
  with torch parity references) for exactly this pipeline.

### D — WER/ASR eval (new infrastructure)

- `tools/eval_asr_wer.py` transcribes TTS output with Qwen3-ASR-0.6B and reports
  WER/CER. **Config trap:** transformers < 5.13 ships no `Qwen3ASR*` classes;
  the eval imports dante's vendored `qwen_asr` package
  (`PYTHONPATH=.../dante/baseline`) and runs in dante's venv (has `nagisa`).
- Reference numbers: official-equivalent short sample WER 20% (serena voice
  quirk: "Autoregressive"→"auto recursive" — a 0.6B ASR/voice interaction, not
  a TTS defect); faster-qwen3-tts full alicia (Ryan) **4.74%**; default serena
  full alicia 38.79% — voice matters more than engine for this metric.

---

## 3. faster-qwen3-tts comparison findings

- Reported wins are upstream-measured (RTX 4090 RTF 0.82→4.78, TTFA
  800→156 ms) and **not locally reproduced**; docs flag an H100 < 4090 anomaly
  (clock-related).
- Same-GPU, matched protocol (alicia, Ryan, rp 1.2, fixed 1,279 completed
  frames, fresh process): whistle default **56.71 s / RTF 0.554** vs
  faster-qwen3-tts 0.3.2 **66.52 s / RTF 0.650** — **whistle ~14.8% faster**,
  while holding exact parity (faster does not).
- Why the head-to-head was fair for us: whistle keeps talker attention on the
  dynamic-cache path; faster uses static full-capacity attention + mask tables
  (same structural tradeoff our frame-graph measured as +8%).
- Version discipline: benchmark the installed 0.3.2, not the 0.2.6 refs
  checkout; fresh process per engine (graph capture + memory interact);
  thermal-throttle discipline (see infra notes).

---

## 4. Environment / infrastructure failures (tooling)

- `HF_HUB_OFFLINE=1` breaks `Qwen3TTSModel.from_pretrained` (the processor
  needs a hub file that isn't cached) — offline mode cannot be used for the
  load path, despite all weights being cached.
- `sox: not found` warning at qwen-tts import is cosmetic (soundfile probes).
- Flaky HF DNS on victoria intermittently re-fetches hub metadata
  ("Fetching 4 files…") — harmless but slow.
- **Syncthing lag of minutes** between a local edit and its presence on
  victoria caused multiple "fixed the bug but the remote ran the old file"
  incidents — always md5-verify the remote copy before launching GPU runs.
- victoria's 40 W mobile RTX 3050 **thermal-throttles after ~2 consecutive
  heavy runs** — order workloads, use fresh processes, report single/min
  numbers with clocks+temps logged (`nvidia-smi` SM MHz / °C).
- transformers API drift: `DynamicCache.key_cache` → `layers[].keys/.values`
  (4.57.3); `CacheLayerMixin` requires `get_mask_sizes`.
- Triton idioms that cost debugging time: no tensor indexing (`out[0, :]`
  unsupported — use a broadcast-multiply-reduce or masking); no
  `ptr.dtype.element_ty` (use the loaded tensor's dtype); branch overlap bugs
  (`is_k = pid >= heads_q` also caught v-heads → OOB `wk` reads → 3.4e38
  parity errors).

## 5. Standing conclusions

1. Scheduling and graph replay are the product; kernels are not the main lever
   at batch one, hidden 2048, on a 6 GB laptop GPU.
2. Bit-identity, not equivalence, is the only safe currency under greedy
   autoregression — the two fastest numbers in the project were deleted.
3. Static caches only pay when fixed shapes are exploited; for eager attention
   they are a measured loss, and mishandled masks make them silent.
4. The promoted path (default/predictor-ffn-graphs + chunked EOS) is the
   fastest exact configuration measured: 56.71 s / RTF 0.554.
5. Next real levers (still unproven): fused predictor kernels that preserve
   the official GEMV accumulation order, larger talker graph regions around
   shape-invariant projections, output-length-bucketed codec capture (ceiling
   ≈3.2%), on-device scheduling (ceiling ≈1.3%).
## 6. Temperature sampling (session 2026-08-21 late)

The checkpoint's `generation_config.json` ships **`do_sample: true`** (temperature
0.9, top_k 50, top_p 1.0, repetition penalty 1.05, `subtalker_*` mirrored) —
greedy was *our* benchmark contract, not the model's native mode. Implemented
as a `sampling={"temperature": …, "top_k": …}` option on the split path
(primary token via top-k + softmax/temperature + `multinomial`; residual
predictor via an eager sampled loop because the captured graphs bake greedy
`argmax`), with `top_p=1.0` a documented no-op.

Measured on alicia full (1,280 frames, 102.4 s audio, speaker Ryan, fresh
process, warmup 1):

| Mode | wall | RTF | WER | CER |
|---|---:|---:|---:|---:|
| greedy (rp 1.2) | 56.72 s | 0.554 | 3.02% | 0.82% |
| sampled t0.9/k50 (rp 1.2) | 74.16 s | 0.724 | 6.47% | 3.40% |
| sampled t0.9/k50 (rp 1.05, official recipe) | 73.55 s | 0.718 | 4.74% | 1.75% |

**Speed regression: ~+30% (56.7 → 73.5–74.2 s).** Causes, in order: (1) the
sampled predictor must run the eager 15-step loop — the CUDA graphs freeze the
greedy `argmax` — which costs ~+11 ms/frame (the V7 graph-vs-eager predictor
gap measured in `exp.md`: 27.4 vs 38.4 ms/frame) ≈ +14 s over 1,279 frames; (2)
the per-frame sampling kernels (top-k, softmax, multinomial) add ~+2–3 ms/frame.
The greedy path is unchanged.

**WER regression: 3.02% → 4.74–6.47%.** Cause: sampling occasionally draws
lower-probability tokens, producing mild mispronunciations the ASR marks as
word errors; greedy `argmax` is the ASR-friendliest policy on this model.
Notably rp 1.05 (the official recipe) beats rp 1.2 under sampling — the
official pair is tuned together.

**Voice finding (why the default changed to Ryan):** the identical greedy path
scores 38.79% WER on serena but 3.02% on Ryan. The voice embedding dominates
this metric on Qwen3-ASR-0.6B (serena's formal read degrades systematically —
e.g. "autoregressive"→"auto regressive" even on the 10-word sample). Default
speaker moved to `ryan` project-wide; the benchmark contract already used Ryan.

**Harness lesson that briefly produced a false "regression":** `tools/profile_tts.py`
takes the positional argument as the *text to synthesize*, not a file —
invoking `tools/profile_tts.py short.txt` synthesizes the literal string "short . txt"
(garbage in → garbage out, 100–150% WER). Use `--text-file`. This misuse
looked exactly like a code regression until the official API bisect proved
the pipeline clean (main split == official bitwise; sandbox == official with
the right text).

**Verdict:** sampling neither helps speed nor WER on this model; greedy + Ryan
+ rp 1.2 remains the recommended configuration (56.7 s, RTF 0.554, WER 3.02%).
Sampling stays available as a separately labeled naturalness experiment; the
official rp 1.05 pair should be used if sampling is ever enabled.

## 7. Stress battery outside alicia (session 2026-08-21 late)

Generated `testdata/*.txt` corpus (short/medium/punct/stories/repeat/too-long)
and ran the trimmed V7 path on each (Ryan, rp 1.2, natural EOS, fresh process):

| text | chars | V7 wall | RTF | WER | notes |
|---|---:|---:|---:|---:|---|
| t_short | 49 | 2.36 s | 0.557 | 0.0% | |
| t_medium | 227 | 8.57 s | 0.541 | 2.56% | |
| t_punct | 367 | 28.66 s | 0.541 | 81.4% | evAL ARTEFACT: TTS spells URLs/numbers verbatim ("h t t p s …"), raw-text WER is meaningless |
| t_story1 | 715 | 23.71 s | 0.540 | 1.55% | **parity vs official: EXACT (549 frames, 1,054,080 samples)** |
| t_story2 | 1297 | 52.99 s | 0.553 | 1.39% | |
| t_repeat | 601 | 25.53 s | 0.546 | 412% | greedy repetition collapse on deliberately repetitive input (known model behavior, not a V7 regression) |
| t_too_long | 8969 | — | — | — | clean capacity `ValueError` (>2048-token cache limit) |

Official latency on the new texts (same protocol): story1 31.82 s / RTF 0.725,
story2 84.91 s / RTF 0.887 → **V7 speedup holds outside alicia: 1.34× and
1.60×**; RTF stayed 0.54–0.56 across every text — no speed regression.

Streaming (`tools/bench_streaming.py`, chunk_size 12): TTFA 0.87–0.96 s across
short→story1; median chunk decode 540–562 ms per 1 s of audio = **1.71–1.78×
realtime**; peak ~2.2 GB. Streaming default speaker was still "serena" after
the Ryan default change — fixed during this pass.

Improvements flagged: (1) documents longer than the 2,048-token cache must be
chunked by the caller (the error is clean but undocumented); (2) repetitive
input still triggers the greedy low-energy collapse — sampling or higher
repetition penalty is the mitigation; (3) the WER bench needs text
normalization (strip URLs/numbers) before symbol-heavy text is scored.

## 8. w8a16 talker-MLP quantization lab (session 2026-08-21 late)

Sandbox `quant.py`: per-channel int8 weights (fp32 scales), dequant-to-fp16
GEMV emulation. Quantized ONLY the talker's MLP linears (self-attention
projections, norms, embeddings, codec head untouched by default); the
predictor never touched. Both variants (MLP-only and MLP+codec_head):

| variant | linears | weight bytes | wall (alicia 1280) | peak | first divergence | WER (alicia) |
|---|---:|---:|---:|---:|---:|---:|
| clean | — | — | 56.78 s | 3037 MB | — | 3.02% |
| mlp | 99 | 622.9→312.4 MB (−49.8%) | 56.79 s | 3003 MB | frame 0, cb 1 | 198.7% |
| mlp+head | 100 | 629.1→315.5 MB (−49.8%) | 57.48 s | 3003 MB | frame 0, cb 1 | 100.0% |

**Verdict: latency unchanged** (dequant-fp16 GEMVs are as fast as bf16 on
tensor cores; batch-one memory-bound anyway), memory gain ~1% of total peak;
**quality catastrophic** — per-channel int8 weight rounding perturbs the
talker hidden state enough to flip a predictor `argmax` at frame 0/codebook 1,
and the greedy trajectory falls into the low-energy collapse (WER up to
~199%). The first-divergence-is-a-residual-codebook pattern is the same
signature as V6/frame-graph/fusion failures — further evidence for the
predictor-brittleness thesis below.

## 9. Why the codebook predictor is so sensitive (thesis)

Every bit-level perturbation in this project — compiled talker (V6), static
attention (frame-graph), QKV fusion, w8a16 talker MLP — diverged FIRST at a
**residual codebook argmax**, never at the primary token. Mechanism:

1. **Sequential residual chain**: book k's token is the argmax of its own
   head, and that token is embedded into book k+1's input → one flip cascades
   through all 15 books within the frame.
2. **Frame mixing**: all 16 codes (primary + residuals) are summed into the
   next frame's talker input → a single residual flip perturbs the whole next
   trajectory.
3. **Thin argmax margins**: residual codebooks encode the hard-to-predict
   remainder of the signal; their logit margins are small, so any numeric
   noise crosses the argmax boundary easily. Primary tokens sit on smoother
   margins with a larger stable plateau.
4. **Zero tolerance of greedy**: no probability hedging — a flipped argmax is
   deterministic and permanent.
5. **Autoregressive compounding**: the frame-0 flip changes the next frame's
   hidden state, which changes the next primary AND all residuals again.

Conclusion: this model can only tolerate **bit-exact** changes anywhere in the
talker→predictor path; any approximate kernel (fusion, quantization, static
attention) must treat the residual argmaxes as the fuse that blows first.
