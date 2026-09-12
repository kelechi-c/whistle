# project notes

`docs/technical_deep_dive.md` is the comprehensive artifact: full codebase map,
benchmark contract, and every optimization (V1–V7) with diagrams. This file
stays the working jotter; the deep dive is the reference for the article.

## article and evidence map

`docs/latency_report.md` is the reviewed central comparison (2026-09-09): one table
per GPU, 0.6B/1.7B across official, Whistle, and isolated faster-qwen3-tts.
Missing 3050 upstream measurements remain explicit. It corrects upstream RTF
units, predictor sampling, streaming-policy mismatch, warmup differences,
in-process parity, and drain timing. The archived Astra/full drafts contain these
tables; the existing Notion page 9c0a1d40-576a-8245-83c4-81099fe5a052 is now
named `astra_draft_whistle` and contains the reviewed draft with images preserved.

`local/article/astra_draft_whistle.md` is the current Astra article, including the
RTX PRO 6000 0.6B/1.7B results and codec-overlap failure. The previous filename
`local/article/notion_draft_astra.md` is kept synchronized. `local/article/full_article.md`
and `local/docs/full_article.md` contain the updated technical narrative.
`local/article/kelechi_draft_whistle.md` preserves the fetched Notion "whistle draft"
prose with local image links. On 2026-09-08, Notion was read only; the user
canceled the upload. The new articles qualify in-process parity, the 1.7B
frame-budget mismatch, unequal audio return boundaries, and invalid drain
timing rather than repeating earlier overclaims.

`docs/shipping_plan.md` records the completed runtime/repository/local
separation. Codec overlap was removed from active inference and profiling on
2026-09-12; batch uses the official codec after generation, while streaming
keeps its required left-context decoder.

`local/article/notion_draft_astra.md` is a completed, separately tagged adaptation
of the user's publication draft. It includes the later timing audit: Track A
also read a pre-codec timestamp, so its ~100 ms figure is not verified TTFA.
The original draft is preserved; the Astra copy includes a hidden local
evidence map for publication preparation.

`local/article/article/whistle_v1_article.md` is the publication clone used for
the current Notion/Sciel article. Its four Luna figures show the historical
Official→V1→V2→V7 ladder, WER-vs-RTF, grouped GPU RTF, and module latency.
The plotted medians and caveats are preserved in `docs/latency_report.md` and
the benchmark JSON artifacts; figure source/audit files are under
`/tmp/whistle-figure-audit/`.
The local directory was reduced to the canonical export and image assets.
The Notion page `whistle_v1_article` (ID
`9c0a1d40-576a-8245-83c4-81099fe5a052`) contains the four uploaded figure
images after the metrics introduction. Sciel's main-path artifact is
`/home/tensor/Shared/code/miscellaneous/sciel/articles/whistle-qwen3-tts.html`
and was pushed as commits `b52bd79` and `2a44226`; its pre-existing dirty files were not
staged.

On 2026-09-12 the canonical clone was re-exported from the fetched Notion
page `whistle_v1_article` (last edited 2026-09-12T14:12:06Z). Its body matches
the page; only signed image URLs become local filenames. The article directory
now contains that markdown export and image/banner assets only. The old
three-panel image was replaced by a page-width latency ladder and a grouped
two-GPU comparison image; the article text is unchanged. The saved Victoria
measurements remain 54.3095s p50 / RTF 0.5583 and 3.02% WER / 0.82% CER. A
fresh battery-powered Victoria run later measured Whistle at 59.6935s p50 /
RTF 0.6136 with exact 1,216-frame codec/audio parity, and official at 85.3378s
/ RTF 0.8772 in one un-warmed timing pass. Fresh ASR was 3.02% WER / 0.82%
CER for both. faster-qwen3-tts remains a historical Ryan/Alicia result at
4.74% WER / 1.75% CER because its Victoria environment lacks Torch for a fresh
run; it is not a same-session comparison.

`local/article/notion_draft.md` is the publication draft; its experiments appendix
draws from `docs/failures_and_trials.md`, `refs/exp.md`, and the later Track A
battery. `local/article/notion_review.md` records remaining main-body revisions.
`local/docs/full_article.md` is the condensed technical narrative and current source
map. It separates the five-run natural-EOS timings from independent parity
checks, historical fixed-budget trials, and streaming measurements. The
1.7B timing artifacts have different frame counts and no recorded parity
check; they are performance evidence only. Local refs/sandbox evidence links
need durable published copies before sharing the articles outside the repo.

## shared runtime choices (src/whistle/config.py)

`config.py` is the single source for the checkpoint id (`CHECKPOINT`), the
canonical speaker/language defaults (`SPEAKER="ryan"`, `LANGUAGE="english"`),
and `MAX_CACHE_LEN=2048`. All entry points (the packaged CLI, profiling tools,
streaming, and server) import these instead of restating them; speaker
defaults are lowercase `ryan` everywhere (serena voice measured 38.8% WER vs
3% for ryan — reproducibility depends on this). `RuntimeConfig.checkpoint`
builds from the same constant.

## src/whistle/inference.py - official-module latency path

`tts_infer` is the batch-one greedy CustomVoice path. It accepts an already
loaded official `Qwen3TTSModel`, so checkpoint loading is outside inference
measurements.

Prompt building and prefill are shared with the streaming path through two
frozen dataclasses: `_prepare` (build_prompt + capacity check + decode-graphs
reset, returns `Prompt`) and `_prefill` (prefill forward + processors + first
token, returns `Prefill`). The batch loop (`tts_infer`) and the streaming loop
(`streaming.stream_tts`) consume these, so the two paths cannot drift.
`_maybe_eos_row` is the shared chunked-EOS scan (`EOS_CHECK_EVERY=8`): it takes
`force=` instead of `final=` — callers must force a scan whenever `codes`
becomes observable (streaming chunk boundaries force it; the batch path only
forces on the final frame).

```
official processor + prompt embeddings
  -> prefill: talker.forward(full prompt, DynamicCache)
  -> first codebook-zero token
  -> decode frame loop:
       talker.forward(one token)
         -> official greedy code predictor x 15 residual codebooks
         -> official embedding reduction and dynamic-cache talker forward
       -> float32 logits processors + argmax
       -> stop before codec EOS
  -> official speech_tokenizer.decode(all frames)
  -> waveform + codec IDs
```

Every head is greedy. Primary logits are converted to float32 before applying
the official repetition and suppression processors, then selected with
`argmax`. EOS token 2150 remains reachable even though ordinary codec tokens
end at 2047. The default repetition penalty is 1.2: lower greedy penalties
collapsed into near-silence after roughly 16 seconds on Alicia, while 1.2
retained energy and matched the official greedy runtime exactly. The output
allocation is trimmed to the natural EOS length.

The prompt block reproduces official role, language, speaker, TTS special
tokens, text, codec padding, and codec BOS. Default decode deliberately uses
the outer official talker forward so predictor cache behavior, bf16 embedding
reduction order, attention state, and hidden-state updates are identical.

## src/whistle/graphs.py - decode capture boundaries

`OfficialPredictor` and `OfficialTalker` own the default DynamicCache
correctness state. `DecodeGraphs` selects these for `official-eager`.

`predictor_loop` contains the complete greedy residual-code sequence. One
`torch.compile(mode="reduce-overhead")` callable covers the entire loop, so
Inductor owns its fusion and cudagraph tree; it is never nested inside a manual
CUDA graph. `PredictorGraph` retains the fixed input/output buffers, positions,
causal masks, and 16-position cache.

`talker_step` is the isolated inner-backbone forward. `TalkerGraph` uses its
`torch.compile(mode="reduce-overhead")` variant by default. The selectable
`cuda-graph` path instead compiles with `max-autotune-no-cudagraphs` and wraps
that callable in one manual CUDA graph. Both variants share the same stable
inputs, compileable `StaticCache`, and explicit per-position mask; the invalid
mask-free static-cache path remains excluded.

`decode_graphs` caches both objects per loaded talker. The predictor embedding
weights are stacked once for experimental static modes. Variable-length
prefill stays eager. `DecodeGraphs` is the boundary that later optimized blocks
can replace only after they pass exact parity. `DecoderFfnGraph.forward`
handles only the decode shape (seq len 1, graph captured) and delegates every
other shape to the official layer; attention output is taken positionally
(`[0]`) since attentions/hidden states are never requested.

The codec calls the official tokenizer model's `decode` implementation and
does not duplicate its chunking. CUDA events separate preparation, prefill,
decode, and codec time with one synchronization after the waveform is enqueued.
`tts_infer` returns both the waveform and the `[frames, codebooks]` codec-ID
tensor on-device. The CLI performs the terminal waveform CPU transfer solely
for WAV writing.

`tests/test_inference.py` runs this same official-module control flow with
tiny official-shaped weights and a tensor-only stub codec; it validates
forward/cache structure, not real speech quality or full-checkpoint latency.

The former model reimplementation and its fixture tooling/tests are
archived under the ignored `stash/nero_reimplementation/` directory.
Active code imports model components only from the installed `qwen-tts`
library. The packaged `src/whistle/cli.py` owns model loading, WAV output, and the
CLI; `src/whistle/inference.py` contains request scheduling; and
`src/whistle/graphs.py` owns reusable decode state. The `tools/profile_tts.py`
compares fixed-length CustomVoice split and official runs.

`tools/profile_tts.py --backend split --check-codec-parity` performs one untimed
official greedy generation after the measurements. It accounts for the
official selected-token/complete-frame offset and requires exact codec shape,
ID, waveform shape, and sample equality. `--talker-mode` defaults to
`official-eager`; `compile` and `cuda-graph` are experimental.

## inference optimization references

`docs/qwen3_tts_official_vs_faster.md` compares the inspected official and
faster-qwen3-tts snapshots. The faster project retains the official model but
replaces nested Hugging Face decode scheduling with a dynamic prefill followed
by static-cache CUDA graphs: one graph for a talker token and one graph for the
complete 15-token residual predictor. Its incremental codec path uses 25-frame
left context. The report also records its batch-one/CUDA-only constraints,
fixed predictor sampling policy, remaining host synchronizations, reported
benchmarks, and the optimization boundaries worth carrying into Whistle.

The v4.1 dynamic/eager-talker diagnostic is noncompetitive: its headline p50
was 91.148 s, while its separate modular median was 100.541 s. Dynamic KV
growth changes buffer addresses and incurs per-step concatenation/reallocation;
the modular measurements also increased across requests. V5 remains the
preferred measured configuration.

The V6 compiled static-cache talker reached 56.577 s p50 but is invalid:
official codec IDs first diverge at frame 1/codebook 13. Frame 0 and the first
13 codebooks of frame 1 match, implicating a small compiled talker hidden-state
difference that later flips a greedy predictor argmax and then compounds.

The repaired greedy reference uses repetition penalty 1.2 and naturally emits
1,216 Alicia frames (97.28 s). All 19,456 codec IDs and 2,334,720 waveform
samples match the official greedy runtime exactly.

The technical report treats numerical codec-token divergence and greedy
low-energy collapse as separate failures with separate causes.

`sandbox/latency_lab/` is an isolated optimization branch of the runtime. Its
best experiment removes nested predictor scheduling with per-codebook eager
CUDA graphs and graphs only the talker's fixed-shape residual FFN blocks. It
retains full 1,280-frame codec and waveform parity while reducing the measured
wall time from an 80.198-second eager reference to 56.936 seconds p50.

The historical fixed-budget V7 benchmark uses Ryan and 1,279 complete frames
for parity with 1,280 official selected tokens. Its three-run p50 is 56.858
seconds at 0.556 RTF; the artifact is
`evidence/v7_exact_graphs_0.6b_alicia.json`.

V7 is promoted to `src/whistle/graphs.py` and is the default `tts_infer` mode.
`PredictorGraphs` owns the prefix-visible predictor cache and its 15
per-codebook graphs; `DecoderFfnGraph` wraps each talker layer while leaving
prefill and variable-length attention eager. `official-eager` remains the
explicit reference mode.

The standalone `mini_qwen3` LLM package moved from `hoot/` to
`/home/tensor/code/ml/decode-lab/src/qwen3_lm` (renamed `qwen3_lm`). It is the
triton kernel surgery lab whose kernels will eventually port back into
Whistle's talker FFN and residual predictor.

## src/whistle/streaming.py + server.py - streaming path

`stream_tts` reuses `_prepare`/`_prefill`/`_maybe_eos_row` from inference.py —
same V7 graphs, same EOS trim semantics, but yields every `chunk_size` frames.
EOS scans are forced at each chunk boundary so a yielded chunk can never
contain stale EOS frames; between boundaries the scan stays on the 8-frame
cadence (no per-frame host sync). Incremental codec decoding keeps a 25-frame
left context and trims its warmup samples per chunk. An EOS hit trims
`frame_count` back to the EOS index; a chunk is yielded only if it still
contains frames, and the loop breaks with `final=True`.

`server.py` serializes generation behind a module-level `threading.Lock`
(`_generate_lock`): decode mutates shared per-model state (rope deltas, graph
input buffers, talker cache), so concurrent requests would interleave writes
into the same CUDA graph buffers. Requests queue on the lock; one GPU serves
one synthesis at a time.


## Track A (2026-08-29, sandbox/track_a — verified on victoria)

`local/docs/literature_landscape.md` maps the field (nari-labs serving SOTA, M*,
megakernels, speech spec-decoding). Track A implements the parity-safe wins:

- **streaming.py**: `ramp_frames=(2,4,8)` chunk-boundary schedule (first
  chunks ship small, later chunks grow to steady `chunk_size`) + RMS
  leading-silence trim on the first chunk + incremental transposed-buffer
  fill. TTFA 507→97 ms (short), 509→100 ms (medium), 146 ms (alicia); cadence
  and emitted codes unchanged; WER CER 0.82% = batch canonical.
- **graphs.py**: `sample_token` + lazily captured second 15-graph predictor
  set per `(temperature, top_k)`; RNG-in-graph gives fresh draws per replay.
  Sampled fixed-1280: 78.4 s → 58.7 s (overhead vs greedy +38% → +3.2%).
- **inference.py**: `temperature`/`top_k` threading; the rejected codec-overlap
  side-stream experiment was removed entirely from active code on 2026-09-12.
  Historical measurements remain in the archived reports and local raw runs.
- KV-prefix caching skipped by analysis (fixed prefix = 8-9 of 20-200+
  prompt tokens; ceiling ~10-50 ms, not worth split-prefill risk).
- Gates all green: parity32 + alicia natural-EOS exact, alicia regression
  rtf 0.555, unit test pass. Battery: `sandbox/track_a/results/battery.log`.
- **Promoted 2026-08-29** to `src/whistle` (graphs/inference/streaming +
  profile CLI --temperature/--top-k) and re-verified on
  victoria: parity32 + alicia natural-EOS exact, sampled-graphs spot +4%,
  TTFA 103.9 ms on the main path (tools/run_promo_check.sh).
- Stage 0 (pytorch-only numerics probe, `sandbox/stage0/`): ULP-level drift
  diverges greedy within frames; collapse into the silence attractor is
  stochastic (~half of trials), not thresholded. Verdict RED for greedy
  kernels; v2 kernel = sampled-only edition (see `local/docs/kernel_plan.md`).

`local/docs/kernel_plan.md` drafts the V8 persistent-kernel path (Track B): memory
arithmetic puts the honest 3050 ceiling at ~1.5-1.75× (predictor re-reads its
weights 15×/frame — sequential dependency makes that traffic compulsory);
Stage 0 sensitivity probe (correct fp32-accum GEMV, is 1e-3 drift stable over
102 s greedy?) decides feasibility before any kernel engineering.

## Microarchitectural Profiling Traces & GPU Bandwidth (2026-09-03)

- **Victoria RTX 3050 6GB Laptop GPU Bandwidth**:
  - Bus: 96-bit GDDR6 @ 5486 MHz (11 Gbps), theoretical peak 131.66 GB/s.
  - Empirical D2D copy bandwidth: 122.39 GB/s (92.9% of theoretical peak).
  - Triad (axpy) bandwidth: 123.88 GB/s; vector add: 121.67 GB/s; scale: 119.40 GB/s.
  - Context: Decode is strictly memory-bandwidth bound (GEMV weight loads dominate); eliminating CPU launch bubbles allows near-peak bandwidth saturation.
- **Trace Analysis (Official vs Whistle 16-frame fixed-pass)**:
  - Official: 2,318 ms wall, 1,761 ms GPU span, 661.8 ms GPU busy, 1,099.3 ms idle bubbles (37.6% duty cycle), 98,825 `cudaLaunchKernel` calls, 565 stream syncs, 375k CPU ops.
  - Whistle: 676 ms wall, 942.7 ms GPU span, 656.7 ms GPU busy, 286.0 ms idle (69.7% duty cycle), 23,229 `cudaLaunchKernel` calls, 660 `cudaGraphLaunch` calls, 87 stream syncs, 99k CPU ops.
  - Speedup: 3.43× wall time, 3.84× bubble reduction, 4.25× fewer kernel launches.
  - Visual artifacts: `docs/inference_trace_comparison.html` + 2x retina screenshots in `out/traces/`.
- **1.7B Variant Benchmark (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, Alicia text, 3 runs p50)**:
  - Official Baseline: 108.415 s p50, RTF 1.060, throughput 0.944× (slower than real-time).
  - Whistle V7: 77.448 s p50, RTF 0.756, throughput 1.322× (faster than real-time, real-time capable).
  - Delta: −30.967 s eliminated (−28.56% latency reduction, +40.04% throughput boost).
  - VRAM: 4,964 MB allocated (fits within 6 GB VRAM with `expandable_segments:True`).

  - Alicia 1,279 frames: 91.722 s official → 71.382 s V1 = 20.340 s cut (22.18% reduction, +28.5% throughput).
- **V3 Dynamic Talker Cache A/B & V7 Lineage**:
  - V3 A/B (50.011 s) cut talker step by 28.43% (18.685 s vs 26.106 s) by using `DynamicCache` over active prefix rather than 2,048-slot padded `StaticCache`.
  - V7 is built directly on V2's dynamic-cache eager talker foundation, adding CUDA graphs only to fixed-shape blocks: 15 predictor codebook graphs (`PrefixStaticLayer`) + 28 talker layer residual FFN graphs (`DecoderFfnGraph`).
- **V5 Failed Config Probe (Greedy vs Sampling)**:
  - Probe: `sandbox/probe_v5.py` on Victoria (compiled predictor loop + static talker graph with `attention_mask=None`).
  - Greedy decoding: Collapses to digital silence after 2 seconds (RMS: [0-2s]=0.0503 → [2-4s]=0.0013 → [4-5.1s]=0.0011; 98% energy loss). Repetitive pad/silence token loops.
  - Sampled decoding (t=0.9, top_k=50): Avoids the silence attractor (RMS: [0-2s]=0.0968, [2-4s]=0.1154, [4-5.1s]=0.1331). Emitted token entropy maintained.

## modal benchmark plan

`local/docs/modal_benchmark_plan.md` is the executor handoff for Modal setup,
RTX PRO 6000 correctness gates, matched 3050 comparisons, benchmark timing
repairs, and article evidence retrieval. The Modal execution artifacts are
tracked in `local/prose/rtx6k_run.md` and the result reports below.

`local/prose/rtx6k_run.md` is the live Modal run log. `local/docs/modal_results.md` is the
compiled technical report, and `local/article/modal_results.md` is the compact
publication supplement. They currently contain completed 0.6B and 1.7B Alicia
batch runs, codec-overlap A/B metrics, official-vs-Whistle streaming latency,
parity gates, and the Modal billing report. `local/tools/modal_bench.py`,
`local/tools/modal_faster_bench.py`, and `local/tools/modal_stream_bench.py`
are the archived Modal adapters for those runs.

`local/tools/victoria_modular_bench.py` runs the same fixed 128-frame CUDA-event module
check over SSH on Victoria. Its saved JSON files under `evidence/` separate
predictor and talker spans for both Whistle and faster-qwen3-tts.

`local/prose/mfu_report.md` records the corrected Victoria official-versus-Whistle fixed-
work MFU measurement. It uses eager shape-counted dense BF16 FLOPs, fresh
post-warmup telemetry, saved Alicia p50 wall times, and the observed bandwidth
calibration.

That provisional MFU report is withdrawn pending the audit in
`local/docs/mfu_measurement_plan.md`: sourced dense GPU peaks, operation accounting,
matched timing and fresh official baselines are required. The plan separates
batch-one measurement from an optional new true-batching implementation.

## publication artifacts

`local/article/article/whistle_v1_article.md` is the local article clone used
for publication. The main-path Sciel copy is
`../miscellaneous/sciel/articles/whistle-qwen3-tts.html` with its assets under
that repository's `articles/images/` directory.

The current Notion export is `local/article/article/whistle_v1_article.md`.
The Luna figure set in that directory is `whistle_wer_rtf_scatter`,
`whistle_gpu_rtf_comparison`, `whistle_module_latency_victoria`, and
`whistle_module_latency_modal` (PNG/SVG). Their audit is
`/tmp/whistle-figure-audit/audit.json`: WER uses Qwen3-ASR-0.6B, official and
Whistle are fresh at 3.0172% WER / 0.8239% CER, and faster's 4.7414% /
3.0896% is historical v0.3.2 quality paired with v0.4.0 timing. Official
module latency is unavailable and is explicitly labelled in the figures.
The current Sciel HTML was rebuilt from this export and pushed as commit
`2a44226`; only image assets and shell metadata/layout changed.

The plot revision on 2026-09-12 regenerates all four figures with a muted-white
background (`#f3f1ed`), no WER/config footnotes, compact module canvases, and
bold `0.6B`/`1.7B` headings. The WER scatter labels are runtime-only; its
historical faster-qwen3-tts WER/timing provenance remains in the audit. The
updated PNGs are used by both Notion and Sciel.

The second 2026-09-12 revision replaces the ad-hoc `/tmp` script with
`tools/make_figures.py`, which reads the evidence JSON directly (plus the Modal
rows of `docs/latency_report.md`) and writes PNG + SVG for all four figures. It
uses SF Pro, resolved from `tools/fonts`, the SF Pro Display download, or
Sciel's own `sfpromedium.otf`, and fails loudly rather than substituting a
different sans. The WER chart is a labelled scatter, the GPU RTF chart is a
dot plot over one shared scale, and both module charts are predictor-to-talker
dumbbells (filled = predictor, hollow = talker) instead of hatched bars. Axis
labels no longer say "median", tick and label spacing is set from explicit
`subplots_adjust` geometry, and SVG keeps glyph outlines so it renders without
SF Pro installed. The Sciel HTML gained matching `figcaption`s, and the Notion
markdown gained the equivalent italic captions. Because the site inverts plot
PNGs on dark palettes, the series ramp stays grayscale with Whistle darkest, so
emphasis survives inversion in every palette/mode combination.
