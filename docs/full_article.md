# Whistle — Qwen3-TTS at 1.8× real-time on a 6 GB laptop GPU

Whistle schedules the official `Qwen3-TTS-12Hz-0.6B-CustomVoice` model more efficiently without replacing its weights or model implementation. On my RTX 3050 Laptop GPU, it generates **97.28 seconds of speech in 54.31 seconds**, compared with **84.05 seconds** for the official runtime: **35.4% less wall time**, or **1.55× the throughput**. Its generation rate is **1.79× real-time**.

The default greedy batch path has separately passed exact codec-token and waveform comparisons against the official greedy configuration. Streaming reaches roughly **100 ms to first audio on short inputs**, but uses a different codec chunk schedule and does **not** promise bit-identical batch waveforms.

This began as a learning project: how much latency could I remove using PyTorch's scheduling and CUDA graph tools? The useful result was learning which boundaries could be optimized while preserving the reference output.

## What the numbers mean

The headline uses five measured runs per engine, excluding warmup and model loading, with `alicia.txt`, Ryan, English, bfloat16, SDPA, batch one, greedy decoding, repetition penalty 1.2, and natural EOS on the RTX 3050 Laptop 6 GB.

| Metric | Official | Whistle |
|---|---:|---:|
| Median generation time | 84.05 s | 54.31 s |
| Audio duration | 97.28 s | 97.28 s |
| Real-time factor: generation time / audio duration | 0.864 | 0.558 |
| Audio seconds generated per wall second | 1.16× | 1.79× |

Sources: [official five-run results](../benchmarks/official_current_p50_5runs.json) and [Whistle five-run results](../benchmarks/v7_current_p50_5runs.json). Those timing files have `codec_parity_checked: false`; the separate [validation log](../sandbox/track_a/results/battery.log) records exact equality for **1,216 frames, 19,456 codec IDs, and 2,334,720 waveform samples**. Timing and correctness are separate pieces of evidence.

The older optimization ladder used 1,279 completed frames, or 102.32 seconds of audio. Later forced-budget experiments often used 1,280 frames, or 102.40 seconds. Those runs explain design choices; their times should not be mixed with the natural-EOS headline.

## Why one audio frame creates so much work

The checkpoint name says `12Hz`; its codec emits **12.5 frames per second**, or one frame per 80 ms of audio. Each frame contains 16 codebook tokens: the talker supplies the primary token and a hidden state, and the code predictor generates the remaining 15 tokens sequentially. The official speech tokenizer converts the completed codes into 24 kHz audio.

```text
prompt → talker prefill → primary token + hidden state
  → predictor: two-token seed, then 14 single-token forwards
  → completed frame: 1 primary + 15 residual codes
  → embed each code through its own table, concatenate, sum
  → next talker step → next primary token + hidden state
  → repeat until EOS → codec decoder → waveform
```

For the 0.6B configuration, 20 talker layers plus 15 passes through a five-layer predictor means approximately **95 transformer-layer evaluations per frame**. That is a layer count, not a CUDA launch count: each layer dispatches several operations. Prompt prefill and the final-frame boundary add small differences to this steady-state accounting.

The historical modular profile attributed about **72% of decode time to the predictor** and **27% to the talker**. Each residual prediction depends on the previous token, so those 15 steps cannot simply be evaluated in parallel. At batch one, the individual matrix-vector operations do little work per launch, and the official nested generation loops add CPU dispatch and bookkeeping between them.

The traces show why scheduling mattered:

![Official runtime trace with frequent dispatch gaps](../article/officialprofile.png)

![Whistle trace with reduced gaps between GPU operations](../article/whistleprofile.png)

There are two costs here: gaps while the GPU waits for submitted work, and memory traffic while the kernels execute. CUDA graphs reduce the first; they do not eliminate the second. Profiler traces illustrate the gaps but are separate from the unprofiled latency measurements.

## The optimizations that survived

### Replace nested generation bookkeeping

V1 replaced the nested Hugging Face generation machinery with explicit prompt preparation, prefill, and decode scheduling around the official modules. The historical wall time fell from **91.72 to 71.38 seconds**. This is an explicit inference loop, not prefill/decode disaggregation across workers or devices.

V2 preallocated intermediates, kept token work on the GPU, and removed unnecessary host round trips. Wall time fell to **67.97 seconds**. These changes also prepared the fixed buffers needed for graph capture. Sources: [baseline](../benchmarks/official_tts_0.6b_alicia.json), [V1](../benchmarks/faster_decode_0.6b_alicia.json), and [V2](../benchmarks/v2_faster_decode_0.6b_alicia.json).

### Capture the fixed parts of decode

A CUDA graph records a sequence of GPU operations and replays it with less CPU dispatch overhead. It preserves the captured operations; it does not automatically fuse them, reduce their memory traffic, or guarantee that the GPU stays fully occupied.

V7 captures two kinds of region:

- **Predictor:** 15 graphs, one per residual position. Each position has a known attention-prefix length. `PrefixStaticLayer` writes KV states into stable backing storage while exposing only the populated prefix, preserving the reference attention shapes.
- **Talker:** one graph per layer around post-attention normalization, the MLP, and residual addition. For the 0.6B configuration this is 20 graphs. The one-token region has fixed shape; variable-length attention remains eager with `DynamicCache`.

That means about **35 graph replays per steady-state frame**, plus eager operations—not two whole-frame graph launches. Prefill also stays eager because its length depends on the request. Capture is reused across requests, while caches and logical positions are reset.

The historical [V7 benchmark](../benchmarks/v7_exact_graphs_0.6b_alicia.json) measured **56.86 seconds**, down from 91.72 seconds for that protocol. Its distinguishing feature is the capture boundary: stable storage and fixed operation shapes without exposing padded attention positions or changing the eager arithmetic.

### Check EOS less often

Reading `token.eq(eos).item()` every frame drains the GPU queue to answer a CPU question. The promoted loop checks for EOS every eight frames and trims the completed output at the first EOS row. It can compute ahead, but those extra rows never reach the codec output. Streaming forces an additional check whenever a chunk becomes visible to the caller.

This preserves stopping semantics while reducing synchronization. It does not remove every host synchronization from the program.

## Experiments and failures

These results come from [failures_and_trials.md](../failures_and_trials.md), the [historical experiment record](../refs/exp.md), and the later [Track A battery](../sandbox/track_a/results/battery.log). The relevant question for each trial is what it changed, what failed, and what the result actually establishes.

### Compilation helped performance, but did not establish the final parity contract

V3's compiled predictor reduced historical wall time to **57.48 seconds**, while its static talker cache increased the talker phase by about **31%**. A [dynamic-talker A/B](../benchmarks/v3_dynamic_talker_cache_ab.json) reached **50.01 seconds**, demonstrating how much padded attention could cost.

That diagnostic is faster than V7. The saved A/B timing file does not establish independent full-sequence exact parity, so it should not be presented as a validated predecessor that V7 somehow outperformed. Historical notes use “valid” inconsistently; the final claim rests on the explicit parity gate.

V6's compiled static talker measured **56.58 seconds** but first differed at **frame 1, codebook 13**. Changed numerical execution is consistent with this failure, but the mismatch alone does not isolate a particular compiler reduction. Compilation is not categorically incompatible with exactness; these candidates failed the required comparison.

### V5: missing cache masking bought a false speedup

V5 measured **48.11 seconds**, then produced near-silence. Its static-cache attention path used `attention_mask=None`, exposing unused padded positions. Attention over those extra positions changes the normalization and therefore the hidden state. The explicit-mask repair, V5.1, took **64.19 seconds**.

This is a semantic attention bug, not merely harmless floating-point drift. A short follow-up probe found that sampling avoided the low-energy loop, but producing audible speech did not repair the attention or recover reference parity.

A different failure occurred with **correct greedy execution and repetition penalties 1.05 or 1.1**: Alicia developed a low-energy repetitive tail. Penalty 1.2 reached natural EOS and matched the equally configured official run. It helped this input; deliberately repetitive text still failed later. Silence alone does not identify the cause.

### One graph or two graphs still paid for padded attention

Later full-frame and dual-graph variants both used static attention over 2,048 slots. They took about **63.1 seconds**, versus **58.28 seconds** for the default in that session, and failed token parity. Short output had 0% ASR WER, yet long output collapsed toward silence after roughly 30 seconds.

The near-identical timings for one and two graphs suggest topology was not the decisive cost. The static-attention workload remained expensive after CPU dispatch was reduced. Short ASR success also failed to predict long-form stability.

### Fusion: distinguish a wrong implementation from changed numerics

An embedding-sum rewrite accidentally used the primary codec table for residual tokens, which each have their own embedding table. Reverting the lookup and preserving the official reduction restored parity. This was an indexing/design bug, not evidence that embedding fusion is impossible.

Other trials tested numerical changes:

| Trial | Observation | Decision |
|---|---|---|
| Combined Q/K/V projection | First mismatch at frame 6, codebook 1 | Reject under exact-parity contract |
| Combined MLP gate/up projection | First 64 frames exact; about 0.4% gain at 128 frames | Leave experimental; no independent full-sequence gate |
| Triton RMSNorm–QKV fusion | No convincing microbenchmark win; 100% WER in the tested graph path | Reject this implementation |

The original Triton measurements showed output differences around 0.004–0.06. A small isolated tensor error is insufficient validation for a long autoregressive sequence. Graph replay already reduces host launch overhead, so a fused kernel must justify itself through remaining execution costs as well as correctness.

### Quantization saved stored weight bytes, not useful latency

The W8A16 trial stored selected talker weights as per-channel int8, then dequantized for fp16 computation. Stored weights fell from about **623 to 312 MB**, but peak allocated memory barely changed (**3,037 to 3,003 MB**) and wall time stayed near **56.8 seconds**. Parity failed at frame 0, codebook 1; the MLP-only run scored **198.7% WER**.

This rejects the tested dequantization approach and its quality tradeoff. It does not establish that an optimized integer kernel, another quantization scheme, or a retrained model could never help. WER can exceed 100% because insertions count as errors against a fixed reference length.

### Codec overlap competed with decode

Running incremental codec work on a second CUDA stream increased wall time from **56.92 to 64.28 seconds** on the 3050. Resource contention outweighed the roughly 1.8-second batch codec phase it could hide.

The tested chunk schedule also changed the waveform: the 64-frame comparison had maximum absolute sample error **0.04425**, despite equal shapes. The overlap option remains default-off. Whether a larger GPU benefits requires another measurement.

### Numerical sensitivity is a finding, not an impossibility proof

The later Stage 0 notes describe replacing selected linear operations with fp32 computation followed by bf16 rounding, and injecting noise before graph capture. Several trajectories diverged within a few frames and collapsed; others diverged but retained healthy energy. The probe code is in [sandbox/stage0](../sandbox/stage0/).

These observations justify full-length testing and caution around arithmetic changes. They do not establish a universal safe error threshold, a 50% failure probability, or that every custom kernel must fail. Likewise, early residual-codebook mismatches suggest a sensitive part of the sequence, but the proposed explanation of smaller residual logit margins needs direct margin measurements.

## Streaming and sampling after V7

### Smaller first chunks reduced time to first audio

The current streaming schedule emits at cumulative frame counts **2, 4, and 8**, then every 12 frames. It also copies only newly produced codes into the codec buffer and optionally trims leading silence from the first chunk.

The saved [battery](../sandbox/track_a/results/battery.log) measured short-input first audio at **97–100 ms**, compared with **502–507 ms** for fixed 12-frame chunks. Medium input measured **100.4 ms**, compared with **509.4 ms**. These are warm local generation measurements, not cold-start, network, playback, or p95 serving latency. On the short sample, disabling silence trimming barely changed TTFA; the earlier chunk boundary accounts for the main improvement.

Streaming retains left context for incremental codec decoding, but its attention context and chunk boundaries differ from batch decoding. Batch waveform equality must not be claimed for this path. The battery's streaming-versus-batch comparison also contains a malformed shape comparison, so that line is not usable parity evidence.

The separate [streaming ASR result](../sandbox/track_a/results/wer_stream_ramp.json) reports **3.02% WER and 0.82% CER** on Alicia with Qwen3-ASR-0.6B. That is an intelligibility check on one sample, not a listening study or proof of waveform equality.

### Sampling no longer requires an eager predictor

The early sampled path fell back to an eager 15-step predictor and cost roughly 30% more than greedy. A later matched battery measured **78.3–78.5 seconds** for eager sampling and **58.7–59.0 seconds** after capturing a second predictor graph set, versus **56.8–57.0 seconds** for greedy.

The graph set is selected by temperature and top-k. Random operations advance across replays; the battery confirms that two sampled runs produce different codes. This recovers most of the scheduling benefit, but is not evidence of exact sampled-output parity with the official engine. The older claim that “sampling is slow because graphs bake in argmax” describes the superseded implementation.

## Limits and generalization

The stress notes report RTF around **0.54–0.56** on additional short and story inputs, with story speedups of **1.34× and 1.60×** over the official runtime. Deliberately repetitive text still caused a greedy collapse. Symbol-heavy text scored poorly under raw-text WER, illustrating why spoken-form normalization matters for numbers and URLs. Ryan and Serena also produced very different ASR scores; the notes do not isolate synthesis quality from recognizer error.

The 1.7B timing artifacts show **108.42 seconds / RTF 1.060** for the [official runtime](../benchmarks/official_1.7b_3runs.json) and **77.45 seconds / RTF 0.756** for [Whistle](../benchmarks/v7_1.7b_3runs.json), with roughly 4.96 GB allocated. However, they contain **1,279 versus 1,280 output frames**, and both record `codec_parity_checked: false`. They suggest useful performance on the larger checkpoint, but are not an exact-output comparison.

The older same-GPU comparison notes record Whistle at 56.71 seconds and faster-qwen3-tts **0.3.2** at 66.52 seconds under their stated protocol. That is a version-specific local result, not a ranking against current runtimes or evidence about faster GPUs. Cross-GPU claims and serving leaderboards need their own matched measurements.

## How the implementation fits together

The source remains the reference for exact signatures and control flow; duplicating hundreds of lines here made the article harder to follow and allowed old code to contradict later changes.

| Component | Responsibility and connection |
|---|---|
| [config.py](../src/whistle/config.py) | Shared checkpoint, speaker, language, and cache-capacity defaults used by entry points |
| [graphs.py](../src/whistle/graphs.py) | Prefix-visible predictor storage, greedy/sample graph sets, talker FFN capture, and per-model reusable decode state |
| [inference.py](../src/whistle/inference.py) | Prompt construction, prefill, frame scheduling, logits processing, EOS trimming, and batch codec decode |
| [streaming.py](../src/whistle/streaming.py) | Reuses request preparation and decode machinery, then emits incremental codec audio at chunk boundaries |
| [server.py](../src/whistle/server.py) | Serializes access to shared model/cache/graph buffers and streams audio to clients |
| [profile_tts.py](../profile_tts.py) | Times both engines, saves phase results, and runs an untimed official parity comparison |

To preserve the greedy reference, the scheduler must carry the prompt layout, position state, per-codebook embedding tables, embedding reduction order, float32 logits processing, and EOS rules faithfully. Stable graph storage is useful only when cache visibility and attention semantics also match. Shared mutable graph buffers require serialized requests.

For the broader code map, see [project.md](../project.md). For historical implementation detail, see [technical_deep_dive.md](technical_deep_dive.md); current source takes precedence over old snapshots.

## Reproducing the headline

Run on the GPU machine, with the environment installed and the model available. These commands request five measured runs after one warmup; they have not been rerun for this editorial update.

```bash
uv run --no-sync python profile_tts.py --text-file alicia.txt --backend split \
  --max-new-tokens 1280 --iterations 5 --warmup 1 --speaker ryan \
  --check-codec-parity --json-out benchmarks/article_v7.json

uv run --no-sync python profile_tts.py --text-file alicia.txt --backend official \
  --max-new-tokens 1280 --iterations 5 --warmup 1 --speaker ryan \
  --json-out benchmarks/article_official.json
```

Use natural EOS on both sides; forced-budget timing alone does not prove official parity. Verify the synced source before remote execution, run each engine in a fresh process, and track GPU clocks and temperature because this laptop throttles under sustained load. Keep profiling separate from timing and inspect phase measurements alongside p50.

Whistle's result is a measured reduction in scheduling overhead with an explicit correctness boundary: roughly 1.8 audio seconds per wall second for the tested 0.6B workload, exact greedy batch output in the recorded checks, and a separately evaluated low-latency streaming path. The failed experiments explain why those boundaries exist.
