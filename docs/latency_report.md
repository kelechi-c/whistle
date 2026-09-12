# Whistle latency report

Reviewed 2026-09-09. Models are **0.6B and 1.7B** (there is no 0.7B checkpoint in this campaign). This report consolidates saved Modal and Victoria measurements.

## Core comparison: RTX 3050 Laptop, 6 GB

Alicia batch generation; historical official/Whistle measurements. Missing means no comparable raw result was located, not zero latency.

| Model | Runtime | Measured passes | Median wall (s) | Wall range (s) | Audio (s) / frames | Median RTF ↓ | Median × real-time ↑ | Peak allocated MiB |
|---|---|---:|---:|---|---|---:|---:|---:|
| 0.6B | Official | 5 | 84.052 | 83.699–84.502 | 97.28 / 1,216 | 0.864 | 1.157 | 2,837 |
| 0.6B | Whistle | 5 | 54.310 | 54.255–54.343 | 97.28 / 1,216 | 0.558 | 1.791 | 3,037 |
| 0.6B | faster-qwen3-tts v0.4.0 (Victoria) | 3 | 66.268 | 66.253–66.454 | 102.40 / 1,280 | 0.647 | 1.545 | Not recorded |
| 1.7B | Official | 3 | 108.415 | 107.758–108.645 | 102.32 / 1,279 | 1.060 | 0.944 | 4,763 |
| 1.7B | Whistle | 3 | 77.448 | 77.440–77.471 | 102.40 / 1,280 | 0.756 | 1.322 | 4,964 |
| 1.7B | faster-qwen3-tts v0.4.0 (Victoria) | 3 | 86.966 | 63.356–86.981 | 73.84–102.40 / 923–1,280 | 0.849 | 1.177 | Not recorded |

Older narrative notes mention faster-qwen3-tts **v0.3.2 at 66.52 s**, paired with Whistle at 56.71 s. The new Victoria v0.4.0 rows above supersede that undocumented note for the current table. The 3050 1.7B official/Whistle timing files still have different frame counts and no recorded parity check.

## Core comparison: RTX PRO 6000 Blackwell Server Edition

Alicia, Ryan, English, bf16, batch one. Each row below ran without another model process sharing its GPU. The models were requested with a 1,280-token/frame cap, whose meaning differs between entrypoints. Loading and warmups are excluded.

| Model | Runtime | Measured passes | Median wall (s) | Wall range (s) | Audio (s) / frames | Median RTF ↓ | Median × real-time ↑ | Peak allocated MiB |
|---|---|---:|---:|---|---|---:|---:|---:|
| 0.6B | Official | 5 | 69.517 | 68.308–71.189 | 97.52 / 1,219 | 0.713 | 1.403 | 2,860 |
| 0.6B | Whistle, overlap off | 5 | 25.386 | 25.139–25.693 | 97.52 / 1,219 | 0.260 | 3.841 | 3,133 |
| 0.6B | faster-qwen3-tts v0.4.0, alone | 5 | 20.412 | 17.539–20.513 | 86.08–102.40 / 1,076–1,280 | 0.199 | 5.016 | Not recorded |
| 1.7B | Official | 5 | 68.327 | 67.394–70.424 | 102.32 / 1,279 | 0.668 | 1.498 | 4,792 |
| 1.7B | Whistle, overlap off | 5 | 27.139 | 26.900–27.574 | 102.40 / 1,280 | 0.265 | 3.773 | 5,065 |
| 1.7B | faster-qwen3-tts v0.4.0, alone | 3 | 24.072 | 23.806–24.157 | 102.40 / 1,280 | 0.235 | 4.254 | Not recorded |

**RTF = wall time / audio duration**; × real-time is its inverse. Both are medians of per-trial ratios. With variable output duration, the ratio of independently computed medians need not equal median RTF. Peak memory is the maximum per-trial allocated memory divided by 2²⁰, not reserved memory or full device usage.

On Modal's PRO 6000, isolated upstream medians were lower than Whistle's by about **19.6% wall time for 0.6B** and **11.3% for 1.7B**. On Victoria's RTX 3050, upstream was slower than Whistle by about **22.2% for 0.6B** and **12.1% for 1.7B** at the saved output lengths. These are observations across different implementations/settings, not validated equal-output speedups.

## Modular GPU check: Modal RTX PRO 6000

I ran a fixed 128-frame CUDA-event diagnostic after a 32-frame warmup. One
predictor call means the complete 15-residual-codebook predictor sequence for
one frame. One talker call means the next-frame talker decode step, including
its attention and the captured or eager post-attention path. Times are summed
GPU event durations; they are not end-to-end request times.

| Model | Runtime | Predictor calls | Predictor total | Predictor / call | Talker calls | Talker total | Talker / call |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.6B | Whistle | 128 | 1,275.3 ms | 9.96 ms | 127 | 1,266.1 ms | 9.97 ms |
| 0.6B | faster-qwen3-tts | 128 | 1,353.1 ms | 10.57 ms | 128 | 653.4 ms | 5.10 ms |
| 1.7B | Whistle | 128 | 1,274.9 ms | 9.96 ms | 127 | 1,264.6 ms | 9.96 ms |
| 1.7B | faster-qwen3-tts | 128 | 1,530.1 ms | 11.95 ms | 128 | 852.5 ms | 6.66 ms |

The faster implementation's talker step is substantially shorter in this
diagnostic, while its complete predictor sequence is slightly longer. This
supports the idea that its larger talker graph can win on the PRO 6000. It
does not explain the Victoria reversal by itself: the modular check was only
run on Modal, and the two packages use different cache, attention, and
predictor policies. Raw checks are `../local/benchmarks/modal/modular_whistle_0.6B.json`,
`modular_whistle_1.7B.json`, `modular_faster_0.6B.json`, and
`modular_faster_1.7B.json`.

## Modular GPU check: Victoria RTX 3050

The same fixed 128-frame CUDA-event diagnostic was run on Victoria after a
32-frame warmup. Whistle's talker wrapper uses eager dynamic-cache attention
with graph-captured FFN regions; faster-qwen3-tts uses one static-cache CUDA
graph for the complete talker step. Predictor calls contain all 15 residual
codebook positions in both implementations.

| Model | Runtime | Predictor calls | Predictor total | Predictor / call | Talker calls | Talker total | Talker / call |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.6B | Whistle | 128 | 3,511.3 ms | 27.43 ms | 127 | 1,592.3 ms | 12.54 ms |
| 0.6B | faster-qwen3-tts | 128 | 3,789.5 ms | 29.61 ms | 128 | 2,630.8 ms | 20.55 ms |
| 1.7B | Whistle | 128 | 3,589.9 ms | 28.05 ms | 127 | 3,613.9 ms | 28.46 ms |
| 1.7B | faster-qwen3-tts | 128 | 3,847.0 ms | 30.06 ms | 128 | 4,603.0 ms | 35.96 ms |

On Victoria, Whistle is faster in both measured module spans. This is the
opposite of the isolated PRO 6000 modular result, where the faster talker
graph was 5.10/6.66 ms per call versus Whistle's 9.97/9.96 ms. The crossover
is consistent with fixed-cache attention and eager-versus-graph boundaries
interacting differently with GPU memory bandwidth and launch cost, but these
traces do not isolate a single causal factor. Raw checks are
`../evidence/victoria_modular_whistle_0.6B.json`,
`victoria_modular_whistle_1.7B.json`, `victoria_modular_faster_0.6B.json`, and
`victoria_modular_faster_1.7B.json`.

## What the review changes

- **Concurrency:** the earlier 41.709 s / 41.466 s upstream results co-ran two model processes on one GPU. They are excluded from the core tables. The observed concurrent/alone ratios are 2.04× (0.6B) and 1.72× (1.7B); differing containers and 0.6B lengths prevent attributing every millisecond to contention. Free VRAM does not imply free compute, memory bandwidth, or CPU dispatch capacity.
- **Upstream generation policy:** the worker passes `do_sample=False` to `generate_custom_voice`, but the inspected upstream `model.py` constructs `PredictorGraph(do_sample=True, top_k=50, temperature=0.9)`; `fast_generate` calls that graph without overriding its policy. The run therefore must not be labeled proven all-greedy. Saved 0.6B outputs vary in length. The exact imported wheel source was not archived, and the local repository snapshot is not a substitute for that missing provenance.
- **Software and attention:** upstream uses qwen-tts-hf / Transformers 5 and requests eager attention; Whistle/official used qwen-tts 0.1.1 / Transformers 4.57.3 and SDPA. The Modal images were not fully locked. Same GPU and text do not make these a controlled runtime-only A/B.
- **Warmup:** Whistle's batch harness uses two complete generation warmups. The upstream Alicia worker uses two shortened warmups: first 64 characters, at most 20 tokens. Three and five measured-pass sets are retained as recorded. The supporting isolated upstream 1.7B five-pass median is 23.958 s, consistent with the three-pass 24.072 s.
- **Stopping and correctness:** capped 1,280-frame output is not verified natural EOS. Serial Whistle checks reported exact IDs/waveforms against an in-process official path that retained Whistle's FFN wrappers; they are not fresh unmodified-model parity tests. Upstream Alicia runs did not save codec-ID parity or WAV quality evidence.
- **Timing boundary:** Whistle returns a completed device waveform; official/upstream return CPU audio. WAV file writing is excluded. These are entrypoint wall measurements, not identical CPU-ready service latencies.
- **Scope:** a definitive fair ranking would require pinned environments, verified predictor policies, consistent warmup/output boundaries, matched frame work, and independent quality/correctness validation. The Victoria cross-check improves the evidence but does not remove those controls.

## Streaming observations (separate workload)

Only **Whistle 0.6B on RTX PRO 6000** has saved Alicia first-chunk measurements here: median **93.100 ms**, trials **103.160, 87.114, 93.100 ms**; median full stream **29.449 s**. Short/medium TTFA medians are **93.144 / 84.924 ms**. Timing starts before iteration and ends after the first tensor's CPU copy. The schedule ramps at cumulative frames **2, 4, 8**, then every 12 frames, with leading-silence trimming enabled. “Chunk size 12” alone was an incomplete description. One short warmup and three repetitions were used; this is not network/playback latency.

The inspected official package returns full audio rather than incremental chunks. Its first available audio therefore coincides with completion; native streaming TTFA is unavailable. The separate streaming-test script left official sampling at its defaults, while Whistle was greedy. Its Alicia completion median **76.645 s** is a different-policy observation, not a fair TTFA speedup baseline. The core official batch result above is the configured greedy measurement.

Upstream short-text chunk-8 TTFA means were **431 ± 95 ms (0.6B)** and **467 ± 45 ms (1.7B)** while both models shared one GPU. They used Aiden, a different prompt and default sampling; they must not be ranked against Whistle's isolated Ryan/Alicia result. Upstream's printed “RTF” values **2.558 / 2.456 are × real-time**, not wall/audio RTF. Its ms/step calculation uses an approximate 12 Hz conversion, so those values are omitted from the core table. Neither 3050 audio-ready TTFA nor Modal Whistle 1.7B TTFA was measured in this campaign.

## Codec overlap remains a failed configuration

On RTX PRO 6000, Whistle overlap off/on medians were **25.386 → 33.290 s (+31.1%)** for 0.6B and **27.139 → 35.532 s (+30.9%)** for 1.7B. Historical 3050 evidence was **56.92 → 64.28 s**. The implementation flushes eight new frames with up to 25 context frames, adding repeated work and launches while the batch codec phase on PRO 6000 is only about 0.22 s. Contention is plausible but no trace decomposes the causes.

The 0.1 ms `codec_drain` is host enqueue duration around an asynchronous wait, not GPU drain time. The side-stream span includes idle waits and cannot be called codec compute time or evidence of concurrent kernel execution. Saved PCM16 off/on WAV RMS differences were 0.00285 / 0.00443 for 0.6B / 1.7B; equal duration is not perceptual equivalence. No codec-ID files were saved for the overlap comparison.

## Evidence index and cost snapshots

3050 raw files:
- `../evidence/official_current_p50_5runs.json`
- `../evidence/v7_current_p50_5runs.json`
- `../evidence/official_1.7b_3runs.json`
- `../evidence/v7_1.7b_3runs.json`
- `../evidence/victoria_whistle_0.6b_3runs.json`
- `../evidence/victoria_whistle_1.7b_3runs.json`
- `../evidence/victoria_faster_0.6b_3runs.json`
- `../evidence/victoria_faster_1.7b_3runs.json`

PRO 6000 directories under `../local/benchmarks/modal/`:
- Official 0.6B: `20260907T225747Z-official-serial-1ef73351`
- Whistle 0.6B: `20260907T214837Z-split-serial-a1d78b3e`
- Official 1.7B: `20260907T225819Z-official-serial-020df549`
- Whistle 1.7B: `20260907T225148Z-split-serial-a666f22e`
- Upstream alone 0.6B: `20260909T041616Z-faster-custom-single-a99d3f31`
- Upstream alone 1.7B, three passes: `20260909T042856Z-faster-custom-single-1.7B-3runs-da2bb490`
- Upstream alone 1.7B, five passes: `20260909T042432Z-faster-custom-single-1.7B-68f1efff`
- Upstream concurrent Alicia: `20260909T034810Z-faster-custom-963812d7`
- Upstream concurrent short text: `20260908T215448Z-faster-custom-ba004f01`
- Streaming 0.6B: `20260908T221919Z-stream-7a43d6a6`
- Overlap 0.6B / 1.7B: `20260907T220526Z-split-overlap-adc1b544` / `20260907T231146Z-split-overlap-e1bd2876`

Batch directories hold result JSON (upstream uses `0.6B_alicia.json` or `1.7B_alicia.json`). Standard Whistle/official runs also have logs and WAVs. Most benchmark data is gitignored: local existence is not publication.

Reported per-app metered costs: upstream isolated 0.6B **$0.15578027**; isolated 1.7B five-pass **$0.15533526**; isolated 1.7B three-pass **$0.14630431**. The earlier $4.29 workspace snapshot predates these runs and must not be presented as the current final campaign bill. Refresh billing before publication.
