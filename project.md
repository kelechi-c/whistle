# project notes

## src/whistle/inference.py - official-module latency path

`tts_infer` is the batch-one greedy CustomVoice path. It accepts an already
loaded official `Qwen3TTSModel`, so checkpoint loading is outside inference
measurements. Its default `official-eager` mode is the correctness reference;
the compiled/static modes remain selectable experiments.

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
can replace only after they pass exact parity.

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
archived under the ignored local `stash/nero_reimplementation/` directory.
Active code imports model components only from the installed `qwen-tts`
library. `src/whistle/infer.py` owns model loading, WAV output, and the CLI;
`src/whistle/inference.py` contains request scheduling; and
`src/whistle/graphs.py` owns reusable decode state. The root `profile_tts.py`
compares fixed-length CustomVoice split and official runs.

`profile_tts.py --backend split --check-codec-parity` performs one untimed
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
`benchmarks/v7_exact_graphs_0.6b_alicia.json`.
