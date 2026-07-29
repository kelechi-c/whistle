# project notes

## src/whistle/inference.py - official-module latency path

`tts_infer` is the single greedy inference hot path for the upcoming optimized
runtime. It accepts an already-loaded official `Qwen3TTSModel`, so checkpoint
loading is outside latency measurements. This path is fixed-length, batch-one,
non-streaming CustomVoice inference with a compiled predictor loop and an
eager dynamic-cache talker. It deliberately has no engine, scheduler, combined
full-frame graph, or stochastic sampling.

```
official processor + prompt embeddings
  -> prefill: talker.forward(full prompt, DynamicCache)
  -> first codebook-zero token
  -> decode frame loop:
       code predictor forward x 15 residual codebooks
       -> sum all 16 codebook embeddings
       -> talker.model forward(one token, growing DynamicCache)
  -> official speech_tokenizer.decode(all frames)
  -> waveform + codec IDs
```

Every talker and predictor head selects `argmax`; the only logit constraints
restrict talker selection to valid codec IDs. Fixed generation avoids the
per-frame EOS `.item()` synchronization entirely. The preparation block
reproduces the official CustomVoice role, language,
speaker, TTS special-token, text, codec-pad, and codec-BOS alignment. Prefill
uses the outer official talker once because it computes mRoPE state, the first
logits, `past_hidden`, and the initial KV cache. Decode calls the predictor
directly so its per-frame cache and each residual head are visible. It then
calls the inner talker backbone directly, because the outer talker would invoke
the predictor internally a second time.

Codes, predictor inputs, positions, and the predictor KV storage are
preallocated GPU tensors. The talker uses a growing `DynamicCache` (populated
KV only, flash SDPA, not capturable); the predictor owns a 16-position
`StaticCache` reset and reused for each frame. Batch-one decode needs no
growing all-ones attention mask, and non-streaming conditioning always adds
`tts_pad`, so neither mask concatenation nor a Python generation-step counter
remains.

The predictor enters its official inner transformer and fixed residual heads
directly. Its separate embedding-table weights are stacked once per module;
one indexed gather and reduction then replaces 15 Python-dispatched embedding
calls when assembling each talker frame.

## src/whistle/graphs.py - decode capture boundaries

`predictor_loop` contains the complete greedy residual-code sequence. One
`torch.compile(mode="reduce-overhead")` callable covers the entire loop, so
Inductor owns its fusion and cudagraph tree; it is never nested inside a manual
CUDA graph. `PredictorGraph` retains the fixed input/output buffers, positions,
causal masks, and 16-position cache.

`TalkerGraph` owns a dynamic `DynamicCache` and runs one eager inner-backbone
forward per token (no CUDA graph). The cache grows per step and holds only
populated KV, so SDPA takes the mask-free flash path (`attention_mask=None`,
`is_compileable=False`, `is_causal` over populated keys) — correct and the
cheapest eager path. A `StaticCache` here is a dead end: its zero-padded KV
needs an explicit mask (slow, ~28 s) or, if the mask is dropped, SDPA attends
over zero slots and the output degenerates to silence (the invalidated v5 bug).
`DynamicCache.update` concatenates per step, so KV addresses change every frame
and cannot be manually captured; the talker can only be graphed by compiling it.

`decode_graphs` caches both objects per loaded talker. The predictor embedding
weights are stacked once per predictor module. Variable-length prefill stays
eager and writes directly into the talker's reset dynamic cache. CPU tests
execute the same blocks eagerly. `DecodeGraphs` is the scheduler boundary that
a later compiled single full-frame graph can replace.

The codec writes chunks directly into one fixed GPU waveform instead of
building a Python list or using the official wrapper's `.cpu().numpy()`
conversion. CUDA events separate preparation, prefill/TTFA, decode, and codec
time with one synchronization only after the completed waveform is enqueued.
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
