# project notes

## faster_decode.py - official-module latency baseline

`tts_infer` is the single greedy inference hot path for the upcoming optimized
runtime. It accepts an already-loaded official `Qwen3TTSModel`, so checkpoint
loading is outside latency measurements. This path is batch-one, non-streaming
CustomVoice inference; it deliberately has no engine, scheduler, static cache,
CUDA graph, compilation, fused operation, or stochastic sampling yet.

```
official processor + prompt embeddings
  -> prefill: talker.forward(full prompt, DynamicCache)
  -> first codebook-zero token
  -> decode frame loop:
       code predictor forward x 15 residual codebooks
       -> sum all 16 codebook embeddings
       -> talker.model forward(one token, existing DynamicCache)
  -> official speech_tokenizer.decode(all frames)
  -> waveform
```

Every talker and predictor head selects `argmax`; the only logit constraints
hide non-code talker IDs and prevent EOS before the minimum frame count. The
preparation block reproduces the official CustomVoice role, language,
speaker, TTS special-token, text, codec-pad, and codec-BOS alignment. Prefill
uses the outer official talker once because it computes mRoPE state, the first
logits, `past_hidden`, and the initial KV cache. Decode calls the predictor
directly so its per-frame cache and each residual head are visible. It then
calls the inner talker backbone directly, because the outer talker would invoke
the predictor internally a second time. The explicit `cache_position`,
`position_ids`, attention-mask growth, and `rope_deltas` are the state that a
later static-cache implementation must preserve.

Timings separate prompt preparation, prefill/TTFA work, autoregressive decode,
codec decode, and total inference. `tests/test_faster_decode.py` runs this same
official-module control flow with the tiny official-shaped weights and a stub
codec; it validates forward/cache structure, not real speech quality or
full-checkpoint latency.

The former `nero/model/` reimplementation and its fixture tooling/tests are
archived under the ignored local `stash/nero_reimplementation/` directory.
Active code imports model components only from the installed `qwen-tts`
library. `nero/infer.py` remains a compatibility entry point, while
`profile_tts.py` compares the explicit `split` path against the untouched
official `GenerationMixin` path.

## inference optimization references

`docs/qwen3_tts_official_vs_faster.md` compares the inspected official and
faster-qwen3-tts snapshots. The faster project retains the official model but
replaces nested Hugging Face decode scheduling with a dynamic prefill followed
by static-cache CUDA graphs: one graph for a talker token and one graph for the
complete 15-token residual predictor. Its incremental codec path uses 25-frame
left context. The report also records its batch-one/CUDA-only constraints,
fixed predictor sampling policy, remaining host synchronizations, reported
benchmarks, and the optimization boundaries worth carrying into Nero.
