# Whistle

Minimal Qwen3-TTS batch-one inference runtime supporting the promoted V7 path
alone: predictor CUDA graphs + talker residual-FFN graphs, eager dynamic talker
attention, chunked-EOS, exact parity with the official runtime (every codec ID
and waveform sample).

## Install / run

Needs CUDA (benchmarks require a GPU). All commands below run on the GPU box
with the project venv:

```bash
cd whistle2   # synced copy on the gpu box
PYTHONPATH=src .venv/bin/python ...   # or: uv run --no-sync python ...
```

## Synthesize

```bash
PYTHONPATH=src .venv/bin/python infer.py "The quick brown fox jumps over the lazy dog." \
    --speaker ryan --out out/whistle.wav
# options: --max-frames (default 1280), --language, --checkpoint, --device, --dtype
```

## Benchmark: V7 vs official latency (simple one-liners)

The canonical comparison runs **both sides at natural EOS** (the official API
stops at the codec EOS regardless of budget, so fixed-token budgets are not
parity-comparable). Both produce the same 1,216-frame / 97.3 s alicia output:

V7 (this repo) with timing JSON, waveform, and exactness gate:

```bash
PYTHONPATH=src .venv/bin/python profile_tts.py --text-file alicia.txt --backend split \
    --max-new-tokens 1280 --iterations 3 --warmup 1 --speaker ryan \
    --check-codec-parity --out out/v7_alicia.wav --json-out benchmarks/v7_latest.json
```

Official `qwen-tts` runtime for the comparison (same protocol):

```bash
PYTHONPATH=src .venv/bin/python profile_tts.py --text-file alicia.txt --backend official \
    --max-new-tokens 1280 --iterations 3 --warmup 1 --speaker ryan \
    --json-out benchmarks/official_latest.json
```

`--check-codec-parity` runs the official API once more and requires every
codec ID and waveform sample to match exactly; with `--fixed-tokens` (a
1,280-frame budget, 102.4 s audio) the parity check is skipped because the
official side always stops at EOS. Reported numbers are p50 over the measured
runs; without `--fixed-tokens` generation stops at the natural EOS.

## Optional quality bench: Qwen3-ASR WER

Transcribes any synthesized WAV with Qwen3-ASR-0.6B and reports WER/CER
against the source text (needs the `qwen_asr` package; the vendored upstream
copy at `dante/baseline` on the gpu box works with transformers < 5.13):

```bash
PYTHONPATH=/path/to/dante/baseline /path/to/dante/.venv/bin/python eval_asr_wer.py \
    --wav out/v7_alicia.wav --ref-file alicia.txt --json-out results/wer_v7.json
```

## Headline numbers (RTX 3050 6 GB Laptop GPU, PyTorch 2.13/CUDA 13, bf16, SDPA, Ryan)

Measured 2026-08-21 on the same protocol (alicia, natural EOS, 1,216 frames /
97.28 s audio, one warmup + one measured run):

| Path | wall | RTF | xrt | parity | WER (Qwen3-ASR-0.6B) |
|---|---:|---:|---:|---:|---:|
| Official `qwen-tts` | 82.91 s | 0.852 | 1.173× | — | — |
| **V7 (this repo)** | **54.08 s** | **0.556** | **1.799×** | **exact (19,456 IDs, 2,334,720 samples)** | **3.02%** |

V7 is 34.8% faster wall-to-wall on identical audio (1.53× faster per audio
second by RTF). At the historical fixed 1,279-frame budget the numbers are
56.7 s / 0.554 / 1.800×.

## Layout

- `src/whistle/inference.py` — `tts_infer`: prompt build, prefill, V7 decode
  loop with chunked EOS, codec decode.
- `src/whistle/graphs.py` — `PrefixStaticLayer`, `PredictorGraphs`,
  `DecoderFfnGraph`, `OfficialTalker`, `decode_graphs`.
- `src/whistle/streaming.py`, `server.py` — chunked streaming + FastAPI server.
- `profile_tts.py` — benchmark harness (split vs official, parity, JSON).
- `infer.py` — single-shot synthesis CLI.
- `eval_asr_wer.py` — optional Qwen3-ASR WER/CER eval.
- `sandbox/` — the experimental lab (frame/dual graphs, Triton fusion,
  sampling) that produced and then rejected the non-V7 ideas.