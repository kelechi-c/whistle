# Whistle

Low-latency inference for **Qwen3-TTS CustomVoice** (0.6B / 1.7B) on a single
consumer GPU.

Whistle keeps the official `qwen-tts` modules, weights, and prompt construction,
and replaces only the decode loop: the talker's post-attention FFN and the
residual codebook predictor positions are captured as fixed-shape CUDA graphs and
replayed once per frame. No custom kernels, no quantisation, no re-training.

![Whistle matches the official runtime's transcript at a lower real-time factor](assets/whistle_wer_rtf_scatter.png)

Alicia letter, 0.6B, bf16, greedy, natural EOS, RTX 3050 Laptop (6 GB), five
measured passes per runtime (2026-09-13):

| runtime | wall | RTF | × real-time | WER |
|---|---:|---:|---:|---:|
| official `qwen-tts` | 83.26 s | 0.856 | 1.17× | 3.02% |
| **whistle** | **54.27 s** | **0.558** | **1.79×** | **3.02%** |

That is **34.8% less wall time**. Codec ids and waveform are bit-identical to the
official runtime in this configuration (natural EOS, all 1,216 frames, compared
against a pristine official model in a separate process). Full tables, per-GPU
results, the streaming numbers, and every measurement caveat are in
[`latency_report.md`](latency_report.md), with the raw records in `evidence/`.
The written article and its full figure set are published separately (Sciel) and
kept out of this repository.

## Requirements

- A CUDA GPU (developed on an RTX 3050 Laptop, 6 GB; ~3 GB peak for 0.6B).
- Python 3.12+, PyTorch with CUDA, and `qwen-tts` (pulled in as a dependency).
- Model weights are downloaded from Hugging Face on first run.

## Install

```bash
uv sync                     # or: pip install -e .
```

## Usage

Synthesize one clip (downloads weights on first run):

```bash
whistle "The quick brown fox jumps over the lazy dog." --out hello.wav
```

Streaming, 12-frame chunks with a ramp so the first audio arrives early:

```python
from qwen_tts import Qwen3TTSModel
from whistle.streaming import stream_tts

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa",
)
for chunk in stream_tts(model, "Hello from Whistle.", chunk_size=12):
    player.write(chunk["audio"].float().cpu(), chunk["sample_rate"])
    if chunk["final"]:
        break
```

First CPU-ready audio on the RTX 3050 is **116 ms** for a short sentence and
**154 ms** for the 1,264-character Alicia letter (chunk ramp 2/4/8, then
12-frame chunks, 25-frame codec context). Streaming the whole letter costs about
8% more wall time than the batch path because each chunk re-decodes its context;
the trade is first audio in 154 ms instead of 54 s.

Optional HTTP streaming server (`audio/wav`, chunked):

```bash
uv sync --extra server
uv run --no-sync python -m whistle.server --port 8000
```

## Reproducing the results

Everything below runs on the GPU box. `alicia.txt` is the benchmark text; both
benchmarks stop at natural EOS unless told otherwise.

```bash
# structural tests (no checkpoint needed)
uv run --no-sync python -m unittest discover -s tests -v

# batch latency + exact parity: two processes, because one 6 GB card
# cannot hold the whistle model and a pristine official model at once
PYTHONPATH=src uv run --no-sync python tools/profile_tts.py --text-file alicia.txt \
    --backend split --iterations 5 --warmup 2 --parity-dir /tmp/whistle-parity
PYTHONPATH=src uv run --no-sync python tools/profile_tts.py --text-file alicia.txt \
    --backend official --iterations 5 --warmup 2 --parity-dir /tmp/whistle-parity

# time to first CPU-ready audio
PYTHONPATH=src uv run --no-sync python tools/bench_streaming.py --text-file alicia.txt

# regenerate the article figures (SF Pro must be installed locally; pass --font-dir)
uv run --no-project --with matplotlib python tools/make_figures.py
```

The parity comparison is exact and the second run exits non-zero on any
difference, including a frame-count difference. Both entrypoints truncate at the
same frame at natural EOS, which is what the recipe above uses. A run that stops
at its frame cap is not directly comparable: the official entrypoint reports one
frame fewer for the same cap (and the codec decoder's lookahead then changes the
last frame's audio), so align the caps when you need a cap-bound comparison.

## Layout

| path | contents |
| --- | --- |
| `src/whistle/` | the runtime: config, CLI, graphs, batch inference, streaming, optional server |
| `tests/` | CPU structural tests for the cache, graph, and token-loop contracts |
| `tools/` | benchmark harnesses, the release gate, and the figure generator |
| `evidence/` | compact benchmark records behind `latency_report.md` and the figures |
| `latency_report.md` | consolidated measurements and their caveats |
