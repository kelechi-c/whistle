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
[`latency_report.md`](latency_report.md). The written article and its full figure
set are published separately (Sciel) and kept out of this repository, together
with the benchmark harnesses and the raw records they produced.

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

## Development

```bash
# structural tests: cpu, tiny random weights, no checkpoint needed
uv run --no-sync python -m unittest discover -s tests -v
```

The benchmark harnesses that produced `latency_report.md` (`profile_tts.py`,
`bench_streaming.py`), the figure generator, the raw result records and the
article bundle are local development material and are not part of this
repository. The report keeps the run settings, medians, ranges and caveats for
every table it states.

## Layout

| path | contents |
| --- | --- |
| `src/whistle/` | the runtime: config, CLI, graphs, batch inference, streaming, optional server |
| `tests/` | CPU structural tests for the cache, graph, and token-loop contracts |
| `assets/` | the chart shown above |
| `alicia.txt` | the long-form benchmark input every quoted number uses |
| `latency_report.md` | consolidated measurements and their caveats |
