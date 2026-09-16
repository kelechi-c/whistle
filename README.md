# whistle

faster inference for **Qwen3-TTS CustomVoice** (0.6B / 1.7B) on a single consumer GPU (tested on RTX 3050).

whistle achieves **1.5x** improvement over the official `qwen-tts` runtime by CUDA graph replay on the **Talker module's post-attention FFN** (20 graphs for 0.6B) and the **residual codebook predictor positions** (15 graphs, one per residual codebook position), which are the specific areas of **fixed shape** execution. Attention is left eager for the talker, while the code predictor module uses **PrefixStaticLayer**, a kvcache implementation which exposes only the active attention prefix slice for each step while maintaining a fixed backing buffer. These **35 graphs are replayed once per frame**, using **native PyTorch optimizations** with **zero custom kernels, quantization, or posttraining** (also no literal `torch.compile`, haha).

![whistle matches the official runtime's transcript/WER at a much lower RTF/latency](assets/whistle_wer_rtf_scatter.png)

## metrics summary

alicia.txt letter, 0.6B, bf16, greedy, RTX 3050 (6 GB), p50 of 5 runs, WER from Qwen3-ASR-0.6B.

| runtime | wall | RTF ↓ | × real-time ↑ | WER ↓ | % gain ↑ |
|---|---:|---:|---:|---:|---:|
| official `qwen-tts` | 83.26 s | 0.856 | 1.17× | 3.02% | — |
| **whistle** | **54.27 s** | **0.558** | **1.79×** | **3.02%** | **34.8%** |

both runtimes transcribe identically with no quality degradation.

tables and charts for other GPUs, model sizes and streaming are in [`latency_report.md`](latency_report.md). benchmark tools are in `tools/`.

**[full technical article](https://kelechi.cc/articles/whistle-qwen3-tts)**

## requirements

- an Nvidia GPU: CUDA only, there is no CPU inference path (0.6B peaks at ~3 GB VRAM, 1.7B at ~5 GB)
- Python 3.12+ and PyTorch 2.4+ (`uv sync` installs both anyways).
- disk: ~2.4 GB (0.6B weights) or ~4.3 GB (1.7B weights), downloaded from Hugging Face on first run.

## install

```bash
git clone https://github.com/kelechi-c/whistle.git
cd whistle && uv sync
```

## usage

synthesize one clip:

```bash
uv run whistle "your life is your canvas, what will you paint?" --out hello.wav
```

Streaming, 12-frame chunks with a ramp so the first audio arrives early:

```python
from qwen_tts import Qwen3TTSModel
from whistle.streaming import stream_tts

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda:0", dtype="bfloat16", attn_implementation="sdpa",
)
for chunk in stream_tts(model, "hello from whistle streaming.", chunk_size=12):
    # chunk["audio"] is (1, 1, samples); squeeze() it if writing with soundfile
    player.write(chunk["audio"].float().cpu(), chunk["sample_rate"])
    if chunk["final"]:
        break
```

first CPU-ready audio on the RTX 3050 is **116 ms** (for short samples/sentences) and **154 ms** (for longer text, like the letter in alicia.txt). uses chunk ramp-up of 2/4/8, then 12-frame chunks, with 25-frame codec context/prefix.

**HTTP streaming server (`audio/wav`, chunked)**:

```bash
uv sync --extra server
uv run python -m whistle.server --port 8000
```

## a little backstory
this is really my first shot at inference engineering/optimizations. I picked this (qwen3-tts) cus I thought it'd be useful locally AND the architecture was interesting/unconventional. 
I am glad I finished this one(since I tend to jump around projects a lot), and even wrote a technical article on it(first proper one I have ever written tbh). 

next steps will be optimizing other audio models, local inference engines, or wait, I should scale up(bigger models/GPUs, TPUs maybe?)...would be more useful in the industry/future.
but still, if I see a small sized model, that fits on the 3050, I will try and perform different surgeries to make it faster than it is already. my current focus is audio models(STT, TTS, music, etc).

Thankfully, current generation LLMs make work and learning much more rapid. 

## acknowledgements

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba's **Qwen** team for the original model architecture/weights/research.
- **[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)** by **Andi Marafioti** for benchmarks and inspiration.
- The test sample at `alicia.txt` is dialogue from the game, *Clair Obscur: Expedition 33*.
- **gpt-5.6-sol** and **deepseek v4-flash** were crucial in code implementations and experiments.

## license
Apache 2.0. See [LICENSE](LICENSE) for details.