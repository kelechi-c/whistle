## Whistle - 1.8x realtime qwen-tts on rtx 3050

**tl:dr** - **whistle** is a runtime for **qwen3**-tts **0.6B**(customvoice 12hz) that achieves **1.8x** realtime generation, using only pytorch internals and dropping RTF from **0.85**(official package) to **0.55** (WER - 3.02%, qwen-asr), on my 6gb rtx 3050 mobile GPU! streaming latency also drops to **100ms** TTFA.

### Intro
this was a learning experiment as I stepped into inference engineering. I wanted to know how fast I could drive qwen3-tts inference using only the optimizations/modules pytorch provides, and maintaining output parity with the official implementation/runtime for the same input text. This could be useful in speech2speech systems/workflows to reduce streaming latency, even if it's a small/old
