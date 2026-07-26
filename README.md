# Whistle

## 0.6B benchmark (based on official repo)
on an RTX 3050 6 GB Laptop GPU,PyTorch 2.13.0, CUDA 13.0, bfloat16, SDPA.

```bash
uv run --no-sync python profile_tts.py \
  --backend official \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --speaker Ryan \
  --max-new-tokens 128 \
  --warmup 1 \
  --iterations 3 \
  --json-out tts_profile_0.6b.json
```

The default profiling text produced 6.160 seconds of 24 kHz audio in every
iteration.

| Metric | Result |
|---|---:|
| Model load | 22.070 s |
| Mean generation latency | 5.558 s |
| Generation latency range | 5.487–5.685 s |
| Mean real-time factor | 0.902 |
| Mean throughput | 1.108× real time |
| Peak allocated GPU memory | 2,246.4 MiB |

The phase measurements are non-overlapping; the code predictor is subtracted
from the inclusive talker measurement.

| Generation phase | Mean latency | Share of wall time |
|---|---:|---:|
| Code predictor | 3.978 s | 71.6% |
| Talker excluding code predictor | 1.460 s | 26.3% |
| Speech codec | 115.09 ms | 2.1% |
| Wrapper overhead | 4.69 ms | 0.08% |