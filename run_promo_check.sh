#!/usr/bin/env bash
# Promotion verification for Track A items 1+2 in main src/whistle.
set -u
cd /home/tensor/whistle2 || exit 1
PY="/home/tensor/whistle2/.venv/bin/python"
short="Sure, I can help with that. Just let me know what time works best for you."

echo "===== unit tests (main package) ====="
"$PY" -m pytest tests -q 2>&1 | tail -2

echo "===== parity32 greedy (MUST pass) ====="
PYTHONPATH=. "$PY" profile_tts.py "$short" --backend split --fixed-tokens \
  --max-new-tokens 32 --iterations 1 --warmup 1 --check-codec-parity 2>&1 | grep -E "parity|rtf"

echo "===== sampled-graphs spot (32 frames, captures on main path) ====="
PYTHONPATH=. "$PY" profile_tts.py "$short" --backend split --fixed-tokens \
  --max-new-tokens 32 --iterations 2 --warmup 1 --temperature 0.9 2>&1 | grep -E "iteration|summary"

echo "===== TTFA spot (ramp+trim defaults, main path) ====="
PYTHONPATH=. "$PY" -c "
import sys, torch
sys.path.insert(0, '.')
from qwen_tts import Qwen3TTSModel
from whistle.streaming import stream_tts
torch.manual_seed(0)
tts = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
    device_map='cuda:0', dtype=torch.bfloat16, attn_implementation='sdpa')
list(stream_tts(tts, 'warmup.', max_new_tokens=16, stop_at_eos=False))
chunks = list(stream_tts(tts, '$short'))
print(f\"ttfa={chunks[0]['cumulative_ms']:.1f}ms first_frames={chunks[0]['chunk_frames']}\")"

echo "===== alicia natural-EOS parity (MUST pass) ====="
PYTHONPATH=. "$PY" profile_tts.py --text-file alicia.txt --backend split \
  --iterations 1 --warmup 0 --check-codec-parity 2>&1 | grep -E "parity|rtf 0"
