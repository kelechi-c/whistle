#!/usr/bin/env bash
# Promotion verification for the shipped runtime and its development tools.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
short="Sure, I can help with that. Just let me know what time works best for you."

echo "===== unit tests (main package) ====="
"$PY" -m pytest tests -q 2>&1 | tail -2

echo "===== parity32 greedy (must pass) ====="
PYTHONPATH=src "$PY" tools/profile_tts.py "$short" --backend split \
  --max-new-tokens 32 --iterations 1 --warmup 1 --check-codec-parity

echo "===== sampled-graphs spot (32 frames, captures on main path) ====="
PYTHONPATH=src "$PY" tools/profile_tts.py "$short" --backend split --fixed-tokens \
  --max-new-tokens 32 --iterations 2 --warmup 1 --temperature 0.9

echo "===== first-audio spot (ramp+trim defaults, main path) ====="
PYTHONPATH=src "$PY" tools/bench_streaming.py --text "$short" --iterations 1 --warmup 1

echo "===== alicia natural-eos parity (must pass) ====="
PYTHONPATH=src "$PY" tools/profile_tts.py --text-file testdata/alicia.txt --backend split \
  --iterations 1 --warmup 0 --check-codec-parity
