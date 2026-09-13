#!/usr/bin/env bash
# Release gate: structural tests, exact natural-EOS parity, sampled capture, streaming.
# Runs on the GPU box from a clone. Override PYTHON/PARITY_DIR if needed.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
PARITY_DIR="${PARITY_DIR:-/tmp/whistle-parity}"
SHORT="Sure, I can help with that. Just let me know what time works best for you."

run_parity() {
    # run_parity <text args...> -- one backend per process into the same directory.
    # Both runs must stop at natural EOS: a cap-bound run is not comparable.
    rm -rf "$PARITY_DIR"
    PYTHONPATH=src "$PY" tools/profile_tts.py "$@" --backend split --parity-dir "$PARITY_DIR"
    PYTHONPATH=src "$PY" tools/profile_tts.py "$@" --backend official --parity-dir "$PARITY_DIR"
}

echo "===== structural tests ====="
PYTHONPATH=src "$PY" -m unittest discover -s tests -q

echo "===== short-text natural-EOS parity (must pass) ====="
run_parity "$SHORT" --max-new-tokens 256 --iterations 1 --warmup 1

echo "===== sampled predictor graphs, captured on the main path ====="
PYTHONPATH=src "$PY" tools/profile_tts.py "$SHORT" --backend split --fixed-tokens \
    --max-new-tokens 32 --iterations 2 --warmup 1 --temperature 0.9

echo "===== first CPU-ready audio ====="
PYTHONPATH=src "$PY" tools/bench_streaming.py --text "$SHORT" --iterations 1 --warmup 1

echo "===== alicia natural-EOS parity (must pass) ====="
run_parity --text-file alicia.txt --iterations 1 --warmup 0
