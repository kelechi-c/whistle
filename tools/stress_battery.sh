#!/usr/bin/env bash
# Stress battery: V7 on non-alicia texts, official compare, WER, and streaming.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
DANTE_PY="${DANTE_PY:-/home/tensor/code/ml/dante/.venv/bin/python}"
FIXTURE_ROOT="${FIXTURE_ROOT:-$ROOT/stash/testdata}"
mkdir -p results out
: > results/stress_summary.txt

section() {
    echo
    echo "===== $1 ====="
    echo "$1" >> results/stress_summary.txt
}

probe() {
    # probe <name> <textfile> [--parity]
    local name="$1" file="$2" parity="${3:-}"
    local json="results/stress_${name}.json"
    local out="out/stress_${name}.wav"
    PYTHONPATH=src "$PY" tools/profile_tts.py --text-file "$file" --backend split \
        --max-new-tokens 1280 --iterations 1 --warmup 1 --speaker Ryan \
        $parity --out "$out" --json-out "$json"
}

wer() {
    PYTHONPATH=/home/tensor/code/ml/dante/baseline "$DANTE_PY" tools/eval_asr_wer.py \
        --wav "$1" --ref-file "$2" --json-out "$3" 2>&1 | grep -E "^reference|^wer:|^cer:" | tr '\n' ' '
    echo
}

section "1. t_short (7 words) — synth + WER"
probe short "$FIXTURE_ROOT/t_short.txt"
wer out/stress_short.wav "$FIXTURE_ROOT/t_short.txt" results/wer_stress_short.json

section "2. t_medium (39 words) — synth + WER"
probe medium "$FIXTURE_ROOT/t_medium.txt"
wer out/stress_medium.wav "$FIXTURE_ROOT/t_medium.txt" results/wer_stress_medium.json

section "3. t_punct (numbers/URLs/symbols) — synth + WER"
probe punct "$FIXTURE_ROOT/t_punct.txt"
wer out/stress_punct.wav "$FIXTURE_ROOT/t_punct.txt" results/wer_stress_punct.json

section "4. t_story1 (127 words) — synth + PARITY + WER"
probe story1 "$FIXTURE_ROOT/t_story1.txt" --check-codec-parity
wer out/stress_story1.wav "$FIXTURE_ROOT/t_story1.txt" results/wer_stress_story1.json

section "5. t_story2 (216 words) — synth + WER"
probe story2 "$FIXTURE_ROOT/t_story2.txt"
wer out/stress_story2.wav "$FIXTURE_ROOT/t_story2.txt" results/wer_stress_story2.json

section "6. t_repeat (200x 'la') — greedy-collapse stress + WER"
probe repeat "$FIXTURE_ROOT/t_repeat.txt"
wer out/stress_repeat.wav "$FIXTURE_ROOT/t_repeat.txt" results/wer_stress_repeat.json

section "7. t_too_long (1560 words) — expect clean capacity error"
PYTHONPATH=src "$PY" tools/profile_tts.py --text-file "$FIXTURE_ROOT/t_too_long.txt" --backend split \
    --max-new-tokens 1280 --iterations 1 --warmup 0 --speaker Ryan 2>&1 | grep -E "ValueError|exceed" | head -2

section "8. official latency on story1 (speed regression check, heavy)"
PYTHONPATH=src "$PY" tools/profile_tts.py --text-file "$FIXTURE_ROOT/t_story1.txt" --backend official \
    --max-new-tokens 1280 --iterations 1 --warmup 1 --speaker Ryan \
    --json-out results/stress_official_story1.json

section "9. official latency on story2 (heavy)"
PYTHONPATH=src "$PY" tools/profile_tts.py --text-file "$FIXTURE_ROOT/t_story2.txt" --backend official \
    --max-new-tokens 1280 --iterations 1 --warmup 1 --speaker Ryan \
    --json-out results/stress_official_story2.json

section "10. streaming latency (short, medium, story1)"
for t in short medium story1; do
    PYTHONPATH=src "$PY" tools/bench_streaming.py --text-file "$FIXTURE_ROOT/t_$t.txt"
done

echo
echo "===== done ====="
cat results/stress_summary.txt
