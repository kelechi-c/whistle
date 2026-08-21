#!/usr/bin/env bash
# Stress battery: V7 on non-alicia texts (short..too-long), official compare,
# WER, and streaming latency. Run on the GPU box.
set -u
cd /home/tensor/whistle2 || exit 1
PY=".venv/bin/python"
DANTE_PY=/home/tensor/code/ml/dante/.venv/bin/python
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
    PYTHONPATH=src $PY profile_tts.py --text-file "$file" --backend split \
        --max-new-tokens 1280 --iterations 1 --warmup 1 --speaker Ryan \
        $parity --out "$out" --json-out "$json"
}

wer() {
    PYTHONPATH=/home/tensor/code/ml/dante/baseline $DANTE_PY eval_asr_wer.py \
        --wav "$1" --ref-file "$2" --json-out "$3" 2>&1 | grep -E "^reference|^wer:|^cer:" | tr '\n' ' '
    echo
}

section "1. t_short (7 words) — synth + WER"
probe short testdata/t_short.txt
wer out/stress_short.wav testdata/t_short.txt results/wer_stress_short.json

section "2. t_medium (39 words) — synth + WER"
probe medium testdata/t_medium.txt
wer out/stress_medium.wav testdata/t_medium.txt results/wer_stress_medium.json

section "3. t_punct (numbers/URLs/symbols) — synth + WER"
probe punct testdata/t_punct.txt
wer out/stress_punct.wav testdata/t_punct.txt results/wer_stress_punct.json

section "4. t_story1 (127 words) — synth + PARITY + WER"
probe story1 testdata/t_story1.txt --check-codec-parity
wer out/stress_story1.wav testdata/t_story1.txt results/wer_stress_story1.json

section "5. t_story2 (216 words) — synth + WER"
probe story2 testdata/t_story2.txt
wer out/stress_story2.wav testdata/t_story2.txt results/wer_stress_story2.json

section "6. t_repeat (200x 'la') — greedy-collapse stress + WER"
probe repeat testdata/t_repeat.txt
wer out/stress_repeat.wav testdata/t_repeat.txt results/wer_stress_repeat.json

section "7. t_too_long (1560 words) — expect clean capacity error"
PYTHONPATH=src $PY profile_tts.py --text-file testdata/t_too_long.txt --backend split \
    --max-new-tokens 1280 --iterations 1 --warmup 0 --speaker Ryan 2>&1 | grep -E "ValueError|exceed" | head -2

section "8. official latency on story1 (speed regression check, heavy)"
PYTHONPATH=src $PY profile_tts.py --text-file testdata/t_story1.txt --backend official \
    --max-new-tokens 1280 --iterations 1 --warmup 1 --speaker Ryan \
    --json-out results/stress_official_story1.json

section "9. official latency on story2 (heavy)"
PYTHONPATH=src $PY profile_tts.py --text-file testdata/t_story2.txt --backend official \
    --max-new-tokens 1280 --iterations 1 --warmup 1 --speaker Ryan \
    --json-out results/stress_official_story2.json

section "10. streaming latency (short, medium, story1)"
for t in short medium story1; do
    PYTHONPATH=src $PY bench_streaming.py --text-file testdata/t_$t.txt
done

echo
echo "===== done ====="
cat results/stress_summary.txt