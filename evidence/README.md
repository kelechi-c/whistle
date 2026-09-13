# Benchmark evidence

Compact records behind `latency_report.md` and the article figures. Values are
as recorded; historical `text_file` fields keep the path used when each run was
made (`alicia.txt` used to live under `testdata/`).

| Evidence | Scope |
| --- | --- |
| `release_whistle_0.6b_5runs.json`, `release_official_0.6b_5runs.json` | Release validation on the RTX 3050, mains power, 2026-09-13: whistle ~0.56 RTF and the official comparison, plus the two-process parity outcome |
| `release_streaming_0.6b.json` | RTX 3050 time to first CPU-ready audio (chunk ramp 2/4/8, then 12-frame chunks) |
| `official_current_p50_5runs.json`, `v7_current_p50_5runs.json`, `official_latest.json`, `v7_latest.json` | Current natural-EOS 0.6B timing snapshots |
| `official_1.7b_3runs.json`, `v7_1.7b_3runs.json` | 1.7B RTX 3050 comparison |
| `correctness_greedy_rp1_2_parity_0.6b_alicia.json` | Exact natural-EOS codec and waveform parity |
| `official_tts_0.6b_alicia.json`, `faster_decode_0.6b_alicia.json`, `v2_faster_decode_0.6b_alicia.json` | Official, V1, and V2 fixed-work baselines |
| `v3_dynamic_talker_cache_ab.json`, `v3_faster_decode_0.6b_alicia*.json` | V3 dynamic-cache and compiled-predictor measurements |
| `v4_faster_decode_0.6b_alicia*.json` | V4 graph measurements |
| `v5_faster_decode_0.6b_alicia*.json`, `v5_1_faster_decode_0.6b_alicia*.json` | V5 and V5.1 compiled/talker graph measurements |
| `v6_compiled_talker_0.6b_alicia_breakdown.json`, `v7_exact_graphs_0.6b_alicia.json` | Invalid V6 diagnostic and promoted V7 fixed-work result |
| `victoria_whistle_*.json`, `victoria_faster_*.json`, `victoria_modular_*.json`, `modular_*.json` | Victoria 0.6B/1.7B and module-latency comparisons |
| `victoria_fresh_official_wer.json`, `victoria_fresh_whistle_wer.json`, `wer_alicia_faster.json` | Qwen3-ASR WER for the quality/speed figure; the ASR harness itself is not part of this tree (it needs a separate Qwen3-ASR install) |

Raw Modal runs, logs, traces, audio, and ad-hoc generator scripts were moved out
of the released tree during the pre-release cleanup (archived locally under
`stash/release-cleanup/`); they are not needed to support the retained numbers.
The written article and its full figure set are kept outside this repository, in
the ignored `article/` directory, and are published separately.
