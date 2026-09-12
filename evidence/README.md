# Curated benchmark evidence

These JSON files are the compact benchmark records cited by the retained
technical reports. They were moved from the former `benchmarks/` tree without
changing their recorded values; historical `text_file` fields therefore retain
the path used when each run was made. New benchmark output belongs under the
ignored `local/benchmarks/` directory.

| Evidence | Scope |
| --- | --- |
| `official_current_p50_5runs.json`, `v7_current_p50_5runs.json`, `official_latest.json`, `v7_latest.json` | Current natural-EOS 0.6B timing snapshots |
| `official_1.7b_3runs.json`, `v7_1.7b_3runs.json` | 1.7B RTX 3050 comparison |
| `correctness_greedy_rp1_2_parity_0.6b_alicia.json` | Exact natural-EOS codec and waveform parity |
| `official_tts_0.6b_alicia.json`, `faster_decode_0.6b_alicia.json`, `v2_faster_decode_0.6b_alicia.json` | Official, V1, and V2 fixed-work baselines |
| `v3_dynamic_talker_cache_ab.json`, `v3_faster_decode_0.6b_alicia*.json` | V3 dynamic-cache and compiled-predictor measurements |
| `v4_faster_decode_0.6b_alicia*.json` | V4 graph measurements |
| `v5_faster_decode_0.6b_alicia*.json`, `v5_1_faster_decode_0.6b_alicia*.json` | V5 and V5.1 compiled/talker graph measurements |
| `v6_compiled_talker_0.6b_alicia_breakdown.json`, `v7_exact_graphs_0.6b_alicia.json` | Invalid V6 diagnostic and promoted V7 fixed-work result |
| `victoria_whistle_*.json`, `victoria_faster_*.json`, `victoria_modular_*.json` | Victoria 0.6B/1.7B and modular comparisons |

Raw Modal/MFU runs, logs, traces, audio, uncited JSON, and generator scripts
remain under `local/benchmarks/` and are intentionally outside the pushed
evidence set.
