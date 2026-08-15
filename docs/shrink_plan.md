# Whistle — shrink & cleanup plan

Goal: reduce the tracked repo to the minimal code that reproduces the V7
result, with **exact parity**, and move everything exploratory/legacy into
`refs/` (gitignored). No shrink lands until the shrunk code passes the
codec-ID + waveform parity gate against the official greedy path.

## Principles

1. **Parity is the gate.** Nothing is swapped into `src/` until the shrunk
   copy passes `--check-codec-parity` (exact codec IDs + exact waveform).
2. **Shrink in the sandbox first** — copy the package, strip it, verify on
   victoria, then swap the verified copy into `src/`. Never edit the live
   package in place during the shrink.
3. **Only two modes survive:** `official-eager` (correctness reference) and
   `predictor-ffn-graphs` (V7). `compile` / `cuda-graph` are numerically
   divergent/invalid and get archived, not maintained.
4. Keep total tracked code **< 2k lines** (currently ~1.9k).

## Current inventory (tracked Python)

| file | lines | fate |
|---|---:|---|
| `src/whistle/graphs.py` | 648 | **strip to ~310** |
| `src/whistle/inference.py` | 316 | keep as-is (scheduler) |
| `src/whistle/config.py` | 42 | **fix typo, drop unused field** |
| `src/whistle/__init__.py` | 8 | keep |
| `infer.py` | 64 | keep (or move into package) |
| `profile_tts.py` | 439 | **drop 2 dead mode choices** |
| `bench_tts.py` | 246 | **move to `refs/`** |
| `tests/test_inference.py` | 149 | **update for removed modes** |

## Phase 1 — sandbox duplicate + shrink + parity

```
cp -r src/whistle sandbox/shrink/src/whistle
cp profile_tts.py sandbox/shrink/
cp infer.py sandbox/shrink/            # CLI, to run the shrunk package
cp pyproject.toml sandbox/shrink/      # so `uv run` picks up the shrunk package
```

Then, inside `sandbox/shrink/src/whistle/graphs.py`, delete (move to
`refs/legacy_graphs.py` for reference):

- `PredictorGraph` class (~404–470) — compiled `StaticCache` predictor loop.
- `TalkerGraph` class (~473–577) — static-cache compiled/manual-graph talker.
- `predictor_loop` + `compiled_predictor_loop` (~128–167).
- `talker_step` + `compiled_talker_step` + `graphable_talker_step` (~169–192).
- `_masks` (~95–121) and `predictor_embedding_weights` (~124–130).
- `StaticCache`, `create_causal_mask`, `create_sliding_window_causal_mask`
  imports (only used by the above).

**Keep:** `PrefixStaticLayer`, `prefix_cache`, `OfficialPredictor`,
`OfficialTalker`, `PredictorGraphs`, `DecoderFfnGraph`, `DecodeGraphs`,
`decode_graphs`. Trim `TalkerMode` to the two valid literals and remove the
dead `else` branch in `DecodeGraphs.__init__`.

`config.py`: delete `frames_per_character` (unused), fix `"bflloat16"` →
`"bfloat16"`.

`profile_tts.py`: drop `compile` / `cuda-graph` from the `--talker-mode`
choice. `tests/test_inference.py`: remove the `cuda-graph` mode assertion.

**Parity gate (on victoria):**

```
cd /home/tensor/whistle2/sandbox/shrink && \
uv run --no-sync python profile_tts.py --text-file alicia.txt \
  --backend split --talker-mode predictor-ffn-graphs --speaker Ryan \
  --lang English --max-new-tokens 1279 --fixed-tokens \
  --repetition-penalty 1.2 --check-codec-parity --iterations 2 --warmup 1
```

Pass = `codec id parity: exact match` + `audio parity: exact match`, and p50
≈ 56.8 s. Also run `python -m unittest tests.test_inference` (CPU structural).

## Phase 2 — swap the verified copy into `src/`

1. Replace `src/whistle/` with `sandbox/shrink/src/whistle/`.
2. Re-run the CPU structural test locally.
3. Re-run the parity gate on victoria against the real `src/`.
4. Commit. Rollback = `git revert` (the pre-shrink tree is the previous commit).

## Phase 3 — move extras to `refs/`

| item | action |
|---|---|
| `bench_tts.py` | `refs/bench_tts_legacy.py` (vanilla qwen-tts benchmark) |
| stripped `graphs.py` legacy block | `refs/legacy_graphs.py` (V3–V6 lineage) |
| `qwen3tts.md` | `refs/` (describes 25 Hz / 32-codebook variant, not the active 12 Hz / 16) |
| `lt-report.md` | `refs/` (old RTX 3060 exploratory run) |
| `exp.md` | `refs/` (experiment journal; deep dive already summarizes it) |
| `qwen_info.txt` | `refs/` |
| `blog_drafts.md`, `reminder.md`, `logs.md`, `logs.txt` | already gitignored; consolidate into `refs/notes/` or leave |

Keep tracked: `report.md`, `results.md` (authoritative benchmark provenance),
`docs/technical_deep_dive.md`, `docs/qwen3_tts_official_vs_faster.md`,
`project.md`, `README.md`, `alicia.txt`, `benchmarks/*.json`.

## Phase 4 — bug fixes + doc touch-ups

- `config.py` dtype typo (fixed in Phase 1).
- `pyproject.toml`: drop the broken `[project.scripts] whistle =
  "whistle.infer:main"` entry (CLI stays at root `infer.py`, run via
  `uv run python infer.py`), or move the CLI into `src/whistle/cli.py` and fix
  the entry point. **Recommendation: drop the entry point** (smallest change).
- README: remove the stale "Qwen3-ASR" mention (TTS-only now).

## Post-shrink target

```
src/whistle/{__init__,config,inference,graphs}.py   ~680 lines
infer.py                                             64
profile_tts.py                                      ~435
tests/test_inference.py                             ~140
pyproject.toml, README.md, project.md, alicia.txt
benchmarks/*.json, docs/*.md, report.md, results.md
```

Tracked Python ≈ **1.3k lines** (from ~1.9k), with the two valid modes and
the parity harness intact.

## Risks

- **Mode removal is one-way** — `compile`/`cuda-graph` are preserved verbatim
  in `refs/legacy_graphs.py`, so nothing is actually lost.
- **`DecodeGraphs` permanently wraps `talker.model.layers`** on CUDA — mode
  switching on a loaded model stays unsupported (documented, unchanged).
- **Parity harness one-frame offset** — keep using the existing
  `max_new_tokens + 1` official alignment; don't "simplify" it (see the V7
  one-frame bug).
