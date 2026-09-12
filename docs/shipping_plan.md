# Shipping separation plan — 2026-09-12

Status: executed in the working tree. The runtime boundary, ignored local
archive, curated evidence set, and codec-overlap removal are all in place.
Changes remain unstaged so the existing dirty work can be reviewed as one diff.

## 1. Installed runtime: exact keep list

| Current path | Why it stays |
| --- | --- |
| `src/whistle/__init__.py` | Package and exported runtime configuration. |
| `src/whistle/config.py` | Shared defaults, cache capacity, device/dtype configuration. |
| `src/whistle/graphs.py` | Predictor graphs, talker FFN graphs, eager attention, buffers/cache reset, sampling. |
| `src/whistle/inference.py` | Prompt/prefill shared with streaming, EOS checks, batch token generation and official codec decode. |
| `src/whistle/streaming.py` | Streaming generation, chunk ramp, left context, audio output. |
| `src/whistle/server.py` | Optional HTTP serving; retain serialization lock and `server` dependency extra. |
| `src/whistle/cli.py` | User-facing synthesis CLI, registered as the `whistle` console script. |

Relationships: CLI → config + inference → graphs + upstream Qwen model;
streaming → shared inference helpers + graphs; server → streaming + config.
Batch codec overlap is removed; streaming left context is required and stays.
No ASR inference implementation is present in this package: `tools/eval_asr_wer.py`
is a quality evaluator using external Qwen3-ASR, not a shipped ASR service.

Runtime needs separately installed dependencies and downloaded model weights.
Keep `qwen-tts`, PyTorch and Transformers compatibility explicit; audit direct
imports rather than relying silently on transitive requirements. Click belongs
to the CLI, SoundFile to WAV output, FastAPI/Uvicorn to the optional server.
Do not bundle checkpoints, CUDA toolchains, or environments in the repository.

`pyproject.toml` and README support building/installing; `uv.lock` supports
reproducible development. They belong in Git, not as imported runtime modules.
Setuptools currently discovers packages under `src`; verify actual wheel AND
sdist contents. A `.gitignore` is not a package inclusion manifest.

## 2. Public repository, outside the runtime

| Current paths | Final disposition |
| --- | --- |
| `README.md`, `pyproject.toml`, `MANIFEST.in`, `uv.lock`, `.gitignore`, `.python-version` | Keep tracked at root. Add license only after owner chooses terms. |
| `tests/__init__.py`, `tests/test_inference.py`, `testdata/alicia.txt` | Keep as development test fixture and parity input. Other stress fixtures are archived under `stash/testdata/`. |
| `tools/profile_tts.py`, `tools/bench_streaming.py`, `tools/eval_asr_wer.py` | Reproducibility and quality tools; excluded from the installed package. |
| `testdata/alicia.txt` | Shared long-form fixture, alongside the other small text fixtures. |
| `tools/run_promo_check.sh`, `tools/stress_battery.sh` | Portable GPU acceptance and stress harnesses. |
| `project.md` | Keep canonical architecture jotter (existing root location); update links after moves. |
| `docs/technical_deep_dive.md`, `docs/qwen3_tts_official_vs_faster.md` | Keep curated technical explanation; check claims and links against current code. |
| `docs/latency_report.md`, `docs/results.md`, `docs/failures_and_trials.md` | Curated performance history; failed overlap experiments remain historical evidence. |
| `evidence/*.json` | Cited benchmark records with the provenance index in `evidence/README.md`. |
| `local/benchmarks/` | Raw Modal/MFU runs, logs, traces, audio, uncited JSON, and generator scripts. |
| `docs/shipping_plan.md` | This completed release-boundary record. |

Don't remove evidence just because it is old. Each public number should resolve
to a tracked, compact artifact or durable external artifact and documented run
settings. Preserve source run IDs and historical timing caveats.

## 3. Local only: do not push or install

- Environments/build/cache: `.venv/`, `.venv-faster/`, `__pycache__/`,
  `.ruff_cache/`, `.pytest_cache/`, `build/`, `dist/`, `wheels/`, `*.egg-info/`.
- Machine/agent state: `.commandcode/`, `.pi/`, `.stfolder/`, `.stignore`,
  `handoff/`, local `AGENTS.md`, `logs.md`, `logs.txt`.
- Experimental code/vendor copies: `refs/`, `stash/`, `sandbox/`.
- Generated output: `out/`, `results/`, raw `local/benchmarks/`, `whistle-test.wav`.
- Working prose and draft publication workspace are under `local/prose/`,
  `local/article/`, and `local/docs/`.
- Ad hoc runners and raw benchmark generators are under `local/tools/` and
  `local/benchmarks/`; promote a portable version into `tools/` only when a
  retained public claim needs it.
- `notion_draft.md` is already deleted in the working tree: do not restore it
  or treat that deletion as this task's change.

## 4. Separation sequence (completed)

1. Snapshot the dirty tree, create the ignored `local/` archive, and move
   working prose, raw runs, drafts, and ad hoc runners without deleting them.
2. Move public tools and fixtures, update README/project/docs references, and
   make the shell harnesses resolve the repository root portably.
3. Replace `.gitignore` with explicit local-only rules while leaving curated
   Markdown, JSON evidence, and fixtures visible to Git.
4. Package the CLI as `whistle`, constrain setuptools discovery to `whistle` and
   `whistle.*`, and keep tools/tests/docs outside the installed runtime.
5. Run syntax, reference, ignore, diff, and package-content checks. No push is
   part of this plan.

Implemented `.gitignore` (root-scoped paths are intentional):

```gitignore
# python, build, and environments
__pycache__/
*.py[cod]
*.egg-info/
/.venv*/
/.ruff_cache/
/.pytest_cache/
/.cache/
/build/
/dist/
/wheels/

# local research and outputs
/local/
/refs/
/stash/
/sandbox/
/benchmarks/
/results/
/out/
/article/
/articles/
/whistle-test.wav

# generated trace pages
docs/*.html

# machine state, credentials, and agent records
/.commandcode/
/.pi/
/.modal/
/.stfolder/
/.stignore
/handoff/
/AGENTS.md
/logs.md
/logs.txt
/.env
/.env.*
!/.env.example
```

Do not blanket-ignore WAV/JSON/Markdown: curated fixtures, results, or public
article assets may need tracking. `.python-version` is visible to Git as the
shared Python pin. `.stignore` remains local because Syncthing and Git have
different policies.

## 5. Verification and remaining release checks

- Local: syntax checks, tiny structural unit tests without checkpoint downloads,
  `git diff --check`, CLI help, import/reference scans, wheel/sdist file listing,
  and `git check-ignore` on intended included/excluded sample paths.
- GPU acceptance on Victoria only: natural-EOS batch codec/waveform parity,
  repeated-request reset, streaming EOS boundaries, sampled graph freshness,
  and warm latency/first CPU-ready audio checks. No inference on this laptop.
- `tools/run_promo_check.sh` now runs a real natural-EOS parity check and uses
  `set -euo pipefail` so a failed check cannot be hidden by output filtering.
- Recheck the archived `local/docs/runtime_review.md` findings against current code:
  CPU sampled dispatch, first-token sampling policy, server initialization /
  warmup locking, and timing labels. Do not assume this older review is current.
- README's blanket exactness claim must be scoped to verified greedy batch
  configurations; streaming and sampled decoding have different contracts.
- Existing phase instrumentation, configuration objects and sampled-graph
  controls are not part of this cleanup. Explain any later refactor to the
  user before changing it, per project instructions.

## Completed in this session

Removed codec-overlap argument, side stream/helper, flush state, alternative
waveform assembly, and side-span/drain metrics from active batch inference.
Removed profiler flags/plumbing and Modal launcher plumbing; new profiler JSON
always describes non-overlapping phases. Callers must stop passing
`overlap_codec` (including `False`); old CLI flags are intentionally unsupported.
Historical reports and ignored experimental snapshots remain archival only.

Verification on 2026-09-12: the six runtime modules, packaged CLI, and tools
compile cleanly; active Python overlap-reference scan and `git diff --check`
pass. The structural unittest cannot import because local `qwen_tts` is absent;
no dependencies were installed and no model/GPU runs were performed. The plan
is visible to Git and records the completed boundary.
