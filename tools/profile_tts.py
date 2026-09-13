#!/usr/bin/env python3
"""Benchmarks the explicit prefill/decode path against the official Qwen path.

Reproduces the numbers in `latency_report.md`: p50 wall time and RTF per backend,
an optional chrome trace, and an exact codec-id and waveform parity check.

Parity compares against a *pristine* official model, so the reference must come
from a separate process: a 6 GB card cannot hold both models at once. Pass the
same `--parity-dir` to one run per backend, matching every other flag, and the
second run reports the comparison:

    uv run --no-sync python tools/profile_tts.py --text-file alicia.txt \\
        --backend split --parity-dir /tmp/whistle-parity
    uv run --no-sync python tools/profile_tts.py --text-file alicia.txt \\
        --backend official --parity-dir /tmp/whistle-parity

The split run writes the ids, waveform and settings as its own reference; the
official run then compares and exits non-zero on any difference, including a
length difference. The stored settings are checked first, so a reference left
over from a different configuration is rejected instead of diffed.

Both entrypoints truncate at the same frame at natural EOS, which is the
recommended parity configuration. A run that stops at its frame *cap* instead is
not comparable as-is: the official entrypoint reports one frame fewer for the same
cap, and the codec decoder's lookahead then changes the last frame's audio, so the
comparison would fail on a real one-frame divergence. Align the caps when you need
a cap-bound comparison (official `--max-new-tokens N+1` against split `N`).
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import pathlib as pl
import statistics
import time
from typing import Any, Callable, Literal

import click
import numpy as np
import soundfile as sf
import torch

from whistle.config import CHECKPOINT, SEED
from whistle.inference import tts_infer

Backend = Literal["split", "official"]
Audio = np.ndarray | torch.Tensor
Generate = Callable[[], "Sample"]
DEFAULT_TEXT = "Autoregressive decoding generates one dependent audio frame at a time."


@dataclass(frozen=True, slots=True)
class Sample:
    """Normalizes backend output for timing, reporting, and audio writing."""

    audio: Audio
    sample_rate: int
    phases: dict[str, float]
    codec_ids: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class Iteration:
    """Stores one unprofiled benchmark measurement."""

    wall_seconds: float
    audio_seconds: float
    rtf: float
    xrt: float
    peak_memory_mb: float
    phases: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Parity:
    """Outcome of a cross-process codec-id and waveform comparison."""

    checked: bool
    passed: bool
    detail: str


def _to_numpy(audio: Any) -> np.ndarray:
    """Moves tensor output to CPU and returns one float32 waveform."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).squeeze()


def _load_model(model_name: str) -> tuple[Any, torch.device, float]:
    """Loads one official CUDA model outside benchmark measurements."""
    if not torch.cuda.is_available():
        raise click.ClickException("profiling requires cuda")
    from qwen_tts import Qwen3TTSModel

    device = torch.device("cuda")
    start = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        model_name, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    torch.cuda.synchronize(device)
    return model, device, time.perf_counter() - start


def _load_split(
    checkpoint: str | None,
    text: str,
    language: str,
    speaker: str,
    max_new_tokens: int,
    repetition_penalty: float,
    fixed_tokens: bool,
    temperature: float | None,
    top_k: int,
) -> tuple[Generate, torch.device, float]:
    """Loads the official model for the explicit prefill/decode path."""
    model, device, load_seconds = _load_model(checkpoint or CHECKPOINT)

    def generate() -> Sample:
        waveform, codec_ids, sample_rate, timings = tts_infer(
            model,
            text,
            language=language,
            speaker=speaker,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            stop_at_eos=not fixed_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        phases = {
            name: timings[name]
            for name in (
                "prepare",
                "prefill",
                "decode",
                "codec",
            )
            if name in timings
        }
        return Sample(waveform[0], sample_rate, phases, codec_ids)

    return generate, device, load_seconds


def _official_sample(
    wrapper: Any,
    text: str,
    language: str,
    speaker: str,
    max_new_tokens: int,
    repetition_penalty: float,
    *,
    fixed_tokens: bool,
) -> Sample:
    """Runs the official API path while retaining its generated codec IDs.

    `fixed_tokens` pins the length for fixed-work comparisons; without it the
    run stops at natural EOS, matching the split backend's default. qwen-tts
    0.1.1 hardcodes `min_new_tokens=2` in its talker kwargs and truncates at the
    config eos id, so the length is pinned by hiding that id for this call.
    """
    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(text)])
    generation = wrapper._merge_generate_kwargs(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        subtalker_dosample=False,
    )
    config = wrapper.model.config.talker_config
    stop_id = config.codec_eos_token_id
    if fixed_tokens:
        config.codec_eos_token_id = -1
        generation["eos_token_id"] = -1
    try:
        codes, _ = wrapper.model.generate(
            input_ids=input_ids,
            instruct_ids=[None],
            languages=[language],
            speakers=[speaker],
            non_streaming_mode=True,
            **generation,
        )
    finally:
        config.codec_eos_token_id = stop_id
    wavs, sample_rate = wrapper.model.speech_tokenizer.decode(
        [{"audio_codes": codes[0]}]
    )
    return Sample(_to_numpy(wavs[0]), int(sample_rate), {}, codes[0])


def _load_official(
    model_name: str,
    text: str,
    language: str,
    speaker: str,
    max_new_tokens: int,
    repetition_penalty: float,
    fixed_tokens: bool,
) -> tuple[Generate, torch.device, float]:
    """Loads the official CustomVoice path for comparison."""
    wrapper, device, load_seconds = _load_model(model_name)

    def generate() -> Sample:
        return _official_sample(
            wrapper,
            text,
            language,
            speaker,
            max_new_tokens,
            repetition_penalty,
            fixed_tokens=fixed_tokens,
        )

    return generate, device, load_seconds


def _compare_codec_ids(generated: torch.Tensor, reference: torch.Tensor) -> str | None:
    """Returns a mismatch description, or None when both id tensors are identical."""
    generated = generated.detach().cpu()
    reference = reference.detach().cpu()
    if generated.ndim != 2 or reference.ndim != 2:
        return (
            f"codec ids must be rank two: generated={tuple(generated.shape)}, "
            f"reference={tuple(reference.shape)}"
        )
    if generated.shape[1] != reference.shape[1]:
        return (
            f"codec id shape mismatch: generated={tuple(generated.shape)}, "
            f"reference={tuple(reference.shape)}"
        )
    shared = min(generated.shape[0], reference.shape[0])
    mismatch = generated[:shared].ne(reference[:shared]).nonzero()
    if mismatch.numel() != 0:
        frame, codebook = mismatch[0].tolist()
        matches = generated[:shared].eq(reference[:shared]).sum().item()
        return (
            f"codec id mismatch at frame {frame}, codebook {codebook}: "
            f"generated={generated[frame, codebook].item()}, "
            f"reference={reference[frame, codebook].item()} "
            f"({matches}/{generated[:shared].numel()} shared ids match)"
        )
    offset = generated.shape[0] - reference.shape[0]
    if offset:
        return (
            f"codec id length differs by {offset:+d} frames "
            f"(generated={generated.shape[0]}, reference={reference.shape[0]}); the official "
            "entrypoint reports one frame fewer for the same frame cap, so align the caps "
            "(run the official backend with --max-new-tokens +1) or compare at natural EOS"
        )
    return None


def _compare_audio(generated: Audio, reference: Audio) -> str | None:
    """Returns a mismatch description, or None when the waveforms are bit-identical."""
    generated_audio = np.reshape(_to_numpy(generated), -1)
    reference_audio = np.reshape(_to_numpy(reference), -1)
    if generated_audio.shape != reference_audio.shape:
        return (
            f"audio shape mismatch: generated={generated_audio.shape}, "
            f"reference={reference_audio.shape}"
        )
    maximum_error = float(np.max(np.abs(generated_audio - reference_audio), initial=0.0))
    if maximum_error != 0.0:
        return f"audio mismatch: max absolute error={maximum_error:.8g}"
    return None


def _fingerprint(settings: dict[str, Any]) -> str:
    """Canonical description of the run settings a reference file belongs to."""
    return json.dumps(settings, sort_keys=True)


def _check_parity(
    directory: pl.Path,
    backend: Backend,
    sample: Sample,
    settings: dict[str, Any],
) -> Parity:
    """Stores this backend's output and compares it against the other backend's.

    A parity check is only complete once both backends have written into the same
    directory; the first run reports `checked=False` and tells the caller what to
    run next. The stored settings are compared first, so a reference left over
    from a different configuration is rejected instead of silently diffed.
    """
    if sample.codec_ids is None:
        raise click.ClickException(f"the {backend} backend did not return codec ids")
    other: Backend = "official" if backend == "split" else "split"
    own = directory / f"{backend}.npz"
    reference_path = directory / f"{other}.npz"
    directory.mkdir(parents=True, exist_ok=True)
    frames = sample.codec_ids.shape[0]
    np.savez(
        own,
        ids=sample.codec_ids.detach().cpu().numpy(),
        audio=np.reshape(_to_numpy(sample.audio), -1),
        settings=_fingerprint(settings),
    )
    if not reference_path.exists():
        return Parity(False, False, f"saved {own}; now run the {other} backend with the same --parity-dir")

    stored = np.load(reference_path)
    stored_settings = json.loads(str(stored["settings"].item()))
    differing = sorted(
        key for key in set(settings) | set(stored_settings)
        if stored_settings.get(key) != settings.get(key)
    )
    if differing:
        return Parity(
            True,
            False,
            f"{other} reference was produced with different settings ({', '.join(differing)}); "
            "delete the directory contents and rerun both backends",
        )

    # Both entrypoints truncate at the same EOS frame; a cap-bound run differs by
    # one frame, and the codec decoder's lookahead then perturbs the last frame's
    # audio, so length must match before the waveform comparison means anything.
    reference_ids = torch.from_numpy(stored["ids"])
    problems = [
        problem
        for problem in (
            _compare_codec_ids(sample.codec_ids, reference_ids),
            _compare_audio(np.reshape(_to_numpy(sample.audio), -1), np.reshape(stored["audio"], -1)),
        )
        if problem is not None
    ]
    if problems:
        return Parity(True, False, f"{backend} vs {other}: " + "; ".join(problems))
    offset = frames - reference_ids.shape[0]
    note = f" ({offset:+d} frame offset)" if offset else ""
    return Parity(
        True,
        True,
        f"{backend} vs {other}: exact codec id and audio match ({frames} frames){note}",
    )

def _benchmark(
    generate: Generate,
    device: torch.device,
    iterations: int,
    warmup: int,
) -> tuple[list[Iteration], Sample]:
    """Runs separate warmups and records every requested benchmark iteration."""
    for index in range(warmup):
        print(f"warmup {index + 1}/{warmup}")
        generate()

    rows: list[Iteration] = []
    sample: Sample | None = None
    for index in range(iterations):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        sample = generate()
        wall = time.perf_counter() - start
        samples = sample.audio.numel() if isinstance(sample.audio, torch.Tensor) else sample.audio.size
        audio_seconds = samples / sample.sample_rate
        phases = sample.phases
        peak = (
            torch.cuda.max_memory_allocated() / 1024**2
            if device.type == "cuda"
            else 0.0
        )
        row = Iteration(
            wall,
            audio_seconds,
            wall / audio_seconds,
            audio_seconds / wall,
            peak,
            phases,
        )
        rows.append(row)
        print(
            f"iteration {index + 1}: {wall:.3f}s wall, {audio_seconds:.3f}s audio, "
            f"rtf {row.rtf:.3f}, xrt {row.xrt:.3f}x"
        )
        if phases:
            print("phases: " + ", ".join(f"{name}={value * 1000:.2f}ms" for name, value in phases.items()))
    if sample is None:
        raise RuntimeError("no benchmark sample was generated")
    return rows, sample


def _trace(generate: Generate, device: torch.device, path: pl.Path) -> None:
    """Runs one diagnostic trace that is deliberately excluded from benchmarks."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as profiler:
        generate()
    profiler.export_chrome_trace(str(path))
    print(f"trace saved to {path}")


@click.command()
@click.argument("text", default=DEFAULT_TEXT)
@click.option(
    "--text-file",
    type=click.Path(path_type=pl.Path, exists=True, dir_okay=False),
    default=None,
    help="read benchmark text from a utf-8 file",
)
@click.option(
    "--backend", type=click.Choice(["split", "official"]), default="split"
)
@click.option(
    "--parity-dir",
    type=click.Path(path_type=pl.Path),
    default=None,
    help="run one backend per invocation into the same directory to verify exact parity",
)
@click.option("--model", default=None, help="official qwen model id or path")
@click.option("--language", default="english", show_default=True)
@click.option("--speaker", default="ryan", show_default=True)
@click.option("--max-new-tokens", type=click.IntRange(min=2), default=1_280)
@click.option("--repetition-penalty", type=click.FloatRange(min=0.001), default=1.2)
@click.option("--temperature", type=click.FloatRange(min=0.01), default=None,
              help="enable do_sample with this temperature (top_k below)")
@click.option("--top-k", type=click.IntRange(min=1), default=50, show_default=True)
@click.option(
    "--fixed-tokens",
    is_flag=True,
    help="ignore eos and emit the complete frame budget",
)
@click.option("--iterations", type=click.IntRange(min=1), default=3)
@click.option("--warmup", type=click.IntRange(min=0), default=1)
@click.option("--out", type=click.Path(path_type=pl.Path), default=None)
@click.option("--json-out", type=click.Path(path_type=pl.Path), default=None)
@click.option("--trace-out", type=click.Path(path_type=pl.Path), default=None)
def main(
    text: str,
    text_file: pl.Path | None,
    backend: Backend,
    parity_dir: pl.Path | None,
    model: str | None,
    language: str,
    speaker: str,
    max_new_tokens: int,
    repetition_penalty: float,
    temperature: float | None,
    top_k: int,
    fixed_tokens: bool,
    iterations: int,
    warmup: int,
    out: pl.Path | None,
    json_out: pl.Path | None,
    trace_out: pl.Path | None,
) -> None:
    """Profiles TEXT synthesis without including optional trace overhead."""
    if text_file is not None:
        text = text_file.read_text(encoding="utf-8")
    torch.manual_seed(SEED)
    if backend == "split":
        loaded = _load_split(
            model,
            text,
            language,
            speaker,
            max_new_tokens,
            repetition_penalty,
            fixed_tokens,
            temperature,
            top_k,
        )
    else:
        loaded = _load_official(
            model or CHECKPOINT,
            text,
            language,
            speaker,
            max_new_tokens,
            repetition_penalty,
            fixed_tokens,
        )
    generate, device, load_seconds = loaded
    print(f"backend: {backend}; device: {device}; load: {load_seconds:.3f}s")
    rows, sample = _benchmark(generate, device, iterations, warmup)
    mean_wall = statistics.mean(row.wall_seconds for row in rows)
    mean_rtf = statistics.mean(row.rtf for row in rows)
    p50_wall = statistics.median(row.wall_seconds for row in rows)
    p50_rtf = statistics.median(row.rtf for row in rows)
    print(
        f"summary: p50 wall {p50_wall:.3f}s; p50 rtf {p50_rtf:.3f}; "
        f"mean wall {mean_wall:.3f}s; mean rtf {mean_rtf:.3f}"
    )
    phase_names = sorted({name for row in rows for name in row.phases})
    mean_phases = {
        name: statistics.mean(row.phases.get(name, 0.0) for row in rows)
        for name in phase_names
    }
    p50_phases = {
        name: statistics.median(row.phases.get(name, 0.0) for row in rows)
        for name in phase_names
    }
    if p50_phases:
        print(
            "p50 phases: "
            + ", ".join(f"{name}={value * 1000:.2f}ms" for name, value in p50_phases.items())
        )

    parity: Parity | None = None
    if parity_dir is not None:
        if temperature is not None:
            raise click.ClickException("parity requires greedy decoding; drop --temperature")
        print("running codec id and audio parity check")
        parity = _check_parity(
            parity_dir,
            backend,
            sample,
            settings={
                "model": model or CHECKPOINT,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "speaker": speaker,
                "language": language,
                "max_new_tokens": max_new_tokens,
                "fixed_tokens": fixed_tokens,
                "repetition_penalty": repetition_penalty,
                "top_k": top_k,
            },
        )
        print(parity.detail)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, _to_numpy(sample.audio), sample.sample_rate)
        print(f"audio saved to {out}")
    if json_out is not None:
        payload = {
            "backend": backend,
            "codec_parity_checked": bool(parity and parity.checked and parity.passed),
            "codec_parity": (
                {"checked": parity.checked, "passed": parity.passed, "detail": parity.detail}
                if parity
                else None
            ),
            "repetition_penalty": repetition_penalty,
            "fixed_tokens": fixed_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "model": model,
            "text_file": str(text_file) if text_file is not None else None,
            "text_characters": len(text),
            "token_count": max_new_tokens,
            "expected_codec_frames": (
                sample.codec_ids.shape[0] if sample.codec_ids is not None else None
            ),
            "load_seconds": load_seconds,
            "iterations": [asdict(row) for row in rows],
            "mean_wall_seconds": mean_wall,
            "mean_rtf": mean_rtf,
            "p50_wall_seconds": p50_wall,
            "p50_rtf": p50_rtf,
            "mean_phases": mean_phases,
            "p50_phases": p50_phases,
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"results saved to {json_out}")
    if trace_out is not None:
        _trace(generate, device, trace_out)
    if parity is not None and parity.checked and not parity.passed:
        raise click.ClickException(parity.detail)


if __name__ == "__main__":
    main()
