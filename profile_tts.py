#!/usr/bin/env python3
"""Low-overhead TTS benchmark for Nero fixtures and official Qwen3-TTS models."""

import json
import time
import click
import torch
import statistics
import numpy as np
import pathlib as pl
import soundfile as sf
from nero.config import RUNTIME
from nero.model.tts import Qwen3TTS
from collections import defaultdict
from typing import Any, Callable, Literal
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, field


Backend = Literal["nero", "official"]
Generate = Callable[[], "Sample"]
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_TEXT = "Autoregressive decoding generates one dependent audio frame at a time."


@dataclass(frozen=True, slots=True)
class Sample:
    """Normalizes backend output for timing, reporting, and audio writing."""

    audio: np.ndarray
    sample_rate: int
    phases: dict[str, float]


@dataclass(frozen=True, slots=True)
class Iteration:
    """Stores one unprofiled benchmark measurement."""

    wall_seconds: float
    audio_seconds: float
    rtf: float
    xrt: float
    peak_memory_mb: float
    phases: dict[str, float] = field(default_factory=dict)


def _synchronize(device: torch.device) -> None:
    """Waits for queued GPU work while remaining a no-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _to_numpy(audio: Any) -> np.ndarray:
    """Moves tensor output to CPU and returns one float32 waveform."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).squeeze()


class OfficialPhaseRecorder(AbstractContextManager["OfficialPhaseRecorder"]):
    """Times nested official talker, predictor, and codec calls with CUDA events."""

    def __init__(self, wrapper: Any) -> None:
        self.wrapper = wrapper
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self.restores: list[Callable[[], None]] = []

    def _patch(self, target: Any, attribute: str, name: str) -> None:
        original = getattr(target, attribute)

        def timed(*args: Any, **kwargs: Any) -> Any:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            self.events[name].append((start, end))
            return result

        setattr(target, attribute, timed)
        self.restores.append(
            lambda: setattr(target, attribute, original)
        )

    def __enter__(self) -> "OfficialPhaseRecorder":
        model = self.wrapper.model
        self._patch(model.talker, "generate", "talker")
        self._patch(model.talker.code_predictor, "generate", "code_predictor")
        self._patch(model.speech_tokenizer, "decode", "codec")
        return self

    def __exit__(self, *exc: object) -> None:
        for restore in reversed(self.restores):
            restore()

    def reset(self) -> None:
        """Drops prior iteration events without changing installed patches."""
        self.events.clear()

    def summary(self) -> dict[str, float]:
        """Synchronizes once and returns inclusive seconds for each component."""
        torch.cuda.synchronize()
        return {
            name: sum(start.elapsed_time(end) for start, end in events) / 1000
            for name, events in self.events.items()
        }


def _official_modular_phases(
    inclusive: dict[str, float], wall_seconds: float
) -> dict[str, float]:
    """Converts nested official timings into non-overlapping wall-time phases."""
    talker_total = inclusive.get("talker", 0.0)
    predictor = inclusive.get("code_predictor", 0.0)
    codec = inclusive.get("codec", 0.0)
    return {
        "talker_excluding_code_predictor": max(0.0, talker_total - predictor),
        "code_predictor": predictor,
        "codec": codec,
        "wrapper_overhead": max(0.0, wall_seconds - talker_total - codec),
    }


def _load_nero(
    checkpoint: str | None,
    text: str,
    language: str,
    speaker: str,
    max_new_tokens: int,
) -> tuple[Generate, torch.device, float, AbstractContextManager[Any]]:
    """Loads a Nero fixture or weight-compatible CustomVoice checkpoint."""
    device = RUNTIME.resolved_device()
    path = pl.Path(checkpoint) if checkpoint is not None else RUNTIME.checkpoint
    nero_weights = path / "model.pt"
    is_fixture = nero_weights.is_file() or (
        checkpoint is None and not RUNTIME.checkpoint.exists()
    )

    if not is_fixture:
        is_hugging_face_id = checkpoint is not None and "/" in checkpoint
        is_hugging_face_dir = (
            path.is_dir()
            and (path / "config.json").is_file()
            and (path / "model.safetensors").is_file()
        )
        if not is_hugging_face_id and not is_hugging_face_dir:
            raise click.ClickException(
                f"{path} is neither a Nero checkpoint nor a Hugging Face "
                "CustomVoice checkpoint"
            )
        from nero.model.custom_voice import Qwen3CustomVoice

        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        start = time.perf_counter()
        model = Qwen3CustomVoice.from_pretrained(
            checkpoint or path,
            device=device,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        _synchronize(device)
        load_seconds = time.perf_counter() - start
        recorder = OfficialPhaseRecorder(model.wrapper)

        def generate_custom_voice() -> Sample:
            wavs, sample_rate = model.generate_custom_voice(
                text,
                language=language,
                speaker=speaker,
                max_new_tokens=max_new_tokens,
            )
            return Sample(_to_numpy(wavs[0]), sample_rate, {})

        return generate_custom_voice, device, load_seconds, recorder

    dtype = RUNTIME.resolved_dtype(device)
    start = time.perf_counter()
    if checkpoint is None and not RUNTIME.checkpoint.exists():
        # Source checkouts do not necessarily include the generated fixture.
        model = Qwen3TTS.tiny(RUNTIME.seed).to(device=device, dtype=dtype).eval()
    else:
        config_path = path / "config.json"
        weights_path = path / "model.pt"
        if not config_path.is_file() or not weights_path.is_file():
            raise click.ClickException(
                f"{path} is not a Nero checkpoint; expected config.json and model.pt"
            )
        model = Qwen3TTS.from_checkpoint(path, device, dtype)
    load_seconds = time.perf_counter() - start
    frames = min(
        RUNTIME.max_frames,
        max(1, round(len(text) * RUNTIME.frames_per_character)),
    )

    def generate() -> Sample:
        result = model.generate(text, frames)
        return Sample(_to_numpy(result.audio[0]), result.sample_rate, result.timings)

    return generate, device, load_seconds, nullcontext()


def _official_method(
    wrapper: Any,
    model_name: str,
    text: str,
    language: str,
    speaker: str,
    ref_audio: pl.Path | None,
    ref_text: str | None,
    max_new_tokens: int,
) -> Callable[[], tuple[Any, int]]:
    """Selects the official wrapper method from the checkpoint family name."""
    name = model_name.lower()
    common = {
        "text": text,
        "language": language,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "subtalker_dosample": False,
    }
    if "voicedesign" in name:
        return lambda: wrapper.generate_voice_design(
            **common, instruct="normal speaking voice."
        )
    if "base" in name:
        if ref_audio is None or ref_text is None:
            raise click.UsageError("base checkpoints require --ref and --ref-text")
        return lambda: wrapper.generate_voice_clone(
            **common, ref_audio=str(ref_audio), ref_text=ref_text
        )
    return lambda: wrapper.generate_custom_voice(**common, speaker=speaker)


def _load_official(
    model_name: str,
    text: str,
    language: str,
    speaker: str,
    ref_audio: pl.Path | None,
    ref_text: str | None,
    max_new_tokens: int,
) -> tuple[Generate, torch.device, float, OfficialPhaseRecorder]:
    """Loads the production Qwen wrapper; CUDA is required to avoid CPU crashes."""
    if not torch.cuda.is_available():
        raise click.ClickException("the official backend requires cuda")
    from qwen_tts import Qwen3TTSModel

    device = torch.device("cuda")
    start = time.perf_counter()
    wrapper = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    _synchronize(device)
    load_seconds = time.perf_counter() - start
    method = _official_method(
        wrapper,
        model_name,
        text,
        language,
        speaker,
        ref_audio,
        ref_text,
        max_new_tokens,
    )
    recorder = OfficialPhaseRecorder(wrapper)

    def generate() -> Sample:
        wavs, sample_rate = method()
        return Sample(_to_numpy(wavs[0]), int(sample_rate), {})

    return generate, device, load_seconds, recorder


def _benchmark(
    generate: Generate,
    device: torch.device,
    iterations: int,
    warmup: int,
    recorder: OfficialPhaseRecorder | None,
) -> tuple[list[Iteration], Sample]:
    """Runs separate warmups and records every requested benchmark iteration."""
    for index in range(warmup):
        print(f"warmup {index + 1}/{warmup}")
        generate()
        _synchronize(device)

    rows: list[Iteration] = []
    sample: Sample | None = None
    for index in range(iterations):
        if recorder is not None:
            recorder.reset()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        sample = generate()
        _synchronize(device)
        wall = time.perf_counter() - start
        audio_seconds = sample.audio.size / sample.sample_rate
        phases = (
            _official_modular_phases(recorder.summary(), wall)
            if recorder is not None
            else sample.phases
        )
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
        _synchronize(device)
    profiler.export_chrome_trace(str(path))
    print(f"trace saved to {path}")


@click.command()
@click.argument("text", default=DEFAULT_TEXT)
@click.option("--backend", type=click.Choice(["nero", "official"]), default="nero")
@click.option("--model", default=None, help="nero checkpoint or official model id")
@click.option("--lang", default="english")
@click.option("--speaker", default="ryan")
@click.option("--ref", type=click.Path(path_type=pl.Path, exists=True), default=None)
@click.option("--ref-text", default=None)
@click.option("--max-new-tokens", type=click.IntRange(min=2), default=512)
@click.option("--iterations", type=click.IntRange(min=1), default=3)
@click.option("--warmup", type=click.IntRange(min=0), default=1)
@click.option("--out", type=click.Path(path_type=pl.Path), default=None)
@click.option("--json-out", type=click.Path(path_type=pl.Path), default=None)
@click.option("--trace-out", type=click.Path(path_type=pl.Path), default=None)
def main(
    text: str,
    backend: Backend,
    model: str | None,
    lang: str,
    speaker: str,
    ref: pl.Path | None,
    ref_text: str | None,
    max_new_tokens: int,
    iterations: int,
    warmup: int,
    out: pl.Path | None,
    json_out: pl.Path | None,
    trace_out: pl.Path | None,
) -> None:
    """Profiles TEXT synthesis without including optional trace overhead."""
    torch.manual_seed(RUNTIME.seed)
    if backend == "nero":
        loaded = _load_nero(model, text, lang, speaker, max_new_tokens)
    else:
        loaded = _load_official(
            model or DEFAULT_MODEL,
            text,
            lang,
            speaker,
            ref,
            ref_text,
            max_new_tokens,
        )
    generate, device, load_seconds, instrumentation = loaded
    recorder = instrumentation if isinstance(instrumentation, OfficialPhaseRecorder) else None
    print(f"backend: {backend}; device: {device}; load: {load_seconds:.3f}s")

    with instrumentation:
        rows, sample = _benchmark(generate, device, iterations, warmup, recorder)
    mean_wall = statistics.mean(row.wall_seconds for row in rows)
    mean_rtf = statistics.mean(row.rtf for row in rows)
    print(f"summary: mean wall {mean_wall:.3f}s; mean rtf {mean_rtf:.3f}")
    phase_names = sorted({name for row in rows for name in row.phases})
    mean_phases = {
        name: statistics.mean(row.phases.get(name, 0.0) for row in rows)
        for name in phase_names
    }
    if mean_phases:
        print(
            "mean phases: "
            + ", ".join(
                f"{name}={value * 1000:.2f}ms"
                for name, value in mean_phases.items()
            )
        )

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, sample.audio, sample.sample_rate)
        print(f"audio saved to {out}")
    if json_out is not None:
        payload = {
            "backend": backend,
            "model": model,
            "load_seconds": load_seconds,
            "iterations": [asdict(row) for row in rows],
            "mean_wall_seconds": mean_wall,
            "mean_rtf": mean_rtf,
            "phase_semantics": "non_overlapping",
            "mean_phases": mean_phases,
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"results saved to {json_out}")
    if trace_out is not None:
        _trace(generate, device, trace_out)


if __name__ == "__main__":
    main()
