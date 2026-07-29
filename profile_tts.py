#!/usr/bin/env python3
"""Compares explicit split inference with the official Qwen generation path."""

from dataclasses import asdict, dataclass, field
import json
import pathlib as pl
import statistics
import time
from typing import Any, Callable, Literal

import click
import numpy as np
import soundfile as sf
import torch

from whistle.config import RUNTIME
from whistle.graphs import TalkerMode
from whistle.inference import tts_infer

Backend = Literal["split", "official"]
Audio = np.ndarray | torch.Tensor
Generate = Callable[[], "Sample"]
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
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
    talker_mode: TalkerMode,
    repetition_penalty: float,
) -> tuple[Generate, torch.device, float, Any]:
    """Loads the official model for the explicit greedy prefill/decode path."""
    model, device, load_seconds = _load_model(checkpoint or DEFAULT_MODEL)

    def generate() -> Sample:
        waveform, codec_ids, sample_rate, timings = tts_infer(
            model,
            text,
            language=language,
            speaker=speaker,
            max_new_tokens=max_new_tokens,
            talker_mode=talker_mode,
            repetition_penalty=repetition_penalty,
        )
        phases = {name: timings[name] for name in ("prepare", "prefill", "decode", "codec")}
        return Sample(waveform[0], sample_rate, phases, codec_ids)

    return generate, device, load_seconds, model


def _official_sample(
    wrapper: Any,
    text: str,
    language: str,
    speaker: str,
    max_new_tokens: int,
    repetition_penalty: float,
) -> Sample:
    """Runs the official API path while retaining its generated codec IDs."""
    input_ids = wrapper._tokenize_texts([wrapper._build_assistant_text(text)])
    generation = wrapper._merge_generate_kwargs(
        min_new_tokens=max_new_tokens,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=repetition_penalty,
        subtalker_dosample=False,
    )
    codes, _ = wrapper.model.generate(
        input_ids=input_ids,
        instruct_ids=[None],
        languages=[language],
        speakers=[speaker],
        non_streaming_mode=True,
        **generation,
    )
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
) -> tuple[Generate, torch.device, float, Any]:
    """Loads fixed-length official CustomVoice inference for comparison."""
    wrapper, device, load_seconds = _load_model(model_name)

    def generate() -> Sample:
        return _official_sample(
            wrapper,
            text,
            language,
            speaker,
            max_new_tokens,
            repetition_penalty,
        )

    return generate, device, load_seconds, wrapper


def _check_codec_parity(split: torch.Tensor, official: torch.Tensor) -> None:
    """Requires identical codec shapes and reports the first differing token."""
    split = split.detach().cpu()
    official = official.detach().cpu()
    if split.ndim != 2 or official.ndim != 2:
        raise click.ClickException(
            f"codec ids must be rank two: split={tuple(split.shape)}, "
            f"official={tuple(official.shape)}"
        )
    if split.shape[1] != official.shape[1]:
        raise click.ClickException(
            f"codec id shape mismatch: split={tuple(split.shape)}, "
            f"official={tuple(official.shape)}"
        )
    shared_frames = min(split.shape[0], official.shape[0])
    shared_split = split[:shared_frames]
    shared_official = official[:shared_frames]
    mismatch = shared_split.ne(shared_official).nonzero()
    if mismatch.numel() != 0:
        frame, codebook = mismatch[0].tolist()
        matches = shared_split.eq(shared_official).sum().item()
        raise click.ClickException(
            f"codec id mismatch at frame {frame}, codebook {codebook}: "
            f"split={split[frame, codebook].item()}, "
            f"official={official[frame, codebook].item()} "
            f"({matches}/{shared_split.numel()} shared ids match)"
        )
    if split.shape != official.shape:
        raise click.ClickException(
            f"codec id length diverges at frame {shared_frames}: "
            f"split={split.shape[0]} frames, official={official.shape[0]} frames"
        )
    print(f"codec id parity: exact match ({split.shape[0]} frames)")


def _check_audio_parity(split: Audio, official: Audio) -> None:
    """Reports exact waveform equality after codec-token parity succeeds."""
    split_audio = _to_numpy(split)
    official_audio = _to_numpy(official)
    if split_audio.shape != official_audio.shape:
        raise click.ClickException(
            f"audio shape mismatch: split={split_audio.shape}, "
            f"official={official_audio.shape}"
        )
    maximum_error = float(np.max(np.abs(split_audio - official_audio), initial=0.0))
    if maximum_error != 0.0:
        raise click.ClickException(
            f"audio mismatch: max absolute error={maximum_error:.8g}"
        )
    print(f"audio parity: exact match ({split_audio.size} samples)")


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
    "--talker-mode",
    type=click.Choice(["official-eager", "compile", "cuda-graph"]),
    default="official-eager",
    show_default=True,
)
@click.option("--check-codec-parity", is_flag=True)
@click.option("--model", default=None, help="official qwen model id or path")
@click.option("--lang", default="english")
@click.option("--speaker", default="serena", show_default=True)
@click.option("--max-new-tokens", type=click.IntRange(min=2), default=1_280)
@click.option("--repetition-penalty", type=click.FloatRange(min=0.001), default=1.2)
@click.option("--iterations", type=click.IntRange(min=1), default=3)
@click.option("--warmup", type=click.IntRange(min=0), default=1)
@click.option("--out", type=click.Path(path_type=pl.Path), default=None)
@click.option("--json-out", type=click.Path(path_type=pl.Path), default=None)
@click.option("--trace-out", type=click.Path(path_type=pl.Path), default=None)
def main(
    text: str,
    text_file: pl.Path | None,
    backend: Backend,
    talker_mode: TalkerMode,
    check_codec_parity: bool,
    model: str | None,
    lang: str,
    speaker: str,
    max_new_tokens: int,
    repetition_penalty: float,
    iterations: int,
    warmup: int,
    out: pl.Path | None,
    json_out: pl.Path | None,
    trace_out: pl.Path | None,
) -> None:
    """Profiles TEXT synthesis without including optional trace overhead."""
    if text_file is not None:
        text = text_file.read_text(encoding="utf-8")
    torch.manual_seed(RUNTIME.seed)
    if backend == "split":
        loaded = _load_split(
            model,
            text,
            lang,
            speaker,
            max_new_tokens,
            talker_mode,
            repetition_penalty,
        )
    else:
        loaded = _load_official(
            model or DEFAULT_MODEL,
            text,
            lang,
            speaker,
            max_new_tokens,
            repetition_penalty,
        )
    generate, device, load_seconds, wrapper = loaded
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
    if mean_phases:
        print(
            "mean phases: "
            + ", ".join(
                f"{name}={value * 1000:.2f}ms"
                for name, value in mean_phases.items()
            )
        )
        print(
            "p50 phases: "
            + ", ".join(
                f"{name}={value * 1000:.2f}ms"
                for name, value in p50_phases.items()
            )
        )

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, _to_numpy(sample.audio), sample.sample_rate)
        print(f"audio saved to {out}")
    if json_out is not None:
        payload = {
            "backend": backend,
            "talker_mode": talker_mode if backend == "split" else None,
            "codec_parity_checked": check_codec_parity,
            "repetition_penalty": repetition_penalty,
            "model": model,
            "text_file": str(text_file) if text_file is not None else None,
            "text_characters": len(text),
            "token_count": max_new_tokens,
            "expected_codec_frames": (
                sample.codec_ids.shape[0]
                if sample.codec_ids is not None
                else max_new_tokens - 1
            ),
            "load_seconds": load_seconds,
            "iterations": [asdict(row) for row in rows],
            "mean_wall_seconds": mean_wall,
            "mean_rtf": mean_rtf,
            "p50_wall_seconds": p50_wall,
            "p50_rtf": p50_rtf,
            "phase_semantics": "non_overlapping",
            "mean_phases": mean_phases,
            "p50_phases": p50_phases,
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"results saved to {json_out}")
    if check_codec_parity:
        if backend != "split" or sample.codec_ids is None:
            raise click.ClickException(
                "codec parity checking requires the split backend"
            )
        print("running untimed official codec id parity check")
        official = _official_sample(
            wrapper,
            text,
            lang,
            speaker,
            max_new_tokens + 1,
            repetition_penalty,
        )
        if official.codec_ids is None:
            raise RuntimeError("official generation did not return codec ids")
        _check_codec_parity(sample.codec_ids, official.codec_ids)
        _check_audio_parity(sample.audio, official.audio)
    if trace_out is not None:
        _trace(generate, device, trace_out)


if __name__ == "__main__":
    main()
