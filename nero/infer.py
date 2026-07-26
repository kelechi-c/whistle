"""Minimal CLI for local fixtures and official CustomVoice checkpoints."""

from dataclasses import replace
import pathlib as pl
import time

import click
import soundfile as sf
import torch

from nero.config import DTypeChoice, DeviceChoice, RUNTIME
from nero.model.custom_voice import Qwen3CustomVoice


def fixture_text_ids(text: str, device: torch.device) -> torch.Tensor:
    """Builds processor-shaped IDs only for the dependency-free tiny fixture."""
    content = list(text.encode("utf-8")) or [0]
    ids = [1, 2, 3, *content, 4, 5, 6, 7, 8]
    return torch.tensor([ids], dtype=torch.long, device=device)


def infer(
    text: str,
    output: pl.Path,
    checkpoint: str,
    language: str,
    max_frames: int,
    device_choice: DeviceChoice,
    dtype_choice: DTypeChoice,
) -> None:
    """Loads a local/Hub checkpoint, generates audio, and reports latency."""
    runtime = replace(RUNTIME, device=device_choice, dtype=dtype_choice)
    device = runtime.resolved_device()
    dtype = runtime.resolved_dtype(device)
    torch.manual_seed(runtime.seed)
    start = time.perf_counter()
    model = Qwen3CustomVoice.from_pretrained(
        checkpoint,
        device=device,
        dtype=dtype,
        local_files_only=pl.Path(checkpoint).is_dir(),
    )
    load_seconds = time.perf_counter() - start
    has_tokenizer = model.tokenizer is not None
    frames = (
        max_frames
        if has_tokenizer
        else min(
            max_frames,
            max(1, round(len(text) * runtime.frames_per_character)),
        )
    )
    start = time.perf_counter()
    result = (
        model.generate_text(
            text,
            language=language,
            max_frames=frames,
            stop_on_eos=True,
        )
        if has_tokenizer
        else model.generate(
            fixture_text_ids(text, device),
            language=language,
            max_frames=frames,
            stop_on_eos=False,
        )
    )
    generation_seconds = time.perf_counter() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result.audio[0, 0].numpy(), result.sample_rate)
    audio_seconds = result.audio.shape[-1] / result.sample_rate
    print(f"audio saved to {output}")
    print(f"device: {device}; frames: {frames}; audio: {audio_seconds:.3f}s")
    print(f"load: {load_seconds:.3f}s; generate: {generation_seconds:.3f}s")
    print(
        f"rtf: {generation_seconds / audio_seconds:.3f}; "
        f"xrt: {audio_seconds / generation_seconds:.3f}x"
    )
    for name, seconds in result.timings.items():
        print(f"{name}: {seconds * 1000:.2f} ms")


@click.command()
@click.argument("text")
@click.option(
    "--checkpoint",
    default=str(RUNTIME.checkpoint),
    show_default=True,
    help="local directory or hugging face model id",
)
@click.option("--language", default="english", show_default=True)
@click.option("--max-frames", type=click.IntRange(min=1), default=RUNTIME.max_frames)
@click.option(
    "--device",
    "device_choice",
    type=click.Choice(["auto", "cpu", "cuda"]),
    default=RUNTIME.device,
    show_default=True,
)
@click.option(
    "--dtype",
    "dtype_choice",
    type=click.Choice(["float32", "float16", "bfloat16"]),
    default=RUNTIME.dtype,
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(path_type=pl.Path),
    default=RUNTIME.output,
    show_default=True,
)
def main(
    text: str,
    checkpoint: str,
    language: str,
    max_frames: int,
    device_choice: DeviceChoice,
    dtype_choice: DTypeChoice,
    out: pl.Path,
) -> None:
    """Generates a WAV from TEXT with a CustomVoice checkpoint."""
    infer(
        text,
        out,
        checkpoint,
        language,
        max_frames,
        device_choice,
        dtype_choice,
    )


if __name__ == "__main__":
    main()
