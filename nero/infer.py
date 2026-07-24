"""Minimal text-to-audio command line entry point."""

import pathlib as pl
import time

import click
import soundfile as sf
import torch

from nero.config import RUNTIME
from nero.model.tts import Qwen3TTS


def infer(text: str, output: pl.Path) -> None:
    """Loads the configured checkpoint, generates audio, and reports latency."""
    device = RUNTIME.resolved_device()
    dtype = RUNTIME.resolved_dtype(device)
    torch.manual_seed(RUNTIME.seed)

    start = time.perf_counter()
    model = Qwen3TTS.from_checkpoint(RUNTIME.checkpoint, device, dtype)
    load_seconds = time.perf_counter() - start
    frames = min(
        RUNTIME.max_frames,
        max(1, round(len(text) * RUNTIME.frames_per_character)),
    )

    start = time.perf_counter()
    result = model.generate(text, frames)
    generation_seconds = time.perf_counter() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, result.audio[0].numpy(), result.sample_rate)
    audio_seconds = result.audio.shape[-1] / result.sample_rate

    print(f"audio saved to {output}")
    print(f"device: {device}; frames: {frames}; audio: {audio_seconds:.3f}s")
    print(f"load: {load_seconds:.3f}s; generate: {generation_seconds:.3f}s")
    print(f"rtf: {generation_seconds / audio_seconds:.3f}; xrt: {audio_seconds / generation_seconds:.3f}x")
    for name, seconds in result.timings.items():
        print(f"{name}: {seconds * 1000:.2f} ms")


@click.command()
@click.argument("text")
@click.option(
    "--out",
    type=click.Path(path_type=pl.Path),
    default=RUNTIME.output,
    show_default=True,
)
def main(text: str, out: pl.Path) -> None:
    """Generates a WAV file from TEXT using choices in nero/config.py."""
    infer(text, out)


if __name__ == "__main__":
    main()
