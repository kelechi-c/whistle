"""CLI for the explicit official-module inference baseline."""

from dataclasses import replace
import pathlib as pl

import click
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import torch

from whistle.config import DTypeChoice, DeviceChoice, RUNTIME
from whistle.inference import tts_infer

CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


@click.command()
@click.argument("text")
@click.option("--checkpoint", default=CHECKPOINT, show_default=True)
@click.option("--speaker", default="Ryan", show_default=True)
@click.option("--language", default="english", show_default=True)
@click.option("--max-frames", type=click.IntRange(min=1), default=RUNTIME.max_frames)
@click.option("--device", "device_choice", type=click.Choice(["auto", "cpu", "cuda"]), default=RUNTIME.device)
@click.option("--dtype", "dtype_choice", type=click.Choice(["float32", "float16", "bfloat16"]), default=RUNTIME.dtype)
@click.option("--out", type=click.Path(path_type=pl.Path), default=RUNTIME.output)
def main(
    text: str,
    checkpoint: str,
    speaker: str,
    language: str,
    max_frames: int,
    device_choice: DeviceChoice,
    dtype_choice: DTypeChoice,
    out: pl.Path,
) -> None:
    """Loads the model, synthesizes TEXT, and writes its waveform."""
    runtime = replace(RUNTIME, device=device_choice, dtype=dtype_choice, max_frames=max_frames, output=out)
    device = runtime.resolved_device()
    torch.manual_seed(runtime.seed)
    model = Qwen3TTSModel.from_pretrained(
        checkpoint,
        device_map=str(device),
        dtype=runtime.resolved_dtype(device),
        attn_implementation="sdpa",
    )
    waveform, codes, sample_rate, timings = tts_infer(
        model, text, speaker=speaker, language=language, max_new_tokens=max_frames
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, waveform[0].float().cpu().numpy(), sample_rate)
    print(f"audio saved to {out}")
    print(
        f"frames: {codes.shape[0]}; codebooks: {codes.shape[1]}; "
        f"prefill: {timings['prefill'] * 1000:.2f} ms; "
        f"decode: {timings['decode'] * 1000:.2f} ms; "
        f"codec: {timings['codec'] * 1000:.2f} ms"
    )


__all__ = ["tts_infer"]


if __name__ == "__main__":
    main()
