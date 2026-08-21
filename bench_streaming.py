#!/usr/bin/env python3
"""Streaming latency bench for the V7 stream path (TTFA + chunk cadence).

Measures, per input text: time-to-first-audio (the first decoded chunk),
per-chunk wall cadence, total wall, emitted frames, and audio duration.
Run on the GPU box:
    PYTHONPATH=src .venv/bin/python bench_streaming.py --text-file testdata/t_medium.txt
"""

import time

import click
import pathlib as pl
import torch
from qwen_tts import Qwen3TTSModel

from whistle.streaming import stream_tts

CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


@click.command()
@click.option("--text-file", type=click.Path(path_type=pl.Path, exists=True), default=None)
@click.option("--text", default=None, help="inline text (positional alternative)")
@click.option("--speaker", default="ryan", show_default=True)
@click.option("--chunk-size", type=click.IntRange(min=1), default=12, show_default=True)
@click.option("--max-new-tokens", type=click.IntRange(min=2), default=1_280)
def main(text_file: pl.Path | None, text: str | None, speaker: str, chunk_size: int, max_new_tokens: int) -> None:
    """Benchmarks stream_tts TTFA and chunk cadence on one text."""
    payload = text if text is not None else text_file.read_text(encoding="utf-8")
    torch.manual_seed(0)
    model = Qwen3TTSModel.from_pretrained(
        CHECKPOINT, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    chunk_audio_sec = chunk_size / 12.5  # 12.5 codec frames per second of audio

    ttft = None
    chunk_times = []
    frames = 0
    for ch in stream_tts(model, payload, speaker=speaker, chunk_size=chunk_size, max_new_tokens=max_new_tokens):
        if ttft is None:
            ttft = ch["ttft_ms"]
        chunk_times.append(ch["chunk_ms"])
        audio_ms = ch["cumulative_ms"]
        frames += ch["chunk_frames"]

    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"text chars: {len(payload)} | frames: {frames} | "
        f"ttfa: {ttft:.1f} ms | total: {ch['cumulative_ms']:.1f} ms | "
        f"audio: {audio_ms:.1f} ms"
    )
    if len(chunk_times) > 1:
        cadence = sorted(chunk_times)[len(chunk_times) // 2]
        print(
            f"chunks: {len(chunk_times)} | median chunk wall: {cadence:.1f} ms "
            f"({chunk_audio_sec * 1000 / cadence:.2f}x realtime chunk decode) | "
            f"peak: {peak:.0f} MB"
        )


if __name__ == "__main__":
    main()