#!/usr/bin/env python3
"""Streaming latency bench for the V7 stream path (TTFA + chunk cadence).

Measures, per input text: time-to-first-audio (the first decoded chunk),
per-chunk wall cadence, total wall, emitted frames, and audio duration.
Run on the GPU box:
    PYTHONPATH=src .venv/bin/python tools/bench_streaming.py --text-file testdata/alicia.txt
"""

import json
import time

import click
import pathlib as pl
import torch
from qwen_tts import Qwen3TTSModel

from whistle.config import CHECKPOINT, SPEAKER
from whistle.streaming import stream_tts


@click.command()
@click.option("--text-file", type=click.Path(path_type=pl.Path, exists=True), default=None)
@click.option("--text", default=None, help="inline text (positional alternative)")
@click.option("--speaker", default=SPEAKER, show_default=True)
@click.option("--chunk-size", type=click.IntRange(min=1), default=12, show_default=True)
@click.option("--max-new-tokens", type=click.IntRange(min=2), default=1_280)
@click.option("--iterations", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--warmup", type=click.IntRange(min=0), default=1, show_default=True)
@click.option("--json-out", type=click.Path(path_type=pl.Path), default=None)
def main(
    text_file: pl.Path | None,
    text: str | None,
    speaker: str,
    chunk_size: int,
    max_new_tokens: int,
    iterations: int,
    warmup: int,
    json_out: pl.Path | None,
) -> None:
    """Benchmarks stream_tts TTFA and chunk cadence on one text."""
    payload = text if text is not None else text_file.read_text(encoding="utf-8")
    torch.manual_seed(0)
    model = Qwen3TTSModel.from_pretrained(
        CHECKPOINT, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    for _ in range(warmup):
        list(stream_tts(model, payload[:64], speaker=speaker, chunk_size=chunk_size, max_new_tokens=64))

    rows = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        first_audio_ms = None
        chunk_times = []
        frames = 0
        last_ready = started
        audio_ms = 0.0
        for ch in stream_tts(model, payload, speaker=speaker, chunk_size=chunk_size, max_new_tokens=max_new_tokens):
            ch["audio"].detach().cpu()
            ready = time.perf_counter()
            if first_audio_ms is None:
                first_audio_ms = (ready - started) * 1000
            chunk_times.append((ready - last_ready) * 1000)
            last_ready = ready
            frames += ch["chunk_frames"]
            audio_ms += ch["audio"].shape[-1] / ch["sample_rate"] * 1000
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - started) * 1000
        rows.append({
            "ttfa_ms": first_audio_ms,
            "total_ms": total_ms,
            "frames": frames,
            "audio_ms": audio_ms,
            "median_chunk_ms": sorted(chunk_times)[len(chunk_times) // 2] if chunk_times else None,
        })

    ttft = sorted(row["ttfa_ms"] for row in rows)[len(rows) // 2]
    total_ms = sorted(row["total_ms"] for row in rows)[len(rows) // 2]
    frames = rows[-1]["frames"]
    audio_ms = rows[-1]["audio_ms"]
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"text chars: {len(payload)} | frames: {frames} | "
        f"ttfa: {ttft:.1f} ms | total: {total_ms:.1f} ms | "
        f"audio: {audio_ms:.1f} ms"
    )
    if rows[-1]["median_chunk_ms"] is not None:
        cadence = sorted(row["median_chunk_ms"] for row in rows)[len(rows) // 2]
        print(
            f"median chunk wall: {cadence:.1f} ms | "
            f"peak: {peak:.0f} MB"
        )
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps({
            "text_characters": len(payload),
            "chunk_size": chunk_size,
            "max_new_tokens": max_new_tokens,
            "iterations": rows,
            "median_ttfa_ms": ttft,
            "median_total_ms": total_ms,
            "peak_memory_mb": peak,
        }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
