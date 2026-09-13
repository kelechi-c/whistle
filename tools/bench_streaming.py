"""
Streaming latency bench: time to first CPU-ready audio, cadence, and total.
Run on a GPU: uv run --no-sync python tools/bench_streaming.py --text-file alicia.txt
"""

import json
import pathlib as pl
import statistics
import time

import click
import torch
from qwen_tts import Qwen3TTSModel

from whistle.config import CHECKPOINT, SEED, SPEAKER
from whistle.streaming import stream_tts


def _ramp(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(",") if part.strip())


@click.command()
@click.option("--text-file", type=click.Path(path_type=pl.Path, exists=True), default=None)
@click.option("--text", default=None, help="inline text instead of --text-file")
@click.option("--speaker", default=SPEAKER, show_default=True)
@click.option("--language", default="english", show_default=True)
@click.option("--chunk-size", type=click.IntRange(min=1), default=12, show_default=True)
@click.option("--ramp", default="2,4,8", show_default=True, help="first-chunk frame schedule (empty string disables)")
@click.option("--no-trim", is_flag=True, help="keep leading silence in the first chunk")
@click.option("--max-new-tokens", type=click.IntRange(min=2), default=1_280)
@click.option("--iterations", type=click.IntRange(min=1), default=3, show_default=True)
@click.option("--warmup", type=click.IntRange(min=0), default=1, show_default=True)
@click.option("--json-out", type=click.Path(path_type=pl.Path), default=None)
def main(
    text_file: pl.Path | None,
    text: str | None,
    speaker: str,
    language: str,
    chunk_size: int,
    ramp: str,
    no_trim: bool,
    max_new_tokens: int,
    iterations: int,
    warmup: int,
    json_out: pl.Path | None,
) -> None:
    """Benchmarks stream_tts first-audio latency and chunk cadence on one text."""
    if text is None and text_file is None:
        raise click.ClickException("provide --text or --text-file")
    payload = text if text is not None else text_file.read_text(encoding="utf-8")
    ramp_frames = _ramp(ramp)
    trim = not no_trim
    torch.manual_seed(SEED)
    model = Qwen3TTSModel.from_pretrained(
        CHECKPOINT, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    stream = dict(
        speaker=speaker, language=language, chunk_size=chunk_size,
        ramp_frames=ramp_frames, trim_leading_silence=trim,
    )
    for index in range(warmup):
        print(f"warmup {index + 1}/{warmup}")
        list(stream_tts(model, payload[:64], max_new_tokens=64, **stream))

    rows = []
    for index in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        first_audio_ms = None
        chunk_times = []
        frames = 0
        last_ready = started
        audio_ms = 0.0
        for chunk in stream_tts(model, payload, max_new_tokens=max_new_tokens, **stream):
            chunk["audio"].detach().cpu()
            ready = time.perf_counter()
            if first_audio_ms is None:
                first_audio_ms = (ready - started) * 1000
            chunk_times.append((ready - last_ready) * 1000)
            last_ready = ready
            frames += chunk["chunk_frames"]
            audio_ms += chunk["audio"].shape[-1] / chunk["sample_rate"] * 1000
        torch.cuda.synchronize()
        row = {
            "first_audio_ms": first_audio_ms,
            "total_ms": (time.perf_counter() - started) * 1000,
            "frames": frames,
            "audio_ms": audio_ms,
            "median_chunk_ms": statistics.median(chunk_times) if chunk_times else None,
        }
        rows.append(row)
        print(
            f"iteration {index + 1}: first audio {row['first_audio_ms']:.1f} ms, "
            f"total {row['total_ms']:.1f} ms, {frames} frames, "
            f"{audio_ms / 1000:.2f} s audio"
        )

    first_audio = statistics.median(row["first_audio_ms"] for row in rows)
    total_ms = statistics.median(row["total_ms"] for row in rows)
    frames = statistics.median(row["frames"] for row in rows)
    audio_ms = statistics.median(row["audio_ms"] for row in rows)
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"text chars: {len(payload)} | frames: {frames} | "
        f"first cpu-ready audio: {first_audio:.1f} ms | total: {total_ms:.1f} ms | "
        f"audio: {audio_ms:.1f} ms | peak: {peak:.0f} MB"
    )
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps({
            "text_file": str(text_file) if text_file is not None else None,
            "text_characters": len(payload),
            "chunk_size": chunk_size,
            "ramp_frames": list(ramp_frames),
            "trim_leading_silence": trim,
            "speaker": speaker,
            "language": language,
            "max_new_tokens": max_new_tokens,
            "warmup": warmup,
            "iterations": rows,
            "median_first_audio_ms": first_audio,
            "median_total_ms": total_ms,
            "peak_memory_mb": peak,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"results saved to {json_out}")


if __name__ == "__main__":
    main()
