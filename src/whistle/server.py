"""Minimal streaming TTS API server (FastAPI + uvicorn).

Endpoints:
- GET /health          model status
- GET /synthesize?text=...&speaker=...&language=...&chunk_size=12
                        streams a WAV (audio/wav) as codec chunks are decoded

Run (on the gpu box):
    uv run --no-sync python -m whistle.server --host 0.0.0.0 --port 8000
Test:
    curl -N 'http://127.0.0.1:8000/synthesize?text=Hello%20world' -o stream.wav
"""

import struct
import threading
from collections.abc import Generator

import click
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from qwen_tts import Qwen3TTSModel

from whistle.config import CHECKPOINT, LANGUAGE, SPEAKER
from whistle.streaming import stream_tts

app = FastAPI(title="whistle-tts", docs_url=None, redoc_url=None)
_model: Qwen3TTSModel | None = None
_device = "unset"
_sample_rate = 24_000
# Decoding mutates shared per-model state (rope deltas, graph input buffers),
# so concurrent requests must not interleave; they queue here instead.
_generate_lock = threading.Lock()


def get_model() -> Qwen3TTSModel:
    """Loads the official model once, lazily on the first request."""
    global _model, _device, _sample_rate
    if _model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _model = Qwen3TTSModel.from_pretrained(
            CHECKPOINT,
            device_map=device,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="sdpa",
        )
        _device = device
        _sample_rate = int(_model.model.speech_tokenizer.model.get_output_sample_rate())
        print(f"whistle server: model loaded on {_device}")
    return _model


def _wav_header(sample_rate: int) -> bytes:
    """WAV header with unknown-size fields (safe for chunked streaming)."""
    return b"".join(
        [
            b"RIFF", struct.pack("<I", 0xFFFFFFFF), b"WAVE",
            b"fmt ", struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16),
            b"data", struct.pack("<I", 0xFFFFFFFF),
        ]
    )


def _wav_chunks(
    text: str, speaker: str, language: str, chunk_size: int
) -> Generator[bytes, None, None]:
    """Yields WAV header + int16 PCM chunks as they are decoded."""
    tts = get_model()
    with _generate_lock:
        yield _wav_header(_sample_rate)
        for chunk in stream_tts(
            tts, text, speaker=speaker, language=language, chunk_size=chunk_size
        ):
            pcm = (chunk["audio"].float().cpu().numpy() * 32767.0).astype("<i2").tobytes()
            yield pcm


@app.get("/health")
def health() -> dict[str, object]:
    """Reports model load state and the device it resolved to."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": _device,
        "sample_rate": _sample_rate,
    }


@app.get("/synthesize")
def synthesize(
    text: str,
    speaker: str = SPEAKER,
    language: str = LANGUAGE,
    chunk_size: int = 12,
) -> StreamingResponse:
    """Streams synthesized speech as audio/wav, chunk by chunk."""
    return StreamingResponse(
        _wav_chunks(text, speaker, language, chunk_size),
        media_type="audio/wav",
        headers={"Cache-Control": "no-cache"},
    )


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", type=click.IntRange(min=1, max=65535), default=8000, show_default=True)
def main(host: str, port: int) -> None:
    """Warms the model, then serves the streaming TTS API."""
    get_model()
    uvicorn.run(app, host=host, port=port, log_level="warning")

if __name__ == "__main__":
    main()
