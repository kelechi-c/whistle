"""Streaming Qwen3-TTS: yields codec chunks plus incremental decoded audio.

``stream_tts`` runs the V7 ``predictor-ffn-graphs`` decode loop but yields a
chunk every ``chunk_size`` frames instead of collecting the whole utterance.
Each chunk is decoded to audio incrementally with the official 25-frame left
context (``decoder(codes[..., start-ctx:end])`` + context trim), so the first
audio chunk is available long before the utterance finishes.

Track A latency work: chunk boundaries follow a ramp schedule — the first
``ramp_frames`` (default 2) frames ship immediately, subsequent chunks grow
stepwise up to the steady-state ``chunk_size`` — cutting TTFA roughly by
ramp[0] frames of decode time while later chunks keep playback headroom. The
first chunk optionally drops leading silence below an RMS threshold (the
nari-style dynamic trim; streaming audio already deviates bitwise from the
official chunked(300, 25) decode at its own boundaries, so this only widens
an existing, inaudible class of deviation). The transposed codec buffer is
filled incrementally instead of re-copied per chunk.

EOS handling matches ``inference.tts_infer``: no per-frame host sync. The
chunked device scan runs at every ``EOS_CHECK_EVERY`` frames, at every chunk
boundary (forced so a chunk never contains stale EOS frames), and at the final
frame; the trim semantics are identical to the batch path.

Yields dicts::

    {"codes": [chunk, 16] int64, "audio": [1, samples] float32,
     "sample_rate": int, "chunk_frames": int, "ttft_ms": float,
     "cumulative_ms": float, "chunk_ms": float, "final": bool}

All tensors stay on-device; the caller transfers what it needs.
"""

import time
from typing import Any, Generator

import click
import torch
from qwen_tts import Qwen3TTSModel

from whistle.config import CHECKPOINT, LANGUAGE, SPEAKER
from whistle.inference import (
    _maybe_eos_row,
    _prefill,
    _prepare,
    _select_token,
)

Chunk = dict[str, Any]


def _trim_leading_silence(
    audio: torch.Tensor,
    sample_rate: int,
    threshold: float = 0.002,
    lead_ms: float = 20.0,
) -> torch.Tensor:
    """Drops samples before the first 10 ms RMS window above ``threshold``.

    Keeps a short lead-in (default 20 ms) so the first phoneme is not clipped;
    quiet audio below the threshold passes through untouched. One tiny host
    sync per request, applied only to the first chunk.
    """
    if audio.shape[-1] == 0:
        return audio
    window = max(1, sample_rate // 100)
    rms = audio.unfold(-1, window, window).pow(2).mean(-1).sqrt()
    loud = (rms > threshold).nonzero()
    if loud.numel() == 0:
        return audio
    onset = int(loud[0, -1]) * window
    lead = min(onset, int(sample_rate * lead_ms / 1000))
    return audio[..., onset - lead:]


def _streaming_decoder(speech_model: torch.nn.Module, left_context: int):
    """Returns a callable that decodes the next codec chunk with left context."""
    decoder = speech_model.decoder
    upsample = int(decoder.total_upsample)
    decoded_frames = 0

    def decode_next(codes_transposed: torch.Tensor) -> torch.Tensor:
        """Decodes codes[..., start-ctx:end] and trims the context warmup samples."""
        nonlocal decoded_frames
        start = decoded_frames
        end = codes_transposed.shape[-1]
        context = min(left_context, start)
        wav = decoder(codes_transposed[..., start - context : end])
        decoded_frames = end
        return wav[..., context * upsample :]

    return decode_next


@torch.inference_mode()
def stream_tts(
    tts: Qwen3TTSModel,
    text: str,
    *,
    speaker: str = SPEAKER,
    language: str = LANGUAGE,
    max_new_tokens: int = 1_280,
    chunk_size: int = 12,
    ramp_frames: tuple[int, ...] = (2, 4, 8),
    trim_leading_silence: bool = True,
    left_context: int = 25,
    repetition_penalty: float = 1.2,
    temperature: float | None = None,
    top_k: int = 50,
    stop_at_eos: bool = True,
) -> Generator[Chunk, None, None]:
    """Streams V7 decode with ramped chunk boundaries and incremental audio.

    Boundaries fire at ``ramp_frames`` counts first, then every ``chunk_size``
    frames; ``ramp_frames=()`` restores the fixed-cadence behavior.
    """
    model = tts.model
    talker = model.talker
    device = next(model.parameters()).device
    sampling = (
        {"temperature": temperature, "top_k": top_k} if temperature is not None else None
    )

    started = time.perf_counter()
    prompt = _prepare(
        tts, text, speaker=speaker, language=language, device=device, max_new_tokens=max_new_tokens
    )
    first = _prefill(tts, prompt, repetition_penalty=repetition_penalty, stop_at_eos=stop_at_eos)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ttft_ms = (time.perf_counter() - started) * 1000

    codes = torch.empty((max_new_tokens, first.num_code_groups), device=device, dtype=torch.long)
    codes_transposed = torch.empty(
        (1, first.num_code_groups, max_new_tokens), device=device, dtype=torch.long
    )
    predictor_input = torch.empty(
        (1, 2, first.hidden_size), device=device, dtype=first.past_hidden.dtype
    )
    speech_model = model.speech_tokenizer.model
    decode_next = _streaming_decoder(speech_model, left_context)
    sample_rate = int(speech_model.get_output_sample_rate())

    token = first.token
    past_hidden = first.past_hidden
    frame_count = 0
    chunk_start_frame = 0
    chunk_started = time.perf_counter()
    schedule = list(ramp_frames)
    next_boundary = schedule.pop(0) if schedule else chunk_size
    emitted = 0

    def chunk_dict(final: bool, now: float) -> Chunk:
        """Assembles one yield, trims first-chunk silence, resets the cadence clock."""
        nonlocal chunk_start_frame, chunk_started, emitted
        audio = decode_next(codes_transposed[..., :frame_count])
        if emitted == 0 and trim_leading_silence:
            audio = _trim_leading_silence(audio, sample_rate)
        emitted += 1
        payload = {
            "codes": codes[chunk_start_frame:frame_count].clone(),
            "audio": audio,
            "sample_rate": sample_rate,
            "chunk_frames": frame_count - chunk_start_frame,
            "ttft_ms": ttft_ms,
            "cumulative_ms": (now - started) * 1000,
            "chunk_ms": (now - chunk_started) * 1000,
            "final": final,
        }
        chunk_start_frame = frame_count
        chunk_started = now
        return payload

    for frame_index in range(max_new_tokens):
        last_id_hidden = prompt.codec_embeddings(token.view(1, 1))
        predictor_input[:, :1].copy_(past_hidden)
        predictor_input[:, 1:].copy_(last_id_hidden)
        residual_codes = prompt.graphs.predictor.run(predictor_input, sampling=sampling)
        codes[frame_index, 0].copy_(token[0])
        codes[frame_index, 1:].copy_(residual_codes[0])
        prompt.primary_history[:, frame_index].copy_(token)
        frame_count = frame_index + 1
        boundary = frame_count == next_boundary
        if boundary:
            next_boundary = schedule.pop(0) if schedule else next_boundary + chunk_size
        is_last = frame_index + 1 == max_new_tokens
        hit = _maybe_eos_row(
            codes,
            frame_count,
            first.eos_token_id,
            stop_at_eos=stop_at_eos,
            force=boundary or is_last,
        )
        if hit is not None:
            frame_count = hit
            is_last = True
        elif not is_last:
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [
                    embedding(residual_codes[:, index : index + 1])
                    for index, embedding in enumerate(first.residual_embeddings)
                ],
                dim=1,
            )
            talker_input = codec_hiddens.sum(dim=1, keepdim=True) + prompt.tts_pad
            past_hidden = prompt.graphs.talker.run(talker_input, prompt.prefill_length + frame_index)
            token = _select_token(
                talker.codec_head(past_hidden),
                prompt.primary_history[:, :frame_count],
                eos_token_id=first.eos_token_id,
                processors=first.processors,
                allow_eos=stop_at_eos,
                sampling=sampling,
            )

        if not (is_last or boundary):
            continue
        if frame_count > chunk_start_frame:
            codes_transposed[..., chunk_start_frame:frame_count].copy_(
                codes[chunk_start_frame:frame_count].t().unsqueeze(0)
            )
            now = time.perf_counter()
            yield chunk_dict(is_last, now)
        if is_last:
            break


@click.command()
@click.argument("text")
@click.option("--checkpoint", default=CHECKPOINT, show_default=True)
@click.option("--speaker", default=SPEAKER, show_default=True)
@click.option("--language", default=LANGUAGE, show_default=True)
@click.option("--max-new-tokens", type=click.IntRange(min=2), default=1_280)
@click.option("--chunk-size", type=click.IntRange(min=1), default=12, show_default=True)
@click.option("--ramp", default="2,4,8", show_default=True, help="comma frame counts for the first chunks (empty string disables)")
@click.option("--no-trim", is_flag=True, help="keep leading silence in the first chunk")
@click.option("--left-context", type=click.IntRange(min=0), default=25, show_default=True)
@click.option("--temperature", type=click.FloatRange(min=0.01), default=None, help="enable do_sample with this temperature")
@click.option("--top-k", type=click.IntRange(min=1), default=50, show_default=True)
def main(
    text: str,
    checkpoint: str,
    speaker: str,
    language: str,
    max_new_tokens: int,
    chunk_size: int,
    ramp: str,
    no_trim: bool,
    left_context: int,
    temperature: float | None,
    top_k: int,
) -> None:
    """Streams TEXT and prints per-chunk latency milestones."""
    ramp_frames = tuple(int(part) for part in ramp.split(",") if part.strip())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts = Qwen3TTSModel.from_pretrained(
        checkpoint,
        device_map=str(device),
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    list(
        stream_tts(
            tts, "warmup.", max_new_tokens=chunk_size + 2, chunk_size=chunk_size,
            ramp_frames=ramp_frames, trim_leading_silence=not no_trim,
            left_context=left_context, stop_at_eos=False,
        )
    )
    torch.cuda.synchronize(device)
    for index, chunk in enumerate(
        stream_tts(tts, text, speaker=speaker, language=language,
                   max_new_tokens=max_new_tokens, chunk_size=chunk_size,
                   ramp_frames=ramp_frames, trim_leading_silence=not no_trim,
                   left_context=left_context, temperature=temperature, top_k=top_k)
    ):
        audio_samples = chunk["audio"].shape[-1]
        print(
            f"chunk {index}: frames={chunk['chunk_frames']} "
            f"cumulative={chunk['cumulative_ms']:.1f}ms "
            f"chunk={chunk['chunk_ms']:.1f}ms audio={audio_samples} samples "
            f"ttft={chunk['ttft_ms']:.1f}ms final={chunk['final']}"
        )


__all__ = ["stream_tts"]


if __name__ == "__main__":
    main()
