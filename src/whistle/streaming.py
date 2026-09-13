"""Streaming Qwen3-TTS: yields codec chunks plus incrementally decoded audio.

``stream_tts`` runs the batch decode loop but yields every ``chunk_size`` frames
instead of collecting the whole utterance. Each chunk is decoded with 25 frames
of left context, so the first audio chunk lands long before the utterance ends.
Chunk boundaries follow a ramp (``ramp_frames``) so the first frames ship early,
and the first chunk optionally drops leading silence below an RMS threshold.

EOS uses the same chunked device scan as the batch path (forced at every chunk
boundary), so no per-frame host sync is needed.

Yields dicts::

    {"codes": [chunk, 16] int64, "audio": [1, samples] float32,
     "sample_rate": int, "chunk_frames": int, "prefill_ms": float,
     "cumulative_ms": float, "chunk_ms": float, "final": bool}

``prefill_ms`` ends before any audio exists. ``cumulative_ms`` and ``chunk_ms``
are host timestamps taken just after that chunk's codec decode is enqueued; the
first chunk is additionally host-synchronized by the silence trim, so the first
``cumulative_ms`` is a true audio-ready milestone and later ones are enqueue
boundaries. Tensors stay on-device, so a consumer measures true first-audio
latency after its own copy.
"""

import time
from typing import Any, Generator

import torch
from qwen_tts import Qwen3TTSModel

from whistle.config import LANGUAGE, SPEAKER
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
    if audio.shape[-1] < window:
        return audio
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
    first = _prefill(
        tts,
        prompt,
        repetition_penalty=repetition_penalty,
        stop_at_eos=stop_at_eos,
        sampling=sampling,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - started) * 1000

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

    def chunk_dict(final: bool) -> Chunk:
        """Decodes the new frames, then records the audio-ready milestone."""
        nonlocal chunk_start_frame, chunk_started, emitted
        audio = decode_next(codes_transposed[..., :frame_count])
        if emitted == 0 and trim_leading_silence:
            audio = _trim_leading_silence(audio, sample_rate)
        now = time.perf_counter()
        emitted += 1
        payload = {
            "codes": codes[chunk_start_frame:frame_count].clone(),
            "audio": audio,
            "sample_rate": sample_rate,
            "chunk_frames": frame_count - chunk_start_frame,
            "prefill_ms": prefill_ms,
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
            yield chunk_dict(is_last)
        if is_last:
            break


__all__ = ["stream_tts"]
