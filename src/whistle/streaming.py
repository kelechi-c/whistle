"""Streaming Qwen3-TTS: yields codec chunks plus incremental decoded audio.

``stream_tts`` runs the V7 ``predictor-ffn-graphs`` decode loop but yields a
chunk every ``chunk_size`` frames instead of collecting the whole utterance.
Each chunk is decoded to audio incrementally with the official 25-frame left
context (``decoder(codes[..., start-ctx:end])`` + context trim), so the first
audio chunk is available long before the utterance finishes.

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
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensLogitsProcessor,
)

from whistle.graphs import decode_graphs
from whistle.inference import _select_token, build_prompt

MAX_CACHE_LEN = 2_048
CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
Chunk = dict[str, Any]


def _streaming_decoder(speech_model: torch.nn.Module, left_context: int):
    """Returns a callable that decodes the next codec chunk with left context."""
    decoder = speech_model.decoder
    upsample = int(decoder.total_upsample)
    decoded_frames = 0

    def decode_next(codes_transposed: torch.Tensor) -> torch.Tensor:
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
    speaker: str = "serena",
    language: str = "english",
    max_new_tokens: int = 1_280,
    chunk_size: int = 12,
    left_context: int = 25,
    repetition_penalty: float = 1.2,
    stop_at_eos: bool = True,
) -> Generator[Chunk, None, None]:
    """Streams V7 decode: one yield per ``chunk_size`` frames with audio."""
    model = tts.model
    talker = model.talker
    predictor = talker.code_predictor
    talker_config = model.config.talker_config
    device = next(model.parameters()).device

    started = time.perf_counter()
    talker_input, attention_mask, tts_pad, prefill_length, codec_embeddings = build_prompt(
        tts, text, language=language, speaker=speaker, device=device
    )
    if prefill_length + max_new_tokens - 1 > MAX_CACHE_LEN:
        raise ValueError("prompt and frames exceed the fixed talker cache capacity")

    graphs = decode_graphs(talker, MAX_CACHE_LEN, "predictor-ffn-graphs")
    graphs.talker.reset(prefill_length)
    talker.rope_deltas = None

    talker_output = talker(
        inputs_embeds=talker_input,
        attention_mask=attention_mask,
        past_key_values=graphs.talker.cache,
        past_hidden=None,
        trailing_text_hidden=tts_pad,
        tts_pad_embed=tts_pad,
        generation_step=None,
        use_cache=True,
        return_dict=True,
    )
    eos_token_id = talker_config.codec_eos_token_id
    suppress_from = talker_config.vocab_size - 1_024
    suppress_tokens = [
        token_id
        for token_id in range(suppress_from, talker_config.vocab_size)
        if token_id != eos_token_id
    ]
    processors = LogitsProcessorList()
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    processors.append(SuppressTokensLogitsProcessor(suppress_tokens, device=device))
    primary_history = torch.empty((1, max_new_tokens), device=device, dtype=torch.long)
    token = _select_token(
        talker_output.logits,
        primary_history[:, :0],
        eos_token_id=eos_token_id,
        processors=processors,
        allow_eos=stop_at_eos,
    )
    past_hidden = talker_output.past_hidden
    graphs.talker.set_rope_deltas(talker.rope_deltas)
    torch.cuda.synchronize(device)
    ttft_ms = (time.perf_counter() - started) * 1000

    num_code_groups = talker_config.num_code_groups
    codes = torch.empty((max_new_tokens, num_code_groups), device=device, dtype=torch.long)
    codes_transposed = torch.empty(
        (1, num_code_groups, max_new_tokens), device=device, dtype=torch.long
    )
    predictor_input = torch.empty(
        (1, 2, talker_config.hidden_size), device=device, dtype=past_hidden.dtype
    )
    residual_embeddings = tuple(predictor.get_input_embeddings())
    speech_model = model.speech_tokenizer.model
    decode_next = _streaming_decoder(speech_model, left_context)
    sample_rate = int(speech_model.get_output_sample_rate())

    frame_count = 0
    chunk_start = time.perf_counter()

    for frame_index in range(max_new_tokens):
        if stop_at_eos and token.eq(eos_token_id).item():
            break
        last_id_hidden = codec_embeddings(token.view(1, 1))
        predictor_input[:, :1].copy_(past_hidden)
        predictor_input[:, 1:].copy_(last_id_hidden)
        residual_codes = graphs.predictor.run(predictor_input)
        codes[frame_index, 0].copy_(token[0])
        codes[frame_index, 1:].copy_(residual_codes[0])
        primary_history[:, frame_index].copy_(token)
        frame_count = frame_index + 1
        is_last = frame_index + 1 == max_new_tokens

        if not is_last:
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [
                    embedding(residual_codes[:, index : index + 1])
                    for index, embedding in enumerate(residual_embeddings)
                ],
                dim=1,
            )
            talker_input = codec_hiddens.sum(dim=1, keepdim=True) + tts_pad
            past_hidden = graphs.talker.run(talker_input, prefill_length + frame_index)
            token = _select_token(
                talker.codec_head(past_hidden),
                primary_history[:, :frame_count],
                eos_token_id=eos_token_id,
                processors=processors,
                allow_eos=stop_at_eos,
            )

        if frame_count % chunk_size == 0:
            codes_transposed[..., :frame_count].copy_(codes[:frame_count].t().unsqueeze(0))
            now = time.perf_counter()
            yield {
                "codes": codes[frame_count - chunk_size : frame_count].clone(),
                "audio": decode_next(codes_transposed[..., :frame_count]),
                "sample_rate": sample_rate,
                "chunk_frames": chunk_size,
                "ttft_ms": ttft_ms,
                "cumulative_ms": (now - started) * 1000,
                "chunk_ms": (now - chunk_start) * 1000,
                "final": is_last,
            }
            chunk_start = now
            if is_last:
                break

    remainder = frame_count % chunk_size
    if remainder and frame_count:
        codes_transposed[..., :frame_count].copy_(codes[:frame_count].t().unsqueeze(0))
        now = time.perf_counter()
        yield {
            "codes": codes[frame_count - remainder : frame_count].clone(),
            "audio": decode_next(codes_transposed[..., :frame_count]),
            "sample_rate": sample_rate,
            "chunk_frames": remainder,
            "ttft_ms": ttft_ms,
            "cumulative_ms": (now - started) * 1000,
            "chunk_ms": (now - chunk_start) * 1000,
            "final": True,
        }


@click.command()
@click.argument("text")
@click.option("--checkpoint", default=CHECKPOINT, show_default=True)
@click.option("--speaker", default="serena", show_default=True)
@click.option("--language", default="english", show_default=True)
@click.option("--max-new-tokens", type=click.IntRange(min=2), default=1_280)
@click.option("--chunk-size", type=click.IntRange(min=1), default=12, show_default=True)
@click.option("--left-context", type=click.IntRange(min=0), default=25, show_default=True)
def main(
    text: str,
    checkpoint: str,
    speaker: str,
    language: str,
    max_new_tokens: int,
    chunk_size: int,
    left_context: int,
) -> None:
    """Streams TEXT and prints per-chunk latency milestones."""
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
            left_context=left_context, stop_at_eos=False,
        )
    )
    torch.cuda.synchronize(device)
    for index, chunk in enumerate(
        stream_tts(tts, text, speaker=speaker, language=language,
                   max_new_tokens=max_new_tokens, chunk_size=chunk_size,
                   left_context=left_context)
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
