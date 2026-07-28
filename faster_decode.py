"""Unoptimized official Qwen3-TTS prefill/decode baseline.

The hot path intentionally stays in one ``tts_infer`` function. It uses the
official model's embeddings, transformer forwards, dynamic caches, heads, and
codec so later optimizations can replace one clearly visible boundary at a time.
"""

from dataclasses import replace
import pathlib as pl
import time

import click
import numpy as np
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import torch

from nero.config import DTypeChoice, DeviceChoice, RUNTIME

CHECKPOINT = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def _synchronize(device: torch.device) -> None:
    """Makes phase timings include queued CUDA work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def tts_infer(
    tts: Qwen3TTSModel,
    text: str,
    *,
    speaker: str = "ryan",
    language: str = "english",
    max_new_tokens: int = 256,
    min_new_tokens: int = 2,
) -> tuple[list[np.ndarray], int, dict[str, float]]:
    """Runs batch-one CustomVoice inference through explicit forward passes.

    Prefill builds the complete non-streaming text/speaker prompt and fills the
    talker's dynamic KV cache. Decode predicts codebook zero from the talker,
    predicts the other codebooks with a fresh predictor cache for that frame,
    then feeds the summed codec embeddings through one cached talker step.
    """
    started = time.perf_counter()
    model = tts.model
    talker = model.talker
    predictor = talker.code_predictor
    config = model.config
    talker_config = config.talker_config
    device = next(model.parameters()).device

    tts._validate_languages([language])
    tts._validate_speakers([speaker])
    input_ids = tts._tokenize_texts([tts._build_assistant_text(text)])[0]
    language_key = language.lower()
    speaker_key = speaker.lower()
    language_id = (
        None
        if language_key == "auto"
        else talker_config.codec_language_id[language_key]
    )
    dialect = talker_config.spk_is_dialect[speaker_key]
    if language_key in {"auto", "chinese"} and dialect:
        language_id = talker_config.codec_language_id[dialect]
    speaker_id = talker_config.spk_id[speaker_key]

    text_embeddings = talker.get_text_embeddings()
    codec_embeddings = talker.get_input_embeddings()
    project_text = talker.text_projection
    token_dtype = input_ids.dtype

    speaker_embed = codec_embeddings(
        torch.tensor(speaker_id, device=device, dtype=token_dtype)
    ).view(1, 1, -1)
    special_text = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device,
        dtype=token_dtype,
    )
    tts_bos, tts_eos, tts_pad = project_text(
        text_embeddings(special_text)
    ).chunk(3, dim=1)

    codec_prefix = (
        [
            talker_config.codec_nothink_id,
            talker_config.codec_think_bos_id,
            talker_config.codec_think_eos_id,
        ]
        if language_id is None
        else [
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
        ]
    )
    codec_prefix_ids = torch.tensor(
        [codec_prefix],
        device=device,
        dtype=token_dtype,
    )
    codec_suffix_ids = torch.tensor(
        [[talker_config.codec_pad_id, talker_config.codec_bos_id]],
        device=device,
        dtype=token_dtype,
    )
    codec_prompt = torch.cat(
        [codec_embeddings(codec_prefix_ids), speaker_embed, codec_embeddings(codec_suffix_ids)],
        dim=1,
    )
    role = project_text(text_embeddings(input_ids[:, :3]))
    codec_header = torch.cat(
        [tts_pad.expand(-1, codec_prompt.shape[1] - 2, -1), tts_bos],
        dim=1,
    ) + codec_prompt[:, :-1]
    spoken_text = torch.cat(
        [project_text(text_embeddings(input_ids[:, 3:-5])), tts_eos],
        dim=1,
    )
    codec_pad = codec_embeddings(
        torch.full(
            (1, spoken_text.shape[1]),
            talker_config.codec_pad_id,
            device=device,
            dtype=token_dtype,
        )
    )
    codec_bos = codec_embeddings(
        torch.tensor(
            [[talker_config.codec_bos_id]], device=device, dtype=token_dtype
        )
    )
    talker_input = torch.cat(
        [role, codec_header, spoken_text + codec_pad, tts_pad + codec_bos],
        dim=1,
    )
    attention_mask = torch.ones(
        talker_input.shape[:2], device=device, dtype=torch.long
    )
    trailing_text = tts_pad
    talker.rope_deltas = None
    _synchronize(device)
    prepare_seconds = time.perf_counter() - started

    # === prefill: variable-length prompt, first codebook-zero token, dynamic KV ===
    phase_started = time.perf_counter()
    talker_output = talker(
        inputs_embeds=talker_input,
        attention_mask=attention_mask,
        past_key_values=None,
        past_hidden=None,
        trailing_text_hidden=trailing_text,
        tts_pad_embed=tts_pad,
        generation_step=None,
        use_cache=True,
        return_dict=True,
    )
    eos_id = talker_config.codec_eos_token_id
    suppress = torch.zeros(
        talker_config.vocab_size, device=device, dtype=torch.bool
    )
    suppress[predictor.config.vocab_size :] = True
    suppress[eos_id] = False
    first_logits = talker_output.logits[:, -1, :].clone()
    first_logits[:, suppress] = -torch.inf
    if min_new_tokens > 0:
        first_logits[:, eos_id] = -torch.inf
    token = first_logits.argmax(dim=-1)
    talker_cache = talker_output.past_key_values
    past_hidden = talker_output.past_hidden
    generation_step = int(talker_output.generation_step)
    _synchronize(device)
    prefill_seconds = time.perf_counter() - phase_started

    # === decode: predictor residual loop, then one cached talker forward per frame ===
    phase_started = time.perf_counter()
    frames: list[torch.Tensor] = []
    predictor_embeddings = predictor.get_input_embeddings()
    for frame_index in range(max_new_tokens):
        if int(token.item()) == eos_id:
            break

        last_id_hidden = codec_embeddings(token.view(1, 1))
        predictor_output = predictor(
            inputs_embeds=torch.cat([past_hidden, last_id_hidden], dim=1),
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        residual = predictor_output.logits[:, -1, :].argmax(dim=-1)
        residual_codes = [residual]
        for _ in range(1, talker_config.num_code_groups - 1):
            predictor_output = predictor(
                input_ids=residual.view(1, 1),
                past_key_values=predictor_output.past_key_values,
                generation_steps=predictor_output.generation_steps,
                use_cache=True,
                return_dict=True,
            )
            residual = predictor_output.logits[:, -1, :].argmax(dim=-1)
            residual_codes.append(residual)

        residual_tensor = torch.stack(residual_codes, dim=1)
        frames.append(torch.cat([token.view(1, 1), residual_tensor], dim=1)[0])
        if frame_index + 1 == max_new_tokens:
            break

        frame_embeddings = [last_id_hidden]
        frame_embeddings.extend(
            embedding(residual_tensor[:, index : index + 1])
            for index, embedding in enumerate(predictor_embeddings)
        )
        talker_input = torch.cat(frame_embeddings, dim=1).sum(dim=1, keepdim=True)
        conditioning = (
            trailing_text[:, generation_step : generation_step + 1]
            if generation_step < trailing_text.shape[1]
            else tts_pad
        )
        talker_input = talker_input + conditioning

        cache_position = torch.tensor(
            [talker_cache.get_seq_length()], device=device, dtype=torch.long
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
            ],
            dim=1,
        )
        position_ids = (
            cache_position[0] + talker.rope_deltas
        ).unsqueeze(0).expand(3, -1, -1)
        backbone_output = talker.model(
            inputs_embeds=talker_input,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=talker_cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        past_hidden = backbone_output.last_hidden_state[:, -1:, :]
        talker_cache = backbone_output.past_key_values
        generation_step += 1
        logits = talker.codec_head(past_hidden)[:, -1, :].clone()
        logits[:, suppress] = -torch.inf
        if len(frames) < min_new_tokens:
            logits[:, eos_id] = -torch.inf
        token = logits.argmax(dim=-1)

    _synchronize(device)
    decode_seconds = time.perf_counter() - phase_started
    if not frames:
        raise RuntimeError("generation stopped before producing an audio frame")
    codes = torch.stack(frames)

    # === codec: unchanged official 12 Hz codes-to-waveform decoder ===
    phase_started = time.perf_counter()
    wavs, sample_rate = model.speech_tokenizer.decode([{"audio_codes": codes}])
    _synchronize(device)
    codec_seconds = time.perf_counter() - phase_started
    timings = {
        "prepare": prepare_seconds,
        "prefill": prefill_seconds,
        "decode": decode_seconds,
        "codec": codec_seconds,
        "total": time.perf_counter() - started,
        "frames": float(codes.shape[0]),
    }
    return wavs, sample_rate, timings


def _load_model(
    checkpoint: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3TTSModel:
    """Loads the official wrapper outside the measured inference path."""
    return Qwen3TTSModel.from_pretrained(
        checkpoint,
        device_map=str(device),
        dtype=dtype,
        attn_implementation="sdpa",
    )


@click.command()
@click.argument("text")
@click.option("--checkpoint", default=CHECKPOINT, show_default=True)
@click.option("--speaker", default="ryan", show_default=True)
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
    speaker: str,
    language: str,
    max_frames: int,
    device_choice: DeviceChoice,
    dtype_choice: DTypeChoice,
    out: pl.Path,
) -> None:
    """Runs the explicit official Qwen3-TTS baseline for TEXT."""
    runtime = replace(
        RUNTIME,
        device=device_choice,
        dtype=dtype_choice,
        max_frames=max_frames,
        output=out,
    )
    device = runtime.resolved_device()
    dtype = runtime.resolved_dtype(device)
    torch.manual_seed(runtime.seed)
    model = _load_model(checkpoint, device, dtype)
    wavs, sample_rate, timings = tts_infer(
        model,
        text,
        speaker=speaker,
        language=language,
        max_new_tokens=max_frames,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wavs[0], sample_rate)
    print(f"audio saved to {out}")
    print(
        f"frames: {int(timings['frames'])}; "
        f"prefill: {timings['prefill'] * 1000:.2f} ms; "
        f"decode: {timings['decode'] * 1000:.2f} ms"
    )
    print(
        f"prepare: {timings['prepare'] * 1000:.2f} ms; "
        f"codec: {timings['codec'] * 1000:.2f} ms; "
        f"total: {timings['total']:.3f} s"
    )


if __name__ == "__main__":
    main()
