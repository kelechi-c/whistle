"""Low-overhead official Qwen3-TTS prefill/decode baseline.

The hot path keeps request scheduling in ``tts_infer`` and delegates the two
fixed-shape decode blocks to ``graphs.py``. Official modules and weights remain
unchanged while decode state and completed outputs stay on-device.
"""

from dataclasses import dataclass
import time
import torch
from qwen_tts import Qwen3TTSModel
from whistle.config import MAX_CACHE_LEN
from whistle.graphs import DecodeGraphs, decode_graphs, sample_token
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensLogitsProcessor,
)

EOS_CHECK_EVERY = 8


def _maybe_eos_row(
    codes: torch.Tensor,
    upto: int,
    eos_token_id: int,
    *,
    stop_at_eos: bool,
    force: bool,
) -> int | None:
    """Returns the first EOS frame index below ``upto``, at the check cadence.

    Replaces the per-frame ``token.eq(eos).item()`` host sync with a device
    scan every ``EOS_CHECK_EVERY`` frames (plus any forced frame), so the GPU
    never drains per frame and the emitted trim stays identical. Callers must
    force a check before any point where ``codes`` becomes observable, such
    as a streaming chunk boundary.
    """
    if not stop_at_eos or (upto % EOS_CHECK_EVERY != 0 and not force):
        return None
    hit = (codes[:upto, 0] == eos_token_id).nonzero()
    return int(hit[0, 0]) if hit.numel() else None


def _select_token(
    logits: torch.Tensor,
    history: torch.Tensor,
    *,
    eos_token_id: int,
    processors: LogitsProcessorList,
    allow_eos: bool,
    sampling: dict[str, float] | None = None,
) -> torch.Tensor:
    """Applies official processors and returns one greedy or sampled token."""
    scores = processors(history, logits[:, -1].to(dtype=torch.float32, copy=True))
    if not allow_eos or history.shape[1] < 2:
        scores[:, eos_token_id] = -torch.inf
    if sampling:
        return sample_token(scores, sampling["temperature"], sampling.get("top_k"))
    return scores.argmax(dim=-1)


@dataclass(frozen=True, slots=True)
class Prompt:
    """Named prompt-build outputs plus the reset shared decode graphs."""

    talker_input: torch.Tensor
    attention_mask: torch.Tensor
    tts_pad: torch.Tensor
    primary_history: torch.Tensor
    codec_embeddings: torch.nn.Module
    graphs: DecodeGraphs
    prefill_length: int


@dataclass(frozen=True, slots=True)
class Prefill:
    """First-frame decode state shared by the batch and streaming loops."""

    token: torch.Tensor
    past_hidden: torch.Tensor
    processors: LogitsProcessorList
    residual_embeddings: tuple[torch.nn.Module, ...]
    eos_token_id: int
    num_code_groups: int
    hidden_size: int


def _prepare(
    tts: Qwen3TTSModel,
    text: str,
    *,
    speaker: str,
    language: str,
    device: torch.device,
    max_new_tokens: int,
) -> Prompt:
    """Builds the prompt tensors, checks cache capacity, and resets decode state."""
    talker = tts.model.talker
    talker_input, attention_mask, tts_pad, prefill_length, codec_embeddings = build_prompt(
        tts, text, language=language, speaker=speaker, device=device
    )
    if prefill_length + max_new_tokens - 1 > MAX_CACHE_LEN:
        raise ValueError("prompt and frames exceed the fixed talker cache capacity")
    graphs = decode_graphs(talker, MAX_CACHE_LEN)
    graphs.talker.reset(prefill_length)
    talker.rope_deltas = None
    return Prompt(
        talker_input=talker_input,
        attention_mask=attention_mask,
        tts_pad=tts_pad,
        primary_history=torch.empty((1, max_new_tokens), device=device, dtype=torch.long),
        codec_embeddings=codec_embeddings,
        graphs=graphs,
        prefill_length=prefill_length,
    )


def _prefill(
    tts: Qwen3TTSModel,
    prompt: Prompt,
    *,
    repetition_penalty: float,
    stop_at_eos: bool,
) -> Prefill:
    """Runs the prefill forward and selects the first primary token."""
    model = tts.model
    talker = model.talker
    talker_config = model.config.talker_config
    talker_output = talker(
        inputs_embeds=prompt.talker_input,
        attention_mask=prompt.attention_mask,
        past_key_values=prompt.graphs.talker.cache,
        past_hidden=None,
        trailing_text_hidden=prompt.tts_pad,
        tts_pad_embed=prompt.tts_pad,
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
    processors.append(
        SuppressTokensLogitsProcessor(suppress_tokens, device=prompt.graphs.talker.device)
    )
    token = _select_token(
        talker_output.logits,
        prompt.primary_history[:, :0],
        eos_token_id=eos_token_id,
        processors=processors,
        allow_eos=stop_at_eos,
    )
    prompt.graphs.talker.set_rope_deltas(talker.rope_deltas)
    return Prefill(
        token=token,
        past_hidden=talker_output.past_hidden,
        processors=processors,
        residual_embeddings=tuple(talker.code_predictor.get_input_embeddings()),
        eos_token_id=eos_token_id,
        num_code_groups=talker_config.num_code_groups,
        hidden_size=talker_config.hidden_size,
    )


def build_prompt(
    tts: Qwen3TTSModel,
    text: str,
    *,
    language: str,
    speaker: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.nn.Module]:
    """Builds the official CustomVoice prompt tensors (verbatim from tts_infer)."""
    talker = tts.model.talker
    config = tts.model.config
    talker_config = config.talker_config

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

    speaker_embed = codec_embeddings(torch.tensor(speaker_id, device=device)).view(1, 1, -1)
    special_text = torch.tensor(
        [[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]],
        device=device,
        dtype=token_dtype,
    )
    tts_bos, tts_eos, tts_pad = project_text(text_embeddings(special_text)).chunk(3, dim=1)

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
    codec_prefix_ids = torch.tensor([codec_prefix], device=device, dtype=token_dtype)
    codec_suffix_ids = torch.tensor(
        [[talker_config.codec_pad_id, talker_config.codec_bos_id]], device=device
    )
    codec_prompt = torch.cat(
        [codec_embeddings(codec_prefix_ids), speaker_embed, codec_embeddings(codec_suffix_ids)],
        dim=1,
    )
    role = project_text(text_embeddings(input_ids[:, :3]))
    codec_header = torch.cat(
        [tts_pad.expand(-1, codec_prompt.shape[1] - 2, -1), tts_bos], dim=1
    ) + codec_prompt[:, :-1]
    spoken_text = torch.cat([project_text(text_embeddings(input_ids[:, 3:-5])), tts_eos], dim=1)
    codec_pad = codec_embeddings(
        torch.full(
            (1, spoken_text.shape[1]),
            talker_config.codec_pad_id,
            device=device,
            dtype=token_dtype,
        )
    )
    codec_bos = codec_embeddings(torch.tensor([[talker_config.codec_bos_id]], device=device))
    talker_input = torch.cat(
        [role, codec_header, spoken_text + codec_pad, tts_pad + codec_bos],
        dim=1,
    )
    attention_mask = torch.ones(talker_input.shape[:2], device=device, dtype=torch.long)
    return talker_input, attention_mask, tts_pad, talker_input.shape[1], codec_embeddings


CODEC_LEFT_CONTEXT = 25


def _flush_codec(
    codes: torch.Tensor,
    decoder: torch.nn.Module,
    upsample: int,
    parts: list[torch.Tensor],
    state: dict[str, int],
    side_stream: torch.cuda.Stream,
    end: int,
) -> None:
    """Decodes frames [state['flushed']:end] on the side stream with left context.

    Replicates the official ``chunked_decode`` inner loop (25-frame left
    context, front-trimmed output) but at an arbitrary flush cadence so codec
    work overlaps the ongoing talker decode. Events keep the codes writes and
    the side-stream reads ordered without host syncs.
    """
    start = state["flushed"]
    if end <= start:
        return
    context = min(CODEC_LEFT_CONTEXT, start)
    ready = torch.cuda.Event()
    ready.record()
    side_stream.wait_event(ready)
    with torch.cuda.stream(side_stream):
        chunk = codes[start - context : end].t().unsqueeze(0).contiguous()
        wav = decoder(chunk)
        parts.append(wav[..., context * upsample :])
    state["flushed"] = end


@torch.inference_mode()
def tts_infer(
    tts: Qwen3TTSModel,
    text: str,
    *,
    speaker: str = "ryan",
    language: str = "english",
    max_new_tokens: int = 1_280,
    stop_at_eos: bool = True,
    repetition_penalty: float = 1.2,
    temperature: float | None = None,
    top_k: int = 50,
    overlap_codec: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float]]:
    """Runs batch-one CustomVoice inference through explicit forward passes.

    Prefill builds the complete non-streaming text/speaker prompt and fills the
    persistent talker cache. Decode emits at most ``max_new_tokens`` frames and
    normally stops before codec EOS. Tokens and waveform remain on-device until
    the caller explicitly transfers the completed outputs.

    ``temperature`` enables official-style do_sample decoding (top-k + softmax
    + multinomial); the predictor runs its captured sampled graph set. When
    ``overlap_codec`` is set (CUDA only), completed frames are decoded
    incrementally on a side stream during the decode loop. That deviates from
    the official chunked(300, 25) decode at chunk boundaries, so the waveform
    is no longer bitwise-identical to the official full decode (codec tokens
    are unchanged); greedy default keeps the exact-parity contract.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")

    started = time.perf_counter()
    device = next(tts.model.parameters()).device
    cuda_timing = device.type == "cuda"
    sampling = (
        {"temperature": temperature, "top_k": top_k} if temperature is not None else None
    )
    phase_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if cuda_timing else None
    if phase_events is not None:
        phase_events[0].record()
    cpu_phase_started = started

    prompt = _prepare(
        tts, text, speaker=speaker, language=language, device=device, max_new_tokens=max_new_tokens
    )
    if phase_events is not None:
        phase_events[1].record()
        prepare_seconds = 0.0
    else:
        now = time.perf_counter()
        prepare_seconds = now - cpu_phase_started
        cpu_phase_started = now

    # === prefill: variable prompt into the selected talker cache ===
    first = _prefill(tts, prompt, repetition_penalty=repetition_penalty, stop_at_eos=stop_at_eos)
    talker = tts.model.talker
    if phase_events is not None:
        phase_events[2].record()
        prefill_seconds = 0.0
    else:
        now = time.perf_counter()
        prefill_seconds = now - cpu_phase_started
        cpu_phase_started = now

    # === decode: official token processing with bounded on-device outputs ===
    codes = torch.empty((max_new_tokens, first.num_code_groups), device=device, dtype=torch.long)
    predictor_input = torch.empty(
        (1, 2, first.hidden_size),
        device=device,
        dtype=first.past_hidden.dtype,
    )
    token = first.token
    past_hidden = first.past_hidden
    frame_count = 0

    speech_model = tts.model.speech_tokenizer.model
    overlap = overlap_codec and cuda_timing
    if overlap:
        side_stream = torch.cuda.Stream(device=device)
        wav_parts: list[torch.Tensor] = []
        flush_state = {"flushed": 0}
        codec_decoder = speech_model.decoder
        upsample = int(codec_decoder.total_upsample)

    for frame_index in range(max_new_tokens):
        last_id_hidden = prompt.codec_embeddings(token.view(1, 1))
        predictor_input[:, :1].copy_(past_hidden)
        predictor_input[:, 1:].copy_(last_id_hidden)
        residual_codes = prompt.graphs.predictor.run(predictor_input, sampling=sampling)
        codes[frame_index, 0].copy_(token[0])
        codes[frame_index, 1:].copy_(residual_codes[0])
        prompt.primary_history[:, frame_index].copy_(token)
        frame_count = frame_index + 1
        hit = _maybe_eos_row(
            codes,
            frame_count,
            first.eos_token_id,
            stop_at_eos=stop_at_eos,
            force=frame_index + 1 == max_new_tokens,
        )
        if hit is not None:
            frame_count = hit
            break
        if frame_index + 1 == max_new_tokens:
            break

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
        if overlap and frame_count % EOS_CHECK_EVERY == 0:
            _flush_codec(codes, codec_decoder, upsample, wav_parts, flush_state, side_stream, frame_count)

    if phase_events is not None:
        phase_events[3].record()
        decode_seconds = 0.0
    else:
        now = time.perf_counter()
        decode_seconds = now - cpu_phase_started
        cpu_phase_started = now

    # === codec: official decoder implementation, kept on-device ===
    codes = codes[:frame_count]
    if frame_count and overlap:
        _flush_codec(codes, codec_decoder, upsample, wav_parts, flush_state, side_stream, frame_count)
        torch.cuda.current_stream(device).wait_stream(side_stream)
        waveform = torch.cat(wav_parts, dim=-1)[0]
    elif frame_count:
        decoded = speech_model.decode(codes.unsqueeze(0), return_dict=False)[0]
        waveform = decoded[0].unsqueeze(0)
    else:
        waveform = torch.empty((1, 0), device=device, dtype=prompt.tts_pad.dtype)

    if phase_events is not None:
        phase_events[4].record()
        phase_events[4].synchronize()
        prepare_seconds = phase_events[0].elapsed_time(phase_events[1]) / 1000
        prefill_seconds = phase_events[1].elapsed_time(phase_events[2]) / 1000
        decode_seconds = phase_events[2].elapsed_time(phase_events[3]) / 1000
        codec_seconds = phase_events[3].elapsed_time(phase_events[4]) / 1000
    else:
        codec_seconds = time.perf_counter() - cpu_phase_started
    total_seconds = time.perf_counter() - started
    timings = {
        "prepare": prepare_seconds,
        "prefill": prefill_seconds,
        "decode": decode_seconds,
        "codec": codec_seconds,
        "total": total_seconds,
        "frames": float(frame_count),
    }
    return waveform, codes, int(speech_model.get_output_sample_rate()), timings
