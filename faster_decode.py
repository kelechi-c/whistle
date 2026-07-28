"""Low-overhead official Qwen3-TTS prefill/decode baseline.

The hot path intentionally stays in one ``tts_infer`` function. It uses the
official embeddings, forwards, dynamic caches, heads, and codec while keeping
decode state and completed audio on-device.
"""

from functools import cache
import time

from qwen_tts import Qwen3TTSModel
import torch
from transformers import StaticCache


# Compiles only the predictor transformer and reuses its wrapper across requests.
@cache
def _compiled_predictor(model: torch.nn.Module) -> torch.nn.Module:
    return torch.compile(model, mode="reduce-overhead")


@torch.inference_mode()
def tts_infer(
    tts: Qwen3TTSModel,
    text: str,
    *,
    speaker: str = "serena",
    language: str = "english",
    max_new_tokens: int = 1_280,
) -> tuple[torch.Tensor, torch.Tensor, int, dict[str, float]]:
    """Runs batch-one CustomVoice inference through explicit forward passes.

    Prefill builds the complete non-streaming text/speaker prompt and fills the
    talker's dynamic KV cache. Decode always emits ``max_new_tokens`` frames:
    every token, codec-ID tensor, and waveform stays on-device until the caller
    explicitly transfers the completed outputs.
    """
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    started = time.perf_counter()
    model = tts.model
    talker = model.talker
    predictor = talker.code_predictor
    config = model.config
    talker_config = config.talker_config
    device = next(model.parameters()).device
    cuda_timing = device.type == "cuda"
    phase_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)] if cuda_timing else None
    if phase_events is not None:
        phase_events[0].record()
    cpu_phase_started = started

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
    prefill_length = talker_input.shape[1]
    talker_cache = StaticCache(
        config=talker.model.config,
        max_cache_len=prefill_length + max_new_tokens - 1,
    )
    talker.rope_deltas = None
    if phase_events is not None:
        phase_events[1].record()
        prepare_seconds = 0.0
    else:
        now = time.perf_counter()
        prepare_seconds = now - cpu_phase_started
        cpu_phase_started = now

    # === prefill: variable-length prompt, first codebook-zero token, dynamic KV ===
    talker_output = talker(
        inputs_embeds=talker_input,
        attention_mask=attention_mask,
        past_key_values=talker_cache,
        past_hidden=None,
        trailing_text_hidden=tts_pad,
        tts_pad_embed=tts_pad,
        generation_step=None,
        use_cache=True,
        return_dict=True,
    )
    code_vocab_size = predictor.config.vocab_size
    token = talker_output.logits[:, -1, :code_vocab_size].argmax(dim=-1)
    past_hidden = talker_output.past_hidden
    if phase_events is not None:
        phase_events[2].record()
        prefill_seconds = 0.0
    else:
        now = time.perf_counter()
        prefill_seconds = now - cpu_phase_started
        cpu_phase_started = now

    # === decode: fixed GPU buffers, predictor residuals, cached talker forwards ===
    num_code_groups = talker_config.num_code_groups
    num_residuals = num_code_groups - 1
    codes = torch.empty((max_new_tokens, num_code_groups), device=device, dtype=torch.long)
    residual_codes = torch.empty((1, num_residuals), device=device, dtype=torch.long)
    predictor_input = torch.empty(
        (1, 2, talker_config.hidden_size),
        device=device,
        dtype=past_hidden.dtype,
    )
    predictor_model = _compiled_predictor(predictor.model)
    predictor_heads = predictor.lm_head
    predictor_projection = predictor.small_to_mtp_projection
    predictor_embedding_weights = torch.stack(tuple(x.weight for x in predictor.get_input_embeddings()))
    residual_indices = torch.arange(num_residuals, device=device)
    predictor_cache = StaticCache(
        config=predictor_model.config,
        max_cache_len=num_residuals + 1,
    )
    predictor_config = predictor_model.config
    predictor_cache.early_initialization(
        batch_size=1,
        num_heads=predictor_config.num_key_value_heads,
        head_dim=predictor_config.head_dim,
        dtype=next(predictor_model.parameters()).dtype,
        device=device,
    )
    predictor_prefill_positions = torch.arange(2, device=device)
    predictor_decode_positions = torch.arange(2, num_residuals + 1, device=device)
    cache_positions = torch.arange(
        prefill_length, prefill_length + max_new_tokens - 1, device=device
    )
    position_ids = (
        cache_positions.to(talker.rope_deltas.dtype).view(1, -1)
        + talker.rope_deltas
    ).unsqueeze(0).expand(3, -1, -1)
    
    for frame_index in range(max_new_tokens):
        last_id_hidden = codec_embeddings(token.view(1, 1))
        predictor_input[:, :1].copy_(past_hidden)
        predictor_input[:, 1:].copy_(last_id_hidden)
        predictor_cache.reset()
        predictor_output = predictor_model(
            inputs_embeds=predictor_projection(predictor_input),
            past_key_values=predictor_cache,
            cache_position=predictor_prefill_positions,
            use_cache=True,
            return_dict=True,
        )
        residual = predictor_heads[0](
            predictor_output.last_hidden_state[:, -1, :]
        ).argmax(dim=-1)
        residual_codes[:, 0].copy_(residual)

        # code predicctor phase/steps
        for residual_index in range(1, num_residuals):
            residual_hidden = predictor_embedding_weights[
                residual_index - 1, residual
            ].unsqueeze(1)
            predictor_output = predictor_model(
                inputs_embeds=predictor_projection(residual_hidden),
                past_key_values=predictor_cache,
                cache_position=predictor_decode_positions[
                    residual_index - 1 : residual_index
                ],
                use_cache=True,
                return_dict=True,
            )
            residual = predictor_heads[residual_index](
                predictor_output.last_hidden_state[:, -1, :]
            ).argmax(dim=-1)
            residual_codes[:, residual_index].copy_(residual)
            
        codes[frame_index, 0].copy_(token[0])
        codes[frame_index, 1:].copy_(residual_codes[0])
        if frame_index + 1 == max_new_tokens:
            break

        residual_hidden = predictor_embedding_weights[
            residual_indices, residual_codes[0]
        ].sum(dim=0).view(1, 1, -1)
        talker_input = last_id_hidden + residual_hidden + tts_pad
        cache_position = cache_positions[frame_index : frame_index + 1]
        backbone_output = talker.model(
            inputs_embeds=talker_input,
            attention_mask=None,
            position_ids=position_ids[:, :, frame_index : frame_index + 1],
            past_key_values=talker_cache,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        past_hidden = backbone_output.last_hidden_state[:, -1:, :]
        token = talker.codec_head(past_hidden)[
            :, -1, :code_vocab_size
        ].argmax(dim=-1)

    if phase_events is not None:
        phase_events[3].record()
        decode_seconds = 0.0
    else:
        now = time.perf_counter()
        decode_seconds = now - cpu_phase_started
        cpu_phase_started = now

    # === codec: fixed GPU output, no official wrapper CPU conversion/list concat ===
    speech_model = model.speech_tokenizer.model
    decoder = speech_model.decoder
    upsample = int(decoder.total_upsample)
    codec_input = codes.unsqueeze(0).transpose(1, 2)
    waveform = torch.empty((1, max_new_tokens * upsample), device=device, dtype=tts_pad.dtype)
    chunk_size = 300
    left_context = 25
    for start_index in range(0, max_new_tokens, chunk_size):
        end_index = min(start_index + chunk_size, max_new_tokens)
        context_start = max(0, start_index - left_context)
        decoded = decoder(codec_input[..., context_start:end_index]).squeeze(1)
        decoded = decoded[..., (start_index - context_start) * upsample :]
        decoded = decoded[..., : (end_index - start_index) * upsample]
        waveform[:, start_index * upsample : end_index * upsample].copy_(decoded)

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
        "frames": float(max_new_tokens),
    }
    return waveform, codes, int(speech_model.get_output_sample_rate()), timings
