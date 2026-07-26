"""Thin wrapper around the installed official Qwen3-TTS 12 Hz decoder."""

from dataclasses import asdict

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder as CodecDecoder,
)

from nero.model.types import CodecConfig


def create_codec_decoder(config: CodecConfig) -> CodecDecoder:
    """Instantiates the official decoder with Nero's immutable config."""
    official_config = Qwen3TTSTokenizerV2DecoderConfig(**asdict(config))
    return CodecDecoder(official_config)


__all__ = ["CodecDecoder", "create_codec_decoder"]
