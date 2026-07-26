"""Immutable official-shaped CustomVoice configuration and inference results."""

from dataclasses import asdict, dataclass, fields
from typing import Any, TypeVar

import torch

ConfigT = TypeVar("ConfigT")


def _known(cls: type[ConfigT], values: dict[str, Any]) -> ConfigT:
    """Builds a dataclass while ignoring Hugging Face bookkeeping fields."""
    names = {field.name for field in fields(cls)}
    return cls(**{name: value for name, value in values.items() if name in names})


@dataclass(frozen=True, slots=True)
class CodePredictorConfig:
    """Defines the residual-codebook transformer from the official config."""

    vocab_size: int = 32
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 1
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    num_code_groups: int = 4
    max_position_embeddings: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CodePredictorConfig":
        """Reads the predictor section of an official or tiny config."""
        return _known(cls, values)


@dataclass(frozen=True, slots=True)
class TalkerConfig:
    """Defines the primary codec transformer and its text projection."""

    vocab_size: int = 48
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    head_dim: int = 8
    text_hidden_size: int = 48
    text_vocab_size: int = 256
    num_code_groups: int = 4
    max_position_embeddings: int = 256
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    codec_eos_token_id: int = 34
    codec_think_id: int = 38
    codec_nothink_id: int = 39
    codec_think_bos_id: int = 40
    codec_think_eos_id: int = 41
    codec_pad_id: int = 32
    codec_bos_id: int = 33
    codec_language_id: dict[str, int] | None = None
    code_predictor_config: CodePredictorConfig = CodePredictorConfig()

    @property
    def mrope_section(self) -> tuple[int, int, int]:
        """Returns three RoPE sections whose sum is half one head."""
        if self.rope_scaling is not None:
            values = self.rope_scaling.get("mrope_section")
            if values is not None:
                return tuple(values)
        quarter = self.head_dim // 4
        return (quarter, quarter // 2, quarter // 2)

    @property
    def mrope_interleaved(self) -> bool:
        """Returns the official interleaved multimodal-RoPE switch."""
        return bool(self.rope_scaling and self.rope_scaling.get("interleaved", False))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TalkerConfig":
        """Reads the talker and nested code-predictor sections."""
        predictor = CodePredictorConfig.from_dict(values["code_predictor_config"])
        remaining = values | {"code_predictor_config": predictor}
        return _known(cls, remaining)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Owns the main official Qwen3-TTS CustomVoice checkpoint structure."""

    talker_config: TalkerConfig = TalkerConfig()
    tts_bos_token_id: int = 253
    tts_eos_token_id: int = 254
    tts_pad_token_id: int = 255
    tts_model_type: str = "custom_voice"
    tokenizer_type: str = "qwen3_tts_tokenizer_12hz"

    def to_dict(self) -> dict[str, Any]:
        """Serializes the nested config in official Hugging Face layout."""
        values = asdict(self)
        values |= {
            "architectures": ["Qwen3TTSForConditionalGeneration"],
            "model_type": "qwen3_tts",
        }
        return values

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        """Reads an official CustomVoice config while ignoring speaker metadata."""
        if values.get("tts_model_type") != "custom_voice":
            raise ValueError("only qwen3-tts custom_voice checkpoints are supported")
        talker = TalkerConfig.from_dict(values["talker_config"])
        return _known(cls, values | {"talker_config": talker})


@dataclass(frozen=True, slots=True)
class CodecConfig:
    """Defines the decoder half of the official 12 Hz speech tokenizer."""

    codebook_size: int = 32
    hidden_size: int = 16
    latent_dim: int = 16
    codebook_dim: int = 16
    decoder_dim: int = 32
    intermediate_size: int = 32
    num_hidden_layers: int = 1
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    head_dim: int = 4
    num_quantizers: int = 4
    num_semantic_quantizers: int = 1
    max_position_embeddings: int = 128
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    sliding_window: int = 16
    layer_scale_initial_scale: float = 0.01
    attention_bias: bool = False
    attention_dropout: float = 0.0
    upsample_rates: tuple[int, ...] = (2, 2)
    upsampling_ratios: tuple[int, ...] = (2,)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CodecConfig":
        """Reads the official speech-tokenizer decoder config."""
        normalized = values | {
            "upsample_rates": tuple(values["upsample_rates"]),
            "upsampling_ratios": tuple(values["upsampling_ratios"]),
        }
        return _known(cls, normalized)


@dataclass(frozen=True, slots=True)
class SpeechTokenizerConfig:
    """Owns codec metadata while intentionally omitting the unused encoder."""

    decoder_config: CodecConfig = CodecConfig()
    output_sample_rate: int = 24_000
    decode_upsample_rate: int = 8

    def to_dict(self) -> dict[str, Any]:
        """Serializes the tiny codec in official speech-tokenizer layout."""
        return {
            "architectures": ["Qwen3TTSTokenizerV2Model"],
            "model_type": "qwen3_tts_tokenizer_12hz",
            "output_sample_rate": self.output_sample_rate,
            "decode_upsample_rate": self.decode_upsample_rate,
            "decoder_config": asdict(self.decoder_config),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SpeechTokenizerConfig":
        """Reads official codec metadata and its nested decoder section."""
        decoder = CodecConfig.from_dict(values["decoder_config"])
        return _known(cls, values | {"decoder_config": decoder})


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Carries generated codec tokens, waveform, and phase timings."""

    audio: torch.Tensor
    codes: torch.Tensor
    sample_rate: int
    timings: dict[str, float]
