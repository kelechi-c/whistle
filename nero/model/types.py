"""Immutable model configuration and generation data."""

from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Defines checkpoint-owned model dimensions, not runtime policy."""

    text_vocab_size: int = 256
    codec_vocab_size: int = 64
    hidden_size: int = 32
    intermediate_size: int = 64
    talker_layers: int = 2
    predictor_layers: int = 1
    attention_heads: int = 4
    key_value_heads: int = 2
    codebooks: int = 4
    sample_rate: int = 24000
    frame_rate: int = 12
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def samples_per_frame(self) -> int:
        """Returns the codec waveform width represented by one token frame."""
        return self.sample_rate // self.frame_rate

    def to_dict(self) -> dict[str, Any]:
        """Converts the immutable configuration to JSON-safe values."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        """Builds a model configuration from checkpoint metadata."""
        return cls(**values)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Carries generated codec tokens, waveform, and non-overlapping timings."""

    audio: torch.Tensor
    codes: torch.Tensor
    sample_rate: int
    timings: dict[str, float]
