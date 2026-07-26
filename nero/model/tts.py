"""Compatibility exports for Nero's CustomVoice-only implementation."""

from nero.model.custom_voice import Qwen3CustomVoice
from nero.model.types import GenerationResult

Qwen3TTS = Qwen3CustomVoice

__all__ = ["GenerationResult", "Qwen3CustomVoice", "Qwen3TTS"]
