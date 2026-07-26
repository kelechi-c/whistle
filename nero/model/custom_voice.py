"""Weight-compatible Qwen3-TTS CustomVoice model exposed through Nero."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class Qwen3CustomVoice:
    """Loads official CustomVoice checkpoints behind a stable Nero API.

    The checkpoint architecture and speech codec live in the ``qwen-tts``
    dependency. Reusing those authoritative primitives keeps safetensor names,
    generation semantics, and waveform decoding exactly compatible while Nero
    owns backend selection and the public inference surface.
    """

    def __init__(self, wrapper: Any) -> None:
        if wrapper.model.tts_model_type != "custom_voice":
            raise ValueError(
                "Qwen3CustomVoice requires a CustomVoice checkpoint, got "
                f"{wrapper.model.tts_model_type!r}"
            )
        self.wrapper = wrapper

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
    ) -> "Qwen3CustomVoice":
        """Loads Hugging Face config, safetensors, tokenizer, and speech codec."""
        from qwen_tts import Qwen3TTSModel

        device_map = f"cuda:{device.index or 0}" if device.type == "cuda" else "cpu"
        wrapper = Qwen3TTSModel.from_pretrained(
            str(checkpoint),
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )
        return cls(wrapper)

    @property
    def model(self) -> torch.nn.Module:
        """Returns the loaded weight-bearing model for instrumentation."""
        return self.wrapper.model

    @property
    def sample_rate(self) -> int:
        """Returns the speech tokenizer's output sample rate."""
        return int(self.model.speech_tokenizer.get_output_sample_rate())

    @torch.inference_mode()
    def generate_custom_voice(
        self,
        text: str | list[str],
        *,
        speaker: str | list[str],
        language: str | list[str] = "english",
        max_new_tokens: int = 512,
        do_sample: bool = False,
        subtalker_dosample: bool = False,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], int]:
        """Generates with the checkpoint's native prompt and decode semantics."""
        return self.wrapper.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            subtalker_dosample=subtalker_dosample,
            **kwargs,
        )

