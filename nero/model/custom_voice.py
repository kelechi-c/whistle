"""Independent, weights-compatible Qwen3-TTS CustomVoice inference surface."""

from __future__ import annotations

import json
import pathlib as pl
import time
from typing import Any

from safetensors import safe_open
import torch
from torch import nn
from transformers import Qwen2Tokenizer

from nero.model.codec import CodecDecoder, create_codec_decoder
from nero.model.layers import RotaryEmbedding
from nero.model.talker import PreparedInput, Talker
from nero.model.types import (
    GenerationResult,
    ModelConfig,
    SpeechTokenizerConfig,
)


class CustomVoiceModel(nn.Module):
    """Matches the official main checkpoint, whose only tensors are `talker.*`."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.talker = Talker(config.talker_config)


class SpeechTokenizer(nn.Module):
    """Matches the official `decoder.*` speech-tokenizer checkpoint subset."""

    def __init__(self, config: SpeechTokenizerConfig) -> None:
        super().__init__()
        self.config = config
        self.decoder = create_codec_decoder(config.decoder_config)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Exposes codec decode independently of autoregressive generation."""
        return self.decoder(codes)


def _checkpoint_path(
    checkpoint: str | pl.Path, local_files_only: bool
) -> pl.Path:
    """Resolves a local directory or downloads only the required HF files."""
    path = pl.Path(checkpoint)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return pl.Path(
        snapshot_download(
            str(checkpoint),
            allow_patterns=(
                "config.json",
                "model.safetensors",
                "merges.txt",
                "speech_tokenizer/config.json",
                "speech_tokenizer/model.safetensors",
                "tokenizer_config.json",
                "vocab.json",
            ),
            local_files_only=local_files_only,
        )
    )


def _materialize_buffers(module: nn.Module, device: torch.device) -> None:
    """Recreates non-persistent RoPE buffers after meta construction."""
    for child in module.modules():
        if isinstance(child, RotaryEmbedding):
            child.materialize(device)
        elif (
            "inv_freq" in child._buffers
            and child._buffers["inv_freq"].device.type == "meta"
            and hasattr(child, "rope_init_fn")
        ):
            inverse, scaling = child.rope_init_fn(child.config, device)
            child.inv_freq = inverse
            child.original_inv_freq = inverse
            child.attention_scaling = scaling


def _load_strict(
    module: nn.Module,
    path: pl.Path,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Assigns only expected safetensors directly into a meta-built module."""
    expected = set(module.state_dict())
    with safe_open(path, framework="pt", device=str(device)) as checkpoint:
        available = set(checkpoint.keys())
        missing = expected - available
        selected = {
            name: checkpoint.get_tensor(name).to(dtype=dtype)
            for name in expected
            if name in available
        }
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"checkpoint is missing {len(missing)} tensors: {preview}")
    module.load_state_dict(selected, strict=True, assign=True)
    _materialize_buffers(module, device)


class Qwen3CustomVoice:
    """Coordinates official-shaped talker generation and codec decoding.

    The wrapper is deliberately not an ``nn.Module``: the official main and
    speech-tokenizer checkpoints remain two independently loadable modules.
    """

    def __init__(
        self,
        model: CustomVoiceModel,
        speech_tokenizer: SpeechTokenizer,
        tokenizer: Any | None = None,
    ) -> None:
        self.model = model
        self.speech_tokenizer = speech_tokenizer
        self.tokenizer = tokenizer

    @property
    def talker(self) -> Talker:
        """Returns the primary token generator for direct optimization work."""
        return self.model.talker

    @property
    def code_predictor(self) -> nn.Module:
        """Returns the residual codebook predictor as a standalone module."""
        return self.talker.code_predictor

    @property
    def codec_decoder(self) -> CodecDecoder:
        """Returns the waveform decoder as a standalone module."""
        return self.speech_tokenizer.decoder

    @property
    def sample_rate(self) -> int:
        """Returns the official codec output sample rate."""
        return self.speech_tokenizer.config.output_sample_rate

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | pl.Path,
        *,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> "Qwen3CustomVoice":
        """Loads official main and decoder weights without importing qwen-tts."""
        root = _checkpoint_path(checkpoint, local_files_only)
        config = ModelConfig.from_dict(
            json.loads((root / "config.json").read_text(encoding="utf-8"))
        )
        codec_config = SpeechTokenizerConfig.from_dict(
            json.loads(
                (root / "speech_tokenizer" / "config.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        with torch.device("meta"):
            model = CustomVoiceModel(config)
            speech_tokenizer = SpeechTokenizer(codec_config)
        _load_strict(model, root / "model.safetensors", device, dtype)
        _load_strict(
            speech_tokenizer,
            root / "speech_tokenizer" / "model.safetensors",
            device,
            dtype,
        )
        model.eval()
        speech_tokenizer.eval()
        tokenizer = (
            Qwen2Tokenizer.from_pretrained(root, local_files_only=True)
            if (root / "tokenizer_config.json").is_file()
            else None
        )
        return cls(model, speech_tokenizer, tokenizer)

    def tokenize(self, text: str) -> torch.Tensor:
        """Reproduces the official assistant-template tokenization for raw text."""
        if self.tokenizer is None:
            raise RuntimeError(
                "this checkpoint has no tokenizer assets; pass prepared text_ids"
            )
        prompt = (
            f"<|im_start|>assistant\n{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        encoded = self.tokenizer(text=prompt, return_tensors="pt")
        return encoded["input_ids"].to(next(self.model.parameters()).device)

    def generate_text(
        self,
        text: str,
        *,
        language: str = "english",
        max_frames: int = 64,
        stop_on_eos: bool = True,
    ) -> GenerationResult:
        """Tokenizes raw text exactly like the official wrapper, then generates."""
        return self.generate(
            self.tokenize(text),
            language=language,
            max_frames=max_frames,
            stop_on_eos=stop_on_eos,
        )

    @torch.inference_mode()
    def generate(
        self,
        text_ids: torch.Tensor,
        *,
        language: str = "english",
        max_frames: int = 64,
        stop_on_eos: bool = True,
    ) -> GenerationResult:
        """Runs the speaker-free, greedy CustomVoice baseline end to end."""
        device = next(self.model.parameters()).device
        text_ids = text_ids.to(device=device, dtype=torch.long)
        start = time.perf_counter()
        prepared = self.talker.prepare_input(
            text_ids,
            tts_bos_token_id=self.model.config.tts_bos_token_id,
            tts_eos_token_id=self.model.config.tts_eos_token_id,
            tts_pad_token_id=self.model.config.tts_pad_token_id,
            language=language,
        )
        prepare_seconds = time.perf_counter() - start
        start = time.perf_counter()
        codes = self.talker.generate_codes(
            prepared, max_frames, stop_on_eos=stop_on_eos
        )
        talker_seconds = time.perf_counter() - start
        start = time.perf_counter()
        audio = self.speech_tokenizer.decode(codes)
        codec_seconds = time.perf_counter() - start
        return GenerationResult(
            audio=audio.cpu(),
            codes=codes.cpu(),
            sample_rate=self.sample_rate,
            timings={
                "prepare": prepare_seconds,
                "talker_and_code_predictor": talker_seconds,
                "codec": codec_seconds,
            },
        )

    def prepare_input(
        self, text_ids: torch.Tensor, language: str = "english"
    ) -> PreparedInput:
        """Exposes prompt construction separately from prefill and decode."""
        return self.talker.prepare_input(
            text_ids.to(next(self.model.parameters()).device),
            tts_bos_token_id=self.model.config.tts_bos_token_id,
            tts_eos_token_id=self.model.config.tts_eos_token_id,
            tts_pad_token_id=self.model.config.tts_pad_token_id,
            language=language,
        )
