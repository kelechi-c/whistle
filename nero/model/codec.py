"""Tiny learned codec decoder used by the CPU fixture."""

import torch
from torch import nn

from nero.model.types import ModelConfig


class CodecDecoder(nn.Module):
    """Maps each multi-codebook frame directly to a short waveform segment."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            nn.Embedding(config.codec_vocab_size, config.hidden_size)
            for _ in range(config.codebooks)
        )
        self.waveform_head = nn.Linear(config.hidden_size, config.samples_per_frame)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Decodes `(batch, frames, codebooks)` token IDs into mono audio."""
        hidden = sum(
            embedding(codes[..., index])
            for index, embedding in enumerate(self.embeddings)
        )
        return self.waveform_head(hidden).tanh().flatten(1)
