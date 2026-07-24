"""Primary codec-token generator for the first codebook."""

import torch
from torch import nn

from nero.model.layers import Decoder
from nero.model.types import ModelConfig


class Talker(nn.Module):
    """Fuses byte-level text and prior audio frames to predict codebook zero."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(config.text_vocab_size, config.hidden_size)
        self.codec_embedding = nn.Embedding(config.codec_vocab_size, config.hidden_size)
        self.model = Decoder(config, config.talker_layers)
        self.codec_head = nn.Linear(config.hidden_size, config.codec_vocab_size, bias=False)

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Embeds the conditioning text once before autoregressive generation."""
        return self.text_embedding(text_ids)

    def forward(
        self, text_embeddings: torch.Tensor, frame_embeddings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the next first-codebook token and its hidden conditioning state."""
        values = torch.cat((text_embeddings, frame_embeddings), dim=1)
        hidden = self.model(values)[:, -1]
        token = self.codec_head(hidden).argmax(dim=-1)
        return token, hidden
