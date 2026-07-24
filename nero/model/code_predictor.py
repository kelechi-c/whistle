"""Residual codec codebook predictor."""

import torch
from torch import nn

from nero.model.layers import Decoder
from nero.model.types import ModelConfig


class CodePredictor(nn.Module):
    """Autoregressively predicts all codebooks after the talker's first token."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        residual_groups = config.codebooks - 1
        self.codebooks = config.codebooks
        self.seed_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.codec_embeddings = nn.ModuleList(
            nn.Embedding(config.codec_vocab_size, config.hidden_size)
            for _ in range(residual_groups)
        )
        self.model = Decoder(config, config.predictor_layers)
        self.lm_heads = nn.ModuleList(
            nn.Linear(config.hidden_size, config.codec_vocab_size, bias=False)
            for _ in range(residual_groups)
        )

    def forward(
        self, first_code: torch.Tensor, talker_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Builds one complete residual-vector-quantizer frame."""
        codes = [first_code]
        values = [self.seed_projection(talker_hidden).unsqueeze(1)]
        for index, head in enumerate(self.lm_heads):
            if index:
                previous_embedding = self.codec_embeddings[index - 1]
                values.append(previous_embedding(codes[-1]).unsqueeze(1))
            hidden = self.model(torch.cat(values, dim=1))[:, -1]
            codes.append(head(hidden).argmax(dim=-1))
        return torch.stack(codes, dim=-1)

    def embed_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """Sums residual-codebook embeddings for the next talker input."""
        return sum(
            embedding(codes[:, index + 1])
            for index, embedding in enumerate(self.codec_embeddings)
        )
