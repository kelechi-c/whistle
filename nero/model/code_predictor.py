"""Official-shaped residual codebook predictor with a frame-level API."""

import torch
from torch import nn

from nero.model.layers import Decoder, KVCache
from nero.model.types import CodePredictorConfig


class CodePredictorModel(Decoder):
    """Adds the 15 official residual-codebook embedding tables to a decoder."""

    def __init__(self, config: CodePredictorConfig, embedding_dim: int) -> None:
        super().__init__(config)
        self.codec_embedding = nn.ModuleList(
            nn.Embedding(config.vocab_size, embedding_dim)
            for _ in range(config.num_code_groups - 1)
        )


class CodePredictor(nn.Module):
    """Completes codebooks 1..N from talker hidden state and codebook zero."""

    def __init__(
        self, config: CodePredictorConfig, talker_hidden_size: int
    ) -> None:
        super().__init__()
        self.config = config
        self.model = CodePredictorModel(config, talker_hidden_size)
        self.lm_head = nn.ModuleList(
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        )
        self.small_to_mtp_projection: nn.Module = (
            nn.Linear(talker_hidden_size, config.hidden_size, bias=True)
            if config.hidden_size != talker_hidden_size
            else nn.Identity()
        )

    @torch.inference_mode()
    def predict_frame(
        self,
        talker_hidden: torch.Tensor,
        first_code_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Greedily predicts one frame's residual codebooks in official order."""
        seed = torch.cat((talker_hidden, first_code_embedding), dim=1)
        values = self.small_to_mtp_projection(seed)
        positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
        hidden, cache = self.model(values, positions)
        code = self.lm_head[0](hidden[:, -1]).argmax(dim=-1)
        codes = [code]
        for index, head in enumerate(self.lm_head[1:], start=1):
            values = self.small_to_mtp_projection(
                self.model.codec_embedding[index - 1](code).unsqueeze(1)
            )
            position = torch.tensor(
                [[index + 1]], dtype=torch.long, device=values.device
            )
            hidden, cache = self.model(values, position, cache)
            code = head(hidden[:, -1]).argmax(dim=-1)
            codes.append(code)
        return torch.stack(codes, dim=-1)

    def embed_residual_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Sums residual embeddings into the talker's hidden dimension."""
        embeddings = tuple(
            embedding(codes[..., index])
            for index, embedding in enumerate(self.model.codec_embedding)
        )
        return torch.stack(embeddings, dim=0).sum(dim=0)

    def forward(
        self,
        values: torch.Tensor,
        position_ids: torch.Tensor,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Exposes the predictor transformer independently for hot-path work."""
        return self.model(self.small_to_mtp_projection(values), position_ids, cache)
