"""End-to-end minimal Qwen3-TTS-shaped model."""

import json
import pathlib as pl
import time

import torch
from torch import nn

from nero.model.code_predictor import CodePredictor
from nero.model.codec import CodecDecoder
from nero.model.talker import Talker
from nero.model.types import GenerationResult, ModelConfig


def _elapsed(start: float) -> float:
    """Returns elapsed wall time for one non-overlapping generation phase."""
    return time.perf_counter() - start


class Qwen3TTS(nn.Module):
    """Coordinates text conditioning, codec-token generation, and audio decode."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.talker = Talker(config)
        self.code_predictor = CodePredictor(config)
        self.codec = CodecDecoder(config)

    @staticmethod
    def tokenize(text: str, device: torch.device) -> torch.Tensor:
        """Uses UTF-8 bytes as a dependency-free, deterministic tiny tokenizer."""
        tokens = list(text.encode("utf-8")) or [0]
        return torch.tensor([tokens], dtype=torch.long, device=device)

    @torch.inference_mode()
    def generate(self, text: str, frames: int) -> GenerationResult:
        """Generates fixed-count codec frames and decodes them into a waveform."""
        if frames < 1:
            raise ValueError("frames must be positive")
        device = next(self.parameters()).device
        text_ids = self.tokenize(text, device)

        start = time.perf_counter()
        text_embeddings = self.talker.encode_text(text_ids)
        timings = {"text_encode": _elapsed(start)}
        frame_embeddings = text_embeddings.new_empty((1, 0, self.config.hidden_size))
        generated: list[torch.Tensor] = []

        talker_seconds = 0.0
        predictor_seconds = 0.0
        for _ in range(frames):
            start = time.perf_counter()
            first_code, hidden = self.talker(text_embeddings, frame_embeddings)
            talker_seconds += _elapsed(start)

            start = time.perf_counter()
            frame = self.code_predictor(first_code, hidden)
            predictor_seconds += _elapsed(start)
            generated.append(frame)
            frame_embedding = self.talker.codec_embedding(frame[:, 0])
            frame_embedding = frame_embedding + self.code_predictor.embed_frame(frame)
            frame_embeddings = torch.cat(
                (frame_embeddings, frame_embedding.unsqueeze(1)),
                dim=1,
            )

        codes = torch.stack(generated, dim=1)
        start = time.perf_counter()
        audio = self.codec(codes)
        timings |= {
            "talker": talker_seconds,
            "code_predictor": predictor_seconds,
            "codec": _elapsed(start),
        }
        return GenerationResult(audio.cpu(), codes.cpu(), self.config.sample_rate, timings)

    def save_checkpoint(self, path: pl.Path) -> None:
        """Writes model dimensions and weights as a self-contained checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_checkpoint(
        cls, path: pl.Path, device: torch.device, dtype: torch.dtype
    ) -> "Qwen3TTS":
        """Loads a Nero-format checkpoint onto the selected runtime device."""
        values = json.loads((path / "config.json").read_text(encoding="utf-8"))
        model = cls(ModelConfig.from_dict(values))
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model.to(device=device, dtype=dtype).eval()

    @classmethod
    def tiny(cls, seed: int = 0) -> "Qwen3TTS":
        """Creates the deterministic random model used to build the CPU fixture."""
        torch.manual_seed(seed)
        return cls(ModelConfig())
