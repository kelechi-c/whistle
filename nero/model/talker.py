"""Official-shaped primary Qwen3-TTS talker and prepared-input generation."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from nero.model.code_predictor import CodePredictor
from nero.model.layers import Decoder, KVCache
from nero.model.types import TalkerConfig


@dataclass(frozen=True, slots=True)
class TalkerState:
    """Carries the hot-path state needed for one subsequent codec frame."""

    cache: KVCache
    hidden: torch.Tensor
    position: int


@dataclass(frozen=True, slots=True)
class PreparedInput:
    """Separates prompt embeddings from text streamed alongside codec frames."""

    prompt: torch.Tensor
    trailing_text: torch.Tensor
    tts_pad: torch.Tensor


class ResizeMLP(nn.Module):
    """Matches the official two-layer biased text projection."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, input_size, bias=True)
        self.linear_fc2 = nn.Linear(input_size, output_size, bias=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.silu(self.linear_fc1(values)))


class TalkerModel(Decoder):
    """Adds official text and codec embeddings to the multimodal decoder."""

    def __init__(self, config: TalkerConfig) -> None:
        super().__init__(
            config,
            multimodal_sections=config.mrope_section,
            multimodal_interleaved=config.mrope_interleaved,
        )
        self.codec_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.text_embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )


class Talker(nn.Module):
    """Predicts codebook zero while delegating residual groups to its predictor."""

    def __init__(self, config: TalkerConfig) -> None:
        super().__init__()
        self.config = config
        self.model = TalkerModel(config)
        self.text_projection = ResizeMLP(
            config.text_hidden_size, config.hidden_size
        )
        self.codec_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.code_predictor = CodePredictor(
            config.code_predictor_config, config.hidden_size
        )

    def _select_first_code(
        self, logits: torch.Tensor, *, allow_eos: bool
    ) -> torch.Tensor:
        """Selects acoustic IDs while treating EOS as control flow, not audio."""
        acoustic_size = self.config.code_predictor_config.vocab_size
        acoustic_logits = logits[..., :acoustic_size]
        if not allow_eos:
            return acoustic_logits.argmax(dim=-1)
        eos_logits = logits[..., self.config.codec_eos_token_id : self.config.codec_eos_token_id + 1]
        selected = torch.cat((acoustic_logits, eos_logits), dim=-1).argmax(dim=-1)
        eos = torch.full_like(selected, self.config.codec_eos_token_id)
        return torch.where(selected == acoustic_size, eos, selected)

    def prepare_input(
        self,
        text_ids: torch.Tensor,
        *,
        tts_bos_token_id: int,
        tts_eos_token_id: int,
        tts_pad_token_id: int,
        language: str = "english",
    ) -> PreparedInput:
        """Builds the official no-speaker streaming prompt for one processed text."""
        if text_ids.ndim != 2 or text_ids.shape[0] != 1 or text_ids.shape[1] < 9:
            raise ValueError("text_ids must have shape (1, length>=9)")
        language_ids = self.config.codec_language_id or {}
        if language != "auto" and language not in language_ids:
            raise ValueError(f"unsupported language: {language}")
        text_specials = torch.tensor(
            [[tts_bos_token_id, tts_eos_token_id, tts_pad_token_id]],
            dtype=text_ids.dtype,
            device=text_ids.device,
        )
        tts_bos, tts_eos, tts_pad = self.text_projection(
            self.model.text_embedding(text_specials)
        ).chunk(3, dim=1)
        if language == "auto":
            codec_prefix = [
                self.config.codec_nothink_id,
                self.config.codec_think_bos_id,
                self.config.codec_think_eos_id,
            ]
        else:
            codec_prefix = [
                self.config.codec_think_id,
                self.config.codec_think_bos_id,
                language_ids[language],
                self.config.codec_think_eos_id,
            ]
        codec_ids = torch.tensor(
            [codec_prefix + [self.config.codec_pad_id, self.config.codec_bos_id]],
            dtype=text_ids.dtype,
            device=text_ids.device,
        )
        codec_embeddings = self.model.codec_embedding(codec_ids)
        role = self.text_projection(self.model.text_embedding(text_ids[:, :3]))
        aligned_prefix = (
            torch.cat(
                (
                    tts_pad.expand(-1, codec_embeddings.shape[1] - 2, -1),
                    tts_bos,
                ),
                dim=1,
            )
            + codec_embeddings[:, :-1]
        )
        first_text = (
            self.text_projection(self.model.text_embedding(text_ids[:, 3:4]))
            + codec_embeddings[:, -1:]
        )
        trailing = torch.cat(
            (
                self.text_projection(self.model.text_embedding(text_ids[:, 4:-5])),
                tts_eos,
            ),
            dim=1,
        )
        return PreparedInput(
            prompt=torch.cat((role, aligned_prefix, first_text), dim=1),
            trailing_text=trailing,
            tts_pad=tts_pad,
        )

    def prefill(self, prompt: torch.Tensor) -> tuple[torch.Tensor, TalkerState]:
        """Runs prompt prefill and returns the first code plus reusable KV state."""
        length = prompt.shape[1]
        positions = torch.arange(length, device=prompt.device)
        position_ids = positions.view(1, 1, -1).expand(3, prompt.shape[0], -1)
        hidden, cache = self.model(prompt, position_ids)
        last_hidden = hidden[:, -1:]
        first_code = self._select_first_code(
            self.codec_head(last_hidden[:, -1]), allow_eos=False
        )
        return first_code, TalkerState(cache, last_hidden, length)

    def decode_step(
        self,
        frame_codes: torch.Tensor,
        text_embedding: torch.Tensor,
        state: TalkerState,
        *,
        allow_eos: bool = False,
    ) -> tuple[torch.Tensor, TalkerState]:
        """Advances the talker by one completed multi-codebook frame."""
        frame_embedding = self.model.codec_embedding(frame_codes[:, 0])
        frame_embedding += self.code_predictor.embed_residual_codes(
            frame_codes[:, 1:]
        )
        values = frame_embedding.unsqueeze(1) + text_embedding
        position_ids = torch.full(
            (3, values.shape[0], 1),
            state.position,
            dtype=torch.long,
            device=values.device,
        )
        hidden, cache = self.model(values, position_ids, state.cache)
        first_code = self._select_first_code(
            self.codec_head(hidden[:, -1]), allow_eos=allow_eos
        )
        return first_code, TalkerState(cache, hidden[:, -1:], state.position + 1)

    @torch.inference_mode()
    def generate_codes(
        self,
        prepared: PreparedInput,
        max_frames: int,
        *,
        stop_on_eos: bool = True,
    ) -> torch.Tensor:
        """Runs the simple greedy CustomVoice loop and returns `(B, K, T)` codes."""
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        first_code, state = self.prefill(prepared.prompt)
        frames: list[torch.Tensor] = []
        for step in range(max_frames):
            if step and bool(
                (first_code == self.config.codec_eos_token_id).all()
            ):
                break
            residual = self.code_predictor.predict_frame(
                state.hidden, self.model.codec_embedding(first_code).unsqueeze(1)
            )
            frame = torch.cat((first_code.unsqueeze(-1), residual), dim=-1)
            frames.append(frame)
            text = (
                prepared.trailing_text[:, step : step + 1]
                if step < prepared.trailing_text.shape[1]
                else prepared.tts_pad
            )
            first_code, state = self.decode_step(
                frame,
                text,
                state,
                allow_eos=stop_on_eos and step >= 1,
            )
        if not frames:
            raise RuntimeError("talker produced eos before any codec frame")
        return torch.stack(frames, dim=-1)
