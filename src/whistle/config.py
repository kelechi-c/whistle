"""Central runtime choices for Whistle inference and profiling."""

from dataclasses import dataclass
import pathlib as pl
from typing import Literal

import torch

DeviceChoice = Literal["auto", "cpu", "cuda"]
DTypeChoice = Literal["float32", "float16", "bfloat16"]


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Holds every mutable runtime choice outside model checkpoint structure."""

    checkpoint: pl.Path = pl.Path("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    output: pl.Path = pl.Path("whistle.wav")
    device: DeviceChoice = "auto"
    dtype: DTypeChoice = "bfloat16"
    seed: int = 0
    max_frames: int = 1_280

    def resolved_device(self) -> torch.device:
        """Selects CUDA only when requested or available under auto mode."""
        use_cuda = self.device == "cuda" or (
            self.device == "auto" and torch.cuda.is_available()
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("cuda was requested but is unavailable")
        return torch.device("cuda" if use_cuda else "cpu")

    def resolved_dtype(self, device: torch.device) -> torch.dtype:
        """Maps the configured dtype and keeps CPU inference broadly supported."""
        dtype = getattr(torch, self.dtype)
        if device.type == "cpu" and dtype == torch.float16:
            raise RuntimeError("float16 cpu inference is unsupported; use float32")
        return dtype


RUNTIME = RuntimeConfig()
