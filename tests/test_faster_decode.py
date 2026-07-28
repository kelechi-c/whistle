"""CPU structural test for the official-module prefill/decode baseline."""

import unittest

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
)
import torch

from faster_decode import tts_infer


class TinyProcessor:
    """Returns the assistant-template token layout expected by prompt slicing."""

    def __call__(
        self, *, text: str, return_tensors: str, padding: bool
    ) -> dict[str, torch.Tensor]:
        del text, return_tensors, padding
        return {
            "input_ids": torch.tensor([[1, 2, 3, 10, 4, 5, 6, 7, 8]])
        }


class TinyCodec:
    """Provides the direct tensor codec surface used by optimized inference."""

    total_upsample = 16

    def __init__(self) -> None:
        self.model = self
        self.decoder = self

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        samples = codes.shape[-1] * self.total_upsample
        return torch.zeros(
            (codes.shape[0], 1, samples),
            device=codes.device,
            dtype=torch.float32,
        )

    def get_output_sample_rate(self) -> int:
        return 24_000


class FasterDecodeTest(unittest.TestCase):
    """Checks cache plumbing and all codebook forwards on tiny CPU weights."""

    def test_tts_infer_runs_split_path(self) -> None:
        predictor = {
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_code_groups": 4,
            "max_position_embeddings": 128,
        }
        talker = {
            "vocab_size": 48,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "text_hidden_size": 48,
            "text_vocab_size": 256,
            "num_code_groups": 4,
            "max_position_embeddings": 256,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "interleaved": True,
            },
            "codec_eos_token_id": 34,
            "codec_think_id": 38,
            "codec_nothink_id": 39,
            "codec_think_bos_id": 40,
            "codec_think_eos_id": 41,
            "codec_pad_id": 32,
            "codec_bos_id": 33,
            "codec_language_id": {"english": 35},
            "spk_id": {"ryan": 37},
            "spk_is_dialect": {"ryan": False},
            "code_predictor_config": predictor,
        }
        config = Qwen3TTSConfig(
            talker_config=talker,
            tokenizer_type="qwen3_tts_tokenizer_12hz",
            tts_model_size="0b6",
            tts_model_type="custom_voice",
            tts_bos_token_id=253,
            tts_eos_token_id=254,
            tts_pad_token_id=255,
        )
        model = Qwen3TTSForConditionalGeneration(config)
        model.load_speech_tokenizer(TinyCodec())
        tts = Qwen3TTSModel(model.eval(), TinyProcessor())

        waveform, sample_rate, timings = tts_infer(
            tts,
            "hi",
            speaker="ryan",
            max_new_tokens=2,
        )

        self.assertEqual(waveform.shape, (1, 32))
        self.assertEqual(waveform.device.type, "cpu")
        self.assertEqual(sample_rate, 24_000)
        self.assertEqual(timings["frames"], 2.0)
        self.assertGreaterEqual(timings["prefill"], 0.0)
        self.assertGreaterEqual(timings["decode"], 0.0)


if __name__ == "__main__":
    unittest.main()
