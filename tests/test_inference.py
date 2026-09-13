"""CPU structural tests for Whistle's official-module inference path.

These exercise cache plumbing, graph reset, EOS cadence, and the CPU eager
sampling fallback on tiny random weights. They do not need a checkpoint.
"""

import unittest
from unittest.mock import patch

import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
from transformers import DynamicCache

from whistle import inference
from whistle.graphs import PredictorGraphs, Talker, decode_graphs
from whistle.inference import _maybe_eos_row, tts_infer
from whistle.streaming import stream_tts


class TinyProcessor:
    """Returns the assistant-template token layout expected by prompt slicing."""

    def __call__(
        self, *, text: str, return_tensors: str, padding: bool
    ) -> dict[str, torch.Tensor]:
        del text, return_tensors, padding
        return {"input_ids": torch.tensor([[1, 2, 3, 10, 4, 5, 6, 7, 8]])}


class TinyDecoder:
    """Stands in for the codec decoder used by the streaming left-context path."""

    total_upsample = 16

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        frames = codes.shape[-1]
        return torch.zeros(frames * self.total_upsample, device=codes.device)


class TinyCodec:
    """Provides the official tensor codec surface used by optimized inference."""

    total_upsample = 16

    def __init__(self) -> None:
        self.model = self
        self.decoder = TinyDecoder()

    def decode(
        self, codes: torch.Tensor, return_dict: bool = False
    ) -> tuple[list[torch.Tensor], int]:
        del return_dict
        samples = codes.shape[1] * self.total_upsample
        audio = torch.zeros(samples, device=codes.device, dtype=torch.float32)
        return ([audio], 24_000)

    def get_output_sample_rate(self) -> int:
        return 24_000


def tiny_tts() -> Qwen3TTSModel:
    """Builds one tiny official config whose vocab exceeds the suppression window."""
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
        "vocab_size": 1_100,
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
    return Qwen3TTSModel(model.eval(), TinyProcessor())


class InferenceTest(unittest.TestCase):
    """Checks cache plumbing, graph reuse, and token-loop contracts on CPU."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.tts = tiny_tts()
        decode_graphs.cache_clear()

    def tearDown(self) -> None:
        decode_graphs.cache_clear()

    def test_split_path_is_repeatable(self) -> None:
        """Two identical requests must produce identical ids after a cache reset."""
        with patch("whistle.inference.MAX_CACHE_LEN", 64):
            waveform, codec_ids, sample_rate, timings = tts_infer(
                self.tts, "hi", speaker="ryan", max_new_tokens=2, stop_at_eos=False
            )
            _, second_ids, _, _ = tts_infer(
                self.tts, "hi", speaker="ryan", max_new_tokens=2, stop_at_eos=False
            )

        graphs = decode_graphs(self.tts.model.talker, 64)
        self.assertIs(graphs, decode_graphs(self.tts.model.talker, 64))
        self.assertIsInstance(graphs.talker, Talker)
        self.assertIsInstance(graphs.predictor, PredictorGraphs)
        self.assertIsInstance(graphs.talker.cache, DynamicCache)
        self.assertEqual(graphs.talker.cache.get_max_cache_shape(), -1)
        self.assertEqual(graphs.predictor.cache.get_max_cache_shape(), -1)
        self.assertTrue(torch.equal(codec_ids, second_ids))
        self.assertEqual(waveform.shape, (1, 32))
        self.assertEqual(codec_ids.shape, (2, 4))
        self.assertEqual(codec_ids.dtype, torch.long)
        self.assertEqual(sample_rate, 24_000)
        self.assertEqual(timings["frames"], 2.0)
        self.assertGreaterEqual(timings["prefill"], 0.0)
        self.assertGreaterEqual(timings["decode"], 0.0)

    def test_request_resets_talker_cache(self) -> None:
        """Each request must start from a fresh talker cache, not the prior one."""
        with patch("whistle.inference.MAX_CACHE_LEN", 64):
            tts_infer(self.tts, "hi", max_new_tokens=2, stop_at_eos=False)
            first_cache = decode_graphs(self.tts.model.talker, 64).talker.cache
            tts_infer(self.tts, "hi", max_new_tokens=2, stop_at_eos=False)
            second_cache = decode_graphs(self.tts.model.talker, 64).talker.cache
        self.assertIsNot(first_cache, second_cache)
        self.assertEqual(first_cache.get_seq_length(), second_cache.get_seq_length())

    def test_capacity_and_budget_are_validated(self) -> None:
        """Oversized prompts and non-positive frame budgets raise before decoding."""
        with patch("whistle.inference.MAX_CACHE_LEN", 8):
            with self.assertRaises(ValueError):
                tts_infer(self.tts, "hi", max_new_tokens=2, stop_at_eos=False)
        with self.assertRaises(ValueError):
            tts_infer(self.tts, "hi", max_new_tokens=0, stop_at_eos=False)
        with self.assertRaises(ValueError):
            tts_infer(self.tts, "hi", max_new_tokens=2, repetition_penalty=0.0)

    def test_sampling_falls_back_to_eager_on_cpu(self) -> None:
        """Sampled decoding must not index a cuda-only captured graph on CPU."""
        with patch("whistle.inference.MAX_CACHE_LEN", 64):
            _, codec_ids, _, _ = tts_infer(
                self.tts,
                "hi",
                max_new_tokens=2,
                stop_at_eos=False,
                temperature=0.9,
                top_k=8,
            )
        self.assertEqual(codec_ids.shape, (2, 4))

    def test_prefill_forwards_sampling_policy(self) -> None:
        """The first token must use the requested sampler, not always argmax."""
        seen: list[tuple[int, dict[str, float] | None]] = []
        real = inference._select_token

        def spy(logits: torch.Tensor, history: torch.Tensor, **kwargs: object) -> torch.Tensor:
            seen.append((history.shape[1], kwargs.get("sampling")))
            return real(logits, history, **kwargs)

        with patch("whistle.inference._select_token", side_effect=spy), patch(
            "whistle.inference.MAX_CACHE_LEN", 64
        ):
            tts_infer(
                self.tts,
                "hi",
                max_new_tokens=2,
                stop_at_eos=False,
                temperature=0.9,
                top_k=8,
            )
        self.assertEqual(seen[0][0], 0)
        self.assertEqual(seen[0][1], {"temperature": 0.9, "top_k": 8})

    def test_streaming_ramp_and_final_chunk(self) -> None:
        """Streaming emits ramp-sized chunks and flags only the last one final."""
        with patch("whistle.inference.MAX_CACHE_LEN", 64):
            chunks = list(
                stream_tts(
                    self.tts,
                    "hi",
                    max_new_tokens=8,
                    chunk_size=4,
                    ramp_frames=(2, 4),
                    stop_at_eos=False,
                )
            )
        self.assertEqual([chunk["chunk_frames"] for chunk in chunks], [2, 2, 4])
        self.assertEqual(sum(chunk["chunk_frames"] for chunk in chunks), 8)
        self.assertTrue(chunks[-1]["final"])
        self.assertFalse(any(chunk["final"] for chunk in chunks[:-1]))

    def test_eos_scan_follows_cadence(self) -> None:
        """The chunked EOS scan only looks at cadence points unless forced."""
        codes = torch.zeros((16, 4), dtype=torch.long)
        codes[11, 0] = 7
        self.assertIsNone(_maybe_eos_row(codes, 8, 7, stop_at_eos=True, force=False))
        self.assertEqual(_maybe_eos_row(codes, 16, 7, stop_at_eos=True, force=False), 11)
        self.assertEqual(_maybe_eos_row(codes, 12, 7, stop_at_eos=True, force=True), 11)
        self.assertIsNone(_maybe_eos_row(codes, 12, 7, stop_at_eos=False, force=True))


if __name__ == "__main__":
    unittest.main()
