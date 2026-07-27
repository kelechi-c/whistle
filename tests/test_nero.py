"""CPU regression tests for the official-shaped tiny CustomVoice path."""

import pathlib as pl
import tempfile
import unittest

import torch

from nero.create_tiny_fixture import create_fixture
from nero.infer import fixture_text_ids
from nero.model.custom_voice import Qwen3CustomVoice

FIXTURE = pl.Path("tests/fixtures/nero_tiny")


class Qwen3CustomVoiceTest(unittest.TestCase):
    """Validates modular generation, strict loading, and official namespaces."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (FIXTURE / "model.safetensors").exists():
            create_fixture(FIXTURE)

    def test_tiny_generation_has_official_shapes(self) -> None:
        model = Qwen3CustomVoice.from_pretrained(
            FIXTURE, local_files_only=True
        )
        text_ids = fixture_text_ids("test", torch.device("cpu"))

        result = model.generate(text_ids, max_frames=2, stop_on_eos=False)

        self.assertEqual(result.codes.shape, (1, 4, 2))
        self.assertEqual(result.audio.shape, (1, 1, 16))
        self.assertLessEqual(result.audio.abs().max(), 1)
        self.assertEqual(
            set(result.timings),
            {"prepare", "talker_and_code_predictor", "codec"},
        )

    def test_major_sections_are_independently_callable(self) -> None:
        model = Qwen3CustomVoice.from_pretrained(
            FIXTURE, local_files_only=True
        )
        prepared = model.prepare_input(
            fixture_text_ids("x", torch.device("cpu"))
        )
        first, state = model.talker.prefill(prepared.prompt)
        residual = model.talker.code_predictor.predict_frame(
            state.hidden,
            model.talker.model.codec_embedding(first).unsqueeze(1),
        )
        codes = torch.cat((first[:, None], residual), dim=-1)[:, :, None]

        audio = model.codec_decoder(codes)

        self.assertEqual(codes.shape, (1, 4, 1))
        self.assertEqual(audio.shape, (1, 1, 8))

    def test_checkpoint_uses_official_weight_names(self) -> None:
        model = Qwen3CustomVoice.from_pretrained(
            FIXTURE, local_files_only=True
        )
        names = set(model.model.state_dict())
        codec_names = set(model.speech_tokenizer.state_dict())

        self.assertIn("talker.model.layers.0.self_attn.q_proj.weight", names)
        self.assertIn(
            "talker.code_predictor.model.codec_embedding.0.weight", names
        )
        self.assertIn("talker.text_projection.linear_fc1.bias", names)
        self.assertIn(
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
            codec_names,
        )
        self.assertIn(
            "decoder.pre_transformer.layers.0.self_attn.q_proj.weight",
            codec_names,
        )

    def test_checkpoint_round_trip_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = pl.Path(directory) / "checkpoint"
            create_fixture(checkpoint)
            first = Qwen3CustomVoice.from_pretrained(
                checkpoint, local_files_only=True
            )
            second = Qwen3CustomVoice.from_pretrained(
                checkpoint, local_files_only=True
            )
            ids = fixture_text_ids("hi", torch.device("cpu"))
            expected = first.generate(ids, max_frames=1, stop_on_eos=False)
            actual = second.generate(ids, max_frames=1, stop_on_eos=False)

        self.assertTrue(torch.equal(actual.codes, expected.codes))
        self.assertTrue(torch.equal(actual.audio, expected.audio))


if __name__ == "__main__":
    unittest.main()
