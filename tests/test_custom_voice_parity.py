"""Opt-in structural load test for the full official CustomVoice checkpoint."""

import os
import unittest

import torch

from nero.model.custom_voice import Qwen3CustomVoice

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
RUN_FULL_LOAD = os.environ.get("RUN_QWEN_TTS_FULL_LOAD") == "1"


@unittest.skipUnless(RUN_FULL_LOAD, "full checkpoint load is opt-in")
class CustomVoiceCompatibilityTest(unittest.TestCase):
    """Confirms strict loading without delegating inference to qwen-tts."""

    def test_official_weights_load_strictly(self) -> None:
        model = Qwen3CustomVoice.from_pretrained(
            MODEL,
            device=torch.device("cpu"),
            dtype=torch.float32,
            local_files_only=True,
        )

        self.assertEqual(model.model.config.tts_model_type, "custom_voice")
        self.assertEqual(model.talker.config.num_code_groups, 16)
        self.assertEqual(model.codec_decoder.config.num_quantizers, 16)


if __name__ == "__main__":
    unittest.main()
