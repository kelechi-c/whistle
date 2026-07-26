"""Opt-in CUDA parity test against the official Qwen3-TTS wrapper."""

from __future__ import annotations

import gc
import os
import unittest

import numpy as np
import torch

from nero.model.custom_voice import Qwen3CustomVoice


MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
RUN_PARITY = os.environ.get("RUN_QWEN_TTS_PARITY") == "1"


@unittest.skipUnless(RUN_PARITY and torch.cuda.is_available(), "CUDA parity test is opt-in")
class CustomVoiceParityTest(unittest.TestCase):
    def test_waveform_is_identical_to_official_wrapper(self) -> None:
        from qwen_tts import Qwen3TTSModel

        common_load = {
            "device_map": "cuda:0",
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "local_files_only": True,
        }
        common_generate = {
            "text": "The same weights and inputs must produce the same speech.",
            "speaker": "Ryan",
            "language": "English",
            "max_new_tokens": 96,
            "do_sample": False,
            "subtalker_dosample": False,
        }

        torch.manual_seed(0)
        official = Qwen3TTSModel.from_pretrained(MODEL, **common_load)
        official_wavs, official_rate = official.generate_custom_voice(**common_generate)
        official_audio = np.asarray(official_wavs[0], dtype=np.float32).copy()
        del official
        gc.collect()
        torch.cuda.empty_cache()

        torch.manual_seed(0)
        nero = Qwen3CustomVoice.from_pretrained(
            MODEL,
            device=torch.device("cuda:0"),
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        nero_wavs, nero_rate = nero.generate_custom_voice(**common_generate)
        nero_audio = np.asarray(nero_wavs[0], dtype=np.float32)

        self.assertEqual(nero_rate, official_rate)
        self.assertEqual(nero_audio.shape, official_audio.shape)
        np.testing.assert_array_equal(nero_audio, official_audio)
        print(
            f"parity: {nero_audio.size} samples at {nero_rate} Hz; "
            "max_abs_difference=0"
        )


if __name__ == "__main__":
    unittest.main()
