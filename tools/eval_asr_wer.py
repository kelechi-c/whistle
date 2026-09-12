#!/usr/bin/env python3
"""Evaluates synthesized TTS audio with Qwen3-ASR: WER/CER vs the source text.

The parity check tells us whether a mode is bit-exact with the official codec;
this script gives a quality number even when parity diverges. It transcribes
the waveform with Qwen3-ASR and reports word/char edit rates against the
normalized reference text. Run it once per WAV and compare the WERs.
"""

import json
import pathlib as pl
import re
from typing import Any

import click
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)

SAMPLE_RATE = 16_000
CHUNK_SEC = 30.0
_ASR_TEXT_TAG = "<asr_text>"

try:  # torchaudio is present on victoria's venv; interpolate is the fallback
    import torchaudio

    def _resample(wav: torch.Tensor, orig: int, target: int) -> torch.Tensor:
        """Resamples a float waveform to ``target`` Hz (torchaudio)."""
        return torchaudio.functional.resample(wav, orig, target)
except ImportError:

    def _resample(wav: torch.Tensor, orig: int, target: int) -> torch.Tensor:
        """Linear interpolation fallback when torchaudio is missing."""
        return torch.nn.functional.interpolate(
            wav.view(1, 1, -1), scale_factor=target / orig
        ).view(-1)


def _build_prompt(processor: Any, language: str | None) -> str:
    """Builds the Qwen3-ASR chat prompt, optionally forcing the language."""
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    base = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return base + f"language {language}{_ASR_TEXT_TAG}" if language else base


def _parse(raw: str, language: str | None) -> str:
    """Extracts plain transcript text from the tagged ASR output."""
    s = str(raw).strip()
    if language:
        return s
    if _ASR_TEXT_TAG not in s:
        return s
    meta, text = s.split(_ASR_TEXT_TAG, 1)
    return text.strip()


def _normalize(text: str) -> list[str]:
    """Lowercases, strips punctuation, and splits into word tokens."""
    words = re.sub(r"[^a-z0-9']+", " ", text.lower()).split()
    return [w.strip("'") for w in words if w.strip("'")]


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Edit distance between two token sequences (rolling DP)."""
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = row
    return prev[-1]


@torch.inference_mode()
def transcribe(wav: torch.Tensor, model: Any, processor: Any, *, language: str | None, max_new_tokens: int) -> str:
    """Runs Qwen3-ASR over 30-second chunks and concatenates the transcripts."""
    prompt = _build_prompt(processor, language)
    chunk_samples = int(SAMPLE_RATE * CHUNK_SEC)
    parts: list[str] = []
    for start in range(0, wav.numel(), chunk_samples):
        chunk = wav[start : start + chunk_samples]
        inputs = processor(text=[prompt], audio=[chunk.numpy()], return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = processor.batch_decode(
            ids.sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        parts.append(_parse(decoded[0], language))
    return " ".join(parts)


@click.command()
@click.option("--wav", type=click.Path(path_type=pl.Path, exists=True), required=True)
@click.option("--ref-text", default=None, help="reference transcript (source text)")
@click.option("--ref-file", type=click.Path(path_type=pl.Path, exists=True), default=None, help="read reference from utf-8 file")
@click.option("--asr-model", default="Qwen/Qwen3-ASR-0.6B", show_default=True)
@click.option("--language", default="english", show_default=True)
@click.option("--max-new-tokens", type=click.IntRange(min=16), default=512)
@click.option("--json-out", type=click.Path(path_type=pl.Path), default=None)
def main(wav: pl.Path, ref_text: str | None, ref_file: pl.Path | None, asr_model: str, language: str, max_new_tokens: int, json_out: pl.Path | None) -> None:
    """Reports WER/CER of WAV against the reference text (Qwen3-ASR)."""
    if ref_text is None and ref_file is None:
        raise click.ClickException("provide --ref-text or --ref-file")
    reference = ref_text if ref_text is not None else ref_file.read_text(encoding="utf-8")

    audio, orig_sr = sf.read(wav, dtype="float32", always_2d=True)
    mono = torch.from_numpy(audio.mean(axis=1))
    if orig_sr != SAMPLE_RATE:
        mono = _resample(mono, orig_sr, SAMPLE_RATE)
    mono = mono.to(torch.float32)

    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        asr_model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    processor = Qwen3ASRProcessor.from_pretrained(asr_model)

    hypothesis = transcribe(mono, model, processor, language=language if language != "auto" else None, max_new_tokens=max_new_tokens)
    ref_words = _normalize(reference)
    hyp_words = _normalize(hypothesis)
    ref_chars = list("".join(ref_words))
    hyp_chars = list("".join(hyp_words))
    word_dist = _levenshtein(ref_words, hyp_words)
    char_dist = _levenshtein(ref_chars, hyp_chars)
    wer = word_dist / len(ref_words) if ref_words else float("nan")
    cer = char_dist / len(ref_chars) if ref_chars else float("nan")

    print(f"reference: {len(ref_words)} words")
    print(f"hypothesis: {len(hyp_words)} words")
    print(f"wer: {wer * 100:.2f}%  ({word_dist} edits / {len(ref_words)} words)")
    print(f"cer: {cer * 100:.2f}%  ({char_dist} edits / {len(ref_chars)} chars)")
    if ref_words != hyp_words:
        print(f"hyp: {' '.join(hyp_words)}")
    if json_out is not None:
        payload = {
            "wav": str(wav),
            "asr_model": asr_model,
            "reference_words": len(ref_words),
            "hypothesis_words": len(hyp_words),
            "word_edit_distance": word_dist,
            "char_edit_distance": char_dist,
            "wer": wer,
            "cer": cer,
            "reference": " ".join(ref_words),
            "hypothesis": " ".join(hyp_words),
        }
        json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"results saved to {json_out}")


if __name__ == "__main__":
    main()