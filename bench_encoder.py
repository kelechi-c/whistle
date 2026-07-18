#!/usr/bin/env python3
"""Standalone Qwen3-ASR audio encoder benchmark on 60s audio.

Isolates the audio tower (conv2d stack + 32-layer transformer)
from the LLM and measures its throughput.

Usage:
    uv run --no-sync python bench_encoder.py <audio.mp3>
    uv run --no-sync python bench_encoder.py <audio.mp3> --profile
"""

import time
from pathlib import Path

import click
import numpy as np
import torch
import torchaudio.functional as Fa
from torchcodec.decoders import AudioDecoder


SAMPLE_RATE = 16000
N_MEL = 128
HOP_LENGTH = 160
WIN_LENGTH = 400
N_FFT = 512


def load_audio(path: str) -> torch.Tensor:
    decoder = AudioDecoder(path)
    samples = decoder.get_all_samples()
    wav = samples.data
    sr = samples.sample_rate
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = Fa.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
    return wav.squeeze(0).to(torch.float32)


def mel_spec(wav: torch.Tensor, n_mels: int = N_MEL) -> torch.Tensor:
    from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

    window = window_function(WIN_LENGTH, "hann", device=wav.device)
    spec = spectrogram(
        wav[None, :], window, frame_length=WIN_LENGTH, hop_length=HOP_LENGTH,
        fft_length=N_FFT, power=2.0, center=True,
    )
    mel_filters = mel_filter_bank(
        int(N_FFT // 2 + 1), n_mels, SAMPLE_RATE, 0.0, None, norm="slaney", mel_scale="slaney",
    )
    mel_spec = mel_filters @ spec
    mel_spec = torch.clamp(mel_spec, min=1e-10).log10()
    max_val = mel_spec.abs().max()
    if max_val > 0:
        mel_spec = torch.clamp(mel_spec, min=max_val * -8.0)
    return mel_spec.T.unsqueeze(0)


@click.command()
@click.argument("audio", type=click.Path(exists=True))
@click.option("--model", default="Qwen/Qwen3-ASR-1.7B", show_default=True)
@click.option("--profile", is_flag=True, help="save torch profiler trace")
@click.option("--seconds", default=60, help="audio duration to test (truncates if longer)")
def main(audio, model, profile, seconds):
    audio_path = str(Path(audio).resolve())

    print("=" * 60)
    print("qwen3-asr audio encoder benchmark")
    print("=" * 60)
    print(f"model: {model}")
    print()

    # load audio and mel
    print("loading audio ...", end=" ", flush=True)
    t0 = time.perf_counter()
    wav = load_audio(audio_path)
    dur = wav.shape[-1] / SAMPLE_RATE
    print(f"done ({dur:.1f}s @ {SAMPLE_RATE}hz)")

    # truncate/pad to target seconds
    target_samples = int(seconds * SAMPLE_RATE)
    if wav.shape[-1] > target_samples:
        wav = wav[:target_samples]
        print(f"truncated to {seconds}s")
    elif wav.shape[-1] < target_samples:
        pad = target_samples - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, pad))
        print(f"padded to {seconds}s")
    print()

    # mel spectrogram
    print("computing mel spectrogram ...", end=" ", flush=True)
    t0 = time.perf_counter()
    mel = mel_spec(wav)
    mel_len = mel.shape[1]
    print(f"done ({mel.shape})")

    print(f"\nmel frames: {mel_len} ({(mel_len * HOP_LENGTH / SAMPLE_RATE):.1f}s equivalent)")

    # load model
    from transformers import AutoConfig, AutoModel, AutoProcessor
    from qwen_asr.core.transformers_backend import Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor

    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)

    print(f"\nloading model {model} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    config = AutoConfig.from_pretrained(model)
    hf_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    ).eval()
    processor = AutoProcessor.from_pretrained(model)
    print(f"done in {time.perf_counter() - t0:.2f}s")

    # extract audio tower
    encoder = hf_model.thinker.audio_tower
    print(f"audio tower: {type(encoder).__name__}")
    print(f"encoder layers: {config.thinker_config.audio_config.encoder_layers}")
    print(f"d_model: {config.thinker_config.audio_config.d_model}")
    print(f"output_dim: {config.thinker_config.audio_config.output_dim}")

    # prepare input: mel of shape (1, T, D)
    mel = mel.to(device=hf_model.device, dtype=hf_model.dtype)
    feature_lens = torch.tensor([mel_len], device=hf_model.device)

    # warmup
    print("\nwarmup ...", end=" ", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = encoder(mel, feature_lens=feature_lens)
    torch.cuda.synchronize()
    print(f"done in {time.perf_counter() - t0:.3f}s")

    # timed run
    n_runs = 5
    print(f"timing {n_runs}x encoder forward ...", end=" ", flush=True)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = encoder(mel, feature_lens=feature_lens)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    print("done")
    print(f"\n  mean: {np.mean(times):.4f}s  min: {np.min(times):.4f}s  max: {np.max(times):.4f}s")
    print(f"  output shape: {out.shape}")
    print(f"  rtf ({dur:.1f}s audio / encoder): {dur / np.mean(times):.1f}x")

    # profile run
    if profile:
        trace_path = f"trace_encoder_{Path(audio).stem}.json"
        print(f"\nprofiling encoder -> {trace_path} ...", end=" ", flush=True)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            _ = encoder(mel, feature_lens=feature_lens)
        torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)
        print("done")
        print(f"trace saved to {trace_path}")
        print("open in chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()
