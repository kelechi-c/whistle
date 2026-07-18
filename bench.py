#!/usr/bin/env python3
"""Qwen3-ASR 0.6B inference + benchmark.

Usage:
    uv run --no-sync python bench.py <audio.mp3> [--model Qwen/Qwen3-ASR-0.6B]
    uv run --no-sync python bench.py --audio-loading-only <audio.mp3>
"""

import time
from pathlib import Path

import click

import numpy as np
import torch

AudioLike = str | tuple[np.ndarray, int]

SAMPLE_RATE = 16000


def benchmark_audio_loading(path: str) -> dict[str, float]:
    import librosa
    import soundfile as sf

    results = {}

    # --- librosa ---
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    audio_lib, sr_lib = librosa.load(path, sr=None, mono=False)
    audio_lib = np.asarray(audio_lib, dtype=np.float32)
    if audio_lib.ndim > 1:
        audio_lib = np.mean(audio_lib, axis=0).astype(np.float32)
    if sr_lib != SAMPLE_RATE:
        audio_lib = librosa.resample(audio_lib, orig_sr=sr_lib, target_sr=SAMPLE_RATE).astype(np.float32)
    t1 = time.perf_counter()
    results["librosa_load+resample"] = t1 - t0
    results["librosa_duration_sec"] = len(audio_lib) / SAMPLE_RATE

    # --- torchaudio ---
    import torchaudio
    import torchaudio.functional as Fa

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    wav_ta, sr_ta = torchaudio.load(path)
    if wav_ta.shape[0] > 1:
        wav_ta = wav_ta.mean(dim=0, keepdim=True)
    if sr_ta != SAMPLE_RATE:
        wav_ta = Fa.resample(wav_ta, orig_freq=sr_ta, new_freq=SAMPLE_RATE)
    audio_ta = wav_ta.squeeze(0).numpy().astype(np.float32)
    t1 = time.perf_counter()
    results["torchaudio_load+resample"] = t1 - t0
    results["torchaudio_duration_sec"] = len(audio_ta) / SAMPLE_RATE

    # --- torchcodec ---
    audio_tc = np.array([], dtype=np.float32)
    try:
        from torchcodec.decoders import AudioDecoder
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        decoder = AudioDecoder(path)
        samples = decoder.get_all_samples()
        wav_tc = samples.data
        sr_tc = samples.sample_rate
        if wav_tc.shape[0] > 1:
            wav_tc = wav_tc.mean(dim=0, keepdim=True)
        if sr_tc != SAMPLE_RATE:
            wav_tc = Fa.resample(wav_tc, orig_freq=sr_tc, new_freq=SAMPLE_RATE)
        audio_tc = wav_tc.squeeze(0).numpy().astype(np.float32)
        t1 = time.perf_counter()
        results["torchcodec_load+resample"] = t1 - t0
        results["torchcodec_duration_sec"] = len(audio_tc) / SAMPLE_RATE
    except Exception as e:
        print(f"warning: torchcodec benchmark failed ({e})")
        results["torchcodec_load+resample"] = None
        results["torchcodec_duration_sec"] = None

    # validate consistency
    max_diff_lib_ta = float(np.max(np.abs(audio_lib[:min(len(audio_lib), len(audio_ta))] - audio_ta[:min(len(audio_lib), len(audio_ta))])))
    results["lib_vs_ta_max_diff"] = float(f"{max_diff_lib_ta:.6f}")
    if "torchcodec_duration_sec" in results and results["torchcodec_duration_sec"] is not None and len(audio_tc) > 0:
        max_diff_lib_tc = float(np.max(np.abs(audio_lib[:min(len(audio_lib), len(audio_tc))] - audio_tc[:min(len(audio_lib), len(audio_tc))])))
        results["lib_vs_tc_max_diff"] = float(f"{max_diff_lib_tc:.6f}")
    else:
        results["lib_vs_tc_max_diff"] = "n/a"

    return results


@torch.no_grad()
def benchmark_inference(
    model_path: str,
    wav: AudioLike,
    context: str = "",
    language: str | None = None,
    max_new_tokens: int = 512,
    compile: bool = False,
) -> dict:
    from qwen_asr import Qwen3ASRModel

    timings: dict[str, float] = {}
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"loading model {model_path} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    asr = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=2,
        max_new_tokens=max_new_tokens,
    )
    timings["model_load"] = time.perf_counter() - t0
    print(f"done in {timings['model_load']:.2f}s")

    # normalize audio
    from qwen_asr.inference.utils import normalize_audio_input

    print("normalizing audio ...", end=" ", flush=True)
    t0 = time.perf_counter()
    wav_np = normalize_audio_input(wav)
    timings["audio_normalize"] = time.perf_counter() - t0
    print(f"done in {timings['audio_normalize']:.3f}s")

    # build text prompt
    print("building prompt ...", end=" ", flush=True)
    t0 = time.perf_counter()
    texts = [asr._build_text_prompt(context=context, force_language=language)]
    timings["build_prompt"] = time.perf_counter() - t0
    print(f"done in {timings['build_prompt']:.4f}s")

    # processor encode
    print("processor encode ...", end=" ", flush=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    inputs = asr.processor(text=texts, audio=[wav_np], return_tensors="pt", padding=True)
    inputs = inputs.to(asr.model.device).to(asr.model.dtype)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    timings["processor_encode"] = time.perf_counter() - t0
    print(f"done in {timings['processor_encode']:.3f}s")

    if compile:
        print("compiling model ...", end=" ", flush=True)
        t0 = time.perf_counter()
        asr.model = torch.compile(asr.model, mode="reduce-overhead")
        timings["compile"] = time.perf_counter() - t0
        print(f"done in {timings['compile']:.2f}s")

    # warmup: same audio, short generation to trigger cuda kernel compilation
    print("warmup ...", end=" ", flush=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    _ = asr.model.generate(**inputs, max_new_tokens=min(32, max_new_tokens))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    timings["warmup"] = time.perf_counter() - t0
    print(f"done in {timings['warmup']:.3f}s")

    # modular: audio encoder forward (neural mel -> hidden states)
    print("audio encoder ...", end=" ", flush=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    if hasattr(asr.model, "get_encoder"):
        encoder = asr.model.get_encoder()
        encoder.eval()
        encoder_out = encoder(
            input_features=inputs["input_features"],
            attention_mask=inputs["feature_attention_mask"],
        )
    elif hasattr(asr.model, "encoder"):
        encoder_out = asr.model.encoder(
            input_features=inputs["input_features"],
            attention_mask=inputs["feature_attention_mask"],
        )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    timings["audio_encoder"] = time.perf_counter() - t0
    print(f"done in {timings['audio_encoder']:.3f}s")

    # modular: model.generate (encoder + LM decode)
    print(f"model generate (max_new={max_new_tokens}) ...", end=" ", flush=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    text_ids = asr.model.generate(**inputs, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    timings["model_generate"] = time.perf_counter() - t0
    print(f"done in {timings['model_generate']:.3f}s")

    # modular: decode
    print("decode output ...", end=" ", flush=True)
    t0 = time.perf_counter()
    decoded = asr.processor.batch_decode(
        text_ids.sequences[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    timings["token_decode"] = time.perf_counter() - t0
    print(f"done in {timings['token_decode']:.4f}s")

    raw = decoded[0]

    from qwen_asr.inference.utils import parse_asr_output

    lang, transcript = parse_asr_output(raw, user_language=language)

    timings["lm_decode"] = timings["model_generate"] - timings["audio_encoder"]

    timings["total_modular"] = sum(
        timings[k] for k in ("audio_normalize", "build_prompt", "processor_encode", "audio_encoder", "model_generate", "token_decode")
    )

    # end2end transcribe
    print("end2end transcribe ...", end=" ", flush=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    results = asr.transcribe(
        audio=wav,
        context=context,
        language=language,
        return_time_stamps=False,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    timings["end2end"] = time.perf_counter() - t0
    print(f"done in {timings['end2end']:.3f}s")

    result = results[0]

    return {
        "timings": timings,
        "language": result.language or lang,
        "transcript": result.text or transcript,
        "raw": raw,
    }


@click.command()
@click.argument("audio", type=click.Path(exists=True))
@click.option("--model", default="Qwen/Qwen3-ASR-1.7B", show_default=True)
@click.option("--lang", default="English", show_default=True, help="force language")
@click.option("--context", default="", help="context/hint text")
@click.option("--max-new-tokens", default=256, show_default=True)
@click.option("--compile", is_flag=True, help="torch.compile model")
@click.option("--audio-loading-only", is_flag=True, help="only benchmark audio loading")
def main(audio, model, lang, context, max_new_tokens, compile, audio_loading_only):
    audio_path = str(Path(audio).resolve())

    print("=" * 60)
    print("qwen3-asr 0.6b benchmark")
    print("=" * 60)
    print(f"audio: {audio_path}")
    print(f"language: {lang}")
    if compile:
        print(f"compile: enabled")
    print()

    # --- audio loading benchmarks ---
    print("[audio processing benchmarks]")
    audio_stats = benchmark_audio_loading(audio_path)
    print(f"audio duration (librosa):  {audio_stats['librosa_duration_sec']:.2f}s")
    print(f"audio duration (torchaud): {audio_stats['torchaudio_duration_sec']:.2f}s")
    print(f"audio duration (tc):       {audio_stats['torchcodec_duration_sec']:.2f}s")
    print()
    print(f"{'method':<30s} {'time (s)':>10s}")
    print(f"{'-'*30} {'-'*10}")
    for key in ("librosa_load+resample", "torchaudio_load+resample", "torchcodec_load+resample"):
        print(f"{key:<30s} {audio_stats[key]:>10.4f}")
    print()
    print(f"librosa vs torchaudio:  max_diff={audio_stats['lib_vs_ta_max_diff']}")
    print(f"librosa vs torchcodec:  max_diff={audio_stats['lib_vs_tc_max_diff']}")
    print()

    if audio_loading_only:
        return

    if not torch.cuda.is_available():
        print("[inference benchmarks] skipping — no GPU detected")
        print("run this on your GPU machine for inference benchmarks\n")
        return

    # --- inference benchmarks ---
    print("[inference benchmarks]")
    result = benchmark_inference(
        model_path=model,
        wav=audio_path,
        context=context,
        language=lang,
        max_new_tokens=max_new_tokens,
        compile=compile,
    )
    print()

    t = result["timings"]
    print(f"{'phase':<30s} {'time (s)':>10s}")
    print(f"{'-'*30} {'-'*10}")
    for key, label in [
        ("model_load", "model load"),
        ("compile", "compile"),
        ("warmup", "warmup"),
        ("audio_normalize", "audio normalize"),
        ("build_prompt", "build prompt"),
        ("processor_encode", "processor encode"),
        ("audio_encoder", "audio encoder"),
        ("lm_decode", "lm decode"),
        ("model_generate", "model generate"),
        ("token_decode", "token decode"),
        ("total_modular", "total (modular)"),
        ("end2end", "end2end transcribe"),
    ]:
        if key in t:
            print(f"{label:<30s} {t[key]:>10.4f}")

    print()
    print(f"end2end / total_modular ratio:  {t['end2end'] / t['total_modular']:.2f}x")
    audio_dur = audio_stats["librosa_duration_sec"]
    if t.get("end2end"):
        print(f"rtf (audio_dur / end2end):      {audio_dur / t['end2end']:.2f}x")
    print()

    print("[transcript]")
    print(f"detected language: {result['language']!r}")
    print()
    for line in result["transcript"].strip().split("\n"):
        print(f"{line}")
    print()
