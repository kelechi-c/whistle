#!/usr/bin/env python3
"""Qwen3-TTS inference benchmark (vanilla qwen_tts package).

Measures generation latency for text-to-speech:
  - model load
  - warmup / prefill
  - end2end generate (including codec decode)
  - optional profiling

Why ref_text is required for voice cloning (Base model):
  The model uses in-context learning (ICL). The reference audio is encoded
  into codec tokens (speaker voice characteristics + prosody), and the
  reference text is tokenized into text tokens. Both are concatenated as a
  prefix prompt before the target text. The model learns to associate the
  codec token sequence with the text token sequence — the transcript
  provides the alignment signal so the model can separate "who is speaking"
  (from codec tokens) from "what is being said" (from text). Without
  ref_text, the model has no way to disentangle speaker identity from content
  in the reference audio.

What autoregressive decode means here:
  At each step, the model generates 16 codec tokens (1 from the talker LM +
  15 from the code predictor) that represent ~83ms of audio (12 Hz frame rate).
  The next step's input depends on the previous step's output — the newly
  generated codec tokens are fed back as context. This is autoregressive
  because each step conditions on all prior steps, and you cannot compute
  step N+1 without finishing step N. The decode loop runs max_new_tokens
  times, generating `max_new_tokens * 83ms` of audio.

Usage:
    uv run --no-sync python bench_tts.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --ref uchechi.mp3 --ref-text "transcript of uchechi"
    uv run --no-sync python bench_tts.py --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --speaker Ryan --profile
    uv run --no-sync python bench_tts.py "Hello world" --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --instruct "cheerful" --profile
    
    --ref uchechi.mp3 --ref-text "Kelechi stop, stop,"
"""

import time
from pathlib import Path

import click
import torch

# Default text that demonstrates autoregressive speech generation.
# Each token feeds into the next — the model must maintain coherence
# across hundreds of autoregressive steps.
AUTOREGRESSIVE_DECODE_TEXT = (
    "This is a test of the autoregressive speech generation pipeline. "
    "Each word is generated one after another, building on everything "
    "that came before. The model maintains consistent voice characteristics, "
    "prosody, and speaking style across the entire utterance, all while "
    "processing the text input and converting it to natural sounding speech. "
    "Let us see how long this takes to generate and how the quality holds up."
)

SAMPLE_RATE = 24000


def benchmark_tts(
    text: str,
    model_path: str,
    language: str = "English",
    speaker: str | None = None,
    instruct: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    max_new_tokens: int = 2048,
    profile: bool = False,
) -> dict:
    from qwen_tts import Qwen3TTSModel

    timings: dict[str, float] = {}
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"loading model {model_path} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=dtype,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
    )
    timings["model_load"] = time.perf_counter() - t0
    print(f"done in {timings['model_load']:.2f}s")

    # warmup: short generate to trigger cuda kernel compilation
    print("warmup (short generate) ...", end=" ", flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    is_base = "Base" in model_path
    is_design = "VoiceDesign" in model_path
    is_custom = not is_base and not is_design

    if is_custom:
        model.generate_custom_voice(
            text="Hello.", language=language,
            speaker=speaker or "Vivian",
            max_new_tokens=min(32, max_new_tokens),
        )
    elif is_design:
        model.generate_voice_design(
            text="Hello.", language=language,
            instruct="Normal.",
            max_new_tokens=min(32, max_new_tokens),
        )
    elif is_base:
        model.generate_voice_clone(
            text="Hello.", language=language,
            ref_audio=ref_audio, ref_text=ref_text,
            max_new_tokens=min(32, max_new_tokens),
        )
    torch.cuda.synchronize()
    timings["warmup"] = time.perf_counter() - t0
    print(f"done in {timings['warmup']:.2f}s")

    # determine generation method and params
    if is_custom:
        gen_fn = model.generate_custom_voice
        gen_kwargs = dict(
            text=text, language=language,
            speaker=speaker or "Vivian",
            max_new_tokens=max_new_tokens,
        )
        if instruct:
            gen_kwargs["instruct"] = instruct
    elif is_design:
        gen_fn = model.generate_voice_design
        gen_kwargs = dict(
            text=text, language=language,
            instruct=instruct or "Normal speaking voice.",
            max_new_tokens=max_new_tokens,
        )
    else:
        gen_fn = model.generate_voice_clone
        gen_kwargs = dict(
            text=text, language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            max_new_tokens=max_new_tokens,
        )

    # generate
    print(f"generating (max_new={max_new_tokens}) ...", end=" ", flush=True)
    torch.cuda.synchronize()
    if profile:
        trace_path = f"trace_tts_{int(time.time())}.json"
        t0 = time.perf_counter()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            wavs, sr = gen_fn(**gen_kwargs)
        torch.cuda.synchronize()
        prof.export_chrome_trace(trace_path)
        timings["generate"] = time.perf_counter() - t0
        print(f"done in {timings['generate']:.3f}s")
        print(f"  trace saved to {trace_path}")
    else:
        t0 = time.perf_counter()
        wavs, sr = gen_fn(**gen_kwargs)
        torch.cuda.synchronize()
        timings["generate"] = time.perf_counter() - t0
        print(f"done in {timings['generate']:.3f}s")

    audio = wavs[0]
    audio_dur = audio.shape[-1] / sr
    timings["audio_duration"] = audio_dur

    return {"timings": timings, "audio": audio, "sr": sr}


@click.command()
@click.argument("text", default=AUTOREGRESSIVE_DECODE_TEXT)
@click.option("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base", show_default=True)
@click.option("--lang", default="English", show_default=True)
@click.option("--speaker", default=None, help="speaker ID (CustomVoice models)")
@click.option("--instruct", default=None, help="voice instruction (CustomVoice/VoiceDesign)")
@click.option("--ref", default="uchechi.mp3", show_default=True, help="reference audio path (Base models)")
@click.option("--ref-text", required=False, help="reference audio transcript (Base models)")
@click.option("--max-new-tokens", default=2048, show_default=True)
@click.option("--profile", is_flag=True, help="save torch profiler trace")
@click.option("--out", default='tts.mp3', help="save generated audio to file")
def main(text, model, lang, speaker, instruct, ref, ref_text, max_new_tokens, profile, out):
    if not torch.cuda.is_available():
        print("error: Qwen3-TTS requires CUDA")
        raise SystemExit(1)

    is_base = "Base" in model
    if is_base and not ref_text:
        print("error: Base models require --ref-text")
        print("")
        print("why? the model uses in-context learning (icl). the reference audio is")
        print("encoded to codec tokens (voice characteristics) and the reference text")
        print("is tokenized to text tokens. both form a prefix prompt that tells the")
        print("model: 'this speaker sounds like X and says Y'. without the transcript,")
        print("the model can't separate speaker identity from content — it doesn't know")
        print("what the reference audio actually says, so it can't learn the mapping")
        print("from codec tokens back to phonemes.")
        raise SystemExit(1)

    print("=" * 60)
    print("qwen3-tts benchmark")
    print("=" * 60)
    print(f"text: {text[:80]}...")
    print(f"language: {lang}")
    print(f"model: {model}")
    if is_base:
        print(f"ref audio: {ref}")
    print()

    result = benchmark_tts(
        text=text, model_path=model, language=lang,
        speaker=speaker, instruct=instruct,
        ref_audio=ref, ref_text=ref_text,
        max_new_tokens=max_new_tokens,
        profile=profile,
    )

    t = result["timings"]
    print()
    print(f"{'phase':<30s} {'time (s)':>10s}")
    print(f"{'-'*30} {'-'*10}")
    for key, label in [
        ("model_load", "model load"),
        ("warmup", "warmup"),
        ("generate", "generate"),
    ]:
        if key in t:
            print(f"{label:<30s} {t[key]:>10.4f}")

    audio_dur = t.get("audio_duration", 0)
    gen_time = t.get("generate", 0)
    if gen_time:
        print()
        print(f"audio duration: {audio_dur:.2f}s")
        print(f"rtf (audio / generate): {audio_dur / gen_time:.2f}x")

    if out:
        import soundfile as sf
        out_path = str(Path(out).resolve())
        sf.write(out_path, result["audio"], result["sr"])
        print(f"\nsaved to {out_path}")


if __name__ == "__main__":
    main()
