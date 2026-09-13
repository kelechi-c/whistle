import pathlib as pl

import click
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from whistle.config import CHECKPOINT, LANGUAGE, OUTPUT, SEED, SPEAKER
from whistle.inference import tts_infer


@click.command()
@click.argument("text")
@click.option("--checkpoint", default=CHECKPOINT, show_default=True)
@click.option("--speaker", default=SPEAKER, show_default=True)
@click.option("--language", default=LANGUAGE, show_default=True)
@click.option("--out", type=click.Path(path_type=pl.Path), default=OUTPUT, show_default=True)
def main(text: str, checkpoint: str, speaker: str, language: str, out: pl.Path) -> None:
    """Synthesizes TEXT with Qwen3-TTS on a cuda device and writes a WAV."""
    if not torch.cuda.is_available():
        raise click.ClickException("cuda is required; whistle runs qwen3-tts on a cuda device")
    torch.manual_seed(SEED)
    model = Qwen3TTSModel.from_pretrained(
        checkpoint, device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    waveform, codes, sample_rate, timings = tts_infer(
        model, text, speaker=speaker, language=language
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, waveform[0].float().cpu().numpy(), sample_rate)
    decode_ms = timings["decode"] * 1000
    codec_ms = timings["codec"] * 1000
    print(f"audio saved to {out}")
    print(f"frames: {codes.shape[0]}; decode: {decode_ms:.1f} ms; codec: {codec_ms:.1f} ms")


if __name__ == "__main__":
    main()
