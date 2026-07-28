"""Compatibility entry point for the official-module inference baseline."""

from faster_decode import main, tts_infer

__all__ = ["tts_infer"]


if __name__ == "__main__":
    main()
