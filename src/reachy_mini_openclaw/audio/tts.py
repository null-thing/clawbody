"""Pluggable local text-to-speech providers."""

import asyncio
import subprocess
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class TTSProvider(Protocol):
    async def synthesize(self, text: str) -> tuple[int, NDArray[np.int16]]: ...

    async def shutdown(self) -> None: ...


class PiperTTS:
    """Offline TTS through Piper's stable raw-audio CLI interface."""

    def __init__(self, executable: str, model_path: str, sample_rate: int):
        self.executable = executable
        self.model_path = model_path
        self.sample_rate = sample_rate

    def _synthesize(self, text: str) -> tuple[int, NDArray[np.int16]]:
        try:
            result = subprocess.run(
                [self.executable, "--model", self.model_path, "--output-raw"],
                input=text.encode("utf-8"),
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Piper executable not found: {self.executable}") from exc
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Piper synthesis failed: {message}") from exc
        audio = np.frombuffer(result.stdout, dtype=np.int16).copy()
        return self.sample_rate, audio.reshape(1, -1)

    async def synthesize(self, text: str) -> tuple[int, NDArray[np.int16]]:
        return await asyncio.to_thread(self._synthesize, text)

    async def shutdown(self) -> None:
        return None


def create_tts_provider() -> TTSProvider:
    from reachy_mini_openclaw.config import config

    if config.TTS_BACKEND != "piper":
        raise ValueError(f"Unsupported TTS_BACKEND: {config.TTS_BACKEND}")
    return PiperTTS(config.PIPER_EXECUTABLE, config.PIPER_MODEL, config.PIPER_SAMPLE_RATE)
