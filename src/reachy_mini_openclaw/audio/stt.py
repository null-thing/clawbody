"""Pluggable local speech-to-text providers."""

import asyncio
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class STTProvider(Protocol):
    async def transcribe(self, audio: NDArray[np.float32], sample_rate: int) -> str: ...

    async def shutdown(self) -> None: ...


class FasterWhisperSTT:
    """Offline STT backed by faster-whisper (loaded lazily)."""

    def __init__(self, model_name: str, device: str, compute_type: str, language: str | None):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _transcribe(self, audio: NDArray[np.float32], sample_rate: int) -> str:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('Install local voice dependencies with: pip install -e ".[local_voice]"') from exc

        if self._model is None:
            self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)

        if sample_rate != 16000:
            from scipy.signal import resample

            count = int(len(audio) * 16000 / sample_rate)
            audio = resample(audio, count).astype(np.float32)
        segments, _ = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    async def transcribe(self, audio: NDArray[np.float32], sample_rate: int) -> str:
        return await asyncio.to_thread(self._transcribe, audio, sample_rate)

    async def shutdown(self) -> None:
        self._model = None


def create_stt_provider() -> STTProvider:
    from reachy_mini_openclaw.config import config

    if config.STT_BACKEND != "faster-whisper":
        raise ValueError(f"Unsupported STT_BACKEND: {config.STT_BACKEND}")
    return FasterWhisperSTT(
        config.STT_MODEL,
        config.STT_DEVICE,
        config.STT_COMPUTE_TYPE,
        config.STT_LANGUAGE,
    )
