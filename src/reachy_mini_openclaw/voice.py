"""Common lifecycle for ClawBody voice backends."""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

AudioFrame = tuple[int, NDArray[np.generic]]


class VoiceHandler(Protocol):
    """The audio lifecycle consumed by :class:`ClawBodyCore`."""

    async def start_up(self) -> None: ...

    async def receive(self, frame: AudioFrame) -> None: ...

    async def emit(self) -> Any: ...

    async def shutdown(self) -> None: ...


def create_voice_handler(*, deps: Any, openclaw_bridge: Any) -> VoiceHandler:
    """Create the configured backend without importing optional providers eagerly."""
    from reachy_mini_openclaw.config import config

    if config.VOICE_BACKEND == "openai":
        from reachy_mini_openclaw.openai_realtime import OpenAIRealtimeHandler

        return OpenAIRealtimeHandler(deps=deps, openclaw_bridge=openclaw_bridge)

    from reachy_mini_openclaw.audio.stt import create_stt_provider
    from reachy_mini_openclaw.audio.tts import create_tts_provider
    from reachy_mini_openclaw.local_voice import LocalVoiceHandler

    if openclaw_bridge is None:
        raise ValueError("OpenClaw is required when VOICE_BACKEND=local")
    return LocalVoiceHandler(
        deps=deps,
        openclaw_bridge=openclaw_bridge,
        stt=create_stt_provider(),
        tts=create_tts_provider(),
        vad_threshold=config.VAD_THRESHOLD,
        silence_ms=config.VAD_SILENCE_MS,
        min_speech_ms=config.VAD_MIN_SPEECH_MS,
    )
