"""Local STT -> OpenClaw -> local TTS voice backend."""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from reachy_mini_openclaw.audio.stt import STTProvider
from reachy_mini_openclaw.audio.tts import TTSProvider
from reachy_mini_openclaw.tools.core_tools import dispatch_tool_call

logger = logging.getLogger(__name__)

LOCAL_VOICE_CONTEXT = """You are speaking through a Reachy Mini robot body.
Keep the spoken response concise and natural. Return ONLY one JSON object with this schema:
{"speech":"words to speak", "actions":[{"name":"emotion", "arguments":{"emotion_name":"happy"}}]}
Allowed action names are look, camera, face_tracking, dance, emotion, stop_moves, and idle.
Actions are optional; use an empty list when no body action is useful. Never invent action names."""


@dataclass
class LocalResponse:
    speech: str
    actions: list[dict[str, Any]] = field(default_factory=list)


def parse_local_response(content: str) -> LocalResponse:
    """Parse the strict action envelope, falling back safely to speech-only text."""
    candidate = content.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1])
            if candidate.lstrip().startswith("json"):
                candidate = candidate.lstrip()[4:].lstrip()
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return LocalResponse(speech=content.strip())
    if not isinstance(payload, dict) or not isinstance(payload.get("speech"), str):
        return LocalResponse(speech=content.strip())
    actions = payload.get("actions", [])
    if not isinstance(actions, list):
        actions = []
    valid_actions = []
    for action in actions:
        if isinstance(action, dict) and isinstance(action.get("name"), str):
            arguments = action.get("arguments", {})
            if isinstance(arguments, dict):
                valid_actions.append({"name": action["name"], "arguments": arguments})
    return LocalResponse(speech=payload["speech"].strip(), actions=valid_actions)


class LocalVoiceHandler:
    """Turn-based local voice handler with VAD and barge-in cancellation."""

    def __init__(
        self,
        *,
        deps: Any,
        openclaw_bridge: Any,
        stt: STTProvider,
        tts: TTSProvider,
        vad_threshold: float,
        silence_ms: int,
        min_speech_ms: int,
    ):
        self.deps = deps
        self.openclaw_bridge = openclaw_bridge
        self.stt = stt
        self.tts = tts
        self.vad_threshold = vad_threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.output_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._audio: list[np.ndarray] = []
        self._sample_rate = 16000
        self._speech_ms = 0.0
        self._silence_ms = 0.0
        self._in_speech = False
        self._processing_task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()

    async def start_up(self) -> None:
        logger.info("Local voice backend ready (STT -> OpenClaw -> TTS)")
        await self._shutdown.wait()

    @staticmethod
    def _mono_float(audio: np.ndarray) -> np.ndarray:
        if audio.ndim == 2:
            if audio.shape[1] > audio.shape[0]:
                audio = audio.T
            audio = audio[:, 0]
        audio = audio.flatten()
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32, copy=False)

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        sample_rate, raw_audio = frame
        audio = self._mono_float(raw_audio)
        if audio.size == 0:
            return
        duration_ms = len(audio) * 1000.0 / sample_rate
        rms = float(np.sqrt(np.mean(np.square(audio))))
        is_speech = rms >= self.vad_threshold

        if is_speech:
            if not self._in_speech:
                self._in_speech = True
                self._speech_ms = 0.0
                self._silence_ms = 0.0
                self._audio.clear()
                await self._interrupt_output()
            self._sample_rate = sample_rate
            self._speech_ms += duration_ms
            self._silence_ms = 0.0
            self._audio.append(audio.copy())
            return

        if not self._in_speech:
            return
        self._audio.append(audio.copy())
        self._silence_ms += duration_ms
        if self._silence_ms >= self.silence_ms:
            utterance = np.concatenate(self._audio)
            speech_ms = self._speech_ms
            self._audio.clear()
            self._in_speech = False
            self._speech_ms = 0.0
            self._silence_ms = 0.0
            if speech_ms >= self.min_speech_ms:
                self._processing_task = asyncio.create_task(
                    self._process_utterance(utterance, self._sample_rate),
                    name="local-voice-turn",
                )

    async def _interrupt_output(self) -> None:
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self.deps.movement_manager.set_processing(False)
        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.reset()

    async def _process_utterance(self, audio: np.ndarray, sample_rate: int) -> None:
        from fastrtc import AdditionalOutputs

        self.deps.movement_manager.set_processing(True)
        try:
            transcript = await self.stt.transcribe(audio, sample_rate)
            if not transcript:
                return
            logger.info("User: %s", transcript)
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

            if not self.openclaw_bridge.is_connected:
                logger.info("OpenClaw disconnected; attempting reconnect")
                if not await self.openclaw_bridge.connect():
                    raise RuntimeError("OpenClaw gateway is unavailable")
            response = await self.openclaw_bridge.chat(transcript, system_context=LOCAL_VOICE_CONTEXT)
            if response.error:
                raise RuntimeError(response.error)
            parsed = parse_local_response(response.content)
            for action in parsed.actions:
                result = await dispatch_tool_call(
                    action["name"], json.dumps(action["arguments"]), self.deps
                )
                if "error" in result:
                    logger.warning("Robot action %s failed: %s", action["name"], result["error"])
            if not parsed.speech:
                return
            logger.info("Assistant: %s", parsed.speech)
            await self.output_queue.put(
                AdditionalOutputs({"role": "assistant", "content": parsed.speech})
            )
            sample_rate, speech_audio = await self.tts.synthesize(parsed.speech)
            flat_audio = speech_audio.astype(np.int16).flatten()
            chunk_size = max(1, sample_rate // 10)
            for offset in range(0, len(flat_audio), chunk_size):
                chunk = flat_audio[offset : offset + chunk_size].reshape(1, -1)
                if self.deps.head_wobbler is not None:
                    encoded = base64.b64encode(chunk.tobytes()).decode("ascii")
                    self.deps.head_wobbler.feed(encoded)
                await self.output_queue.put((sample_rate, chunk))
        except asyncio.CancelledError:
            logger.debug("Local voice turn interrupted")
            raise
        except Exception as exc:
            logger.error("Local voice turn failed: %s", exc)
        finally:
            self.deps.movement_manager.set_processing(False)

    async def emit(self) -> Any:
        return await self.output_queue.get()

    async def shutdown(self) -> None:
        self._shutdown.set()
        await self._interrupt_output()
        await self.stt.shutdown()
        await self.tts.shutdown()
