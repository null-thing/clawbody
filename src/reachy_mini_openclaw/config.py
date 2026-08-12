"""Configuration management for Reachy Mini OpenClaw.

Handles environment variables and configuration settings for the application.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
_project_root = Path(__file__).parent.parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # Voice backend selection
    VOICE_BACKEND: str = field(default_factory=lambda: os.getenv("VOICE_BACKEND", "local").lower())
    STT_BACKEND: str = field(default_factory=lambda: os.getenv("STT_BACKEND", "faster-whisper").lower())
    TTS_BACKEND: str = field(default_factory=lambda: os.getenv("TTS_BACKEND", "piper").lower())

    # OpenAI Configuration (only required by VOICE_BACKEND=openai)
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    OPENAI_MODEL: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-realtime-1.5"))
    OPENAI_VOICE: str = field(default_factory=lambda: os.getenv("OPENAI_VOICE", "cedar"))

    # Local voice configuration
    STT_MODEL: str = field(default_factory=lambda: os.getenv("STT_MODEL", "base.en"))
    STT_DEVICE: str = field(default_factory=lambda: os.getenv("STT_DEVICE", "cpu"))
    STT_COMPUTE_TYPE: str = field(default_factory=lambda: os.getenv("STT_COMPUTE_TYPE", "int8"))
    STT_LANGUAGE: str | None = field(default_factory=lambda: os.getenv("STT_LANGUAGE") or None)
    PIPER_EXECUTABLE: str = field(default_factory=lambda: os.getenv("PIPER_EXECUTABLE", "piper"))
    PIPER_MODEL: str = field(default_factory=lambda: os.getenv("PIPER_MODEL", "en_US-lessac-medium.onnx"))
    PIPER_SAMPLE_RATE: int = field(default_factory=lambda: int(os.getenv("PIPER_SAMPLE_RATE", "22050")))
    VAD_THRESHOLD: float = field(default_factory=lambda: float(os.getenv("VAD_THRESHOLD", "0.015")))
    VAD_SILENCE_MS: int = field(default_factory=lambda: int(os.getenv("VAD_SILENCE_MS", "700")))
    VAD_MIN_SPEECH_MS: int = field(default_factory=lambda: int(os.getenv("VAD_MIN_SPEECH_MS", "250")))

    # OpenClaw Gateway Configuration
    OPENCLAW_GATEWAY_URL: str = field(default_factory=lambda: os.getenv("OPENCLAW_GATEWAY_URL", "ws://localhost:18789"))
    OPENCLAW_TOKEN: str | None = field(default_factory=lambda: os.getenv("OPENCLAW_TOKEN"))
    OPENCLAW_AGENT_ID: str = field(default_factory=lambda: os.getenv("OPENCLAW_AGENT_ID", "main"))
    # Session key for OpenClaw - uses "main" to share context with WhatsApp and other channels
    # Format: agent:<agent_id>:<session_key>, but we only need the session key part here
    OPENCLAW_SESSION_KEY: str = field(default_factory=lambda: os.getenv("OPENCLAW_SESSION_KEY", "main"))

    # Robot Configuration
    ROBOT_NAME: str | None = field(default_factory=lambda: os.getenv("ROBOT_NAME"))

    # Feature Flags
    ENABLE_OPENCLAW_TOOLS: bool = field(default_factory=lambda: os.getenv("ENABLE_OPENCLAW_TOOLS", "true").lower() == "true")
    ENABLE_CAMERA: bool = field(default_factory=lambda: os.getenv("ENABLE_CAMERA", "true").lower() == "true")
    ENABLE_FACE_TRACKING: bool = field(default_factory=lambda: os.getenv("ENABLE_FACE_TRACKING", "true").lower() == "true")

    # Face Tracking Configuration
    # Options: "yolo", "mediapipe", or None for auto-detect
    HEAD_TRACKER_TYPE: str | None = field(default_factory=lambda: os.getenv("HEAD_TRACKER_TYPE", "yolo"))

    # Local Vision Processing
    ENABLE_LOCAL_VISION: bool = field(default_factory=lambda: os.getenv("ENABLE_LOCAL_VISION", "false").lower() == "true")
    LOCAL_VISION_MODEL: str = field(default_factory=lambda: os.getenv("LOCAL_VISION_MODEL", "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"))
    VISION_DEVICE: str = field(default_factory=lambda: os.getenv("VISION_DEVICE", "auto"))  # "auto", "cuda", "mps", "cpu"
    HF_HOME: str = field(default_factory=lambda: os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

    # Custom Profile (for personality customization)
    CUSTOM_PROFILE: str | None = field(default_factory=lambda: os.getenv("REACHY_MINI_CUSTOM_PROFILE"))

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if self.VOICE_BACKEND not in {"local", "openai"}:
            errors.append("VOICE_BACKEND must be 'local' or 'openai'")
        if self.VOICE_BACKEND == "openai" and not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required when VOICE_BACKEND=openai")
        if self.VOICE_BACKEND == "local":
            if self.STT_BACKEND != "faster-whisper":
                errors.append("STT_BACKEND must be 'faster-whisper'")
            if self.TTS_BACKEND != "piper":
                errors.append("TTS_BACKEND must be 'piper'")
        return errors


# Global configuration instance
config = Config()


def set_custom_profile(profile: str | None) -> None:
    """Update the custom profile at runtime."""
    global config
    config.CUSTOM_PROFILE = profile
    os.environ["REACHY_MINI_CUSTOM_PROFILE"] = profile or ""


def set_face_tracking_enabled(enabled: bool) -> None:
    """Enable or disable face tracking at runtime."""
    global config
    config.ENABLE_FACE_TRACKING = enabled


def set_local_vision_enabled(enabled: bool) -> None:
    """Enable or disable local vision processing at runtime."""
    global config
    config.ENABLE_LOCAL_VISION = enabled
