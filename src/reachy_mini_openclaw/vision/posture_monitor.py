"""Local forward-head posture monitoring from the existing camera stream.

Frames are processed in memory only.  This module never writes frames to disk
and never sends them to OpenClaw or another network service.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Landmark:
    """A normalized pose landmark used by the posture heuristic."""

    x: float
    y: float
    visibility: float = 1.0


@dataclass(frozen=True)
class PostureAssessment:
    """Result of one posture estimate."""

    forward_head_ratio: float
    turtle_neck_suspected: bool


class PoseDetector(Protocol):
    """Minimal detector interface, making the monitor independent of MediaPipe."""

    def detect(self, frame: NDArray[np.uint8]) -> Mapping[str, Landmark] | None: ...

    def close(self) -> None: ...


def assess_forward_head(
    landmarks: Mapping[str, Landmark],
    threshold: float,
    min_visibility: float = 0.5,
) -> PostureAssessment | None:
    """Estimate forward-head posture from ears, shoulders, and hips.

    The horizontal ear-to-shoulder displacement is normalized by torso length,
    so the threshold is stable across camera distances.  It is most useful with
    the Reachy camera placed to the user's side or at a three-quarter angle.
    """

    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(name not in landmarks or landmarks[name].visibility < min_visibility for name in required):
        return None

    ears = [
        landmark
        for name in ("left_ear", "right_ear")
        if (landmark := landmarks.get(name)) is not None and landmark.visibility >= min_visibility
    ]
    if not ears:
        return None

    left_shoulder, right_shoulder = landmarks["left_shoulder"], landmarks["right_shoulder"]
    left_hip, right_hip = landmarks["left_hip"], landmarks["right_hip"]
    shoulder = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
    hip = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
    ear = (sum(point.x for point in ears) / len(ears), sum(point.y for point in ears) / len(ears))

    torso_length = math.dist(shoulder, hip)
    if torso_length < 0.03:
        return None

    ratio = abs(ear[0] - shoulder[0]) / torso_length
    return PostureAssessment(
        forward_head_ratio=ratio,
        turtle_neck_suspected=ratio >= threshold,
    )


class MediaPipePoseDetector:
    """MediaPipe Tasks Pose adapter; imported only when monitoring starts."""

    def __init__(self, model_path: str, min_detection_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise ImportError(
                "Posture monitoring requires MediaPipe. Install with: pip install -e '.[mediapipe_vision]'"
            ) from exc

        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"Posture model not found: {model_path}. Download pose_landmarker_lite.task and set POSTURE_MODEL_PATH."
            )
        self._mp = mp
        self._vision = vision
        self._last_timestamp_ms = 0
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._pose = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame: NDArray[np.uint8]) -> Mapping[str, Landmark] | None:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame[:, :, ::-1].copy())
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._pose.detect_for_video(image, timestamp_ms)
        if not result.pose_landmarks:
            return None

        points = result.pose_landmarks[0]
        ids = self._vision.PoseLandmark
        names = {
            "left_ear": ids.LEFT_EAR,
            "right_ear": ids.RIGHT_EAR,
            "left_shoulder": ids.LEFT_SHOULDER,
            "right_shoulder": ids.RIGHT_SHOULDER,
            "left_hip": ids.LEFT_HIP,
            "right_hip": ids.RIGHT_HIP,
        }
        landmarks = {}
        for name, index in names.items():
            point = points[index]
            visibility = getattr(point, "visibility", None)
            landmarks[name] = Landmark(point.x, point.y, 1.0 if visibility is None else visibility)
        return landmarks

    def close(self) -> None:
        self._pose.close()


class PostureMonitor:
    """Poll the shared camera buffer and raise a bounded local posture alert."""

    def __init__(
        self,
        camera_worker: object,
        on_turtle_neck: Callable[[], None],
        threshold: float = 0.20,
        sustain_seconds: float = 4.0,
        cooldown_seconds: float = 30.0,
        check_interval: float = 0.5,
        model_path: str = "models/mediapipe/pose_landmarker_lite.task",
        detector_factory: Callable[[], PoseDetector] | None = None,
    ) -> None:
        self.camera_worker = camera_worker
        self.on_turtle_neck = on_turtle_neck
        self.threshold = threshold
        self.sustain_seconds = sustain_seconds
        self.cooldown_seconds = cooldown_seconds
        self.check_interval = check_interval
        self.detector_factory = detector_factory or (lambda: MediaPipePoseDetector(model_path))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start monitoring, returning False if the optional detector is unavailable."""
        if self._thread is not None and self._thread.is_alive():
            return True
        try:
            self._detector = self.detector_factory()
        except (ImportError, FileNotFoundError) as exc:
            logger.warning("Posture monitoring is disabled: %s", exc)
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="posture-monitor", daemon=True)
        self._thread.start()
        logger.info("Local posture monitor started (no frames are stored or transmitted)")
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        detector = getattr(self, "_detector", None)
        if detector is not None:
            detector.close()
            self._detector = None
        logger.info("Local posture monitor stopped")

    def _run(self) -> None:
        bad_since: float | None = None
        last_alert = float("-inf")
        while not self._stop_event.is_set():
            try:
                frame = self.camera_worker.get_latest_frame()
                landmarks = self._detector.detect(frame) if frame is not None else None
                assessment = (
                    assess_forward_head(landmarks, self.threshold) if landmarks is not None else None
                )
                now = time.monotonic()
                if assessment is not None and assessment.turtle_neck_suspected:
                    bad_since = bad_since or now
                    if now - bad_since >= self.sustain_seconds and now - last_alert >= self.cooldown_seconds:
                        logger.info("Sustained forward-head posture detected (ratio=%.2f)", assessment.forward_head_ratio)
                        self.on_turtle_neck()
                        last_alert = now
                else:
                    bad_since = None
            except Exception as exc:
                logger.warning("Posture monitor frame skipped: %s", exc)
            self._stop_event.wait(self.check_interval)
