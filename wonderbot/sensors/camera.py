from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List

from .base import SensorObservation, SensorStatus
from .vision_lite import (
    VisionLiteAnalyzer,
    VisionLiteMetrics,
    brightness_phrase as _brightness_phrase,
    format_vision_lite_text,
    metrics_metadata,
)
from ..perception import ImageCaptioner


class CameraUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class CameraMetrics:
    motion: float
    brightness_delta: float
    brightness: float
    contrast: float


class OpenCVCameraAdapter:
    name = "camera"

    def __init__(
        self,
        index: int = 0,
        width: int = 320,
        height: int = 240,
        motion_threshold: float = 0.08,
        brightness_threshold: float = 0.05,
        min_salience: float = 0.12,
        captioner: ImageCaptioner | None = None,
        caption_interval_seconds: float = 3.0,
        caption_salience_threshold: float = 0.22,
        caption_min_chars: int = 12,
        vision_lite_enabled: bool = True,
        analysis_width: int = 320,
        motion_pixel_threshold: float = 18.0,
        scene_change_threshold: float = 0.22,
        backend_min_salience: float = 0.35,
        sharpness_reference: float = 120.0,
        face_hint_enabled: bool = False,
        face_hint_min_interval_seconds: float = 5.0,
        state_change_cooldown_seconds: float = 3.0,
    ) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise CameraUnavailableError(
                "opencv-python and numpy are not installed. Install with: pip install -e .[vision]"
            ) from exc
        self._cv2 = cv2
        self._np = np
        self.motion_threshold = motion_threshold
        self.brightness_threshold = brightness_threshold
        self.min_salience = min_salience
        self.captioner = captioner
        self.caption_interval_seconds = caption_interval_seconds
        self.caption_salience_threshold = caption_salience_threshold
        self.caption_min_chars = caption_min_chars
        self.vision_lite_enabled = bool(vision_lite_enabled)
        self._vision = VisionLiteAnalyzer(
            cv2,
            np,
            motion_threshold=motion_threshold,
            brightness_threshold=brightness_threshold,
            min_salience=min_salience,
            analysis_width=analysis_width,
            motion_pixel_threshold=motion_pixel_threshold,
            scene_change_threshold=scene_change_threshold,
            backend_min_salience=backend_min_salience,
            sharpness_reference=sharpness_reference,
            face_hint_enabled=face_hint_enabled,
            face_hint_min_interval_seconds=face_hint_min_interval_seconds,
            state_change_cooldown_seconds=state_change_cooldown_seconds,
        )
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise CameraUnavailableError(f"Could not open camera index {index}.")
        if width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._prev_gray = None
        self._prev_brightness = None
        self._last_text = ""
        self._last_caption = ""
        self._last_caption_at = 0.0

    def read_frame(self):
        ok, frame = self._cap.read()
        if not ok:
            raise CameraUnavailableError("Failed to read frame from camera.")
        return frame

    def poll(self) -> List[SensorObservation]:
        frame = self.read_frame()
        if self.vision_lite_enabled:
            return self._poll_vision_lite(frame)
        return self._poll_legacy(frame)

    def status(self) -> SensorStatus:
        detail = "camera adapter active"
        if self.vision_lite_enabled:
            detail += "; vision-lite metrics active"
            if self._vision.face_hint_enabled:
                if self._vision.face_hint_available:
                    detail += "; OpenCV face-ish hints enabled"
                else:
                    detail += "; OpenCV face-ish hints requested but cascade unavailable"
        if self.captioner is not None:
            detail += f"; captioning via {getattr(self.captioner, 'model_name', 'captioner')}"
        return SensorStatus(source=self.name, enabled=True, available=True, detail=detail)

    def close(self) -> None:
        self._cap.release()

    def _poll_vision_lite(self, frame) -> List[SensorObservation]:
        was_initialized = self._vision.initialized
        metrics = self._vision.analyze(frame)
        if not was_initialized:
            return []
        if not self._vision.should_report(metrics):
            return []

        text = format_vision_lite_text(metrics)
        metadata = metrics_metadata(metrics, remote_bridge=False)
        caption = self._maybe_caption(frame, metrics.salience)
        if caption:
            text = f"{text} Scene impression: {caption}."
            metadata["caption"] = caption
        if text == self._last_text and metrics.salience < 0.45 and not metrics.visual_state_changed:
            return []
        self._last_text = text
        return [
            SensorObservation(
                source=self.name,
                text=text,
                salience=round(metrics.salience, 6),
                metadata=metadata,
            )
        ]

    def _poll_legacy(self, frame) -> List[SensorObservation]:
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        metrics = self._analyze(gray)
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_brightness = metrics.brightness
            return []

        salience = min(
            1.0,
            max(
                metrics.motion * 4.2,
                metrics.brightness_delta * 3.6,
                max(0.0, metrics.contrast - 0.18) * 0.8,
            ),
        )
        self._prev_gray = gray
        self._prev_brightness = metrics.brightness
        if salience < self.min_salience:
            return []

        motion_phrase = "strong motion" if metrics.motion >= self.motion_threshold * 2.0 else "noticeable motion"
        if metrics.motion < self.motion_threshold:
            motion_phrase = "subtle motion"

        light_phrase = "lighting shift" if metrics.brightness_delta >= self.brightness_threshold else "stable lighting"
        brightness_phrase = _brightness_phrase(metrics.brightness)
        texture_phrase = "busy visual texture" if metrics.contrast >= 0.38 else "simple visual texture"
        text = f"camera sees {motion_phrase} with {light_phrase} in a {brightness_phrase} and {texture_phrase}."
        metadata = {
            "motion": round(metrics.motion, 6),
            "brightness_delta": round(metrics.brightness_delta, 6),
            "brightness": round(metrics.brightness, 6),
            "contrast": round(metrics.contrast, 6),
            "vision_lite": False,
        }
        caption = self._maybe_caption(frame, salience)
        if caption:
            text = f"{text} Scene impression: {caption}."
            metadata["caption"] = caption
        if text == self._last_text and salience < 0.45:
            return []
        self._last_text = text
        return [
            SensorObservation(
                source=self.name,
                text=text,
                salience=round(salience, 6),
                metadata=metadata,
            )
        ]

    def _analyze(self, gray) -> CameraMetrics:
        brightness = float(gray.mean()) / 255.0
        contrast = min(1.0, float(gray.std()) / 96.0)
        motion = 0.0
        brightness_delta = 0.0
        if self._prev_gray is not None:
            diff = self._cv2.absdiff(gray, self._prev_gray)
            motion = float(diff.mean()) / 255.0
            brightness_delta = abs(brightness - float(self._prev_brightness or 0.0))
        return CameraMetrics(
            motion=motion,
            brightness_delta=brightness_delta,
            brightness=brightness,
            contrast=contrast,
        )

    def _maybe_caption(self, frame, salience: float) -> str | None:
        if self.captioner is None or salience < self.caption_salience_threshold:
            return None
        now = time.monotonic()
        if self._last_caption and (now - self._last_caption_at) < self.caption_interval_seconds:
            return None
        try:
            result = self.captioner.caption(frame)
        except Exception:
            return None
        if result is None:
            return None
        caption = _clean_generated_text(result.text)
        if len(caption) < self.caption_min_chars:
            return None
        if _normalize_text(caption) == _normalize_text(self._last_caption):
            return None
        self._last_caption = caption
        self._last_caption_at = now
        return caption



def _clean_generated_text(text: str) -> str:
    return " ".join(str(text).strip().split()).strip(" .")



def _normalize_text(text: str) -> str:
    return _clean_generated_text(text).lower()
