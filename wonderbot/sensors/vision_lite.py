from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class VisionLiteMetrics:
    brightness: float
    brightness_delta: float
    contrast: float
    contrast_delta: float
    sharpness: float
    sharpness_raw: float
    blur: float
    edge_density: float
    edge_delta: float
    texture: float
    motion_mean: float
    motion_ratio: float
    motion_magnitude: float
    scene_change_score: float
    salience: float
    backend_hint: bool
    visual_state: str
    visual_state_changed: bool
    face_hint_enabled: bool
    face_hint_available: bool
    faceish_count: int
    remote_timestamp_ms: int = 0


class VisionLiteAnalyzer:
    """Small OpenCV-only visual-change analyzer for WonderBot live-lite.

    This deliberately avoids captioning, object recognition, Torch, BLIP, and any
    heavyweight model. It only reports low-level visual signals: motion, light,
    sharpness, texture, broad scene change, and optional OpenCV Haar face-ish
    hints when enabled and available.
    """

    def __init__(
        self,
        cv2: Any,
        np: Any,
        *,
        motion_threshold: float = 0.08,
        brightness_threshold: float = 0.05,
        min_salience: float = 0.12,
        analysis_width: int = 320,
        motion_pixel_threshold: float = 18.0,
        scene_change_threshold: float = 0.22,
        backend_min_salience: float = 0.35,
        sharpness_reference: float = 120.0,
        face_hint_enabled: bool = False,
        face_hint_min_interval_seconds: float = 5.0,
        state_change_cooldown_seconds: float = 3.0,
    ) -> None:
        self.cv2 = cv2
        self.np = np
        self.motion_threshold = max(0.0, float(motion_threshold))
        self.brightness_threshold = max(0.0, float(brightness_threshold))
        self.min_salience = max(0.0, float(min_salience))
        self.analysis_width = max(80, int(analysis_width))
        self.motion_pixel_threshold = max(1.0, float(motion_pixel_threshold))
        self.scene_change_threshold = max(0.0, float(scene_change_threshold))
        self.backend_min_salience = max(self.min_salience, float(backend_min_salience))
        self.sharpness_reference = max(1.0, float(sharpness_reference))
        self.face_hint_enabled = bool(face_hint_enabled)
        self.face_hint_min_interval_seconds = max(0.5, float(face_hint_min_interval_seconds))
        self.state_change_cooldown_seconds = max(0.0, float(state_change_cooldown_seconds))

        self._prev_gray = None
        self._prev_brightness: float | None = None
        self._prev_contrast: float | None = None
        self._prev_edge_density: float | None = None
        self._last_visual_state = ""
        self._last_state_change_at = 0.0
        self._last_face_check_at = 0.0
        self._last_faceish_count = 0
        self._face_cascade = self._load_face_cascade()

    @property
    def initialized(self) -> bool:
        return self._prev_gray is not None

    @property
    def face_hint_available(self) -> bool:
        return self._face_cascade is not None

    def analyze(self, frame: Any, *, remote_timestamp_ms: int = 0) -> VisionLiteMetrics:
        gray = self._to_gray(frame)
        gray = self._resize_for_analysis(gray)

        brightness = float(gray.mean()) / 255.0
        contrast = clamp01(float(gray.std()) / 96.0)
        sharpness_raw = float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var())
        sharpness = clamp01(sharpness_raw / self.sharpness_reference)
        blur = clamp01(1.0 - sharpness)

        edges = self.cv2.Canny(gray, 40, 120)
        edge_density = float((edges > 0).mean()) if edges.size else 0.0
        texture = clamp01((contrast * 0.58) + (edge_density * 3.5))

        motion_mean = 0.0
        motion_ratio = 0.0
        motion_magnitude = 0.0
        brightness_delta = 0.0
        contrast_delta = 0.0
        edge_delta = 0.0

        if self._prev_gray is not None:
            diff = self.cv2.absdiff(gray, self._prev_gray)
            motion_mean = float(diff.mean()) / 255.0
            motion_ratio = float((diff >= self.motion_pixel_threshold).mean()) if diff.size else 0.0
            motion_magnitude = clamp01(max(motion_mean * 4.2, motion_ratio * 2.4))
            brightness_delta = abs(brightness - float(self._prev_brightness or 0.0))
            contrast_delta = abs(contrast - float(self._prev_contrast or 0.0))
            edge_delta = abs(edge_density - float(self._prev_edge_density or 0.0))

        scene_change_score = clamp01(
            max(
                motion_mean * 5.2,
                motion_ratio * 2.8,
                brightness_delta * 3.6,
                contrast_delta * 1.4,
                edge_delta * 2.2,
            )
        )

        faceish_count = self._detect_faceish(gray)
        face_salience = 0.30 if faceish_count > 0 else 0.0

        salience = clamp01(
            max(
                scene_change_score,
                motion_magnitude,
                brightness_delta * 3.6,
                max(0.0, texture - 0.55) * 0.65,
                face_salience,
            )
        )

        visual_state = self._visual_state_signature(
            brightness=brightness,
            sharpness=sharpness,
            texture=texture,
            motion_magnitude=motion_magnitude,
            scene_change_score=scene_change_score,
            faceish_count=faceish_count,
        )
        visual_state_changed = self._state_changed(visual_state)

        backend_hint = bool(
            salience >= self.backend_min_salience
            or scene_change_score >= self.scene_change_threshold
            or (visual_state_changed and salience >= max(self.min_salience, self.backend_min_salience * 0.72))
            or (faceish_count > 0 and visual_state_changed)
        )

        self._prev_gray = gray
        self._prev_brightness = brightness
        self._prev_contrast = contrast
        self._prev_edge_density = edge_density

        return VisionLiteMetrics(
            brightness=brightness,
            brightness_delta=brightness_delta,
            contrast=contrast,
            contrast_delta=contrast_delta,
            sharpness=sharpness,
            sharpness_raw=sharpness_raw,
            blur=blur,
            edge_density=edge_density,
            edge_delta=edge_delta,
            texture=texture,
            motion_mean=motion_mean,
            motion_ratio=motion_ratio,
            motion_magnitude=motion_magnitude,
            scene_change_score=scene_change_score,
            salience=salience,
            backend_hint=backend_hint,
            visual_state=visual_state,
            visual_state_changed=visual_state_changed,
            face_hint_enabled=self.face_hint_enabled,
            face_hint_available=self.face_hint_available,
            faceish_count=faceish_count,
            remote_timestamp_ms=int(remote_timestamp_ms or 0),
        )

    def should_report(self, metrics: VisionLiteMetrics) -> bool:
        return bool(metrics.salience >= self.min_salience or metrics.visual_state_changed)

    def _to_gray(self, frame: Any) -> Any:
        if getattr(frame, "ndim", 0) == 2:
            return frame
        return self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)

    def _resize_for_analysis(self, gray: Any) -> Any:
        height, width = gray.shape[:2]
        if width <= self.analysis_width:
            return gray
        scale = self.analysis_width / float(width)
        target = (self.analysis_width, max(1, int(height * scale)))
        return self.cv2.resize(gray, target, interpolation=self.cv2.INTER_AREA)

    def _load_face_cascade(self):
        if not self.face_hint_enabled:
            return None
        try:
            cascade_path = getattr(self.cv2.data, "haarcascades", "") + "haarcascade_frontalface_default.xml"
            cascade = self.cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                return None
            return cascade
        except Exception:
            return None

    def _detect_faceish(self, gray: Any) -> int:
        if not self.face_hint_enabled or self._face_cascade is None:
            return 0
        now = time.monotonic()
        if (now - self._last_face_check_at) < self.face_hint_min_interval_seconds:
            return int(self._last_faceish_count)
        self._last_face_check_at = now
        try:
            found = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(32, 32),
            )
            count = int(len(found))
        except Exception:
            count = 0
        self._last_faceish_count = count
        return count

    def _visual_state_signature(
        self,
        *,
        brightness: float,
        sharpness: float,
        texture: float,
        motion_magnitude: float,
        scene_change_score: float,
        faceish_count: int,
    ) -> str:
        return ":".join(
            [
                _bucket(brightness, (0.22, 0.42, 0.64, 0.80)),
                _bucket(sharpness, (0.18, 0.36, 0.62, 0.82)),
                _bucket(texture, (0.18, 0.36, 0.58, 0.78)),
                _bucket(motion_magnitude, (0.05, 0.14, 0.28, 0.50)),
                _bucket(scene_change_score, (0.08, 0.18, 0.32, 0.55)),
                "faceish" if faceish_count > 0 else "no-faceish",
            ]
        )

    def _state_changed(self, visual_state: str) -> bool:
        now = time.monotonic()
        if not self._last_visual_state:
            self._last_visual_state = visual_state
            self._last_state_change_at = now
            return False
        if visual_state == self._last_visual_state:
            return False
        if (now - self._last_state_change_at) < self.state_change_cooldown_seconds:
            return False
        self._last_visual_state = visual_state
        self._last_state_change_at = now
        return True


def metrics_metadata(metrics: VisionLiteMetrics, *, remote_bridge: bool = False) -> dict[str, object]:
    data: dict[str, object] = {
        "vision_lite": True,
        "brightness": round(metrics.brightness, 6),
        "brightness_delta": round(metrics.brightness_delta, 6),
        "contrast": round(metrics.contrast, 6),
        "contrast_delta": round(metrics.contrast_delta, 6),
        "sharpness": round(metrics.sharpness, 6),
        "sharpness_raw": round(metrics.sharpness_raw, 3),
        "blur": round(metrics.blur, 6),
        "edge_density": round(metrics.edge_density, 6),
        "edge_delta": round(metrics.edge_delta, 6),
        "texture": round(metrics.texture, 6),
        "motion": round(metrics.motion_mean, 6),
        "motion_mean": round(metrics.motion_mean, 6),
        "motion_ratio": round(metrics.motion_ratio, 6),
        "motion_magnitude": round(metrics.motion_magnitude, 6),
        "scene_change_score": round(metrics.scene_change_score, 6),
        "backend_hint": bool(metrics.backend_hint),
        "visual_state": metrics.visual_state,
        "visual_state_changed": bool(metrics.visual_state_changed),
        "face_hint_enabled": bool(metrics.face_hint_enabled),
        "face_hint_available": bool(metrics.face_hint_available),
        "faceish_count": int(metrics.faceish_count),
        "memory_eligible": bool(metrics.backend_hint),
    }
    if remote_bridge:
        data["remote_bridge"] = True
        data["remote_timestamp_ms"] = int(metrics.remote_timestamp_ms or 0)
    return data


def format_vision_lite_text(metrics: VisionLiteMetrics) -> str:
    parts = [
        f"camera sees {motion_phrase(metrics)}",
        scene_change_phrase(metrics),
        light_phrase(metrics),
        brightness_phrase(metrics.brightness),
        f"{sharpness_phrase(metrics.sharpness)} focus",
        f"{texture_phrase(metrics.texture, metrics.edge_density)} texture",
    ]
    text = "; ".join(part for part in parts if part).strip()
    if metrics.visual_state_changed:
        text += "; visual state changed"
    if metrics.face_hint_enabled:
        if metrics.face_hint_available and metrics.faceish_count > 0:
            text += f"; OpenCV face-ish hint: {metrics.faceish_count} face-like region(s)"
        elif not metrics.face_hint_available:
            text += "; OpenCV face-ish hint unavailable"
    return text + "."


def brightness_phrase(value: float) -> str:
    if value < 0.18:
        return "very dark scene"
    if value < 0.34:
        return "dark scene"
    if value < 0.46:
        return "dim scene"
    if value > 0.86:
        return "very bright scene"
    if value > 0.70:
        return "bright scene"
    return "mid-lit scene"


def light_phrase(metrics: VisionLiteMetrics) -> str:
    delta = metrics.brightness_delta
    if delta >= 0.18:
        return "major lighting shift"
    if delta >= 0.08:
        return "clear lighting shift"
    if delta >= 0.035:
        return "small lighting shift"
    return "stable lighting"


def motion_phrase(metrics: VisionLiteMetrics) -> str:
    value = max(metrics.motion_magnitude, metrics.motion_ratio * 2.0, metrics.motion_mean * 4.0)
    if value >= 0.62:
        return "strong motion"
    if value >= 0.34:
        return "clear motion"
    if value >= 0.16:
        return "noticeable motion"
    if value >= 0.055:
        return "subtle motion"
    return "almost no motion"


def scene_change_phrase(metrics: VisionLiteMetrics) -> str:
    score = metrics.scene_change_score
    if score >= 0.62:
        return "major scene change"
    if score >= 0.34:
        return "clear scene change"
    if score >= 0.16:
        return "small scene change"
    return "stable scene"


def sharpness_phrase(value: float) -> str:
    if value < 0.16:
        return "blurry"
    if value < 0.34:
        return "soft"
    if value > 0.78:
        return "very sharp"
    if value > 0.55:
        return "sharp"
    return "clear enough"


def texture_phrase(texture: float, edge_density: float) -> str:
    value = max(texture, edge_density * 3.5)
    if value >= 0.78:
        return "dense"
    if value >= 0.55:
        return "busy"
    if value >= 0.28:
        return "moderate"
    return "simple"


def _bucket(value: float, thresholds: tuple[float, ...]) -> str:
    for index, threshold in enumerate(thresholds):
        if value < threshold:
            return str(index)
    return str(len(thresholds))
