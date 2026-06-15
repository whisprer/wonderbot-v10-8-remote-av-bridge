from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class FaceLiteRegion:
    """Normalized description of one OpenCV face-like region.

    This is intentionally phrased as face-like rather than identity/person truth.
    Haar cascades are useful hints, not reliable semantic perception.
    """

    x: float
    y: float
    width: float
    height: float
    center_x: float
    center_y: float
    area_ratio: float

    def as_metadata(self) -> dict[str, float]:
        return {
            "x": round(self.x, 6),
            "y": round(self.y, 6),
            "width": round(self.width, 6),
            "height": round(self.height, 6),
            "center_x": round(self.center_x, 6),
            "center_y": round(self.center_y, 6),
            "area_ratio": round(self.area_ratio, 6),
        }


@dataclass(slots=True)
class FaceLiteDetection:
    enabled: bool
    available: bool
    count: int
    confidence: float
    stable: bool
    appeared: bool
    lost: bool
    status: str
    area_ratio: float
    center_x: float
    center_y: float
    regions: tuple[FaceLiteRegion, ...]
    checked_now: bool


@dataclass(slots=True)
class ExpressionLiteDetection:
    enabled: bool
    available: bool
    attempted: bool
    hint: str
    confidence: float
    changed: bool
    smileish: bool
    neutralish: bool
    unclear: bool
    status: str
    evidence: tuple[str, ...]
    smile_count: int
    smile_area_ratio: float
    checked_now: bool


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
    faceish_confidence: float
    faceish_stable: bool
    faceish_appeared: bool
    faceish_lost: bool
    faceish_status: str
    faceish_area_ratio: float
    faceish_center_x: float
    faceish_center_y: float
    faceish_regions: tuple[FaceLiteRegion, ...]
    faceish_checked_now: bool
    expression_hint_enabled: bool
    expression_hint_available: bool
    expression_hint: str
    expression_confidence: float
    expression_changed: bool
    expression_smileish: bool
    expression_neutralish: bool
    expression_unclear: bool
    expression_status: str
    expression_evidence: tuple[str, ...]
    expression_smile_count: int
    expression_smile_area_ratio: float
    expression_checked_now: bool
    remote_timestamp_ms: int = 0


class VisionLiteAnalyzer:
    """Small OpenCV-only visual-change analyzer for WonderBot live-lite.

    This deliberately avoids captioning, object recognition, Torch, BLIP, and any
    heavyweight model. It only reports low-level visual signals: motion, light,
    sharpness, texture, broad scene change, and optional OpenCV Haar Face-Lite
    presence and Expression-Lite hints when enabled and available.

    Face-Lite is intentionally conservative: it reports possible face-like
    regions, not identity or person truth. Expression-Lite is even more
    conservative: it reports weak smile-ish / neutral-ish / unclear cues, not
    emotional truth.
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
        expression_hint_enabled: bool = False,
        expression_hint_min_interval_seconds: float = 5.0,
        expression_hint_min_face_confidence: float = 0.38,
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
        self.expression_hint_enabled = bool(expression_hint_enabled and self.face_hint_enabled)
        self.expression_hint_min_interval_seconds = max(0.5, float(expression_hint_min_interval_seconds))
        self.expression_hint_min_face_confidence = max(0.0, float(expression_hint_min_face_confidence))
        self.state_change_cooldown_seconds = max(0.0, float(state_change_cooldown_seconds))

        self._prev_gray = None
        self._prev_brightness: float | None = None
        self._prev_contrast: float | None = None
        self._prev_edge_density: float | None = None
        self._last_visual_state = ""
        self._last_state_change_at = 0.0
        self._last_face_check_at = 0.0
        self._last_faceish_count = 0
        self._last_face_detection = FaceLiteDetection(
            enabled=self.face_hint_enabled,
            available=False,
            count=0,
            confidence=0.0,
            stable=False,
            appeared=False,
            lost=False,
            status="disabled" if not self.face_hint_enabled else "unavailable",
            area_ratio=0.0,
            center_x=0.0,
            center_y=0.0,
            regions=(),
            checked_now=False,
        )
        self._last_face_present = False
        self._last_face_center: tuple[float, float] | None = None
        self._last_face_area_ratio = 0.0
        self._last_expression_check_at = 0.0
        self._last_expression_hint = ""
        self._last_expression_confidence = 0.0
        self._last_expression_detection = ExpressionLiteDetection(
            enabled=self.expression_hint_enabled,
            available=False,
            attempted=False,
            hint="",
            confidence=0.0,
            changed=False,
            smileish=False,
            neutralish=False,
            unclear=False,
            status="disabled" if not self.expression_hint_enabled else "unavailable",
            evidence=(),
            smile_count=0,
            smile_area_ratio=0.0,
            checked_now=False,
        )
        self._face_cascade = self._load_face_cascade()
        self._smile_cascade = self._load_smile_cascade()

    @property
    def initialized(self) -> bool:
        return self._prev_gray is not None

    @property
    def face_hint_available(self) -> bool:
        return self._face_cascade is not None

    @property
    def expression_hint_available(self) -> bool:
        return bool(self.expression_hint_enabled and self._smile_cascade is not None)

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

        face = self._detect_faceish(gray)
        expression = self._detect_expressionish(
            gray,
            face=face,
            brightness=brightness,
            sharpness=sharpness,
        )
        face_salience = self._face_salience(face)
        expression_salience = self._expression_salience(expression)

        salience = clamp01(
            max(
                scene_change_score,
                motion_magnitude,
                brightness_delta * 3.6,
                max(0.0, texture - 0.55) * 0.65,
                face_salience,
                expression_salience,
            )
        )

        visual_state = self._visual_state_signature(
            brightness=brightness,
            sharpness=sharpness,
            texture=texture,
            motion_magnitude=motion_magnitude,
            scene_change_score=scene_change_score,
            faceish_count=face.count,
            faceish_status=face.status,
            faceish_center_x=face.center_x,
            faceish_center_y=face.center_y,
            expression_hint=expression.hint,
            expression_status=expression.status,
        )
        visual_state_changed = self._state_changed(visual_state)

        face_backend_event = bool(
            self.face_hint_enabled
            and face.available
            and face.confidence >= 0.36
            and (face.appeared or face.lost or (face.stable and visual_state_changed))
        )
        expression_backend_event = bool(
            self.expression_hint_enabled
            and expression.available
            and expression.changed
            and expression.confidence >= 0.45
            and expression.hint in {"smile-ish", "neutral-ish", "unclear"}
        )

        backend_hint = bool(
            salience >= self.backend_min_salience
            or scene_change_score >= self.scene_change_threshold
            or (visual_state_changed and salience >= max(self.min_salience, self.backend_min_salience * 0.72))
            or face_backend_event
            or expression_backend_event
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
            faceish_count=face.count,
            faceish_confidence=face.confidence,
            faceish_stable=face.stable,
            faceish_appeared=face.appeared,
            faceish_lost=face.lost,
            faceish_status=face.status,
            faceish_area_ratio=face.area_ratio,
            faceish_center_x=face.center_x,
            faceish_center_y=face.center_y,
            faceish_regions=face.regions,
            faceish_checked_now=face.checked_now,
            expression_hint_enabled=self.expression_hint_enabled,
            expression_hint_available=self.expression_hint_available,
            expression_hint=expression.hint,
            expression_confidence=expression.confidence,
            expression_changed=expression.changed,
            expression_smileish=expression.smileish,
            expression_neutralish=expression.neutralish,
            expression_unclear=expression.unclear,
            expression_status=expression.status,
            expression_evidence=expression.evidence,
            expression_smile_count=expression.smile_count,
            expression_smile_area_ratio=expression.smile_area_ratio,
            expression_checked_now=expression.checked_now,
            remote_timestamp_ms=int(remote_timestamp_ms or 0),
        )

    def should_report(self, metrics: VisionLiteMetrics) -> bool:
        face_event = bool(metrics.faceish_appeared or metrics.faceish_lost)
        expression_event = bool(metrics.expression_changed and metrics.expression_confidence >= 0.34)
        return bool(metrics.salience >= self.min_salience or metrics.visual_state_changed or face_event or expression_event)

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

    def _load_smile_cascade(self):
        if not self.expression_hint_enabled:
            return None
        try:
            cascade_path = getattr(self.cv2.data, "haarcascades", "") + "haarcascade_smile.xml"
            cascade = self.cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                return None
            return cascade
        except Exception:
            return None

    def _detect_faceish(self, gray: Any) -> FaceLiteDetection:
        if not self.face_hint_enabled:
            detection = FaceLiteDetection(
                enabled=False,
                available=False,
                count=0,
                confidence=0.0,
                stable=False,
                appeared=False,
                lost=False,
                status="disabled",
                area_ratio=0.0,
                center_x=0.0,
                center_y=0.0,
                regions=(),
                checked_now=False,
            )
            self._last_face_detection = detection
            return detection

        if self._face_cascade is None:
            detection = FaceLiteDetection(
                enabled=True,
                available=False,
                count=0,
                confidence=0.0,
                stable=False,
                appeared=False,
                lost=False,
                status="unavailable",
                area_ratio=0.0,
                center_x=0.0,
                center_y=0.0,
                regions=(),
                checked_now=False,
            )
            self._last_face_detection = detection
            return detection

        now = time.monotonic()
        if (now - self._last_face_check_at) < self.face_hint_min_interval_seconds:
            cached = self._last_face_detection
            return FaceLiteDetection(
                enabled=cached.enabled,
                available=cached.available,
                count=cached.count,
                confidence=cached.confidence,
                stable=cached.stable,
                appeared=False,
                lost=False,
                status=cached.status,
                area_ratio=cached.area_ratio,
                center_x=cached.center_x,
                center_y=cached.center_y,
                regions=cached.regions,
                checked_now=False,
            )

        self._last_face_check_at = now
        regions = self._run_face_cascade(gray)
        count = len(regions)
        present = count > 0
        previous_present = bool(self._last_face_present)
        appeared = bool(present and not previous_present)
        lost = bool((not present) and previous_present)

        primary = regions[0] if regions else None
        center_x = float(primary.center_x) if primary else 0.0
        center_y = float(primary.center_y) if primary else 0.0
        area_ratio = float(primary.area_ratio) if primary else 0.0
        stable = False

        if present and previous_present and self._last_face_center is not None:
            dx = center_x - self._last_face_center[0]
            dy = center_y - self._last_face_center[1]
            center_distance = float((dx * dx + dy * dy) ** 0.5)
            area_delta = abs(area_ratio - self._last_face_area_ratio)
            stable = bool(center_distance <= 0.12 and area_delta <= 0.10)

        confidence = self._face_confidence(count=count, area_ratio=area_ratio, stable=stable, appeared=appeared)
        status = self._face_status(
            present=present,
            confidence=confidence,
            stable=stable,
            appeared=appeared,
            lost=lost,
        )

        if present:
            self._last_face_center = (center_x, center_y)
            self._last_face_area_ratio = area_ratio
        else:
            self._last_face_center = None
            self._last_face_area_ratio = 0.0

        self._last_face_present = present
        self._last_faceish_count = count

        detection = FaceLiteDetection(
            enabled=True,
            available=True,
            count=count,
            confidence=confidence,
            stable=stable,
            appeared=appeared,
            lost=lost,
            status=status,
            area_ratio=area_ratio,
            center_x=center_x,
            center_y=center_y,
            regions=tuple(regions),
            checked_now=True,
        )
        self._last_face_detection = detection
        return detection

    def _run_face_cascade(self, gray: Any) -> list[FaceLiteRegion]:
        if self._face_cascade is None:
            return []
        try:
            equalized = self.cv2.equalizeHist(gray)
            found = self._face_cascade.detectMultiScale(
                equalized,
                scaleFactor=1.08,
                minNeighbors=5,
                minSize=(32, 32),
                flags=getattr(self.cv2, "CASCADE_SCALE_IMAGE", 0),
            )
        except Exception:
            return []

        height, width = gray.shape[:2]
        frame_area = max(1.0, float(width * height))
        regions: list[FaceLiteRegion] = []

        for raw in found[:8]:
            x, y, w, h = [float(v) for v in raw]
            if w <= 0.0 or h <= 0.0:
                continue
            area_ratio = clamp01((w * h) / frame_area)
            # Filter tiny and implausibly huge rectangles. These are hints, not truth.
            if area_ratio < 0.008 or area_ratio > 0.72:
                continue
            cx = clamp01((x + (w / 2.0)) / max(1.0, float(width)))
            cy = clamp01((y + (h / 2.0)) / max(1.0, float(height)))
            regions.append(
                FaceLiteRegion(
                    x=clamp01(x / max(1.0, float(width))),
                    y=clamp01(y / max(1.0, float(height))),
                    width=clamp01(w / max(1.0, float(width))),
                    height=clamp01(h / max(1.0, float(height))),
                    center_x=cx,
                    center_y=cy,
                    area_ratio=area_ratio,
                )
            )

        regions.sort(key=lambda region: region.area_ratio, reverse=True)
        return regions[:4]

    def _detect_expressionish(
        self,
        gray: Any,
        *,
        face: FaceLiteDetection,
        brightness: float,
        sharpness: float,
    ) -> ExpressionLiteDetection:
        if not self.expression_hint_enabled:
            detection = ExpressionLiteDetection(
                enabled=False,
                available=False,
                attempted=False,
                hint="",
                confidence=0.0,
                changed=False,
                smileish=False,
                neutralish=False,
                unclear=False,
                status="disabled",
                evidence=(),
                smile_count=0,
                smile_area_ratio=0.0,
                checked_now=False,
            )
            self._last_expression_detection = detection
            return detection

        if self._smile_cascade is None:
            detection = ExpressionLiteDetection(
                enabled=True,
                available=False,
                attempted=False,
                hint="",
                confidence=0.0,
                changed=False,
                smileish=False,
                neutralish=False,
                unclear=False,
                status="unavailable",
                evidence=("OpenCV smile cascade unavailable",),
                smile_count=0,
                smile_area_ratio=0.0,
                checked_now=False,
            )
            self._last_expression_detection = detection
            return detection

        now = time.monotonic()
        if (now - self._last_expression_check_at) < self.expression_hint_min_interval_seconds:
            cached = self._last_expression_detection
            return ExpressionLiteDetection(
                enabled=cached.enabled,
                available=cached.available,
                attempted=False,
                hint=cached.hint,
                confidence=cached.confidence,
                changed=False,
                smileish=cached.smileish,
                neutralish=cached.neutralish,
                unclear=cached.unclear,
                status=cached.status,
                evidence=cached.evidence,
                smile_count=cached.smile_count,
                smile_area_ratio=cached.smile_area_ratio,
                checked_now=False,
            )

        self._last_expression_check_at = now
        evidence: list[str] = []

        if not face.available:
            evidence.append("face detector unavailable")
            return self._store_expression_detection(
                hint="",
                confidence=0.0,
                status="face-unavailable",
                evidence=tuple(evidence),
                smile_count=0,
                smile_area_ratio=0.0,
                checked_now=True,
            )

        if face.count <= 0 or not face.regions:
            evidence.append("no face-like region available")
            return self._store_expression_detection(
                hint="",
                confidence=0.0,
                status="no-face",
                evidence=tuple(evidence),
                smile_count=0,
                smile_area_ratio=0.0,
                checked_now=True,
            )

        if face.confidence < self.expression_hint_min_face_confidence:
            evidence.append(f"face confidence below expression threshold ({face.confidence:.2f})")
            return self._store_expression_detection(
                hint="unclear",
                confidence=max(0.18, face.confidence * 0.45),
                status="face-too-weak",
                evidence=tuple(evidence),
                smile_count=0,
                smile_area_ratio=0.0,
                checked_now=True,
            )

        if brightness < 0.16:
            evidence.append("scene too dark for reliable expression hint")
        if sharpness < 0.18:
            evidence.append("face region too soft for reliable expression hint")

        primary = face.regions[0]
        roi = self._face_roi(gray, primary)
        if roi is None or getattr(roi, "size", 0) <= 0:
            evidence.append("face crop unavailable")
            return self._store_expression_detection(
                hint="unclear",
                confidence=0.22,
                status="crop-unavailable",
                evidence=tuple(evidence),
                smile_count=0,
                smile_area_ratio=0.0,
                checked_now=True,
            )

        smile_count, smile_area_ratio = self._run_smile_cascade(roi)
        if smile_count > 0:
            evidence.append(f"OpenCV smile-ish feature count={smile_count}")
            confidence = self._expression_confidence(
                face_confidence=face.confidence,
                smile_count=smile_count,
                smile_area_ratio=smile_area_ratio,
                brightness=brightness,
                sharpness=sharpness,
            )
            return self._store_expression_detection(
                hint="smile-ish",
                confidence=confidence,
                status="smile-ish",
                evidence=tuple(evidence),
                smile_count=smile_count,
                smile_area_ratio=smile_area_ratio,
                checked_now=True,
            )

        if evidence:
            return self._store_expression_detection(
                hint="unclear",
                confidence=0.26,
                status="unclear",
                evidence=tuple(evidence),
                smile_count=0,
                smile_area_ratio=0.0,
                checked_now=True,
            )

        evidence.append("face-like region present; no smile-ish feature found")
        confidence = clamp01(0.22 + (face.confidence * 0.28) + (0.08 if face.stable else 0.0))
        return self._store_expression_detection(
            hint="neutral-ish",
            confidence=min(0.52, confidence),
            status="neutral-ish",
            evidence=tuple(evidence),
            smile_count=0,
            smile_area_ratio=0.0,
            checked_now=True,
        )

    def _store_expression_detection(
        self,
        *,
        hint: str,
        confidence: float,
        status: str,
        evidence: tuple[str, ...],
        smile_count: int,
        smile_area_ratio: float,
        checked_now: bool,
    ) -> ExpressionLiteDetection:
        confidence = clamp01(confidence)
        changed = bool(hint and hint != self._last_expression_hint and confidence >= 0.28)
        detection = ExpressionLiteDetection(
            enabled=True,
            available=self.expression_hint_available,
            attempted=True,
            hint=hint,
            confidence=confidence,
            changed=changed,
            smileish=(hint == "smile-ish"),
            neutralish=(hint == "neutral-ish"),
            unclear=(hint == "unclear"),
            status=status,
            evidence=evidence,
            smile_count=max(0, int(smile_count)),
            smile_area_ratio=clamp01(smile_area_ratio),
            checked_now=bool(checked_now),
        )
        if hint:
            self._last_expression_hint = hint
            self._last_expression_confidence = confidence
        self._last_expression_detection = detection
        return detection

    def _face_roi(self, gray: Any, region: FaceLiteRegion):
        height, width = gray.shape[:2]
        x0 = max(0, int((region.x + region.width * 0.12) * width))
        x1 = min(width, int((region.x + region.width * 0.88) * width))
        # Smile cascades are less noisy when focused on the lower half of a face-like region.
        y0 = max(0, int((region.y + region.height * 0.45) * height))
        y1 = min(height, int((region.y + region.height * 0.92) * height))
        if x1 <= x0 or y1 <= y0:
            return None
        roi = gray[y0:y1, x0:x1]
        if getattr(roi, "size", 0) <= 0:
            return None
        try:
            return self.cv2.equalizeHist(roi)
        except Exception:
            return roi

    def _run_smile_cascade(self, roi: Any) -> tuple[int, float]:
        if self._smile_cascade is None:
            return 0, 0.0
        try:
            found = self._smile_cascade.detectMultiScale(
                roi,
                scaleFactor=1.7,
                minNeighbors=18,
                minSize=(18, 8),
                flags=getattr(self.cv2, "CASCADE_SCALE_IMAGE", 0),
            )
        except Exception:
            return 0, 0.0
        roi_h, roi_w = roi.shape[:2]
        roi_area = max(1.0, float(roi_w * roi_h))
        areas: list[float] = []
        for raw in found[:6]:
            _x, _y, w, h = [float(v) for v in raw]
            if w <= 0.0 or h <= 0.0:
                continue
            area_ratio = clamp01((w * h) / roi_area)
            if 0.01 <= area_ratio <= 0.75:
                areas.append(area_ratio)
        if not areas:
            return 0, 0.0
        return len(areas), max(areas)

    def _expression_confidence(
        self,
        *,
        face_confidence: float,
        smile_count: int,
        smile_area_ratio: float,
        brightness: float,
        sharpness: float,
    ) -> float:
        confidence = 0.30
        confidence += min(0.22, face_confidence * 0.25)
        confidence += min(0.18, max(0, smile_count - 1) * 0.06)
        confidence += min(0.16, smile_area_ratio * 0.75)
        if brightness < 0.16:
            confidence -= 0.10
        if sharpness < 0.18:
            confidence -= 0.10
        return clamp01(min(0.82, confidence))

    def _expression_salience(self, expression: ExpressionLiteDetection) -> float:
        if not expression.enabled or not expression.available:
            return 0.0
        if expression.changed and expression.confidence >= 0.45:
            return clamp01(max(0.30, expression.confidence * 0.58))
        if expression.smileish and expression.confidence >= 0.38:
            return clamp01(max(0.24, expression.confidence * 0.42))
        return 0.0

    def _face_confidence(self, *, count: int, area_ratio: float, stable: bool, appeared: bool) -> float:
        if count <= 0:
            return 0.0
        confidence = 0.34
        confidence += min(0.28, area_ratio * 2.2)
        confidence += min(0.14, max(0, count - 1) * 0.05)
        if stable:
            confidence += 0.12
        if appeared:
            confidence += 0.04
        return clamp01(min(0.86, confidence))

    def _face_status(
        self,
        *,
        present: bool,
        confidence: float,
        stable: bool,
        appeared: bool,
        lost: bool,
    ) -> str:
        if lost:
            return "lost"
        if not present:
            return "none"
        if appeared:
            return "appeared"
        if stable:
            return "stable"
        if confidence >= 0.62:
            return "present"
        return "possible"

    def _face_salience(self, face: FaceLiteDetection) -> float:
        if not face.enabled or not face.available:
            return 0.0
        if face.appeared or face.lost:
            return clamp01(max(0.36, face.confidence * 0.70))
        if face.stable and face.count > 0:
            return clamp01(max(0.20, face.confidence * 0.34))
        if face.count > 0:
            return clamp01(max(0.26, face.confidence * 0.42))
        return 0.0

    def _visual_state_signature(
        self,
        *,
        brightness: float,
        sharpness: float,
        texture: float,
        motion_magnitude: float,
        scene_change_score: float,
        faceish_count: int,
        faceish_status: str,
        faceish_center_x: float,
        faceish_center_y: float,
        expression_hint: str,
        expression_status: str,
    ) -> str:
        face_part = "no-faceish"
        if faceish_count > 0:
            face_part = ":".join(
                [
                    "faceish",
                    str(min(3, int(faceish_count))),
                    _bucket(faceish_center_x, (0.33, 0.66)),
                    _bucket(faceish_center_y, (0.33, 0.66)),
                    str(faceish_status),
                ]
            )
        elif faceish_status == "lost":
            face_part = "faceish-lost"

        expression_part = "no-expression"
        if expression_hint:
            expression_part = ":".join(["expression", expression_hint, expression_status])

        return ":".join(
            [
                _bucket(brightness, (0.22, 0.42, 0.64, 0.80)),
                _bucket(sharpness, (0.18, 0.36, 0.62, 0.82)),
                _bucket(texture, (0.18, 0.36, 0.58, 0.78)),
                _bucket(motion_magnitude, (0.05, 0.14, 0.28, 0.50)),
                _bucket(scene_change_score, (0.08, 0.18, 0.32, 0.55)),
                face_part,
                expression_part,
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
    regions = [region.as_metadata() for region in metrics.faceish_regions]
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
        "face_lite": bool(metrics.face_hint_enabled),
        "face_lite_is_inferred": True,
        "face_lite_status": metrics.faceish_status,
        "face_lite_count": int(metrics.faceish_count),
        "face_lite_confidence": round(metrics.faceish_confidence, 6),
        "face_lite_stable": bool(metrics.faceish_stable),
        "face_lite_appeared": bool(metrics.faceish_appeared),
        "face_lite_lost": bool(metrics.faceish_lost),
        "face_lite_area_ratio": round(metrics.faceish_area_ratio, 6),
        "face_lite_center_x": round(metrics.faceish_center_x, 6),
        "face_lite_center_y": round(metrics.faceish_center_y, 6),
        "face_lite_regions": regions,
        "face_lite_checked_now": bool(metrics.faceish_checked_now),
        "expression_lite": bool(metrics.expression_hint_enabled),
        "expression_lite_is_inferred": True,
        "expression_lite_available": bool(metrics.expression_hint_available),
        "expression_lite_hint": metrics.expression_hint,
        "expression_lite_confidence": round(metrics.expression_confidence, 6),
        "expression_lite_changed": bool(metrics.expression_changed),
        "expression_lite_smileish": bool(metrics.expression_smileish),
        "expression_lite_neutralish": bool(metrics.expression_neutralish),
        "expression_lite_unclear": bool(metrics.expression_unclear),
        "expression_lite_status": metrics.expression_status,
        "expression_lite_evidence": list(metrics.expression_evidence),
        "expression_lite_smile_count": int(metrics.expression_smile_count),
        "expression_lite_smile_area_ratio": round(metrics.expression_smile_area_ratio, 6),
        "expression_lite_checked_now": bool(metrics.expression_checked_now),
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
    face = face_lite_phrase(metrics)
    if face:
        parts.append(face)
    expression = expression_lite_phrase(metrics)
    if expression:
        parts.append(expression)
    text = "; ".join(part for part in parts if part).strip()
    if metrics.visual_state_changed:
        text += "; visual state changed"
    return text + "."


def face_lite_phrase(metrics: VisionLiteMetrics) -> str:
    if not metrics.face_hint_enabled:
        return ""
    if not metrics.face_hint_available:
        return "Face-Lite unavailable"
    if metrics.faceish_lost:
        return "Face-Lite: previously seen face-like region no longer detected"
    if metrics.faceish_count <= 0:
        return ""

    count = int(metrics.faceish_count)
    region_word = "region" if count == 1 else "regions"
    confidence = metrics.faceish_confidence
    if metrics.faceish_appeared:
        prefix = "new possible"
    elif metrics.faceish_stable:
        prefix = "stable possible"
    elif confidence >= 0.62:
        prefix = "possible"
    else:
        prefix = "faint possible"

    location = _face_location_phrase(metrics.faceish_center_x, metrics.faceish_center_y)
    confidence_text = f"confidence={confidence:.2f}"
    return f"Face-Lite: {prefix} face-like {region_word} detected ({count}, {location}, {confidence_text})"


def expression_lite_phrase(metrics: VisionLiteMetrics) -> str:
    if not metrics.expression_hint_enabled:
        return ""
    if not metrics.face_hint_enabled or not metrics.face_hint_available:
        return ""
    if not metrics.expression_hint_available:
        return "Expression-Lite unavailable"
    if not metrics.expression_hint:
        return ""
    if metrics.expression_confidence < 0.28 and not metrics.expression_changed:
        return ""

    confidence = metrics.expression_confidence
    if metrics.expression_smileish:
        return f"Expression-Lite: possible smile-ish expression cue (confidence={confidence:.2f})"
    if metrics.expression_neutralish and (metrics.expression_changed or confidence >= 0.38):
        return f"Expression-Lite: possible neutral-ish expression cue (confidence={confidence:.2f})"
    if metrics.expression_unclear and (metrics.expression_changed or confidence >= 0.34):
        return f"Expression-Lite: expression cue unclear (confidence={confidence:.2f})"
    return ""


def _face_location_phrase(center_x: float, center_y: float) -> str:
    horizontal = "center"
    if center_x < 0.38:
        horizontal = "left"
    elif center_x > 0.62:
        horizontal = "right"

    vertical = "middle"
    if center_y < 0.38:
        vertical = "upper"
    elif center_y > 0.62:
        vertical = "lower"

    if horizontal == "center" and vertical == "middle":
        return "near center"
    return f"{vertical}-{horizontal} frame"


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
