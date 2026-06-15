from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Iterable


@dataclass(slots=True, frozen=True)
class AffectEstimate:
    """A cautious, non-diagnostic affect estimate from text/audio cues.

    This is deliberately an inference layer, not a claim about the user's actual
    internal state. Downstream code should phrase these estimates as "possible"
    and keep them out of automatic memory promotion unless explicitly reviewed.
    """

    label: str = "neutral"
    confidence: float = 0.0
    evidence: tuple[str, ...] = field(default_factory=tuple)
    valence: float = 0.0
    arousal: float = 0.0
    should_report: bool = False

    def metadata(self) -> dict[str, object]:
        return {
            "emotion_lite": True,
            "affect_is_inferred": True,
            "affect_label": self.label,
            "affect_confidence": round(float(self.confidence), 4),
            "affect_valence": round(float(self.valence), 4),
            "affect_arousal": round(float(self.arousal), 4),
            "affect_should_report": bool(self.should_report),
            "affect_evidence": list(self.evidence),
        }


class EmotionLiteEstimator:
    """Small rule-based text/audio affect estimator.

    It intentionally avoids heavy ML dependencies and avoids strong emotional
    claims. It works best as journal/debug metadata and prompt seasoning for
    backend-worthy speech observations.
    """

    def __init__(self, min_confidence: float = 0.32) -> None:
        self.min_confidence = max(0.0, min(1.0, float(min_confidence)))

    def estimate(
        self,
        transcript: str,
        *,
        salience: float = 0.0,
        rms: float | None = None,
        peak: float | None = None,
        zcr: float | None = None,
    ) -> AffectEstimate:
        raw = " ".join(str(transcript or "").split())
        norm = _normalize(raw)
        if not norm:
            return AffectEstimate(label="neutral", confidence=0.0, evidence=(), should_report=False)

        scores: dict[str, float] = {
            "neutral": 0.0,
            "uncertain": 0.0,
            "amused": 0.0,
            "frustrated": 0.0,
            "tired": 0.0,
            "engaged": 0.0,
            "distressed-lite": 0.0,
        }
        evidence: list[str] = []

        def add(label: str, amount: float, why: str) -> None:
            scores[label] = scores.get(label, 0.0) + float(amount)
            if why not in evidence:
                evidence.append(why)

        _match_any(norm, _UNCERTAIN_PATTERNS, lambda pat: add("uncertain", 0.22, _evidence_text("uncertainty phrase", pat)))
        _match_any(norm, _AMUSED_PATTERNS, lambda pat: add("amused", 0.30, _evidence_text("amusement cue", pat)))
        _match_any(norm, _SMILE_PATTERNS, lambda pat: add("amused", 0.24, _evidence_text("smile/joy cue", pat)))
        _match_any(norm, _FRUSTRATED_PATTERNS, lambda pat: add("frustrated", 0.28, _evidence_text("frustration cue", pat)))
        _match_any(norm, _TIRED_PATTERNS, lambda pat: add("tired", 0.28, _evidence_text("tiredness cue", pat)))
        _match_any(norm, _ENGAGED_PATTERNS, lambda pat: add("engaged", 0.20, _evidence_text("engagement cue", pat)))
        _match_any(norm, _POSITIVE_PATTERNS, lambda pat: add("engaged", 0.26, _evidence_text("positive cue", pat)))
        _match_any(norm, _DISTRESSED_PATTERNS, lambda pat: add("distressed-lite", 0.34, _evidence_text("distress cue", pat)))

        question_count = raw.count("?")
        exclamation_count = raw.count("!")
        if question_count:
            add("uncertain", min(0.20, 0.08 * question_count), "question punctuation")
        if exclamation_count:
            if scores["frustrated"] >= scores["amused"] and scores["frustrated"] > 0.0:
                add("frustrated", min(0.16, 0.06 * exclamation_count), "emphatic punctuation")
            else:
                add("engaged", min(0.14, 0.05 * exclamation_count), "emphatic punctuation")

        word_count = len(norm.split())
        if word_count <= 3 and question_count:
            add("uncertain", 0.08, "short question")
        if word_count >= 10 and scores["engaged"] > 0.0:
            add("engaged", 0.05, "longer engaged utterance")

        audio_salience = max(0.0, min(1.0, float(salience or 0.0)))
        audio_arousal = _audio_arousal(audio_salience=audio_salience, rms=rms, peak=peak, zcr=zcr)
        if audio_arousal >= 0.72:
            evidence.append("high audio salience")
            if scores["frustrated"] > 0.0:
                scores["frustrated"] += 0.06
            elif scores["amused"] > 0.0:
                scores["amused"] += 0.06
            elif scores["engaged"] > 0.0:
                scores["engaged"] += 0.07
        elif audio_arousal <= 0.24 and scores["tired"] > 0.0:
            scores["tired"] += 0.05
            evidence.append("low audio arousal")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        label, top_score = ranked[0]
        runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

        if top_score <= 0.0:
            confidence = min(0.26, 0.14 + audio_salience * 0.08)
            return AffectEstimate(
                label="neutral",
                confidence=round(confidence, 4),
                evidence=tuple(evidence[:5]),
                valence=0.0,
                arousal=round(audio_arousal, 4),
                should_report=False,
            )

        if label != "distressed-lite" and scores["distressed-lite"] >= 0.34 and scores["distressed-lite"] >= top_score - 0.05:
            label = "distressed-lite"
            top_score = scores[label]

        if label not in {"distressed-lite", "frustrated"} and abs(top_score - runner_up_score) < 0.07 and scores["uncertain"] > 0.0:
            label = "uncertain"
            top_score = max(top_score, scores[label])
            if "mixed cues" not in evidence:
                evidence.append("mixed cues")

        confidence = 0.18 + top_score + audio_salience * 0.10 + min(0.12, len(evidence) * 0.025)
        confidence = max(0.0, min(0.88, confidence))
        should_report = bool(confidence >= self.min_confidence and label != "neutral")

        return AffectEstimate(
            label=label,
            confidence=round(confidence, 4),
            evidence=tuple(evidence[:6]),
            valence=round(_label_valence(label), 4),
            arousal=round(max(audio_arousal, _label_arousal(label)), 4),
            should_report=should_report,
        )


def format_affect_suffix(estimate: AffectEstimate) -> str:
    label = estimate.label.replace("-", " ")
    return f"Affect estimate: possibly {label} (confidence={estimate.confidence:.2f})."


def apply_emotion_lite_to_text_and_metadata(
    text: str,
    metadata: dict[str, object],
    estimator: EmotionLiteEstimator | None,
    *,
    salience: float,
    append_to_text: bool = True,
) -> tuple[str, dict[str, object]]:
    if estimator is None:
        return text, metadata

    transcript = str(metadata.get("transcript") or metadata.get("transcript_fragment") or "")
    estimate = estimator.estimate(
        transcript,
        salience=salience,
        rms=_coerce_optional_float(metadata.get("utterance_rms") or metadata.get("rms")),
        peak=_coerce_optional_float(metadata.get("utterance_peak") or metadata.get("peak")),
        zcr=_coerce_optional_float(metadata.get("utterance_zcr") or metadata.get("zcr")),
    )
    metadata.update(estimate.metadata())
    if append_to_text and estimate.should_report:
        text = f"{text} {format_affect_suffix(estimate)}"
    return text, metadata



@dataclass(slots=True, frozen=True)
class VisualAffectContext:
    """A recent visual expression cue eligible for cautious affect fusion.

    This remains explicitly inferential: OpenCV face/expression hints are weak
    context, never emotional truth and never identity recognition.
    """

    hint: str
    confidence: float
    face_confidence: float
    face_count: int
    expression_status: str
    expression_evidence: tuple[str, ...]
    observed_at: float

    def is_fusable(self, now: float, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False
        if now - float(self.observed_at) > ttl_seconds:
            return False
        if self.face_count <= 0:
            return False
        if self.hint not in {"smile-ish", "neutral-ish", "unclear"}:
            return False
        return self.confidence >= 0.24 or self.hint == "unclear"


@dataclass(slots=True, frozen=True)
class MultimodalAffectEstimate:
    label: str
    confidence: float
    sources: tuple[str, ...]
    evidence: tuple[str, ...]
    mixed_signals: bool = False
    should_report: bool = False
    visual_age_seconds: float = 0.0

    def metadata(self) -> dict[str, object]:
        return {
            "multimodal_affect": True,
            "multimodal_affect_is_inferred": True,
            "multimodal_affect_label": self.label,
            "multimodal_affect_confidence": round(float(self.confidence), 4),
            "multimodal_affect_sources": list(self.sources),
            "multimodal_affect_evidence": list(self.evidence),
            "multimodal_affect_mixed_signals": bool(self.mixed_signals),
            "multimodal_affect_should_report": bool(self.should_report),
            "multimodal_affect_visual_age_seconds": round(float(self.visual_age_seconds), 3),
        }


def extract_visual_affect_context(
    metadata: dict[str, object] | None,
    *,
    observed_at: float,
) -> VisualAffectContext | None:
    """Extract a Face/Expression-Lite visual cue from camera metadata."""

    if not isinstance(metadata, dict):
        return None
    if not bool(metadata.get("expression_lite")):
        return None
    if not bool(metadata.get("expression_lite_available")):
        return None
    if int(_coerce_optional_float(metadata.get("face_lite_count")) or 0) <= 0:
        return None

    hint = str(metadata.get("expression_lite_hint") or "").strip().lower()
    if hint not in {"smile-ish", "neutral-ish", "unclear"}:
        return None

    confidence = _coerce_optional_float(metadata.get("expression_lite_confidence")) or 0.0
    face_confidence = _coerce_optional_float(metadata.get("face_lite_confidence")) or 0.0
    if confidence < 0.24 and hint != "unclear":
        return None
    if face_confidence < 0.28:
        return None

    return VisualAffectContext(
        hint=hint,
        confidence=max(0.0, min(1.0, float(confidence))),
        face_confidence=max(0.0, min(1.0, float(face_confidence))),
        face_count=max(0, int(_coerce_optional_float(metadata.get("face_lite_count")) or 0)),
        expression_status=str(metadata.get("expression_lite_status") or ""),
        expression_evidence=tuple(_metadata_string_list(metadata.get("expression_lite_evidence"))[:6]),
        observed_at=float(observed_at),
    )


def apply_multimodal_affect_to_text_and_metadata(
    text: str,
    metadata: dict[str, object],
    visual_context: VisualAffectContext | None,
    *,
    now: float,
    max_visual_age_seconds: float = 8.0,
    min_confidence: float = 0.34,
    append_to_text: bool = True,
) -> tuple[str, dict[str, object]]:
    """Fuse text/audio Emotion-Lite with recent visual Expression-Lite context.

    Text/audio remains primary. Visual expression cues only adjust or qualify the
    affect estimate, and disagreements become "mixed/uncertain" rather than a
    strong claim.
    """

    if visual_context is None or not visual_context.is_fusable(now, max_visual_age_seconds):
        return text, metadata
    if not isinstance(metadata, dict):
        return text, metadata
    if not bool(metadata.get("emotion_lite")):
        return text, metadata

    base_label = str(metadata.get("affect_label") or "neutral").strip().lower()
    base_confidence = _coerce_optional_float(metadata.get("affect_confidence")) or 0.0
    base_evidence = _metadata_string_list(metadata.get("affect_evidence"))
    visual_age = max(0.0, float(now) - float(visual_context.observed_at))

    estimate = fuse_affect_estimates(
        base_label=base_label,
        base_confidence=base_confidence,
        base_should_report=bool(metadata.get("affect_should_report")),
        base_evidence=base_evidence,
        visual_context=visual_context,
        visual_age_seconds=visual_age,
        min_confidence=min_confidence,
    )

    metadata.update(estimate.metadata())
    metadata.update(
        {
            "multimodal_affect_base_label": base_label,
            "multimodal_affect_base_confidence": round(float(base_confidence), 4),
            "multimodal_affect_visual_hint": visual_context.hint,
            "multimodal_affect_visual_confidence": round(float(visual_context.confidence), 4),
            "multimodal_affect_face_confidence": round(float(visual_context.face_confidence), 4),
            "multimodal_affect_face_count": int(visual_context.face_count),
        }
    )

    if append_to_text and estimate.should_report:
        text = _strip_prior_affect_suffix(text)
        text = f"{text} {format_multimodal_affect_suffix(estimate)}"

    return text, metadata


def fuse_affect_estimates(
    *,
    base_label: str,
    base_confidence: float,
    base_should_report: bool,
    base_evidence: list[str],
    visual_context: VisualAffectContext,
    visual_age_seconds: float,
    min_confidence: float = 0.34,
) -> MultimodalAffectEstimate:
    base_label = (base_label or "neutral").strip().lower()
    base_confidence = max(0.0, min(1.0, float(base_confidence or 0.0)))
    visual_confidence = max(0.0, min(1.0, float(visual_context.confidence)))
    face_confidence = max(0.0, min(1.0, float(visual_context.face_confidence)))

    sources = ("text/audio emotion-lite", "visual expression-lite", "face-lite gate")
    evidence = list(base_evidence[:4])
    evidence.append(f"visual expression cue: {visual_context.hint}")
    evidence.append(f"face-lite confidence: {face_confidence:.2f}")
    for item in visual_context.expression_evidence:
        if item and item not in evidence:
            evidence.append(f"visual evidence: {item}")

    positive = {"engaged", "amused"}
    negative = {"frustrated", "distressed-lite", "tired"}
    mixed = False

    hint = visual_context.hint

    if hint == "smile-ish":
        if base_label in positive:
            label = "engaged/amused" if base_label == "engaged" else "amused"
            confidence = base_confidence + visual_confidence * 0.18 + face_confidence * 0.05
        elif base_label in negative:
            label = "uncertain"
            mixed = True
            confidence = max(0.34, min(0.72, max(base_confidence, visual_confidence) - 0.03 + face_confidence * 0.04))
            evidence.append("mixed cues: smile-ish visual cue conflicts with negative text/audio estimate")
        elif base_label == "uncertain":
            label = "uncertain/engaged"
            confidence = max(base_confidence, 0.28 + visual_confidence * 0.30 + face_confidence * 0.08)
            evidence.append("smile-ish visual cue softens uncertainty")
        else:
            label = "engaged/amused"
            confidence = 0.18 + visual_confidence * 0.42 + face_confidence * 0.10

    elif hint == "neutral-ish":
        if base_label in positive and base_confidence >= 0.50:
            label = "uncertain"
            mixed = True
            confidence = max(0.34, min(0.62, base_confidence - 0.06 + visual_confidence * 0.10))
            evidence.append("mixed cues: neutral-ish visual cue conflicts with positive text/audio estimate")
        elif base_label in ({"uncertain"} | negative):
            label = base_label
            confidence = min(0.88, base_confidence + visual_confidence * 0.05)
            evidence.append("neutral-ish visual cue does not override text/audio estimate")
        else:
            label = "neutral"
            confidence = max(base_confidence, visual_confidence * 0.35)

    else:  # unclear
        if base_label in negative or base_label == "uncertain":
            label = base_label
            confidence = max(base_confidence, 0.34 + visual_confidence * 0.08)
            evidence.append("visual expression unclear; preserving text/audio estimate")
        elif base_should_report and base_label in positive:
            label = base_label
            confidence = max(0.34, base_confidence - 0.04)
            evidence.append("visual expression unclear; slightly damped text/audio estimate")
        else:
            label = "uncertain"
            confidence = max(0.34, min(0.56, base_confidence * 0.55 + visual_confidence * 0.35))
            evidence.append("visual expression unclear")

    confidence = max(0.0, min(0.92, float(confidence)))
    if label == "neutral":
        should_report = False
    else:
        should_report = bool(confidence >= min_confidence or mixed)

    return MultimodalAffectEstimate(
        label=label,
        confidence=round(confidence, 4),
        sources=sources,
        evidence=tuple(evidence[:8]),
        mixed_signals=mixed,
        should_report=should_report,
        visual_age_seconds=visual_age_seconds,
    )


def format_multimodal_affect_suffix(estimate: MultimodalAffectEstimate) -> str:
    label = estimate.label.replace("-", " ")
    source_phrase = "text/audio + expression-lite cue"
    if estimate.mixed_signals:
        return (
            f"Multimodal affect estimate: mixed cues, possibly {label} "
            f"(confidence={estimate.confidence:.2f}; {source_phrase})."
        )
    return (
        f"Multimodal affect estimate: possibly {label} "
        f"(confidence={estimate.confidence:.2f}; {source_phrase})."
    )


def _strip_prior_affect_suffix(text: str) -> str:
    cleaned = re.sub(r"\s+Affect estimate: possibly .*?\(confidence=[0-9.]+\)\.", "", str(text)).strip()
    return cleaned or str(text)


def _metadata_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9'?!%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _match_any(text: str, patterns: Iterable[str], callback) -> None:
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.search(pattern[3:], text):
                callback(pattern[3:])
        elif pattern in text:
            callback(pattern)


def _evidence_text(kind: str, pattern: str) -> str:
    cleaned = pattern.replace("\b", "").replace(".*", "…")
    cleaned = cleaned.strip("^$()?:")
    if len(cleaned) > 34:
        cleaned = cleaned[:31] + "..."
    return f"{kind}: {cleaned}"


def _coerce_optional_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _audio_arousal(*, audio_salience: float, rms: float | None, peak: float | None, zcr: float | None) -> float:
    parts = [max(0.0, min(1.0, audio_salience))]
    if rms is not None:
        parts.append(max(0.0, min(1.0, rms * 8.0)))
    if peak is not None:
        parts.append(max(0.0, min(1.0, peak * 2.0)))
    if zcr is not None:
        if 0.03 <= zcr <= 0.22:
            parts.append(0.44)
        elif zcr > 0.22:
            parts.append(0.58)
        else:
            parts.append(0.26)
    return sum(parts) / max(1, len(parts))


def _label_valence(label: str) -> float:
    return {
        "amused": 0.62,
        "engaged": 0.34,
        "neutral": 0.0,
        "uncertain": -0.12,
        "tired": -0.24,
        "frustrated": -0.46,
        "distressed-lite": -0.58,
    }.get(label, 0.0)


def _label_arousal(label: str) -> float:
    return {
        "amused": 0.58,
        "engaged": 0.52,
        "neutral": 0.22,
        "uncertain": 0.38,
        "tired": 0.18,
        "frustrated": 0.68,
        "distressed-lite": 0.62,
    }.get(label, 0.25)


_UNCERTAIN_PATTERNS = (
    "don't know",
    "dont know",
    "not sure",
    "unsure",
    "maybe",
    "i guess",
    "i think",
    "reckon",
    "what to do",
    "confused",
    "hmm",
    "erm",
    "uh",
    "not certain",
)

_AMUSED_PATTERNS = (
    "lol",
    "haha",
    "hehe",
    "funny",
    "made me laugh",
    "that's hilarious",
    "thats hilarious",
    "amusing",
    "lmao",
    ":)",
)

_SMILE_PATTERNS = (
    "yay",
    "woo",
    "wooo",
    "smile",
    "smiling",
    "cheerful",
    "fun",
    "joy",
    "joyful",
)

_FRUSTRATED_PATTERNS = (
    "fuck",
    "fucking",
    "shit",
    "damn",
    "bloody",
    "wtf",
    "broken",
    "bronked",
    "not working",
    "doesn't work",
    "doesnt work",
    "failed",
    "error",
    "wrong",
    "again",
    "stupid",
    "annoying",
    "frustrating",
    "can't get",
    "cant get",
)

_TIRED_PATTERNS = (
    "tired",
    "exhausted",
    "sleepy",
    "worn out",
    "drained",
    "burned out",
    "burnt out",
    "knackered",
    "frazzled",
    "too tired",
)

_ENGAGED_PATTERNS = (
    "let's",
    "lets",
    "go with",
    "move on",
    "ready",
    "yup",
    "yes",
    "cool",
    "nice",
    "great",
    "excellent",
    "thoughts",
    "reckon",
    "we can",
    "let us",
)


_POSITIVE_PATTERNS = (
    "happy",
    "happier",
    "happiness",
    "pleased",
    "delighted",
    "glad",
    "lovely",
    "brilliant",
    "love it",
    "love this",
    "i love",
    "working",
    "it works",
    "that works",
)

_DISTRESSED_PATTERNS = (
    "i don't know what to do",
    "i dont know what to do",
    "can't cope",
    "cant cope",
    "can't do this",
    "cant do this",
    "hopeless",
    "panicking",
    "panic",
    "scared",
    "afraid",
    "overwhelmed",
    "desperate",
)
