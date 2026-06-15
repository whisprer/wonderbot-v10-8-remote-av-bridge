from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable


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
        _match_any(norm, _FRUSTRATED_PATTERNS, lambda pat: add("frustrated", 0.28, _evidence_text("frustration cue", pat)))
        _match_any(norm, _TIRED_PATTERNS, lambda pat: add("tired", 0.28, _evidence_text("tiredness cue", pat)))
        _match_any(norm, _ENGAGED_PATTERNS, lambda pat: add("engaged", 0.20, _evidence_text("engagement cue", pat)))
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
                scores["amused"] += 0.05
            elif scores["engaged"] > 0.0:
                scores["engaged"] += 0.05
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
