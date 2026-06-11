from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import math
import unicodedata
from typing import Dict, Iterable, List, Sequence


@dataclass(slots=True)
class SegmentEvent:
    index: int
    text: str
    start: int
    end: int
    signature: str
    phase_shift: float
    priority: float
    byte_ids: List[int]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class EventCodec:
    BOS = 1
    EOS = 2
    BYTES_BASE = 256

    def __init__(
        self,
        dim: int = 192,
        ngram: int = 3,
        window_chars: int = 16,
        min_segment_chars: int = 4,
        cosine_drop: float = 0.14,
        lowercase: bool = False,
        nfkc: bool = True,
    ) -> None:
        self.dim = dim
        self.ngram = ngram
        self.window_chars = window_chars
        self.min_segment_chars = min_segment_chars
        self.cosine_drop = cosine_drop
        self.lowercase = lowercase
        self.nfkc = nfkc

    def normalize(self, text: str) -> str:
        if self.nfkc:
            text = unicodedata.normalize("NFKC", text)
        if self.lowercase:
            text = text.lower()
        return text

    def encode_lossless(self, text: str, with_markers: bool = True) -> List[int]:
        body = [self.BYTES_BASE + b for b in text.encode("utf-8", errors="strict")]
        if with_markers:
            return [self.BOS, *body, self.EOS]
        return body

    def decode_lossless(self, ids: Sequence[int], with_markers: bool = True) -> str:
        buf = []
        for token in ids:
            if with_markers and token in (self.BOS, self.EOS):
                continue
            if token < self.BYTES_BASE:
                raise ValueError(f"Non-byte token encountered in lossless decode: {token}")
            buf.append(token - self.BYTES_BASE)
        return bytes(buf).decode("utf-8", errors="strict")

    def vectorize(self, text: str) -> List[float]:
        text = self.normalize(text)
        if not text:
            return [0.0] * self.dim
        buckets = [0.0] * self.dim
        grams = self._ngrams(text)
        if not grams:
            grams = [text]
        for gram in grams:
            idx = self._bucket(gram)
            buckets[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in buckets))
        if norm == 0:
            return buckets
        return [v / norm for v in buckets]

    def signature(self, text: str) -> str:
        payload = self.normalize(text).encode("utf-8", errors="strict")
        return hashlib.blake2b(payload, digest_size=16, person=b"wonderbot-seg").hexdigest()

    def segment_text(self, text: str) -> List[str]:
        text = self.normalize(text)
        if not text:
            return []
        if len(text) <= self.min_segment_chars:
            return [text]

        features = [self.vectorize(text[max(0, i - self.window_chars): i + 1]) for i in range(len(text))]
        cuts: List[int] = [0]
        anchor = features[min(self.window_chars - 1, len(features) - 1)]
        previous_cos = 1.0

        for i in range(1, len(text)):
            current = features[i]
            cosine = _cosine(anchor, current)
            anchor = [0.92 * a + 0.08 * c for a, c in zip(anchor, current)]
            distance = previous_cos - cosine
            boundary_char = text[i - 1]
            next_char = text[i]
            boundary_ok = (
                boundary_char.isspace()
                or next_char.isspace()
                or boundary_char in ",.;:!?()[]{}<>/|"
                or next_char in ",.;:!?()[]{}<>/|"
            )
            if i - cuts[-1] >= self.min_segment_chars and distance >= self.cosine_drop and boundary_ok:
                cuts.append(i)
            previous_cos = cosine

        if cuts[-1] != len(text):
            cuts.append(len(text))

        segments: List[str] = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            chunk = text[start:end].strip()
            if chunk:
                segments.append(chunk)
        return segments or [text]

    def analyze_text(self, text: str) -> List[SegmentEvent]:
        normalized = self.normalize(text)
        if not normalized:
            return []
        raw_segments = self.segment_text(normalized)
        events: List[SegmentEvent] = []
        cursor = 0
        previous_vector = None
        for index, segment in enumerate(raw_segments):
            start = normalized.find(segment, cursor)
            end = start + len(segment)
            cursor = end
            vector = self.vectorize(segment)
            phase_shift = 0.0 if previous_vector is None else max(0.0, 1.0 - _cosine(previous_vector, vector))
            priority = self._priority(segment, phase_shift)
            event = SegmentEvent(
                index=index,
                text=segment,
                start=start,
                end=end,
                signature=self.signature(segment),
                phase_shift=round(phase_shift, 6),
                priority=round(priority, 6),
                byte_ids=self.encode_lossless(segment),
            )
            events.append(event)
            previous_vector = vector
        return events

    def summarize_features(self, text: str) -> Dict[str, float]:
        events = self.analyze_text(text)
        if not events:
            return {"segments": 0.0, "mean_priority": 0.0, "max_phase_shift": 0.0}
        mean_priority = sum(e.priority for e in events) / len(events)
        max_phase = max(e.phase_shift for e in events)
        return {
            "segments": float(len(events)),
            "mean_priority": round(mean_priority, 6),
            "max_phase_shift": round(max_phase, 6),
        }

    def _ngrams(self, text: str) -> List[str]:
        if len(text) < self.ngram:
            return [text] if text else []
        return [text[i : i + self.ngram] for i in range(len(text) - self.ngram + 1)]

    def _bucket(self, gram: str) -> int:
        digest = hashlib.blake2b(gram.encode("utf-8", errors="strict"), digest_size=8, person=b"wonderbot").digest()
        return int.from_bytes(digest, "little") % self.dim

    def _priority(self, segment: str, phase_shift: float) -> float:
        length_score = min(len(segment) / 48.0, 1.0)
        punctuation_score = 0.12 if any(ch in segment for ch in "?!:") else 0.0
        numeric_score = 0.08 if any(ch.isdigit() for ch in segment) else 0.0
        uppercase_score = 0.08 if sum(ch.isupper() for ch in segment) >= 2 else 0.0
        novelty_hint = min(phase_shift * 0.9, 0.5)
        return min(1.0, 0.25 + length_score * 0.35 + punctuation_score + numeric_score + uppercase_score + novelty_hint)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return dot / (na * nb)
