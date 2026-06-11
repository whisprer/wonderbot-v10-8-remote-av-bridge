from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import time
import uuid
from typing import Dict, Iterable, List, Optional

from .event_codec import EventCodec, SegmentEvent


@dataclass(slots=True)
class MemoryItem:
    id: str
    text: str
    source: str
    created_at_ms: int
    priority: float
    importance: float
    novelty: float
    protected: bool
    status: str
    signature: str
    vector: List[float]
    segments: List[Dict[str, object]]
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class MemoryStore:
    def __init__(
        self,
        codec: EventCodec,
        path: str = "state/memory.json",
        max_active_items: int = 2000,
        protect_identity: bool = True,
        importance_threshold: float = 0.36,
        min_novelty: float = 0.08,
    ) -> None:
        self.codec = codec
        self.path = Path(path)
        self.max_active_items = max_active_items
        self.protect_identity = protect_identity
        self.importance_threshold = importance_threshold
        self.min_novelty = min_novelty
        self.items: List[MemoryItem] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.items = []
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.items = [MemoryItem(**item) for item in data]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [item.to_dict() for item in self.items]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, text: str, source: str, metadata: Optional[Dict[str, object]] = None) -> MemoryItem:
        text = text.strip()
        if not text:
            raise ValueError("Cannot store empty memory text.")
        vector = self.codec.vectorize(text)
        signature = self.codec.signature(text)
        segments = [event.to_dict() for event in self.codec.analyze_text(text)]
        importance = self._importance(text, source)
        novelty = self._novelty(vector)
        protected = self._protected(text)
        priority = min(1.0, 0.45 * importance + 0.35 * novelty + 0.20 * self._segment_priority(segments))
        item = MemoryItem(
            id=str(uuid.uuid4()),
            text=text,
            source=source,
            created_at_ms=_ts_ms(),
            priority=round(priority, 6),
            importance=round(importance, 6),
            novelty=round(novelty, 6),
            protected=protected,
            status="active",
            signature=signature,
            vector=vector,
            segments=segments,
            metadata=dict(metadata or {}),
        )
        self.items.append(item)
        self._consolidate_if_needed()
        return item

    def search(self, query: str, k: int = 5, include_archived: bool = False) -> List[MemoryItem]:
        if not query.strip():
            return []
        qv = self.codec.vectorize(query)
        results: List[tuple[float, MemoryItem]] = []
        for item in self.items:
            if item.status != "active" and not include_archived:
                continue
            sim = _cosine(qv, item.vector)
            age_penalty = self._age_penalty(item.created_at_ms)
            score = 0.68 * sim + 0.22 * item.priority + 0.10 * age_penalty
            results.append((score, item))
        results.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in results[: max(0, k)]]

    def top_memories(self, limit: int = 10) -> List[MemoryItem]:
        active = [item for item in self.items if item.status == "active"]
        return sorted(active, key=lambda item: (item.priority, item.importance, item.created_at_ms), reverse=True)[:limit]

    def stats(self) -> Dict[str, object]:
        active = sum(1 for item in self.items if item.status == "active")
        archived = sum(1 for item in self.items if item.status == "archived")
        protected = sum(1 for item in self.items if item.protected)
        return {
            "total": len(self.items),
            "active": active,
            "archived": archived,
            "protected": protected,
        }

    def _consolidate_if_needed(self) -> None:
        active = [item for item in self.items if item.status == "active"]
        if len(active) <= self.max_active_items:
            return
        scored = sorted(active, key=lambda item: (item.priority, item.importance, item.created_at_ms))
        overflow = len(active) - self.max_active_items
        archived = 0
        for item in scored:
            if archived >= overflow:
                break
            if item.protected:
                continue
            item.status = "archived"
            archived += 1

    def _importance(self, text: str, source: str) -> float:
        length_score = min(len(text) / 220.0, 1.0)
        punctuation_score = 0.1 if any(ch in text for ch in "?!") else 0.0
        source_score = 0.12 if source in {"user", "assistant"} else 0.04
        return min(1.0, 0.18 + 0.6 * length_score + punctuation_score + source_score)

    def _novelty(self, vector: List[float]) -> float:
        active = [item for item in self.items if item.status == "active"]
        if not active:
            return 1.0
        max_sim = max(_cosine(vector, item.vector) for item in active)
        novelty = max(0.0, 1.0 - max_sim)
        if novelty < self.min_novelty:
            return novelty
        return novelty

    def _segment_priority(self, segments: List[Dict[str, object]]) -> float:
        if not segments:
            return 0.0
        values = [float(segment["priority"]) for segment in segments]
        return sum(values) / len(values)

    def _protected(self, text: str) -> bool:
        if not self.protect_identity:
            return False
        lowered = text.lower()
        triggers = ["my name is", "call me", "i am ", "i'm "]
        return any(trigger in lowered for trigger in triggers)

    def _age_penalty(self, created_at_ms: int) -> float:
        age_seconds = max(0.0, (_ts_ms() - created_at_ms) / 1000.0)
        return max(0.0, 1.0 - min(age_seconds / 86400.0, 1.0))


def _ts_ms() -> int:
    return int(time.time() * 1000)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
