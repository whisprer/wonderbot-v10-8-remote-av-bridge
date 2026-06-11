from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import time
import uuid
from typing import Dict, Iterable, List, Optional, Sequence

from .event_codec import EventCodec
from .memory import MemoryItem


@dataclass(slots=True)
class LongTermMemoryEntry:
    id: str
    text: str
    kind: str
    created_at_ms: int
    updated_at_ms: int
    last_accessed_ms: int
    strength: float
    uses: int
    status: str
    signature: str
    vector: List[float]
    evidence: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_memory_item(self) -> MemoryItem:
        priority = min(1.0, 0.60 * self.strength + 0.25 * min(1.0, self.uses / 8.0) + 0.15)
        now = _now_ms()
        return MemoryItem(
            id=f"ltm:{self.id}",
            text=self.text,
            source=f"ltm/{self.kind}",
            created_at_ms=self.created_at_ms,
            priority=round(priority, 6),
            importance=round(min(1.0, self.strength), 6),
            novelty=0.18,
            protected=bool(self.metadata.get("protected", False)),
            status="active",
            signature=self.signature,
            vector=list(self.vector),
            segments=[],
            metadata={
                "ltm": True,
                "kind": self.kind,
                "strength": self.strength,
                "uses": self.uses,
                "age_seconds": max(0.0, (now - self.created_at_ms) / 1000.0),
                **self.metadata,
            },
        )


@dataclass(slots=True)
class LongTermMemoryStatus:
    path: str
    total_entries: int
    active_entries: int
    archived_entries: int
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class LongTermMemoryStore:
    def __init__(self, codec: EventCodec, path: str = "state/long_term_memory.json") -> None:
        self.codec = codec
        self.path = Path(path)
        self.entries: List[LongTermMemoryEntry] = []
        self.last_sleep_ms = 0
        self.last_dream_ms = 0
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.entries = []
            self.last_sleep_ms = 0
            self.last_dream_ms = 0
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            self.last_sleep_ms = 0
            self.last_dream_ms = 0
            payloads = data
        else:
            self.last_sleep_ms = int(data.get("last_sleep_ms", 0))
            self.last_dream_ms = int(data.get("last_dream_ms", 0))
            payloads = data.get("entries", [])
        self.entries = [LongTermMemoryEntry(**entry) for entry in payloads if isinstance(entry, dict)]
        for entry in self.entries:
            if len(entry.vector) != self.codec.dim:
                entry.vector = self.codec.vectorize(entry.text)
                entry.signature = self.codec.signature(entry.text)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_sleep_ms": self.last_sleep_ms,
            "last_dream_ms": self.last_dream_ms,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def stats(self) -> Dict[str, object]:
        by_kind: Dict[str, int] = {}
        active_entries = 0
        archived_entries = 0
        for entry in self.entries:
            by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
            if entry.status == "active":
                active_entries += 1
            else:
                archived_entries += 1
        return {
            "total": len(self.entries),
            "active": active_entries,
            "archived": archived_entries,
            "by_kind": dict(sorted(by_kind.items())),
            "last_sleep_ms": self.last_sleep_ms,
            "last_dream_ms": self.last_dream_ms,
        }

    def status(self) -> LongTermMemoryStatus:
        active = sum(1 for entry in self.entries if entry.status == "active")
        archived = sum(1 for entry in self.entries if entry.status != "active")
        detail = "empty long-term memory"
        if self.entries:
            detail = f"{active} active entries across {len({entry.kind for entry in self.entries})} kinds"
        return LongTermMemoryStatus(
            path=str(self.path),
            total_entries=len(self.entries),
            active_entries=active,
            archived_entries=archived,
            detail=detail,
        )

    def latest(self, kind: str | None = None, limit: int = 10, active_only: bool = True) -> List[LongTermMemoryEntry]:
        out = self.entries
        if kind is not None:
            out = [entry for entry in out if entry.kind == kind]
        if active_only:
            out = [entry for entry in out if entry.status == "active"]
        return sorted(out, key=lambda entry: (entry.strength, entry.updated_at_ms), reverse=True)[: max(0, limit)]

    def search(self, query: str, k: int = 5, active_only: bool = True) -> List[LongTermMemoryEntry]:
        if not query.strip():
            return []
        qv = self.codec.vectorize(query)
        scored: List[tuple[float, LongTermMemoryEntry]] = []
        for entry in self.entries:
            if active_only and entry.status != "active":
                continue
            sim = _cosine(qv, entry.vector)
            strength_term = min(1.0, entry.strength)
            use_term = min(1.0, entry.uses / 10.0)
            recency_term = _recency_term(entry.last_accessed_ms or entry.updated_at_ms)
            score = 0.62 * sim + 0.22 * strength_term + 0.10 * use_term + 0.06 * recency_term
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = [entry for _, entry in scored[: max(0, k)]]
        self.record_access(results)
        return results

    def add_or_reinforce(
        self,
        text: str,
        kind: str,
        strength: float,
        evidence: Optional[Sequence[str]] = None,
        source_ids: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> LongTermMemoryEntry:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise ValueError("Cannot add empty long-term memory entry.")
        signature = self.codec.signature(cleaned)
        now = _now_ms()
        existing = next((entry for entry in self.entries if entry.signature == signature and entry.kind == kind), None)
        if existing is not None:
            existing.updated_at_ms = now
            existing.last_accessed_ms = now
            existing.uses += 1
            existing.status = "active"
            existing.strength = round(min(1.0, existing.strength + max(0.02, strength * 0.35)), 6)
            for ev in evidence or []:
                if ev not in existing.evidence:
                    existing.evidence.append(ev)
            for source_id in source_ids or []:
                if source_id not in existing.source_ids:
                    existing.source_ids.append(source_id)
            if metadata:
                existing.metadata.update(metadata)
            return existing
        entry = LongTermMemoryEntry(
            id=str(uuid.uuid4()),
            text=cleaned,
            kind=kind,
            created_at_ms=now,
            updated_at_ms=now,
            last_accessed_ms=now,
            strength=round(max(0.01, min(1.0, strength)), 6),
            uses=1,
            status="active",
            signature=signature,
            vector=self.codec.vectorize(cleaned),
            evidence=list(evidence or []),
            source_ids=list(source_ids or []),
            metadata=dict(metadata or {}),
        )
        self.entries.append(entry)
        return entry

    def record_access(self, entries: Iterable[LongTermMemoryEntry]) -> None:
        now = _now_ms()
        for entry in entries:
            entry.last_accessed_ms = now
            entry.uses += 1
            entry.strength = round(min(1.0, entry.strength + 0.01), 6)

    def decay(self, decay_rate: float = 0.03, archive_below: float = 0.12) -> int:
        archived = 0
        for entry in self.entries:
            if entry.status != "active":
                continue
            recency = _recency_term(entry.last_accessed_ms)
            use_bonus = min(0.06, entry.uses * 0.005)
            decayed = max(0.0, entry.strength * (1.0 - decay_rate) + 0.03 * recency + use_bonus)
            entry.strength = round(min(1.0, decayed), 6)
            if entry.strength < archive_below and entry.uses <= 2 and not entry.metadata.get("pinned", False):
                entry.status = "archived"
                archived += 1
        return archived


def _now_ms() -> int:
    return int(time.time() * 1000)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _recency_term(ts_ms: int) -> float:
    if ts_ms <= 0:
        return 0.0
    age_seconds = max(0.0, (_now_ms() - ts_ms) / 1000.0)
    return max(0.0, 1.0 - min(age_seconds / (7.0 * 86400.0), 1.0))
