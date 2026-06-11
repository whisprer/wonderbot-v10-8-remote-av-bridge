from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import time
import uuid
from typing import Dict, Iterable, List, Optional

from .event_codec import EventCodec
from .memory import MemoryItem


@dataclass(slots=True)
class SelfModelEntry:
    id: str
    kind: str
    text: str
    source: str
    created_at_ms: int
    updated_at_ms: int
    strength: float
    status: str
    signature: str
    vector: List[float]
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def to_memory_item(self, priority_boost: float = 0.0) -> MemoryItem:
        priority = max(0.05, min(1.0, self.strength + priority_boost))
        return MemoryItem(
            id=f"self:{self.id}",
            text=f"self/{self.kind}: {self.text}",
            source=f"self/{self.kind}",
            created_at_ms=self.updated_at_ms,
            priority=round(priority, 6),
            importance=round(max(0.45, self.strength), 6),
            novelty=round(max(0.05, 1.0 - self.strength * 0.5), 6),
            protected=True,
            status="active",
            signature=f"self:{self.signature}",
            vector=list(self.vector),
            segments=[],
            metadata={"self_kind": self.kind, **self.metadata},
        )


@dataclass(slots=True)
class SelfModelStatus:
    path: str
    total_entries: int
    active_entries: int
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class SelfModelStore:
    def __init__(self, codec: EventCodec, path: str = "state/self_model.json") -> None:
        self.codec = codec
        self.path = Path(path)
        self.entries: List[SelfModelEntry] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.entries = []
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        payloads = data if isinstance(data, list) else data.get("entries", [])
        self.entries = []
        for payload in payloads:
            entry = SelfModelEntry(**payload)
            if len(entry.vector) != self.codec.dim:
                entry.vector = self.codec.vectorize(entry.text)
                entry.signature = self.codec.signature(f"{entry.kind}:{entry.text}")
            self.entries.append(entry)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [entry.to_dict() for entry in self.entries]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_or_reinforce(
        self,
        kind: str,
        text: str,
        source: str = "user",
        strength: float = 0.72,
        evidence: Optional[Iterable[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> SelfModelEntry:
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise ValueError("Cannot add empty self-model entry.")
        signature = self.codec.signature(f"{kind}:{cleaned}")
        now = _now_ms()
        existing = next((entry for entry in self.entries if entry.kind == kind and entry.signature == signature and entry.status == "active"), None)
        if existing is not None:
            existing.updated_at_ms = now
            existing.strength = round(min(1.0, existing.strength + max(0.03, strength * 0.25)), 6)
            for item in evidence or []:
                if item not in existing.evidence:
                    existing.evidence.append(item)
            if metadata:
                existing.metadata.update(metadata)
            return existing
        entry = SelfModelEntry(
            id=str(uuid.uuid4()),
            kind=kind,
            text=cleaned,
            source=source,
            created_at_ms=now,
            updated_at_ms=now,
            strength=round(max(0.05, min(1.0, strength)), 6),
            status="active",
            signature=signature,
            vector=self.codec.vectorize(cleaned),
            evidence=list(evidence or []),
            metadata=dict(metadata or {}),
        )
        self.entries.append(entry)
        return entry

    def latest(self, kind: str | None = None, limit: int = 10, active_only: bool = True) -> List[SelfModelEntry]:
        out = self.entries
        if kind is not None:
            out = [entry for entry in out if entry.kind == kind]
        if active_only:
            out = [entry for entry in out if entry.status == "active"]
        return sorted(out, key=lambda entry: (entry.strength, entry.updated_at_ms), reverse=True)[: max(0, limit)]

    def search(self, query: str, k: int = 5, active_only: bool = True) -> List[SelfModelEntry]:
        if not query.strip():
            return self.latest(limit=k, active_only=active_only)
        qv = self.codec.vectorize(query)
        scored: List[tuple[float, SelfModelEntry]] = []
        for entry in self.entries:
            if active_only and entry.status != "active":
                continue
            sim = _cosine(qv, entry.vector)
            kind_bonus = 0.06 if entry.kind in {"preference", "constraint", "identity"} else 0.0
            score = 0.74 * sim + 0.20 * entry.strength + kind_bonus
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[: max(0, k)]]

    def context_items(self, query: str, limit: int = 3) -> List[MemoryItem]:
        entries = self.search(query, k=limit) if query.strip() else self.latest(limit=limit)
        return [entry.to_memory_item(priority_boost=0.12 if entry.kind in {"preference", "constraint", "identity"} else 0.02) for entry in entries]

    def stats(self) -> Dict[str, object]:
        by_kind: Dict[str, int] = {}
        for entry in self.entries:
            by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
        return {"total": len(self.entries), "by_kind": dict(sorted(by_kind.items()))}

    def status(self) -> SelfModelStatus:
        active = sum(1 for entry in self.entries if entry.status == "active")
        detail = "empty self model"
        if self.entries:
            detail = f"{active} active entries across {len({entry.kind for entry in self.entries})} kinds"
        return SelfModelStatus(path=str(self.path), total_entries=len(self.entries), active_entries=active, detail=detail)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
