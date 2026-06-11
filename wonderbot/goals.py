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
class GoalEntry:
    id: str
    title: str
    detail: str
    kind: str
    status: str
    priority: float
    progress: float
    created_at_ms: int
    updated_at_ms: int
    last_active_ms: int
    focus_count: int
    signature: str
    vector: List[float]
    tags: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def combined_text(self) -> str:
        return self.title if not self.detail else f"{self.title}. {self.detail}"

    def to_memory_item(self, focused: bool = False) -> MemoryItem:
        priority = self.priority + (0.16 if focused else 0.04)
        return MemoryItem(
            id=f"goal:{self.id}",
            text=f"goal/{self.status}: {self.combined_text()}",
            source=f"goal/{self.status}",
            created_at_ms=self.updated_at_ms,
            priority=round(max(0.05, min(1.0, priority)), 6),
            importance=round(max(0.45, self.priority), 6),
            novelty=round(max(0.05, 1.0 - self.progress * 0.5), 6),
            protected=True,
            status="active",
            signature=f"goal:{self.signature}",
            vector=list(self.vector),
            segments=[],
            metadata={"goal_id": self.id, "goal_status": self.status, "goal_kind": self.kind, **self.metadata},
        )


@dataclass(slots=True)
class GoalStatus:
    path: str
    total_entries: int
    active_entries: int
    focused_goal_id: str
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class GoalStore:
    ACTIVE_STATUSES = {"active", "pending", "blocked"}

    def __init__(self, codec: EventCodec, path: str = "state/goals.json") -> None:
        self.codec = codec
        self.path = Path(path)
        self.entries: List[GoalEntry] = []
        self.focus_goal_id: str = ""
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.entries = []
            self.focus_goal_id = ""
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.focus_goal_id = "" if isinstance(data, list) else str(data.get("focus_goal_id", ""))
        payloads = data if isinstance(data, list) else data.get("entries", [])
        self.entries = []
        for payload in payloads:
            entry = GoalEntry(**payload)
            if len(entry.vector) != self.codec.dim:
                entry.vector = self.codec.vectorize(entry.combined_text())
                entry.signature = self.codec.signature(entry.combined_text())
            self.entries.append(entry)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "focus_goal_id": self.focus_goal_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_goal(
        self,
        title: str,
        detail: str = "",
        kind: str = "goal",
        status: str = "active",
        priority: float = 0.68,
        progress: float = 0.0,
        tags: Optional[Sequence[str]] = None,
        evidence: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
        focus: bool = False,
    ) -> GoalEntry:
        title = " ".join(title.strip().split())
        detail = " ".join(detail.strip().split())
        if not title:
            raise ValueError("Cannot add empty goal title.")
        combined = title if not detail else f"{title}. {detail}"
        signature = self.codec.signature(combined)
        now = _now_ms()
        existing = next((entry for entry in self.entries if entry.signature == signature and entry.status in self.ACTIVE_STATUSES), None)
        if existing is not None:
            existing.updated_at_ms = now
            existing.last_active_ms = now
            existing.priority = round(min(1.0, max(existing.priority, priority)), 6)
            existing.progress = round(max(existing.progress, progress), 6)
            for tag in tags or []:
                if tag not in existing.tags:
                    existing.tags.append(tag)
            for item in evidence or []:
                if item not in existing.evidence:
                    existing.evidence.append(item)
            if metadata:
                existing.metadata.update(metadata)
            if focus:
                self.focus_goal_id = existing.id
                existing.focus_count += 1
            return existing
        entry = GoalEntry(
            id=str(uuid.uuid4()),
            title=title,
            detail=detail,
            kind=kind,
            status=status,
            priority=round(max(0.05, min(1.0, priority)), 6),
            progress=round(max(0.0, min(1.0, progress)), 6),
            created_at_ms=now,
            updated_at_ms=now,
            last_active_ms=now,
            focus_count=1 if focus else 0,
            signature=signature,
            vector=self.codec.vectorize(combined),
            tags=list(tags or []),
            evidence=list(evidence or []),
            metadata=dict(metadata or {}),
        )
        self.entries.append(entry)
        if focus or not self.focus_goal_id:
            self.focus_goal_id = entry.id
        return entry

    def latest(self, status: str | None = None, limit: int = 10) -> List[GoalEntry]:
        out = self.entries
        if status is not None:
            out = [entry for entry in out if entry.status == status]
        return sorted(out, key=lambda entry: (entry.status in self.ACTIVE_STATUSES, entry.priority, entry.updated_at_ms), reverse=True)[: max(0, limit)]

    def search(self, query: str, k: int = 5, statuses: Optional[Sequence[str]] = None) -> List[GoalEntry]:
        pool = self.entries
        if statuses is not None:
            status_set = set(statuses)
            pool = [entry for entry in pool if entry.status in status_set]
        if not query.strip():
            return self.latest(limit=k)
        qv = self.codec.vectorize(query)
        scored: List[tuple[float, GoalEntry]] = []
        for entry in pool:
            sim = _cosine(qv, entry.vector)
            focus_bonus = 0.10 if entry.id == self.focus_goal_id else 0.0
            score = 0.66 * sim + 0.22 * entry.priority + 0.07 * (1.0 - entry.progress) + focus_bonus
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[: max(0, k)]]

    def get(self, goal_id_prefix: str) -> Optional[GoalEntry]:
        goal_id_prefix = goal_id_prefix.strip().lower()
        if not goal_id_prefix:
            return None
        exact = next((entry for entry in self.entries if entry.id.lower() == goal_id_prefix), None)
        if exact is not None:
            return exact
        return next((entry for entry in self.entries if entry.id.lower().startswith(goal_id_prefix)), None)

    def set_status(self, goal_id_prefix: str, status: str, progress: Optional[float] = None, note: str = "") -> Optional[GoalEntry]:
        entry = self.get(goal_id_prefix)
        if entry is None:
            return None
        now = _now_ms()
        entry.status = status
        entry.updated_at_ms = now
        entry.last_active_ms = now
        if progress is not None:
            entry.progress = round(max(0.0, min(1.0, progress)), 6)
        if note:
            entry.evidence.append(note)
        if status not in self.ACTIVE_STATUSES and self.focus_goal_id == entry.id:
            self.focus_goal_id = self._fallback_focus_id(excluding=entry.id)
        return entry

    def set_focus(self, goal_id_prefix: str) -> Optional[GoalEntry]:
        entry = self.get(goal_id_prefix)
        if entry is None:
            return None
        self.focus_goal_id = entry.id
        entry.focus_count += 1
        entry.last_active_ms = _now_ms()
        return entry

    def focused(self) -> Optional[GoalEntry]:
        if not self.focus_goal_id:
            return None
        return next((entry for entry in self.entries if entry.id == self.focus_goal_id), None)

    def note_evidence(self, text: str, query: str = "") -> List[str]:
        matched = self.search(query or text, k=3, statuses=["active", "pending", "blocked"])
        hit_ids: List[str] = []
        for entry in matched:
            if _lexical_overlap(entry.combined_text(), text) < 0.12 and query:
                continue
            entry.last_active_ms = _now_ms()
            entry.updated_at_ms = entry.last_active_ms
            if text not in entry.evidence:
                entry.evidence.append(text)
            hit_ids.append(entry.id)
        return hit_ids

    def context_items(self, query: str, limit: int = 3) -> List[MemoryItem]:
        focused = self.focused()
        items: List[MemoryItem] = []
        if focused is not None and focused.status in self.ACTIVE_STATUSES:
            items.append(focused.to_memory_item(focused=True))
        for entry in self.search(query, k=max(1, limit), statuses=["active", "pending", "blocked"]):
            if focused is not None and entry.id == focused.id:
                continue
            items.append(entry.to_memory_item(focused=False))
        deduped: List[MemoryItem] = []
        seen: set[str] = set()
        for item in items:
            if item.signature in seen:
                continue
            seen.add(item.signature)
            deduped.append(item)
        return deduped[: max(0, limit)]

    def queue(self, limit: int = 10) -> List[GoalEntry]:
        active = [entry for entry in self.entries if entry.status in self.ACTIVE_STATUSES]
        return sorted(active, key=lambda entry: (entry.id == self.focus_goal_id, entry.priority, 1.0 - entry.progress, entry.updated_at_ms), reverse=True)[: max(0, limit)]

    def stats(self) -> Dict[str, object]:
        by_status: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        for entry in self.entries:
            by_status[entry.status] = by_status.get(entry.status, 0) + 1
            by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
        return {
            "total": len(self.entries),
            "focus_goal_id": self.focus_goal_id,
            "by_status": dict(sorted(by_status.items())),
            "by_kind": dict(sorted(by_kind.items())),
        }

    def status(self) -> GoalStatus:
        active = sum(1 for entry in self.entries if entry.status in self.ACTIVE_STATUSES)
        detail = "empty goal store"
        if self.entries:
            focused = self.focused()
            if focused is not None:
                detail = f"{active} active goals; focused on {focused.title}"
            else:
                detail = f"{active} active goals"
        return GoalStatus(path=str(self.path), total_entries=len(self.entries), active_entries=active, focused_goal_id=self.focus_goal_id, detail=detail)

    def _fallback_focus_id(self, excluding: str = "") -> str:
        for entry in self.queue(limit=10):
            if entry.id != excluding:
                return entry.id
        return ""


def _now_ms() -> int:
    return int(time.time() * 1000)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _lexical_overlap(a: str, b: str) -> float:
    aset = {token for token in a.lower().split() if len(token) > 2}
    bset = {token for token in b.lower().split() if len(token) > 2}
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / max(len(aset), len(bset))
