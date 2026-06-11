from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import time
import uuid
from typing import Dict, List, Optional


@dataclass(slots=True)
class JournalEntry:
    id: str
    kind: str
    text: str
    created_at_ms: int
    score: float
    status: str
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class JournalStatus:
    path: str
    last_consolidated_ms: int
    total_entries: int
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class JournalStore:
    def __init__(self, path: str = 'state/journal.json') -> None:
        self.path = Path(path)
        self.last_consolidated_ms = 0
        self.entries: List[JournalEntry] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.last_consolidated_ms = 0
            self.entries = []
            return
        data = json.loads(self.path.read_text(encoding='utf-8'))
        if isinstance(data, list):
            self.last_consolidated_ms = 0
            self.entries = [JournalEntry(**entry) for entry in data if isinstance(entry, dict)]
            return
        self.last_consolidated_ms = int(data.get('last_consolidated_ms', 0))
        self.entries = [JournalEntry(**entry) for entry in data.get('entries', []) if isinstance(entry, dict)]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'last_consolidated_ms': self.last_consolidated_ms,
            'entries': [entry.to_dict() for entry in self.entries],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    def add(
        self,
        kind: str,
        text: str,
        score: float = 0.5,
        evidence: Optional[List[str]] = None,
        status: str = 'active',
        metadata: Optional[Dict[str, object]] = None,
    ) -> JournalEntry:
        entry = JournalEntry(
            id=str(uuid.uuid4()),
            kind=kind,
            text=' '.join(text.strip().split()),
            created_at_ms=_now_ms(),
            score=round(float(score), 6),
            status=status,
            evidence=list(evidence or []),
            metadata=dict(metadata or {}),
        )
        self.entries.append(entry)
        return entry

    def latest(self, kind: str | None = None, limit: int = 10, active_only: bool = True) -> List[JournalEntry]:
        out = self.entries
        if kind is not None:
            out = [entry for entry in out if entry.kind == kind]
        if active_only:
            out = [entry for entry in out if entry.status == 'active']
        return sorted(out, key=lambda entry: (entry.created_at_ms, entry.score), reverse=True)[:max(0, limit)]

    def stats(self) -> Dict[str, object]:
        by_kind: Dict[str, int] = {}
        for entry in self.entries:
            by_kind[entry.kind] = by_kind.get(entry.kind, 0) + 1
        return {
            'total': len(self.entries),
            'last_consolidated_ms': self.last_consolidated_ms,
            'by_kind': dict(sorted(by_kind.items())),
        }

    def status(self) -> JournalStatus:
        detail = 'empty journal'
        if self.entries:
            detail = f'{len(self.entries)} entries across {len({entry.kind for entry in self.entries})} kinds'
        return JournalStatus(
            path=str(self.path),
            last_consolidated_ms=self.last_consolidated_ms,
            total_entries=len(self.entries),
            detail=detail,
        )


def _now_ms() -> int:
    return int(time.time() * 1000)
