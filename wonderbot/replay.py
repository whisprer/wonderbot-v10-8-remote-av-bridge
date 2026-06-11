from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import Any, List, Optional, Set


@dataclass(slots=True)
class ReplayStatus:
    enabled: bool
    path: str
    detail: str


class ReplayLogger:
    def __init__(self, path: str, enabled: bool = True, flush_each_write: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(path)
        self.flush_each_write = flush_each_write
        self._handle = None
        self._detail = 'logging disabled'
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open('a', encoding='utf-8')
            self._detail = f'writing replay events to {self.path}'

    def log(self, kind: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self._handle is None:
            return
        record = {
            'ts_ms': int(time.time() * 1000),
            'kind': kind,
            'payload': payload,
        }
        self._handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        if self.flush_each_write:
            self._handle.flush()

    def read_recent(self, limit: int = 50, kinds: Optional[Set[str]] = None) -> List[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = [line for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]
        out: List[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kinds is not None and record.get('kind') not in kinds:
                continue
            out.append(record)
            if len(out) >= max(0, limit):
                break
        return list(reversed(out))

    def status(self) -> ReplayStatus:
        return ReplayStatus(enabled=self.enabled, path=str(self.path), detail=self._detail)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None
