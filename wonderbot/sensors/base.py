from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol


@dataclass(slots=True)
class SensorObservation:
    source: str
    text: str
    salience: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SensorStatus:
    source: str
    enabled: bool
    available: bool
    detail: str


class SensorAdapter(Protocol):
    name: str

    def poll(self) -> List[SensorObservation]:
        ...

    def status(self) -> SensorStatus:
        ...

    def close(self) -> None:
        ...
