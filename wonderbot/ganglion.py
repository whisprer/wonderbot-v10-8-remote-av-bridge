from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass(slots=True)
class GanglionState:
    tick: int
    mean_value: float
    max_value: float
    bleed: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class CABus:
    def __init__(self, height: int = 8, width: int = 8, channels: int = 8, bleed: float = 0.03) -> None:
        self.height = height
        self.width = width
        self.channels = channels
        self.bleed = bleed
        self.state = [
            [[0.0 for _ in range(width)] for _ in range(height)]
            for _ in range(channels)
        ]

    def reset(self) -> None:
        for channel in range(self.channels):
            for y in range(self.height):
                for x in range(self.width):
                    self.state[channel][y][x] = 0.0

    def inject(self, channel_idx: int, patch: List[List[float]], y: int = 0, x: int = 0) -> None:
        channel = channel_idx % self.channels
        for i, row in enumerate(patch):
            for j, value in enumerate(row):
                yy = (y + i) % self.height
                xx = (x + j) % self.width
                self.state[channel][yy][xx] = float(value)

    def read_patch(self, y: int = 0, x: int = 0, h: int = 4, w: int = 4) -> List[List[List[float]]]:
        out: List[List[List[float]]] = []
        for channel in range(self.channels):
            plane = []
            for i in range(h):
                row = []
                for j in range(w):
                    yy = (y + i) % self.height
                    xx = (x + j) % self.width
                    row.append(self.state[channel][yy][xx])
                plane.append(row)
            out.append(plane)
        return out

    def tick(self, row_to_shift: int) -> None:
        self._shift_row(row_to_shift % self.height)
        self._ca_update()
        self._crosstalk()

    def snapshot(self) -> GanglionState:
        values = [value for channel in self.state for row in channel for value in row]
        if not values:
            return GanglionState(tick=0, mean_value=0.0, max_value=0.0, bleed=self.bleed)
        return GanglionState(
            tick=0,
            mean_value=round(sum(values) / len(values), 6),
            max_value=round(max(values), 6),
            bleed=self.bleed,
        )

    def _shift_row(self, row_idx: int) -> None:
        for channel in range(self.channels):
            row = self.state[channel][row_idx]
            self.state[channel][row_idx] = [row[-1], *row[:-1]]

    def _ca_update(self) -> None:
        next_state = [
            [[0.0 for _ in range(self.width)] for _ in range(self.height)]
            for _ in range(self.channels)
        ]
        for channel in range(self.channels):
            for y in range(self.height):
                for x in range(self.width):
                    current = self.state[channel][y][x]
                    neighbors = self._neighbor_sum(channel, y, x)
                    value = _sigmoid(0.9 * current + 0.15 * neighbors)
                    next_state[channel][y][x] = value
        self.state = next_state

    def _neighbor_sum(self, channel: int, y: int, x: int) -> float:
        total = 0.0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy = (y + dy) % self.height
                xx = (x + dx) % self.width
                total += self.state[channel][yy][xx]
        return total

    def _crosstalk(self) -> None:
        if self.bleed <= 0.0 or self.channels <= 1:
            return
        channel_means = []
        for channel in range(self.channels):
            values = [value for row in self.state[channel] for value in row]
            channel_means.append(sum(values) / len(values))
        mean_all = sum(channel_means) / len(channel_means)
        for channel in range(self.channels):
            delta = (mean_all - channel_means[channel]) * self.bleed
            for y in range(self.height):
                for x in range(self.width):
                    self.state[channel][y][x] = max(0.0, min(1.0, self.state[channel][y][x] + delta))


class Ganglion:
    def __init__(self, height: int = 8, width: int = 8, channels: int = 8, bleed: float = 0.03) -> None:
        self.t = 0
        self.bus = CABus(height=height, width=width, channels=channels, bleed=bleed)

    def reset(self) -> None:
        self.t = 0
        self.bus.reset()

    def tick(self, count: int = 1) -> None:
        for _ in range(max(1, count)):
            self.bus.tick(self.t % self.bus.height)
            self.t += 1

    def write_signature(self, signature: str, channel_idx: int | None = None) -> None:
        bytes_ = bytes.fromhex(signature[:16])
        patch = _bytes_to_patch(bytes_)
        channel = (channel_idx if channel_idx is not None else self.t) % self.bus.channels
        self.bus.inject(channel, patch, y=self.t % self.bus.height, x=(self.t * 3) % self.bus.width)

    def state_summary(self) -> GanglionState:
        snap = self.bus.snapshot()
        return GanglionState(tick=self.t, mean_value=snap.mean_value, max_value=snap.max_value, bleed=snap.bleed)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = pow(2.718281828459045, -x)
        return 1.0 / (1.0 + z)
    z = pow(2.718281828459045, x)
    return z / (1.0 + z)


def _bytes_to_patch(data: bytes) -> List[List[float]]:
    values = list(data)
    while len(values) < 16:
        values.append(0)
    values = values[:16]
    patch: List[List[float]] = []
    for i in range(0, 16, 4):
        patch.append([round(v / 255.0, 6) for v in values[i:i + 4]])
    return patch
