from __future__ import annotations

from dataclasses import asdict, dataclass, field
import itertools
from typing import Dict, List, Sequence

from .journal import JournalEntry, JournalStore
from .longterm import LongTermMemoryEntry, LongTermMemoryStore
from .memory import MemoryItem, MemoryStore
from .replay import ReplayLogger


@dataclass(slots=True)
class SleepReport:
    promoted_count: int = 0
    promoted_texts: List[str] = field(default_factory=list)
    dream_count: int = 0
    dreams: List[str] = field(default_factory=list)
    archived_count: int = 0
    reinforced_existing: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class MemoryLifecycle:
    def __init__(
        self,
        memory: MemoryStore,
        journal: JournalStore,
        longterm: LongTermMemoryStore,
        replay: ReplayLogger,
        promotion_limit: int = 6,
        min_promotion_strength: float = 0.58,
        dream_limit: int = 3,
        dream_similarity_min: float = 0.32,
        dream_similarity_max: float = 0.82,
        archive_decay_rate: float = 0.03,
        archive_below_strength: float = 0.12,
        auto_sleep_every_explicit_turns: int = 8,
    ) -> None:
        self.memory = memory
        self.journal = journal
        self.longterm = longterm
        self.replay = replay
        self.promotion_limit = promotion_limit
        self.min_promotion_strength = min_promotion_strength
        self.dream_limit = dream_limit
        self.dream_similarity_min = dream_similarity_min
        self.dream_similarity_max = dream_similarity_max
        self.archive_decay_rate = archive_decay_rate
        self.archive_below_strength = archive_below_strength
        self.auto_sleep_every_explicit_turns = auto_sleep_every_explicit_turns
        self._explicit_turns_since_last_sleep = 0

    def note_explicit_turn(self) -> bool:
        self._explicit_turns_since_last_sleep += 1
        return self._explicit_turns_since_last_sleep >= self.auto_sleep_every_explicit_turns

    def sleep(self, force: bool = True) -> SleepReport:
        if not force and self._explicit_turns_since_last_sleep < self.auto_sleep_every_explicit_turns:
            return SleepReport()
        report = SleepReport()
        candidates = self._promotion_candidates()[: self.promotion_limit]
        for strength, source_kind, text, evidence, source_ids, metadata in candidates:
            before = self._existing_count()
            entry = self.longterm.add_or_reinforce(
                text=text,
                kind=source_kind,
                strength=strength,
                evidence=evidence,
                source_ids=source_ids,
                metadata=metadata,
            )
            after = self._existing_count()
            if after == before:
                report.reinforced_existing += 1
            else:
                report.promoted_count += 1
            report.promoted_texts.append(entry.text)
            self.replay.log("ltm_promote", entry.to_dict())

        dreams = self._dream_entries(limit=self.dream_limit)
        for dream_text, source_ids in dreams:
            entry = self.longterm.add_or_reinforce(
                text=dream_text,
                kind="dream",
                strength=0.52,
                evidence=[dream_text],
                source_ids=source_ids,
                metadata={"generated": True},
            )
            report.dream_count += 1
            report.dreams.append(entry.text)
            self.replay.log("ltm_dream", entry.to_dict())

        report.archived_count = self.longterm.decay(
            decay_rate=self.archive_decay_rate,
            archive_below=self.archive_below_strength,
        )
        self.longterm.last_sleep_ms = _now_ms()
        if report.dream_count:
            self.longterm.last_dream_ms = _now_ms()
        self.longterm.save()
        self._explicit_turns_since_last_sleep = 0
        self.replay.log("sleep_cycle", report.to_dict())
        return report

    def dream(self, force: bool = True) -> SleepReport:
        report = SleepReport()
        dreams = self._dream_entries(limit=self.dream_limit)
        for dream_text, source_ids in dreams:
            entry = self.longterm.add_or_reinforce(
                text=dream_text,
                kind="dream",
                strength=0.50,
                evidence=[dream_text],
                source_ids=source_ids,
                metadata={"generated": True, "dream_only": True},
            )
            report.dream_count += 1
            report.dreams.append(entry.text)
            self.replay.log("ltm_dream", entry.to_dict())
        if report.dream_count:
            self.longterm.last_dream_ms = _now_ms()
            self.longterm.save()
        self.replay.log("dream_cycle", report.to_dict())
        return report

    def _promotion_candidates(self) -> List[tuple[float, str, str, List[str], List[str], Dict[str, object]]]:
        candidates: List[tuple[float, str, str, List[str], List[str], Dict[str, object]]] = []
        journal_entries = self.journal.latest(limit=24)
        for entry in journal_entries:
            strength = self._journal_strength(entry)
            if strength < self.min_promotion_strength:
                continue
            candidates.append(
                (
                    strength,
                    f"journal/{entry.kind}",
                    entry.text,
                    list(entry.evidence[:4]),
                    [entry.id],
                    {"journal_kind": entry.kind},
                )
            )
        for item in self.memory.top_memories(limit=16):
            if item.source not in {"user", "assistant", "camera", "microphone"}:
                continue
            if item.metadata.get("memory_kind") in {"summary", "reflection"}:
                continue
            strength = min(0.96, 0.52 * item.priority + 0.30 * item.importance + 0.18 * item.novelty)
            if strength < self.min_promotion_strength:
                continue
            kind = f"episodic/{item.source}"
            candidates.append(
                (
                    round(strength, 6),
                    kind,
                    item.text,
                    [item.text[:160]],
                    [item.id],
                    {"from_memory": True, "source": item.source},
                )
            )
        candidates.sort(key=lambda row: (row[0], len(row[2])), reverse=True)
        deduped: List[tuple[float, str, str, List[str], List[str], Dict[str, object]]] = []
        seen: set[str] = set()
        for row in candidates:
            key = row[2].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    def _dream_entries(self, limit: int) -> List[tuple[str, List[str]]]:
        candidates = self.longterm.latest(limit=18)
        if len(candidates) < 2:
            return []
        dreams: List[tuple[float, str, List[str]]] = []
        for left, right in itertools.combinations(candidates, 2):
            if left.kind == "dream" and right.kind == "dream":
                continue
            similarity = _cosine(left.vector, right.vector)
            if similarity < self.dream_similarity_min or similarity > self.dream_similarity_max:
                continue
            text = self._render_dream(left, right)
            novelty = abs(left.strength - right.strength) * 0.2 + (1.0 - abs(similarity - 0.52))
            score = 0.72 * novelty + 0.28 * ((left.strength + right.strength) / 2.0)
            dreams.append((score, text, [left.id, right.id]))
        dreams.sort(key=lambda row: row[0], reverse=True)
        out: List[tuple[str, List[str]]] = []
        seen: set[str] = set()
        for _, text, source_ids in dreams:
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((text, source_ids))
            if len(out) >= max(0, limit):
                break
        return out

    def _journal_strength(self, entry: JournalEntry) -> float:
        kind_base = {
            "belief": 0.84,
            "task": 0.78,
            "summary": 0.74,
            "thread": 0.72,
            "reflection": 0.68,
        }.get(entry.kind, 0.62)
        evidence_bonus = min(0.08, len(entry.evidence) * 0.02)
        return round(min(1.0, kind_base * 0.65 + entry.score * 0.25 + evidence_bonus), 6)

    def _render_dream(self, left: LongTermMemoryEntry, right: LongTermMemoryEntry) -> str:
        left_focus = _focus_phrase(left.text)
        right_focus = _focus_phrase(right.text)
        if left.kind.startswith("journal/task") or right.kind.startswith("journal/task"):
            return f"Dream synthesis: connect {left_focus} with {right_focus} as one continuous plan rather than two separate threads."
        if left.kind.startswith("journal/belief") or right.kind.startswith("journal/belief"):
            return f"Dream synthesis: treat {left_focus} as a governing preference that should shape work on {right_focus}."
        return f"Dream synthesis: the system keeps linking {left_focus} with {right_focus}, suggesting a durable adjacency worth preserving."

    def _existing_count(self) -> int:
        return len(self.longterm.entries)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _focus_phrase(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "the current thread"
    lowered = cleaned.lower()
    tokens = [token for token in lowered.replace("—", " ").replace("-", " ").split() if len(token) > 2]
    if not tokens:
        return cleaned[:96]
    return " ".join(tokens[:8])[:96]


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
