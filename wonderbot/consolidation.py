from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Dict, Iterable, List, Sequence

from .journal import JournalStore
from .memory import MemoryItem, MemoryStore
from .replay import ReplayLogger


@dataclass(slots=True)
class ConsolidationReport:
    summary: str = ""
    reflection: str = ""
    tasks: List[str] = field(default_factory=list)
    beliefs: List[str] = field(default_factory=list)
    threads: List[str] = field(default_factory=list)
    created_entry_ids: List[str] = field(default_factory=list)
    consolidated_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class MemoryConsolidator:
    def __init__(
        self,
        memory: MemoryStore,
        journal: JournalStore,
        replay: ReplayLogger,
        summary_min_items: int = 4,
        summary_window_items: int = 12,
        max_summary_sentences: int = 3,
        task_limit: int = 4,
        belief_limit: int = 4,
        thread_limit: int = 4,
        auto_every_explicit_turns: int = 3,
    ) -> None:
        self.memory = memory
        self.journal = journal
        self.replay = replay
        self.summary_min_items = max(2, summary_min_items)
        self.summary_window_items = max(self.summary_min_items, summary_window_items)
        self.max_summary_sentences = max(1, max_summary_sentences)
        self.task_limit = max(1, task_limit)
        self.belief_limit = max(1, belief_limit)
        self.thread_limit = max(1, thread_limit)
        self.auto_every_explicit_turns = max(1, auto_every_explicit_turns)
        self._explicit_turns_since_last = 0

    def note_explicit_turn(self) -> bool:
        self._explicit_turns_since_last += 1
        return self._explicit_turns_since_last >= self.auto_every_explicit_turns

    def consolidate(self, force: bool = False, reflect_only: bool = False) -> ConsolidationReport:
        recent = self._recent_unconsolidated_memories()
        if not force and not reflect_only and len(recent) < self.summary_min_items:
            return ConsolidationReport(consolidated_count=len(recent))

        report = ConsolidationReport(consolidated_count=len(recent))
        if not reflect_only and recent:
            summary = self._build_summary(recent)
            if summary:
                entry = self.journal.add(
                    "summary",
                    summary,
                    score=self._score_for_recent(recent),
                    evidence=[item.text for item in recent[:4]],
                    metadata={"source_count": len(recent)},
                )
                self._mark_memory_consolidated(recent, entry.id, "summary")
                self.memory.add(
                    summary,
                    source="assistant",
                    metadata={
                        "memory_kind": "summary",
                        "journal_entry_id": entry.id,
                        "consolidated": True,
                    },
                )
                report.summary = summary
                report.created_entry_ids.append(entry.id)
                self.replay.log("journal_summary", entry.to_dict())

            for task in self._extract_tasks(recent)[: self.task_limit]:
                entry = self.journal.add(
                    "task",
                    task,
                    score=0.78,
                    evidence=self._evidence_for_text(task, recent),
                    metadata={"open": True},
                )
                report.created_entry_ids.append(entry.id)
                report.tasks.append(task)
                self.replay.log("journal_task", entry.to_dict())

            for belief in self._extract_beliefs(recent)[: self.belief_limit]:
                entry = self.journal.add(
                    "belief",
                    belief,
                    score=0.72,
                    evidence=self._evidence_for_text(belief, recent),
                )
                report.created_entry_ids.append(entry.id)
                report.beliefs.append(belief)
                self.replay.log("journal_belief", entry.to_dict())

            for thread in self._extract_threads(recent)[: self.thread_limit]:
                entry = self.journal.add(
                    "thread",
                    thread,
                    score=0.67,
                    evidence=self._evidence_for_text(thread, recent),
                    metadata={"open": True},
                )
                report.created_entry_ids.append(entry.id)
                report.threads.append(thread)
                self.replay.log("journal_thread", entry.to_dict())

            self.journal.last_consolidated_ms = max(item.created_at_ms for item in recent)
            self._explicit_turns_since_last = 0

        reflection = self._build_reflection()
        if reflection:
            entry = self.journal.add(
                "reflection",
                reflection,
                score=0.74,
                evidence=[report.summary] if report.summary else [],
                metadata={"reflect_only": reflect_only},
            )
            self.memory.add(
                reflection,
                source="assistant",
                metadata={
                    "memory_kind": "reflection",
                    "journal_entry_id": entry.id,
                    "consolidated": True,
                },
            )
            report.reflection = reflection
            report.created_entry_ids.append(entry.id)
            self.replay.log("journal_reflection", entry.to_dict())
        self.journal.save()
        self.memory.save()
        return report

    def _recent_unconsolidated_memories(self) -> List[MemoryItem]:
        candidates = [
            item
            for item in self.memory.items
            if item.status == "active"
            and item.created_at_ms > self.journal.last_consolidated_ms
            and not bool(item.metadata.get("consolidated"))
            and item.metadata.get("memory_kind") not in {"summary", "reflection"}
            and item.source in {"user", "camera", "microphone", "assistant"}
        ]
        candidates.sort(key=lambda item: item.created_at_ms, reverse=True)
        return list(reversed(candidates[: self.summary_window_items]))

    def _build_summary(self, items: Sequence[MemoryItem]) -> str:
        if not items:
            return ""
        user_focuses = [self._focus_phrase(item.text) for item in items if item.source == "user"]
        sensor_focuses = [self._focus_phrase(item.text) for item in items if item.source in {"camera", "microphone"}]
        assistant_focuses = [self._focus_phrase(item.text) for item in items if item.source == "assistant"]
        top_user = _top_phrases(user_focuses, limit=2)
        top_sensor = _top_phrases(sensor_focuses, limit=1)
        top_assistant = _top_phrases(assistant_focuses, limit=1)
        sentences: List[str] = []
        if top_user:
            sentences.append(f"The recent session concentrated on {', '.join(top_user)}.")
        if top_sensor:
            sentences.append(f"Live input was pulled toward {top_sensor[0]}.")
        if top_assistant:
            sentences.append(
                f"The agent kept returning to {top_assistant[0]} as the nearest anchored reply thread."
            )
        if not sentences:
            compact = "; ".join(self._focus_phrase(item.text) for item in items[:3] if self._focus_phrase(item.text))
            if compact:
                sentences.append(f"The recent session revolved around {compact}.")
        return " ".join(sentences[: self.max_summary_sentences]).strip()

    def _extract_tasks(self, items: Sequence[MemoryItem]) -> List[str]:
        found: List[str] = []
        seen: set[str] = set()
        patterns = [
            r"\b(?:need to|needs to|let's|bring on|next|please|could you|can you|we should|should)\s+([^.!?]{4,120})",
            r"\bphase\s+([0-9]+[^.!?]{0,80})",
        ]
        for item in items:
            if item.source != "user":
                continue
            lowered = item.text.lower()
            for pattern in patterns:
                for match in re.finditer(pattern, lowered):
                    phrase = match.group(1).strip(" .,:;!?")
                    if not phrase:
                        continue
                    task = _clean_sentence(phrase)
                    if pattern.startswith(r"\bphase"):
                        task = f"Advance {task}"
                    if task not in seen and len(task) >= 4:
                        seen.add(task)
                        found.append(task)
        return found

    def _extract_beliefs(self, items: Sequence[MemoryItem]) -> List[str]:
        found: List[str] = []
        seen: set[str] = set()
        patterns = [
            r"\b(?:i prefer|prefer)\s+([^.!?]{4,120})",
            r"\b(?:i want|want)\s+([^.!?]{4,120})",
            r"\b(?:call me|my name is)\s+([^.!?]{2,80})",
            r"\b(?:always|never)\s+([^.!?]{4,120})",
        ]
        for item in items:
            if item.source != "user":
                continue
            lowered = item.text.lower()
            for pattern in patterns:
                for match in re.finditer(pattern, lowered):
                    phrase = _clean_sentence(match.group(0).strip(" .,:;!?"))
                    if phrase and phrase not in seen:
                        seen.add(phrase)
                        found.append(phrase)
        return found

    def _extract_threads(self, items: Sequence[MemoryItem]) -> List[str]:
        found: List[str] = []
        seen: set[str] = set()
        for item in items:
            if item.source not in {"user", "assistant"}:
                continue
            text = item.text.strip()
            lowered = text.lower()
            if "?" in text or any(marker in lowered for marker in ["next", "later", "still", "unfinished", "todo", "to do"]):
                thread = self._focus_phrase(text)
                if thread and thread not in seen:
                    seen.add(thread)
                    found.append(thread)
        return found

    def _build_reflection(self) -> str:
        recent_turns = self.replay.read_recent(limit=18, kinds={"turn", "spontaneous_response"})
        if not recent_turns:
            return ""
        imagination_total = 0
        imagination_accepted = 0
        suppressed = 0
        sensor_turns = 0
        for record in recent_turns:
            payload = record.get("payload", {})
            metadata = payload.get("backend_metadata", {}) or {}
            if "imagination" in metadata:
                imagination_total += 1
                if bool(metadata.get("imagination")):
                    imagination_accepted += 1
            if payload.get("inhibition_reason"):
                suppressed += 1
            if payload.get("source") in {"camera", "microphone"}:
                sensor_turns += 1
        if imagination_total == 0 and suppressed == 0 and sensor_turns == 0:
            return ""
        parts: List[str] = []
        if imagination_total:
            ratio = imagination_accepted / imagination_total
            if ratio >= 0.66:
                parts.append(
                    "The recent loop allowed controlled imaginative sidesteps often enough to stay inventive without looking loose."
                )
            elif ratio <= 0.2:
                parts.append(
                    "The recent loop stayed tightly anchored and rarely accepted imaginative sidesteps, which is safe but may be leaving useful adjacent ideas on the table."
                )
            else:
                parts.append(
                    "The recent loop balanced grounded replies with occasional imaginative sidesteps instead of defaulting to either rigidity or drift."
                )
        if suppressed:
            parts.append(
                f"{suppressed} recent turns were deliberately suppressed by cooldown or salience policy, which suggests the anti-chatter guardrails are active."
            )
        if sensor_turns:
            parts.append(
                f"{sensor_turns} recent turns came from live sensors, so the current behavior is being shaped by real-world input rather than text alone."
            )
        return " ".join(parts).strip()

    def _mark_memory_consolidated(self, items: Sequence[MemoryItem], entry_id: str, kind: str) -> None:
        for item in items:
            item.metadata["consolidated"] = True
            item.metadata["journal_entry_id"] = entry_id
            item.metadata["journal_kind"] = kind

    def _focus_phrase(self, text: str) -> str:
        lowered = " ".join(text.strip().split()).lower()
        if not lowered:
            return ""
        patterns = [
            r"\b(?:about|regarding|focused on|centered on|working on|for)\s+([^,.!?]{4,90})",
            r"\b(?:design|build|make|wire|stabilize|improve|tune|keep|advance)\s+([^,.!?]{4,90})",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered)
            if match:
                phrase = match.group(1).strip(" .,:;!?")
                if phrase:
                    return phrase[:96]
        quoted = re.findall(r'"([^"]{3,96})"', text)
        if quoted:
            return quoted[0].strip().lower()[:96]
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-']+", lowered)
        content = [token for token in tokens if token not in _STOPWORDS and len(token) > 2]
        return " ".join(content[:6])[:96]

    def _evidence_for_text(self, needle: str, items: Sequence[MemoryItem]) -> List[str]:
        lowered = needle.lower()
        keywords = [piece for piece in lowered.split() if len(piece) > 2][:3]
        evidence = [item.text for item in items if any(piece in item.text.lower() for piece in keywords)]
        return evidence[:3]

    def _score_for_recent(self, items: Sequence[MemoryItem]) -> float:
        if not items:
            return 0.0
        return sum(item.priority for item in items) / len(items)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "do", "does", "for", "from", "get", "got", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "just", "like", "more", "my", "now", "of", "on", "or", "our", "so", "stuff", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "up", "use", "was", "we", "what", "when", "where", "which", "with", "would", "you", "your", "fren", "please", "could", "can"
}


def _clean_sentence(text: str) -> str:
    text = " ".join(text.strip().split())
    return text[:1].upper() + text[1:] if text else ""


def _top_phrases(values: Iterable[str], limit: int = 3) -> List[str]:
    counts: Dict[str, int] = {}
    for value in values:
        cleaned = value.strip(" .,:;!?")
        if len(cleaned) < 4:
            continue
        counts[cleaned] = counts.get(cleaned, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [value for value, _ in ranked[: max(0, limit)]]
