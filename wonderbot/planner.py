from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import math
import re
import time
import uuid
from typing import Dict, Iterable, List, Optional, Sequence

from .event_codec import EventCodec
from .memory import MemoryItem


@dataclass(slots=True)
class PlanStep:
    id: str
    title: str
    order: int
    status: str
    progress: float
    action_intent: str
    dependency_ids: List[str]
    blocker_note: str
    created_at_ms: int
    updated_at_ms: int
    completed_at_ms: int
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def short(self) -> str:
        return self.title


@dataclass(slots=True)
class PlanEntry:
    id: str
    title: str
    detail: str
    goal_id: str
    status: str
    priority: float
    progress: float
    created_at_ms: int
    updated_at_ms: int
    last_active_ms: int
    focus_count: int
    signature: str
    vector: List[float]
    action_intents: List[str] = field(default_factory=list)
    steps: List[PlanStep] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def combined_text(self) -> str:
        return self.title if not self.detail else f"{self.title}. {self.detail}"

    def to_memory_item(self, focused: bool = False) -> MemoryItem:
        open_steps = sum(1 for step in self.steps if step.status not in {"done", "cancelled"})
        next_bits = [step.title for step in self.steps if step.status in {"todo", "doing", "blocked"}][:2]
        suffix = f" Next: {'; '.join(next_bits)}" if next_bits else ""
        text = f"plan/{self.status}: {self.combined_text()} ({open_steps} open steps).{suffix}".strip()
        priority = self.priority + (0.18 if focused else 0.05)
        return MemoryItem(
            id=f"plan:{self.id}",
            text=text,
            source=f"plan/{self.status}",
            created_at_ms=self.updated_at_ms,
            priority=round(max(0.05, min(1.0, priority)), 6),
            importance=round(max(0.45, self.priority), 6),
            novelty=round(max(0.05, 1.0 - self.progress * 0.45), 6),
            protected=True,
            status="active",
            signature=f"plan:{self.signature}",
            vector=list(self.vector),
            segments=[],
            metadata={
                "plan_id": self.id,
                "plan_status": self.status,
                "goal_id": self.goal_id,
                "action_intents": list(self.action_intents),
                **self.metadata,
            },
        )


@dataclass(slots=True)
class PlanStatus:
    path: str
    total_entries: int
    active_entries: int
    focused_plan_id: str
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class OutcomeUpdate:
    matched_plan_id: str = ""
    matched_step_id: str = ""
    matched_goal_id: str = ""
    outcome_kind: str = ""
    status: str = ""
    progress: float = 0.0
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class PlanStore:
    ACTIVE_STATUSES = {"active", "pending", "blocked"}
    STEP_ACTIVE_STATUSES = {"todo", "doing", "blocked"}

    def __init__(self, codec: EventCodec, path: str = "state/plans.json") -> None:
        self.codec = codec
        self.path = Path(path)
        self.entries: List[PlanEntry] = []
        self.focus_plan_id: str = ""
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.entries = []
            self.focus_plan_id = ""
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.focus_plan_id = "" if isinstance(data, list) else str(data.get("focus_plan_id", ""))
        payloads = data if isinstance(data, list) else data.get("entries", [])
        self.entries = []
        for payload in payloads:
            payload = dict(payload)
            payload["steps"] = [step if isinstance(step, PlanStep) else PlanStep(**step) for step in payload.get("steps", [])]
            entry = PlanEntry(**payload)
            if len(entry.vector) != self.codec.dim:
                entry.vector = self.codec.vectorize(entry.combined_text())
                entry.signature = self.codec.signature(entry.combined_text())
            self.entries.append(entry)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "focus_plan_id": self.focus_plan_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_plan(
        self,
        title: str,
        detail: str = "",
        goal_id: str = "",
        status: str = "active",
        priority: float = 0.68,
        focus: bool = False,
        action_intents: Optional[Sequence[str]] = None,
        evidence: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> PlanEntry:
        title = _clean_text(title)
        detail = _clean_text(detail)
        if not title:
            raise ValueError("Cannot add empty plan title.")
        combined = title if not detail else f"{title}. {detail}"
        signature = self.codec.signature(combined)
        now = _now_ms()
        existing = next((entry for entry in self.entries if entry.signature == signature and entry.status in self.ACTIVE_STATUSES), None)
        if existing is not None:
            existing.updated_at_ms = now
            existing.last_active_ms = now
            existing.priority = round(min(1.0, max(existing.priority, priority)), 6)
            if goal_id and not existing.goal_id:
                existing.goal_id = goal_id
            for intent in action_intents or []:
                if intent and intent not in existing.action_intents:
                    existing.action_intents.append(intent)
            for item in evidence or []:
                if item not in existing.evidence:
                    existing.evidence.append(item)
            if metadata:
                existing.metadata.update(metadata)
            if focus:
                self.focus_plan_id = existing.id
                existing.focus_count += 1
            return existing
        entry = PlanEntry(
            id=str(uuid.uuid4()),
            title=title,
            detail=detail,
            goal_id=goal_id,
            status=status,
            priority=round(max(0.05, min(1.0, priority)), 6),
            progress=0.0,
            created_at_ms=now,
            updated_at_ms=now,
            last_active_ms=now,
            focus_count=1 if focus else 0,
            signature=signature,
            vector=self.codec.vectorize(combined),
            action_intents=[intent for intent in (action_intents or []) if intent],
            evidence=list(evidence or []),
            metadata=dict(metadata or {}),
        )
        self.entries.append(entry)
        if focus or not self.focus_plan_id:
            self.focus_plan_id = entry.id
        return entry

    def add_step(
        self,
        plan_id_prefix: str,
        text: str,
        action_intent: str = "",
        dependency_ids: Optional[Sequence[str]] = None,
        status: str = "todo",
        evidence: Optional[Sequence[str]] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Optional[PlanStep]:
        plan = self.get(plan_id_prefix)
        if plan is None:
            return None
        title = _clean_text(text)
        if not title:
            raise ValueError("Cannot add empty step title.")
        existing = next((step for step in plan.steps if _normalize_key(step.title) == _normalize_key(title)), None)
        now = _now_ms()
        if existing is not None:
            existing.updated_at_ms = now
            if action_intent and not existing.action_intent:
                existing.action_intent = action_intent
            for item in evidence or []:
                if item not in existing.evidence:
                    existing.evidence.append(item)
            if dependency_ids:
                for dep in dependency_ids:
                    if dep and dep not in existing.dependency_ids:
                        existing.dependency_ids.append(dep)
            if metadata:
                existing.metadata.update(metadata)
            plan.updated_at_ms = now
            plan.last_active_ms = now
            if existing.action_intent and existing.action_intent not in plan.action_intents:
                plan.action_intents.append(existing.action_intent)
            self._recompute_progress(plan)
            return existing
        step = PlanStep(
            id=str(uuid.uuid4()),
            title=title,
            order=len(plan.steps),
            status=status,
            progress=1.0 if status == "done" else (0.45 if status == "doing" else 0.0),
            action_intent=action_intent or _infer_action_intent(title),
            dependency_ids=[dep for dep in (dependency_ids or []) if dep],
            blocker_note="",
            created_at_ms=now,
            updated_at_ms=now,
            completed_at_ms=now if status == "done" else 0,
            evidence=list(evidence or []),
            metadata=dict(metadata or {}),
        )
        plan.steps.append(step)
        if step.action_intent and step.action_intent not in plan.action_intents:
            plan.action_intents.append(step.action_intent)
        plan.updated_at_ms = now
        plan.last_active_ms = now
        self._recompute_progress(plan)
        return step

    def add_dependency(self, plan_id_prefix: str, step_id_prefix: str, dependency_step_id_prefix: str) -> Optional[PlanStep]:
        plan = self.get(plan_id_prefix)
        if plan is None:
            return None
        step = self._get_step(plan, step_id_prefix)
        dep = self._get_step(plan, dependency_step_id_prefix)
        if step is None or dep is None or step.id == dep.id:
            return None
        if dep.id not in step.dependency_ids:
            step.dependency_ids.append(dep.id)
        step.updated_at_ms = _now_ms()
        self._recompute_progress(plan)
        return step

    def set_status(self, plan_id_prefix: str, status: str, note: str = "") -> Optional[PlanEntry]:
        plan = self.get(plan_id_prefix)
        if plan is None:
            return None
        now = _now_ms()
        plan.status = status
        plan.updated_at_ms = now
        plan.last_active_ms = now
        if note:
            plan.evidence.append(note)
        if status == "done":
            plan.progress = 1.0
            for step in plan.steps:
                if step.status != "done":
                    step.status = "done"
                    step.progress = 1.0
                    step.updated_at_ms = now
                    step.completed_at_ms = now
        if status not in self.ACTIVE_STATUSES and self.focus_plan_id == plan.id:
            self.focus_plan_id = self._fallback_focus_id(excluding=plan.id)
        return plan

    def set_focus(self, plan_id_prefix: str) -> Optional[PlanEntry]:
        plan = self.get(plan_id_prefix)
        if plan is None:
            return None
        self.focus_plan_id = plan.id
        plan.focus_count += 1
        plan.last_active_ms = _now_ms()
        return plan

    def set_step_status(
        self,
        plan_id_prefix: str,
        step_id_prefix: str,
        status: str,
        progress: Optional[float] = None,
        note: str = "",
        blocker_note: str = "",
    ) -> Optional[PlanStep]:
        plan = self.get(plan_id_prefix)
        if plan is None:
            return None
        step = self._get_step(plan, step_id_prefix)
        if step is None:
            return None
        now = _now_ms()
        step.status = status
        step.updated_at_ms = now
        if progress is None:
            if status == "done":
                step.progress = 1.0
            elif status == "doing":
                step.progress = max(step.progress, 0.45)
            elif status == "blocked":
                step.progress = min(max(step.progress, 0.05), 0.75)
            else:
                step.progress = min(step.progress, 0.2)
        else:
            step.progress = round(max(0.0, min(1.0, progress)), 6)
        if status == "done":
            step.completed_at_ms = now
        if note and note not in step.evidence:
            step.evidence.append(note)
        if blocker_note:
            step.blocker_note = blocker_note
        elif status != "blocked":
            step.blocker_note = ""
        plan.updated_at_ms = now
        plan.last_active_ms = now
        if note and note not in plan.evidence:
            plan.evidence.append(note)
        self._recompute_progress(plan)
        return step

    def get(self, plan_id_prefix: str) -> Optional[PlanEntry]:
        needle = plan_id_prefix.strip().lower()
        if not needle:
            return None
        exact = next((entry for entry in self.entries if entry.id.lower() == needle), None)
        if exact is not None:
            return exact
        return next((entry for entry in self.entries if entry.id.lower().startswith(needle)), None)

    def latest(self, status: str | None = None, limit: int = 10) -> List[PlanEntry]:
        out = self.entries
        if status is not None:
            out = [entry for entry in out if entry.status == status]
        return sorted(out, key=lambda entry: (entry.status in self.ACTIVE_STATUSES, entry.priority, entry.updated_at_ms), reverse=True)[: max(0, limit)]

    def search(self, query: str, k: int = 5, statuses: Optional[Sequence[str]] = None) -> List[PlanEntry]:
        pool = self.entries
        if statuses is not None:
            status_set = set(statuses)
            pool = [entry for entry in pool if entry.status in status_set]
        if not query.strip():
            return self.latest(limit=k)
        qv = self.codec.vectorize(query)
        scored: List[tuple[float, PlanEntry]] = []
        for entry in pool:
            sim = _cosine(qv, entry.vector)
            focus_bonus = 0.12 if entry.id == self.focus_plan_id else 0.0
            score = 0.62 * sim + 0.18 * entry.priority + 0.14 * (1.0 - entry.progress) + focus_bonus
            scored.append((score, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[: max(0, k)]]

    def focused(self) -> Optional[PlanEntry]:
        if not self.focus_plan_id:
            return None
        return next((entry for entry in self.entries if entry.id == self.focus_plan_id), None)

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

    def queue(self, limit: int = 10) -> List[PlanEntry]:
        active = [entry for entry in self.entries if entry.status in self.ACTIVE_STATUSES]
        return sorted(active, key=lambda entry: (entry.id == self.focus_plan_id, entry.priority, 1.0 - entry.progress, entry.updated_at_ms), reverse=True)[: max(0, limit)]

    def executable_steps(self, limit: int = 8) -> List[tuple[PlanEntry, PlanStep]]:
        pairs: List[tuple[float, PlanEntry, PlanStep]] = []
        for plan in self.queue(limit=max(1, limit * 2)):
            for step in plan.steps:
                if step.status not in {"todo", "doing"}:
                    continue
                if not self._step_dependencies_satisfied(plan, step):
                    continue
                urgency = 0.60 * plan.priority + 0.25 * (1.0 - plan.progress) + (0.10 if plan.id == self.focus_plan_id else 0.0) + (0.05 if step.status == "doing" else 0.0)
                pairs.append((urgency, plan, step))
        pairs.sort(key=lambda item: item[0], reverse=True)
        return [(plan, step) for _, plan, step in pairs[: max(0, limit)]]

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

    def note_outcome(self, text: str) -> Optional[OutcomeUpdate]:
        outcome_kind = _outcome_kind(text)
        if not outcome_kind:
            return None
        candidate = self._best_outcome_match(text)
        if candidate is None:
            return None
        plan, step = candidate
        note = _clean_text(text)
        if outcome_kind == "done":
            updated_step = self.set_step_status(plan.id, step.id, status="done", progress=1.0, note=note)
            if updated_step is None:
                return None
            plan = self.get(plan.id) or plan
            status = plan.status
        elif outcome_kind == "blocked":
            updated_step = self.set_step_status(plan.id, step.id, status="blocked", note=note, blocker_note=note)
            if updated_step is None:
                return None
            plan = self.get(plan.id) or plan
            if plan.status not in {"done", "cancelled"}:
                plan.status = "blocked"
            status = "blocked"
        else:
            updated_step = self.set_step_status(plan.id, step.id, status="doing", progress=max(step.progress, 0.45), note=note)
            if updated_step is None:
                return None
            plan = self.get(plan.id) or plan
            if plan.status == "pending":
                plan.status = "active"
            status = "doing"
        return OutcomeUpdate(
            matched_plan_id=plan.id,
            matched_step_id=updated_step.id,
            matched_goal_id=plan.goal_id,
            outcome_kind=outcome_kind,
            status=status,
            progress=plan.progress,
            note=note,
        )

    def stats(self) -> Dict[str, object]:
        by_status: Dict[str, int] = {}
        total_steps = 0
        by_intent: Dict[str, int] = {}
        for entry in self.entries:
            by_status[entry.status] = by_status.get(entry.status, 0) + 1
            total_steps += len(entry.steps)
            for intent in entry.action_intents:
                if intent:
                    by_intent[intent] = by_intent.get(intent, 0) + 1
        return {
            "total": len(self.entries),
            "focus_plan_id": self.focus_plan_id,
            "total_steps": total_steps,
            "by_status": dict(sorted(by_status.items())),
            "by_intent": dict(sorted(by_intent.items())),
        }

    def status(self) -> PlanStatus:
        active = sum(1 for entry in self.entries if entry.status in self.ACTIVE_STATUSES)
        detail = "empty plan store"
        if self.entries:
            focused = self.focused()
            if focused is not None:
                detail = f"{active} active plans; focused on {focused.title}"
            else:
                detail = f"{active} active plans"
        return PlanStatus(path=str(self.path), total_entries=len(self.entries), active_entries=active, focused_plan_id=self.focus_plan_id, detail=detail)

    def _get_step(self, plan: PlanEntry, step_id_prefix: str) -> Optional[PlanStep]:
        needle = step_id_prefix.strip().lower()
        if not needle:
            return None
        exact = next((step for step in plan.steps if step.id.lower() == needle), None)
        if exact is not None:
            return exact
        return next((step for step in plan.steps if step.id.lower().startswith(needle)), None)

    def _step_dependencies_satisfied(self, plan: PlanEntry, step: PlanStep) -> bool:
        if not step.dependency_ids:
            return True
        done_ids = {candidate.id for candidate in plan.steps if candidate.status == "done"}
        return all(dep in done_ids for dep in step.dependency_ids)

    def _recompute_progress(self, plan: PlanEntry) -> None:
        if not plan.steps:
            plan.progress = 0.0
            if plan.status not in {"cancelled", "done"}:
                plan.status = "active"
            return
        total = len(plan.steps)
        done = sum(1 for step in plan.steps if step.status == "done")
        blocked = sum(1 for step in plan.steps if step.status == "blocked")
        plan.progress = round(sum(step.progress for step in plan.steps) / max(1, total), 6)
        if done == total:
            plan.status = "done"
        elif blocked and done < total:
            plan.status = "blocked"
        elif any(step.status == "doing" for step in plan.steps):
            plan.status = "active"
        else:
            plan.status = "active"
        plan.vector = self.codec.vectorize(plan.combined_text())

    def _fallback_focus_id(self, excluding: str = "") -> str:
        for entry in self.queue(limit=10):
            if entry.id != excluding:
                return entry.id
        return ""

    def _best_outcome_match(self, text: str) -> Optional[tuple[PlanEntry, PlanStep]]:
        lowered = text.lower()
        tokens = _tokens(lowered)
        best: tuple[float, PlanEntry, PlanStep] | None = None
        for plan in self.queue(limit=10):
            for step in plan.steps:
                if step.status in {"done", "cancelled"}:
                    continue
                overlap = _lexical_overlap(" ".join(tokens), step.title.lower())
                dep_penalty = 0.12 if step.dependency_ids and not self._step_dependencies_satisfied(plan, step) else 0.0
                focus_bonus = 0.08 if plan.id == self.focus_plan_id else 0.0
                doing_bonus = 0.06 if step.status == "doing" else 0.0
                score = overlap + focus_bonus + doing_bonus - dep_penalty
                if step.action_intent and step.action_intent in lowered:
                    score += 0.18
                if any(token in lowered for token in _tokens(plan.title)):
                    score += 0.08
                if best is None or score > best[0]:
                    best = (score, plan, step)
        if best is None or best[0] < 0.18:
            return None
        return best[1], best[2]


_ACTION_VERBS = [
    "build", "add", "fix", "test", "verify", "refactor", "wire", "document", "integrate", "train",
    "debug", "stabilize", "implement", "measure", "ship", "run", "design", "capture", "improve",
    "create", "update", "patch", "support", "harden", "tune", "profile", "promote", "sleep", "dream",
]


_OUTCOME_DONE = [
    "done", "completed", "finished", "fixed", "implemented", "wired", "added", "works now", "working now",
    "solved", "shipped", "passed", "merged", "talked", "started talking", "it works",
]
_OUTCOME_BLOCKED = [
    "blocked", "stuck", "failing", "failed", "broke", "broken", "can't", "cannot", "won't", "error", "issue",
    "problem", "doesn't work", "doesnt work", "crash", "crashed",
]
_OUTCOME_DOING = [
    "working on", "implementing", "building", "adding", "testing", "debugging", "tuning", "wiring", "trying",
    "progress", "in progress", "currently", "now on", "next up",
]


def _clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokens(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9][a-z0-9_\-']+", text.lower()) if len(token) > 2]


def _infer_action_intent(text: str) -> str:
    lowered = text.lower()
    for verb in _ACTION_VERBS:
        if re.search(rf"\b{re.escape(verb)}\b", lowered):
            return verb
    words = _tokens(lowered)
    return words[0] if words else ""


def _outcome_kind(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in _OUTCOME_BLOCKED):
        return "blocked"
    if any(token in lowered for token in _OUTCOME_DONE):
        return "done"
    if any(token in lowered for token in _OUTCOME_DOING):
        return "doing"
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
    aset = set(_tokens(a))
    bset = set(_tokens(b))
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / max(len(aset), len(bset))
