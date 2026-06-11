from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import shlex
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .agent import WonderBot
    from .planner import PlanEntry, PlanStep


@dataclass(slots=True)
class ActionResult:
    success: bool
    summary: str
    payload: Dict[str, Any] = field(default_factory=dict)
    status_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolSpec:
    name: str
    detail: str
    read_only: bool = False
    intents: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionRecord:
    id: str
    tool_name: str
    source: str
    dry_run: bool
    allowed: bool
    success: bool
    started_at_ms: int
    finished_at_ms: int
    summary: str
    args: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    plan_id: str = ""
    step_id: str = ""
    status_hint: str = ""
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionStatus:
    path: str
    total_runs: int
    successful_runs: int
    dry_runs: int
    last_tool_name: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionStore:
    def __init__(self, path: str = "state/action_runs.json") -> None:
        self.path = Path(path)
        self.runs: List[ExecutionRecord] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.runs = []
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            payloads = data
        else:
            payloads = data.get("runs", [])
        self.runs = [payload if isinstance(payload, ExecutionRecord) else ExecutionRecord(**payload) for payload in payloads if isinstance(payload, dict)]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"runs": [run.to_dict() for run in self.runs]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append(self, run: ExecutionRecord) -> None:
        self.runs.append(run)

    def latest(self, limit: int = 10) -> List[ExecutionRecord]:
        return list(sorted(self.runs, key=lambda item: item.finished_at_ms, reverse=True)[: max(0, limit)])

    def status(self) -> ExecutionStatus:
        successful = sum(1 for run in self.runs if run.success)
        dry_runs = sum(1 for run in self.runs if run.dry_run)
        last = self.latest(1)
        detail = "empty execution log"
        if last:
            detail = f"last: {last[0].tool_name} ({'dry-run' if last[0].dry_run else 'commit'})"
        return ExecutionStatus(
            path=str(self.path),
            total_runs=len(self.runs),
            successful_runs=successful,
            dry_runs=dry_runs,
            last_tool_name=last[0].tool_name if last else "",
            detail=detail,
        )


class ActionRegistry:
    def __init__(self, agent: 'WonderBot', path: str = "state/action_runs.json", default_dry_run: bool = True, auto_mark_step_doing: bool = True, auto_mark_done_on_success: bool = False) -> None:
        self.agent = agent
        self.store = ExecutionStore(path=path)
        self.default_dry_run = default_dry_run
        self.auto_mark_step_doing = auto_mark_step_doing
        self.auto_mark_done_on_success = auto_mark_done_on_success
        self._tools = self._build_specs()
        self._tool_map: Dict[str, ToolSpec] = {}
        for spec in self._tools:
            self._tool_map[spec.name] = spec
            for alias in spec.aliases:
                self._tool_map[alias] = spec

    def save(self) -> None:
        self.store.save()

    def status(self) -> ExecutionStatus:
        return self.store.status()

    def list_tools(self) -> List[ToolSpec]:
        return list(self._tools)

    def latest_runs(self, limit: int = 10) -> List[ExecutionRecord]:
        return self.store.latest(limit=limit)

    def execute_tool(self, tool_name: str, args: Optional[Dict[str, Any]] = None, *, dry_run: Optional[bool] = None, source: str = "manual", plan_id: str = "", step_id: str = "") -> ExecutionRecord:
        args = dict(args or {})
        spec = self.resolve_tool(tool_name)
        started = _now_ms()
        if spec is None:
            run = ExecutionRecord(
                id=str(uuid.uuid4()),
                tool_name=tool_name,
                source=source,
                dry_run=self.default_dry_run if dry_run is None else bool(dry_run),
                allowed=False,
                success=False,
                started_at_ms=started,
                finished_at_ms=_now_ms(),
                summary=f"unknown tool: {tool_name}",
                args=args,
                error="unknown-tool",
                plan_id=plan_id,
                step_id=step_id,
            )
            self._record(run)
            return run
        actual_dry_run = self.default_dry_run if dry_run is None else bool(dry_run)
        try:
            result = self._dispatch(spec.name, args=args, dry_run=actual_dry_run)
            run = ExecutionRecord(
                id=str(uuid.uuid4()),
                tool_name=spec.name,
                source=source,
                dry_run=actual_dry_run,
                allowed=True,
                success=result.success,
                started_at_ms=started,
                finished_at_ms=_now_ms(),
                summary=result.summary,
                args=args,
                result=result.payload,
                plan_id=plan_id,
                step_id=step_id,
                status_hint=result.status_hint,
            )
        except Exception as exc:  # pragma: no cover - defensive
            run = ExecutionRecord(
                id=str(uuid.uuid4()),
                tool_name=spec.name,
                source=source,
                dry_run=actual_dry_run,
                allowed=True,
                success=False,
                started_at_ms=started,
                finished_at_ms=_now_ms(),
                summary=f"tool {spec.name} failed: {exc}",
                args=args,
                error=str(exc),
                plan_id=plan_id,
                step_id=step_id,
            )
        self._record(run)
        return run

    def execute_plan_step(self, plan_id_prefix: str, step_id_prefix: str, *, dry_run: Optional[bool] = None, source: str = "plan-step") -> ExecutionRecord:
        plan = self.agent.plans.get(plan_id_prefix)
        if plan is None:
            return self.execute_tool("noop", {"reason": "no matching plan", "plan_id": plan_id_prefix, "step_id": step_id_prefix}, dry_run=True, source=source)
        step = self.agent.plans._get_step(plan, step_id_prefix)
        if step is None:
            return self.execute_tool("noop", {"reason": "no matching step", "plan_id": plan.id, "step_id": step_id_prefix}, dry_run=True, source=source, plan_id=plan.id)
        tool_name, inferred_args = self.resolve_step_tool(plan, step)
        if actual_dry_run(dry_run, self.default_dry_run) is False and self.auto_mark_step_doing and step.status == "todo":
            self.agent.set_plan_step_status(plan.id, step.id, status="doing", note="tool execution started")
        run = self.execute_tool(tool_name, inferred_args, dry_run=dry_run, source=source, plan_id=plan.id, step_id=step.id)
        if not run.dry_run:
            if run.success and (run.status_hint == "done" or (self.auto_mark_done_on_success and step.status not in {"done", "cancelled"})):
                self.agent.set_plan_step_status(plan.id, step.id, status="done", note=run.summary)
            elif not run.success:
                self.agent.set_plan_step_status(plan.id, step.id, status="blocked", note=run.summary, blocker_note=run.summary)
        return run

    def resolve_tool(self, name: str) -> Optional[ToolSpec]:
        return self._tool_map.get(name.strip().lower())

    def resolve_step_tool(self, plan: 'PlanEntry', step: 'PlanStep') -> tuple[str, Dict[str, Any]]:
        explicit = _extract_explicit_tool(step.title)
        if explicit:
            return explicit, _extract_explicit_args(step.title)
        intent = (step.action_intent or "").strip().lower()
        title = step.title.lower()
        if intent in {"sleep", "dream", "reflect", "consolidate", "diagnostics", "remember", "search", "speak", "watch", "sense"}:
            return _intent_to_tool(intent), _args_from_step_title(intent, step.title, plan)
        if "diagnostic" in title:
            return "diagnostics", {}
        if title.startswith("remember ") or title.startswith("search "):
            return "remember", {"query": step.title.split(" ", 1)[1]}
        if title.startswith("speak "):
            return "speak", {"text": step.title.split(" ", 1)[1]}
        if title.startswith("goal add "):
            return "goal_add", {"text": step.title.split(" ", 2)[2] if len(step.title.split(" ", 2)) > 2 else ""}
        if title.startswith("plan add "):
            return "plan_add", {"text": step.title.split(" ", 2)[2] if len(step.title.split(" ", 2)) > 2 else ""}
        return "noop", {"reason": f"manual step: {step.title}", "plan_title": plan.title, "step_title": step.title}

    def _record(self, run: ExecutionRecord) -> None:
        self.store.append(run)
        self.agent.replay.log("tool_run", run.to_dict())

    def _dispatch(self, tool_name: str, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        handler = getattr(self, f"_tool_{tool_name}", None)
        if handler is None:
            return ActionResult(success=False, summary=f"tool {tool_name} is not implemented", status_hint="blocked")
        return handler(args=args, dry_run=dry_run)

    def _build_specs(self) -> List[ToolSpec]:
        return [
            ToolSpec(name="noop", detail="Record a manual or unresolved action without mutating state.", read_only=True, intents=["manual"], aliases=["manual"]),
            ToolSpec(name="diagnostics", detail="Collect runtime diagnostics and state summary.", read_only=True, intents=["diagnostics", "diagnose"], aliases=["diag"]),
            ToolSpec(name="search_memory", detail="Search short-term memory.", read_only=True, intents=["search", "remember"], aliases=["search"]),
            ToolSpec(name="remember", detail="Search long-term memory.", read_only=True, intents=["remember", "search"], aliases=["ltm_search"]),
            ToolSpec(name="sense", detail="Poll live sensors once.", read_only=False, intents=["sense", "watch"], aliases=[]),
            ToolSpec(name="watch", detail="Advance live sensor/watch ticks.", read_only=False, intents=["watch", "sense"], aliases=[]),
            ToolSpec(name="consolidate", detail="Run memory consolidation.", read_only=False, intents=["consolidate"], aliases=[]),
            ToolSpec(name="reflect", detail="Run reflection only.", read_only=False, intents=["reflect"], aliases=[]),
            ToolSpec(name="sleep", detail="Run sleep promotion into long-term memory.", read_only=False, intents=["sleep", "promote"], aliases=[]),
            ToolSpec(name="dream", detail="Run dream/rehearsal synthesis.", read_only=False, intents=["dream"], aliases=[]),
            ToolSpec(name="speak", detail="Send text to the configured speaker.", read_only=False, intents=["speak", "say"], aliases=["say"]),
            ToolSpec(name="note", detail="Write a note into memory.", read_only=False, intents=["note", "capture"], aliases=["capture"]),
            ToolSpec(name="goal_add", detail="Create a new goal.", read_only=False, intents=["goal", "add"], aliases=[]),
            ToolSpec(name="goal_focus", detail="Focus an existing goal.", read_only=False, intents=["focus"], aliases=[]),
            ToolSpec(name="plan_add", detail="Create a new plan.", read_only=False, intents=["plan", "build"], aliases=[]),
            ToolSpec(name="plan_focus", detail="Focus an existing plan.", read_only=False, intents=["plan"], aliases=[]),
            ToolSpec(name="step_done", detail="Mark a plan step done.", read_only=False, intents=["done", "complete"], aliases=[]),
            ToolSpec(name="step_doing", detail="Mark a plan step doing.", read_only=False, intents=["doing", "work"], aliases=[]),
            ToolSpec(name="step_block", detail="Mark a plan step blocked.", read_only=False, intents=["block", "blocked"], aliases=[]),
        ]

    def _tool_noop(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        reason = str(args.get("reason", "manual action required"))
        return ActionResult(success=True, summary=f"no-op: {reason}", payload={"args": args})

    def _tool_diagnostics(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        diagnostics = self.agent.diagnostics()
        payload = {
            "backend": diagnostics.get("backend_name", ""),
            "memory_total": diagnostics.get("memory_stats", {}).get("total", 0),
            "goal_total": diagnostics.get("extra", {}).get("goals", {}).get("total_entries", 0),
            "plan_total": diagnostics.get("extra", {}).get("plans", {}).get("total_entries", 0),
            "action_total": diagnostics.get("extra", {}).get("actions", {}).get("total_runs", 0),
        }
        return ActionResult(success=True, summary="collected runtime diagnostics", payload=payload)

    def _tool_search_memory(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", 6) or 6)
        hits = self.agent.memory.search(query, k=limit) if query else self.agent.memory.top_memories(limit)
        payload = {"results": [item.text for item in hits]}
        summary = f"found {len(hits)} short-term memory hits"
        return ActionResult(success=True, summary=summary, payload=payload)

    def _tool_remember(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        query = str(args.get("query", "")).strip()
        limit = int(args.get("limit", 6) or 6)
        hits = self.agent.longterm.search(query, k=limit) if query else self.agent.longterm.latest(limit=limit)
        payload = {"results": [entry.text for entry in hits]}
        summary = f"found {len(hits)} long-term memory hits"
        return ActionResult(success=True, summary=summary, payload=payload)

    def _tool_sense(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        if dry_run:
            return ActionResult(success=True, summary="would poll live sensors once", payload={"ticks": 1})
        turns = self.agent.poll_sensors()
        payload = {"turns": [turn.to_dict() for turn in turns]}
        return ActionResult(success=True, summary=f"polled sensors and received {len(turns)} salient turns", payload=payload)

    def _tool_watch(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        ticks = max(1, int(args.get("ticks", 1) or 1))
        if dry_run:
            return ActionResult(success=True, summary=f"would advance {ticks} live ticks", payload={"ticks": ticks})
        turns = self.agent.idle_tick(ticks)
        payload = {"ticks": ticks, "turns": [turn.to_dict() for turn in turns]}
        return ActionResult(success=True, summary=f"advanced {ticks} ticks and produced {len(turns)} turns", payload=payload)

    def _tool_consolidate(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        if dry_run:
            return ActionResult(success=True, summary="would run memory consolidation", payload={})
        report = self.agent.consolidate(force=True)
        payload = report.to_dict()
        summary = report.summary or "ran memory consolidation"
        return ActionResult(success=True, summary=summary, payload=payload, status_hint="done")

    def _tool_reflect(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        if dry_run:
            return ActionResult(success=True, summary="would run reflection", payload={})
        report = self.agent.reflect(force=True)
        payload = report.to_dict()
        summary = report.reflection or "ran reflection"
        return ActionResult(success=True, summary=summary, payload=payload, status_hint="done")

    def _tool_sleep(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        if dry_run:
            return ActionResult(success=True, summary="would run sleep promotion", payload={})
        report = self.agent.sleep(force=True)
        payload = report.to_dict()
        summary = f"sleep promoted {len(report.promoted_texts)} entries"
        return ActionResult(success=True, summary=summary, payload=payload, status_hint="done")

    def _tool_dream(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        count = max(1, int(args.get("count", 1) or 1))
        if dry_run:
            return ActionResult(success=True, summary=f"would run {count} dream cycle(s)", payload={"count": count})
        combined: List[Dict[str, Any]] = []
        for _ in range(count):
            report = self.agent.dream(force=True)
            combined.append(report.to_dict())
        return ActionResult(success=True, summary=f"ran {count} dream cycle(s)", payload={"reports": combined}, status_hint="done")

    def _tool_speak(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        text = str(args.get("text", "")).strip()
        if not text:
            return ActionResult(success=False, summary="speak tool requires text", status_hint="blocked")
        if dry_run:
            return ActionResult(success=True, summary=f"would speak: {text}", payload={"text": text})
        if not self.agent.speaker.status().available:
            return ActionResult(success=False, summary="voice output is unavailable", status_hint="blocked")
        self.agent.speaker.say(text)
        self.agent.replay.log("tool_voice_output", {"text": text})
        return ActionResult(success=True, summary=f"spoke: {text}", payload={"text": text}, status_hint="done")

    def _tool_note(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        text = str(args.get("text", "")).strip()
        if not text:
            return ActionResult(success=False, summary="note tool requires text", status_hint="blocked")
        if dry_run:
            return ActionResult(success=True, summary=f"would store note: {text}", payload={"text": text})
        self.agent._store_memory(text, source="tool", metadata={"tool": "note"})
        return ActionResult(success=True, summary=f"stored note: {text}", payload={"text": text}, status_hint="done")

    def _tool_goal_add(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        text = str(args.get("text", "")).strip()
        if not text:
            return ActionResult(success=False, summary="goal_add requires text", status_hint="blocked")
        if dry_run:
            return ActionResult(success=True, summary=f"would add goal: {text}", payload={"text": text})
        entry = self.agent.add_goal(text)
        return ActionResult(success=True, summary=f"added goal {entry.id[:8]}: {entry.title}", payload={"goal": entry.to_dict()}, status_hint="done")

    def _tool_goal_focus(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        goal_id = str(args.get("goal_id", "")).strip()
        if not goal_id:
            return ActionResult(success=False, summary="goal_focus requires goal_id", status_hint="blocked")
        if dry_run:
            return ActionResult(success=True, summary=f"would focus goal {goal_id}", payload={"goal_id": goal_id})
        entry = self.agent.focus_goal(goal_id)
        if entry is None:
            return ActionResult(success=False, summary="no matching goal", status_hint="blocked")
        return ActionResult(success=True, summary=f"focused goal {entry.id[:8]}: {entry.title}", payload={"goal": entry.to_dict()}, status_hint="done")

    def _tool_plan_add(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        text = str(args.get("text", "")).strip()
        goal_id = str(args.get("goal_id", "")).strip()
        if not text:
            return ActionResult(success=False, summary="plan_add requires text", status_hint="blocked")
        if dry_run:
            return ActionResult(success=True, summary=f"would add plan: {text}", payload={"text": text, "goal_id": goal_id})
        entry = self.agent.add_plan(text, goal_id=goal_id)
        return ActionResult(success=True, summary=f"added plan {entry.id[:8]}: {entry.title}", payload={"plan": entry.to_dict()}, status_hint="done")

    def _tool_plan_focus(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        plan_id = str(args.get("plan_id", "")).strip()
        if not plan_id:
            return ActionResult(success=False, summary="plan_focus requires plan_id", status_hint="blocked")
        if dry_run:
            return ActionResult(success=True, summary=f"would focus plan {plan_id}", payload={"plan_id": plan_id})
        entry = self.agent.focus_plan(plan_id)
        if entry is None:
            return ActionResult(success=False, summary="no matching plan", status_hint="blocked")
        return ActionResult(success=True, summary=f"focused plan {entry.id[:8]}: {entry.title}", payload={"plan": entry.to_dict()}, status_hint="done")

    def _tool_step_done(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        plan_id = str(args.get("plan_id", "")).strip()
        step_id = str(args.get("step_id", "")).strip()
        note = str(args.get("note", "")).strip()
        if not plan_id or not step_id:
            return ActionResult(success=False, summary="step_done requires plan_id and step_id", status_hint="blocked")
        if dry_run:
            return ActionResult(success=True, summary=f"would mark {plan_id}/{step_id} done", payload={"plan_id": plan_id, "step_id": step_id})
        _, step = self.agent.set_plan_step_status(plan_id, step_id, status="done", note=note or "tool step_done")
        if step is None:
            return ActionResult(success=False, summary="no matching step", status_hint="blocked")
        return ActionResult(success=True, summary=f"marked step {step.id[:8]} done", payload={"step_id": step.id}, status_hint="done")

    def _tool_step_doing(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        plan_id = str(args.get("plan_id", "")).strip()
        step_id = str(args.get("step_id", "")).strip()
        note = str(args.get("note", "")).strip()
        if not plan_id or not step_id:
            return ActionResult(success=False, summary="step_doing requires plan_id and step_id", status_hint="blocked")
        if dry_run:
            return ActionResult(success=True, summary=f"would mark {plan_id}/{step_id} doing", payload={"plan_id": plan_id, "step_id": step_id})
        _, step = self.agent.set_plan_step_status(plan_id, step_id, status="doing", note=note or "tool step_doing")
        if step is None:
            return ActionResult(success=False, summary="no matching step", status_hint="blocked")
        return ActionResult(success=True, summary=f"marked step {step.id[:8]} doing", payload={"step_id": step.id}, status_hint="doing")

    def _tool_step_block(self, *, args: Dict[str, Any], dry_run: bool) -> ActionResult:
        plan_id = str(args.get("plan_id", "")).strip()
        step_id = str(args.get("step_id", "")).strip()
        note = str(args.get("note", "")).strip()
        if not plan_id or not step_id:
            return ActionResult(success=False, summary="step_block requires plan_id and step_id", status_hint="blocked")
        if dry_run:
            return ActionResult(success=True, summary=f"would mark {plan_id}/{step_id} blocked", payload={"plan_id": plan_id, "step_id": step_id})
        _, step = self.agent.set_plan_step_status(plan_id, step_id, status="blocked", note=note or "tool step_block", blocker_note=note or "tool step_block")
        if step is None:
            return ActionResult(success=False, summary="no matching step", status_hint="blocked")
        return ActionResult(success=True, summary=f"marked step {step.id[:8]} blocked", payload={"step_id": step.id}, status_hint="blocked")


def parse_kv_args(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for token in shlex.split(text):
        if "=" not in token:
            out.setdefault("_", []).append(token)
            continue
        key, value = token.split("=", 1)
        out[key.strip()] = _coerce_value(value.strip())
    return out


def actual_dry_run(value: Optional[bool], default: bool) -> bool:
    return default if value is None else bool(value)


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _intent_to_tool(intent: str) -> str:
    mapping = {
        "diagnostics": "diagnostics",
        "diagnose": "diagnostics",
        "search": "search_memory",
        "remember": "remember",
        "sleep": "sleep",
        "dream": "dream",
        "reflect": "reflect",
        "consolidate": "consolidate",
        "sense": "sense",
        "watch": "watch",
        "speak": "speak",
    }
    return mapping.get(intent, intent)


def _extract_explicit_tool(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered.startswith("tool:"):
        remainder = stripped.split(":", 1)[1].strip()
        return remainder.split(None, 1)[0].strip().lower()
    if lowered.startswith("act:"):
        remainder = stripped.split(":", 1)[1].strip()
        return remainder.split(None, 1)[0].strip().lower()
    return ""


def _extract_explicit_args(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if ":" not in stripped:
        return {}
    remainder = stripped.split(":", 1)[1].strip()
    parts = remainder.split(None, 1)
    if len(parts) == 1:
        return {}
    return parse_kv_args(parts[1])


def _args_from_step_title(intent: str, title: str, plan: 'PlanEntry') -> Dict[str, Any]:
    stripped = title.strip()
    lowered = stripped.lower()
    if intent in {"remember", "search"}:
        if lowered.startswith(intent + " "):
            return {"query": stripped[len(intent) + 1 :].strip()}
        return {"query": stripped}
    if intent == "speak":
        if lowered.startswith("speak "):
            return {"text": stripped[6:].strip()}
        return {"text": stripped}
    if intent == "watch":
        tokens = stripped.split()
        count = 1
        for token in tokens:
            if token.isdigit():
                count = max(1, int(token))
                break
        return {"ticks": count}
    return {"plan_id": plan.id, "step_title": title}


def _now_ms() -> int:
    return int(time.time() * 1000)
