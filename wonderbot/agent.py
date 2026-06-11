from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
import time
from typing import Dict, List, Optional

from .config import WonderBotConfig
from .consolidation import ConsolidationReport, MemoryConsolidator
from .diagnostics import collect_runtime_diagnostics
from .event_codec import EventCodec, SegmentEvent
from .ganglion import Ganglion
from .journal import JournalStore
from .goals import GoalEntry, GoalStore
from .planner import PlanEntry, PlanStore
from .execution import ActionRegistry, ExecutionRecord
from .lifecycle import MemoryLifecycle, SleepReport
from .llm_backends import create_backend
from .longterm import LongTermMemoryStore
from .memory import MemoryItem, MemoryStore
from .replay import ReplayLogger
from .resonance import ResonanceField
from .selfmodel import SelfModelStore
from .sensors import SensorHub, SensorObservation, build_sensor_hub
from .tts import build_speaker


@dataclass(slots=True)
class AgentTurn:
    stimulus: str
    response: Optional[str]
    resonance: float
    tick: int
    recalled: List[str]
    spontaneous: bool
    backend: str
    source: str = "user"
    salience: float = 0.0
    mode: str = "observe"
    inhibition_reason: str = ""
    backend_metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class FocusEntry:
    text: str
    source: str
    salience: float
    tick: int
    created_at_ms: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class FocusState:
    mode: str = "observe"
    active_focus: str = ""
    goal_anchor: str = ""
    last_updated_ms: int = 0
    recent: List[FocusEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "active_focus": self.active_focus,
            "goal_anchor": self.goal_anchor,
            "last_updated_ms": self.last_updated_ms,
            "recent": [entry.to_dict() for entry in self.recent],
        }


WonderBotConfigAlias = WonderBotConfig


class WonderBot:
    def __init__(self, config: WonderBotConfig, sensor_hub: SensorHub | None = None) -> None:
        self.config = config
        if not self.config.backend.hf_device:
            self.config.backend.hf_device = self.config.runtime.hf_llm_device or self.config.runtime.default_device
        if not self.config.backend.hf_device_map:
            self.config.backend.hf_device_map = self.config.runtime.hf_llm_device_map
        if (not self.config.backend.hf_torch_dtype or self.config.backend.hf_torch_dtype == 'auto') and self.config.runtime.hf_llm_torch_dtype:
            self.config.backend.hf_torch_dtype = self.config.runtime.hf_llm_torch_dtype
        if not self.config.backend.hf_offload_dir:
            self.config.backend.hf_offload_dir = self.config.runtime.offload_dir
        if not self.config.backend.hf_load_in_4bit and self.config.runtime.hf_llm_load_in_4bit:
            self.config.backend.hf_load_in_4bit = True
        if self.config.backend.hf_quant_type == 'nf4' and self.config.runtime.hf_llm_quant_type:
            self.config.backend.hf_quant_type = self.config.runtime.hf_llm_quant_type
        if self.config.backend.hf_compute_dtype == 'float16' and self.config.runtime.hf_llm_compute_dtype:
            self.config.backend.hf_compute_dtype = self.config.runtime.hf_llm_compute_dtype
        if self.config.backend.hf_double_quant is True and self.config.runtime.hf_llm_double_quant is not None:
            self.config.backend.hf_double_quant = self.config.runtime.hf_llm_double_quant
        if not self.config.tts.device:
            self.config.tts.device = self.config.runtime.tts_device or self.config.runtime.default_device
        if not self.config.tts.hf_device:
            self.config.tts.hf_device = self.config.tts.device
        self.codec = EventCodec(
            dim=config.codec.dim,
            ngram=config.codec.ngram,
            window_chars=config.codec.window_chars,
            min_segment_chars=config.codec.min_segment_chars,
            cosine_drop=config.codec.cosine_drop,
            lowercase=config.codec.lowercase,
            nfkc=config.codec.nfkc,
        )
        self.memory = MemoryStore(
            codec=self.codec,
            path=config.memory.path,
            max_active_items=config.memory.max_active_items,
            protect_identity=config.memory.protect_identity,
            importance_threshold=config.memory.importance_threshold,
            min_novelty=config.memory.min_novelty,
        )
        self.journal = JournalStore(path=config.journal.path)
        self.longterm = LongTermMemoryStore(codec=self.codec, path=config.longterm.path)
        self.self_model = SelfModelStore(codec=self.codec, path=config.selfmodel.path)
        self.goals = GoalStore(codec=self.codec, path=config.goals.path)
        self.plans = PlanStore(codec=self.codec, path=config.plans.path)
        self.ganglion = Ganglion(
            height=config.ganglion.height,
            width=config.ganglion.width,
            channels=config.ganglion.channels,
            bleed=config.ganglion.bleed,
        )
        self.resonance = ResonanceField(
            sigma=config.resonance.sigma,
            tau=config.resonance.tau,
            alpha=config.resonance.alpha,
            prime_count=config.resonance.prime_count,
        )
        self.backend = create_backend(config=config.backend, codec=self.codec)
        self.sensor_hub = sensor_hub if sensor_hub is not None else build_sensor_hub(config)
        self.replay = ReplayLogger(
            path=config.logging.path,
            enabled=config.logging.enabled,
            flush_each_write=config.logging.flush_each_write,
        )
        self.actions = ActionRegistry(
            agent=self,
            path=config.execution.path,
            default_dry_run=config.execution.default_dry_run,
            auto_mark_step_doing=config.execution.auto_mark_step_doing,
            auto_mark_done_on_success=config.execution.auto_mark_done_on_success,
        )
        self.consolidator = MemoryConsolidator(
            memory=self.memory,
            journal=self.journal,
            replay=self.replay,
            summary_min_items=config.consolidation.summary_min_items,
            summary_window_items=config.consolidation.summary_window_items,
            max_summary_sentences=config.consolidation.max_summary_sentences,
            task_limit=config.consolidation.task_limit,
            belief_limit=config.consolidation.belief_limit,
            thread_limit=config.consolidation.thread_limit,
            auto_every_explicit_turns=config.consolidation.auto_every_explicit_turns,
        )
        self.lifecycle = MemoryLifecycle(
            memory=self.memory,
            journal=self.journal,
            longterm=self.longterm,
            replay=self.replay,
            promotion_limit=config.sleep.promotion_limit,
            min_promotion_strength=config.sleep.min_promotion_strength,
            dream_limit=config.sleep.dream_limit,
            dream_similarity_min=config.sleep.dream_similarity_min,
            dream_similarity_max=config.sleep.dream_similarity_max,
            archive_decay_rate=config.sleep.archive_decay_rate,
            archive_below_strength=config.sleep.archive_below_strength,
            auto_sleep_every_explicit_turns=config.sleep.auto_sleep_every_explicit_turns,
        )
        self.speaker = build_speaker(config.tts)
        self.voice_enabled = config.tts.enabled and self.speaker.status().available
        self.last_turn: Optional[AgentTurn] = None
        self.last_consolidation_report: Optional[ConsolidationReport] = None
        self.last_sleep_report: Optional[SleepReport] = None
        self.focus_state = FocusState(last_updated_ms=_now_ms())
        self._seed_default_self_model()
        self._sync_goal_focus_anchor()
        self._idle_counter = 0
        self._last_response_ms = 0
        self._last_spontaneous_ms = 0
        self._last_sensor_response_ms = 0
        self._last_source_response_ms: Dict[str, int] = {}
        self._last_stimulus_seen_ms: Dict[str, int] = {}
        self._log_runtime_startup()

    def observe(self, stimulus: str, source: str = "user", explicit: bool = True) -> AgentTurn:
        return self._observe_common(
            stimulus=stimulus,
            source=source,
            explicit=explicit,
            source_salience=0.0,
            metadata=None,
        )

    def observe_sensor(self, observation: SensorObservation) -> AgentTurn:
        return self._observe_common(
            stimulus=observation.text,
            source=observation.source,
            explicit=False,
            source_salience=observation.salience,
            metadata={**observation.metadata, "sensor": True, "salience": observation.salience},
        )

    def poll_sensors(self) -> List[AgentTurn]:
        turns: List[AgentTurn] = []
        for observation in self.sensor_hub.poll():
            self.replay.log(
                "sensor_observation",
                {
                    "source": observation.source,
                    "text": observation.text,
                    "salience": observation.salience,
                    "metadata": observation.metadata,
                    "tick": self.ganglion.t,
                },
            )
            if observation.salience < self.config.live.sensor_memory_threshold:
                self.replay.log(
                    "sensor_dropped",
                    {
                        "source": observation.source,
                        "salience": observation.salience,
                        "threshold": self.config.live.sensor_memory_threshold,
                        "text": observation.text,
                    },
                )
                continue
            turn = self.observe_sensor(observation)
            turns.append(turn)
        return turns

    def idle_tick(self, count: int = 1) -> List[AgentTurn]:
        turns: List[AgentTurn] = []
        for _ in range(max(1, count)):
            self.ganglion.tick()
            sensor_turns = self.poll_sensors() if self.config.live.enabled else []
            turns.extend(sensor_turns)
            self._idle_counter += 1
            if self._idle_counter >= self.config.agent.spontaneous_interval:
                now = _now_ms()
                if self._live_speech_mode_active():
                    self.replay.log(
                        "spontaneous_suppressed",
                        {
                            "reason": "live speech mode active",
                            "tick": self.ganglion.t,
                        },
                    )
                    self._idle_counter = 0
                    continue
                if now - self._last_spontaneous_ms < int(self.config.stability.spontaneous_cooldown_seconds * 1000):
                    self.replay.log(
                        "spontaneous_suppressed",
                        {
                            "reason": "spontaneous cooldown active",
                            "tick": self.ganglion.t,
                        },
                    )
                    self._idle_counter = 0
                    continue
                memories = self._recall_context("")
                self.focus_state.mode = "reflect"
                result = self.backend.generate(
                    stimulus="",
                    memories=memories,
                    style=self.config.agent.response_style,
                    spontaneous=True,
                )
                if result.text.strip():
                    text = result.text.strip()
                    self._store_memory(text, source="assistant", metadata={"spontaneous": True, **result.metadata})
                    self._note_response(source="assistant", stimulus="", spontaneous=True)
                    self._update_focus(text=text, source="assistant", salience=0.0, explicit=False)
                    turn = AgentTurn(
                        stimulus="",
                        response=text,
                        resonance=0.0,
                        tick=self.ganglion.t,
                        recalled=[memory.text for memory in memories],
                        spontaneous=True,
                        backend=result.backend_name,
                        source="assistant",
                        salience=0.0,
                        mode="reflect",
                        backend_metadata=result.metadata,
                    )
                    turns.append(turn)
                    self.last_turn = turn
                    self._emit_voice(turn)
                    self.replay.log("spontaneous_response", turn.to_dict())
                self._idle_counter = 0
        return turns

    def consolidate(self, force: bool = True) -> ConsolidationReport:
        report = self.consolidator.consolidate(force=force, reflect_only=False)
        self.last_consolidation_report = report
        self.replay.log("consolidation", report.to_dict())
        return report

    def reflect(self, force: bool = True) -> ConsolidationReport:
        report = self.consolidator.consolidate(force=force, reflect_only=True)
        self.last_consolidation_report = report
        self.replay.log("reflection", report.to_dict())
        return report

    def sleep(self, force: bool = True) -> SleepReport:
        report = self.lifecycle.sleep(force=force)
        self.last_sleep_report = report
        return report

    def dream(self, force: bool = True) -> SleepReport:
        report = self.lifecycle.dream(force=force)
        self.last_sleep_report = report
        return report

    def add_goal(self, text: str, priority: float | None = None, focus: bool | None = None) -> GoalEntry:
        title, detail = _split_goal_text(text)
        entry = self.goals.add_goal(
            title=title,
            detail=detail,
            priority=priority if priority is not None else self.config.goals.default_priority,
            focus=self.config.goals.auto_focus_new_goal if focus is None else focus,
            evidence=[text],
            metadata={"captured": "manual"},
        )
        self._sync_goal_focus_anchor()
        self.replay.log("goal_add", entry.to_dict())
        return entry

    def set_goal_status(self, goal_id_prefix: str, status: str, progress: float | None = None, note: str = "") -> GoalEntry | None:
        entry = self.goals.set_status(goal_id_prefix, status=status, progress=progress, note=note)
        if entry is not None:
            self._sync_goal_focus_anchor()
            self.replay.log("goal_status", {"id": entry.id, "status": entry.status, "progress": entry.progress, "note": note})
        return entry

    def focus_goal(self, goal_id_prefix: str) -> GoalEntry | None:
        entry = self.goals.set_focus(goal_id_prefix)
        if entry is not None:
            self._sync_goal_focus_anchor()
            self.replay.log("goal_focus", {"id": entry.id, "title": entry.title})
        return entry

    def add_plan(self, text: str, goal_id: str = "", focus: bool | None = None, priority: float | None = None) -> PlanEntry:
        title, detail = _split_goal_text(text)
        linked_goal = self.goals.get(goal_id) if goal_id else self.goals.focused()
        entry = self.plans.add_plan(
            title=title,
            detail=detail,
            goal_id=linked_goal.id if linked_goal is not None else (goal_id or ""),
            priority=priority if priority is not None else (linked_goal.priority if linked_goal is not None else self.config.plans.default_priority),
            focus=True if focus is None else focus,
            action_intents=_extract_action_intents(text),
            evidence=[text],
            metadata={"captured": "manual"},
        )
        self._sync_goal_progress_from_plan(entry, reason="manual-plan-add")
        self._sync_goal_focus_anchor()
        self.replay.log("plan_add", entry.to_dict())
        return entry

    def add_plan_step(self, plan_id_prefix: str, text: str) -> tuple[PlanEntry | None, object | None]:
        plan = self.plans.get(plan_id_prefix)
        if plan is None:
            return None, None
        step = self.plans.add_step(plan.id, text, evidence=[text], metadata={"captured": "manual"})
        plan = self.plans.get(plan.id)
        if plan is not None:
            self._sync_goal_progress_from_plan(plan, reason="manual-step-add")
        if step is not None:
            self.replay.log("plan_step_add", {"plan_id": plan.id if plan else plan_id_prefix, "step_id": step.id, "title": step.title})
        return plan, step

    def set_plan_status(self, plan_id_prefix: str, status: str, note: str = "") -> PlanEntry | None:
        entry = self.plans.set_status(plan_id_prefix, status=status, note=note)
        if entry is not None:
            self._sync_goal_progress_from_plan(entry, reason="plan-status")
            self._sync_goal_focus_anchor()
            self.replay.log("plan_status", {"id": entry.id, "status": entry.status, "progress": entry.progress, "note": note})
        return entry

    def focus_plan(self, plan_id_prefix: str) -> PlanEntry | None:
        entry = self.plans.set_focus(plan_id_prefix)
        if entry is not None:
            self._sync_goal_focus_anchor()
            self.replay.log("plan_focus", {"id": entry.id, "title": entry.title})
        return entry

    def set_plan_step_status(self, plan_id_prefix: str, step_id_prefix: str, status: str, note: str = "", blocker_note: str = "") -> tuple[PlanEntry | None, object | None]:
        step = self.plans.set_step_status(plan_id_prefix, step_id_prefix, status=status, note=note, blocker_note=blocker_note)
        plan = self.plans.get(plan_id_prefix)
        if plan is not None:
            self._sync_goal_progress_from_plan(plan, reason="plan-step-status")
        if step is not None:
            self.replay.log("plan_step_status", {"plan_id": plan.id if plan else plan_id_prefix, "step_id": step.id, "status": step.status, "progress": step.progress, "note": note, "blocker_note": blocker_note})
        return plan, step

    def add_plan_dependency(self, plan_id_prefix: str, step_id_prefix: str, dependency_step_id_prefix: str) -> tuple[PlanEntry | None, object | None]:
        step = self.plans.add_dependency(plan_id_prefix, step_id_prefix, dependency_step_id_prefix)
        plan = self.plans.get(plan_id_prefix)
        if step is not None:
            self.replay.log("plan_step_dependency", {"plan_id": plan.id if plan else plan_id_prefix, "step_id": step.id, "dependency_ids": list(step.dependency_ids)})
        return plan, step

    def run_tool(self, tool_name: str, args: Optional[Dict[str, object]] = None, dry_run: Optional[bool] = None, source: str = "manual") -> ExecutionRecord:
        return self.actions.execute_tool(tool_name, dict(args or {}), dry_run=dry_run, source=source)

    def run_plan_step(self, plan_id_prefix: str, step_id_prefix: str, dry_run: Optional[bool] = None, source: str = "plan-step") -> ExecutionRecord:
        return self.actions.execute_plan_step(plan_id_prefix, step_id_prefix, dry_run=dry_run, source=source)

    def capture_self_statement(self, kind: str, text: str, source: str = "user", strength: float = 0.76) -> None:
        entry = self.self_model.add_or_reinforce(kind=kind, text=text, source=source, strength=strength, evidence=[text])
        self.replay.log("self_model_write", {"kind": kind, "text": entry.text, "strength": entry.strength})

    def save(self) -> None:
        self.memory.save()
        self.journal.save()
        self.longterm.save()
        self.self_model.save()
        self.goals.save()
        self.plans.save()
        self.actions.save()
        self.replay.log(
            "save",
            {
                "memory_path": self.config.memory.path,
                "journal_path": self.config.journal.path,
                "longterm_path": self.config.longterm.path,
                "self_model_path": self.config.selfmodel.path,
                "goals_path": self.config.goals.path,
                "plans_path": self.config.plans.path,
                "actions_path": self.config.execution.path,
                "tick": self.ganglion.t,
            },
        )

    def close(self) -> None:
        self.sensor_hub.close()
        self.save()
        self.speaker.close()
        self.replay.close()

    def diagnostics(self) -> Dict[str, object]:
        sensor_statuses = [asdict(status) for status in self.sensor_hub.status()]
        return collect_runtime_diagnostics(
            config=self.config,
            backend_name=getattr(self.backend, "name", type(self.backend).__name__),
            sensor_statuses=sensor_statuses,
            speaker_status=asdict(self.speaker.status()),
            replay_status=asdict(self.replay.status()),
            focus_state=self.focus_state.to_dict(),
            memory_stats=self.memory.stats(),
            extra={
                "journal": self.journal.status().to_dict(),
                "longterm": self.longterm.status().to_dict(),
                "self_model": self.self_model.status().to_dict(),
                "goals": self.goals.status().to_dict(),
                "plans": self.plans.status().to_dict(),
                "actions": self.actions.status().to_dict(),
                "consolidation": self.config.consolidation.__dict__ if hasattr(self.config.consolidation, "__dict__") else asdict(self.config.consolidation),
                "sleep": self.config.sleep.__dict__ if hasattr(self.config.sleep, "__dict__") else asdict(self.config.sleep),
            },
        )

    def set_voice_enabled(self, enabled: bool) -> bool:
        available = self.speaker.status().available
        self.voice_enabled = bool(enabled and available)
        self.replay.log("voice_toggle", {"enabled": self.voice_enabled, "available": available})
        return self.voice_enabled

    def state_summary(self) -> Dict[str, object]:
        return {
            "tick": self.ganglion.t,
            "ganglion": self.ganglion.state_summary().to_dict(),
            "memory": self.memory.stats(),
            "journal": {
                "status": self.journal.status().to_dict(),
                "stats": self.journal.stats(),
                "latest_summary": [entry.to_dict() for entry in self.journal.latest("summary", limit=1)],
                "latest_reflection": [entry.to_dict() for entry in self.journal.latest("reflection", limit=1)],
            },
            "longterm": {
                "status": self.longterm.status().to_dict(),
                "stats": self.longterm.stats(),
                "latest": [entry.to_dict() for entry in self.longterm.latest(limit=3)],
            },
            "self_model": {
                "status": self.self_model.status().to_dict(),
                "stats": self.self_model.stats(),
                "latest": [entry.to_dict() for entry in self.self_model.latest(limit=5)],
            },
            "goals": {
                "status": self.goals.status().to_dict(),
                "stats": self.goals.stats(),
                "focused": self.goals.focused().to_dict() if self.goals.focused() else None,
                "queue": [entry.to_dict() for entry in self.goals.queue(limit=5)],
            },
            "plans": {
                "status": self.plans.status().to_dict(),
                "stats": self.plans.stats(),
                "focused": self.plans.focused().to_dict() if self.plans.focused() else None,
                "queue": [entry.to_dict() for entry in self.plans.queue(limit=5)],
                "next_steps": [
                    {"plan_id": plan.id, "plan_title": plan.title, "step": step.to_dict()}
                    for plan, step in self.plans.executable_steps(limit=5)
                ],
            },
            "focus": self.focus_state.to_dict(),
            "actions": {
                "status": self.actions.status().to_dict(),
                "latest": [run.to_dict() for run in self.actions.latest_runs(limit=5)],
                "tools": [tool.to_dict() for tool in self.actions.list_tools()],
            },
            "voice": {
                "enabled": self.voice_enabled,
                "status": asdict(self.speaker.status()),
            },
            "replay": asdict(self.replay.status()),
            "sensors": [asdict(status) for status in self.sensor_hub.status()],
            "last_turn": self.last_turn.to_dict() if self.last_turn else None,
            "last_consolidation": self.last_consolidation_report.to_dict() if self.last_consolidation_report else None,
            "last_sleep": self.last_sleep_report.to_dict() if self.last_sleep_report else None,
        }

    def _observe_common(
        self,
        stimulus: str,
        source: str,
        explicit: bool,
        source_salience: float,
        metadata: Optional[Dict[str, object]],
    ) -> AgentTurn:
        stimulus = stimulus.strip()
        now = _now_ms()
        response_stimulus = self._effective_stimulus(stimulus, source=source, explicit=explicit, metadata=metadata)
        events = self.codec.analyze_text(response_stimulus)
        if stimulus and explicit and source == "user":
            self._capture_phase8_signals(stimulus)
            self._capture_phase9_outcomes(stimulus)
        memory_eligible = bool((metadata or {}).get("memory_eligible", False if (source == "microphone" and not (metadata or {}).get("transcript")) else True))
        if stimulus and memory_eligible:
            self._store_memory(stimulus, source=source, metadata={"explicit": explicit, **(metadata or {})})
        self._update_focus(text=response_stimulus or stimulus, source=source, salience=source_salience, explicit=explicit)
        resonance_value = self._ingest_events(events)
        drive = max(resonance_value, min(1.0, source_salience * self.config.live.sensor_reaction_gain))
        recalled_items = self._recall_context(response_stimulus, source=source, explicit=explicit, metadata=metadata)
        recalled = [item.text for item in recalled_items]
        microphone_has_transcript = bool((metadata or {}).get("transcript")) if source == "microphone" else True
        transcript_word_count = len(str((metadata or {}).get("transcript") or "").split()) if source == "microphone" else 0
        should_answer = explicit or (
            microphone_has_transcript
            and (
                (source == "microphone" and transcript_word_count >= self.config.microphone.transcript_reply_min_words)
                or drive >= self.config.live.sensor_reaction_threshold
                or self.resonance.should_react(
                    drive,
                    self.config.agent.reaction_threshold,
                    explicit=explicit,
                )
            )
        )
        response = None
        inhibition_reason = ""
        if source == "microphone" and not explicit and not microphone_has_transcript:
            _, inhibition_reason = self._response_allowed(
                source=source,
                explicit=explicit,
                source_salience=source_salience,
                stimulus=response_stimulus,
                now_ms=now,
                metadata=metadata,
            )
        backend_name = self.backend.name if hasattr(self.backend, "name") else type(self.backend).__name__
        backend_metadata: Dict[str, object] = {}
        self.focus_state.mode = "think" if should_answer else "observe"
        if should_answer:
            allowed, inhibition_reason = self._response_allowed(
                source=source,
                explicit=explicit,
                source_salience=source_salience,
                stimulus=response_stimulus,
                now_ms=now,
                metadata=metadata,
            )
            if allowed:
                result = self.backend.generate(
                    stimulus=response_stimulus,
                    memories=recalled_items,
                    style=self.config.agent.response_style,
                    spontaneous=False,
                )
                response = result.text.strip()
                backend_name = result.backend_name
                backend_metadata = dict(result.metadata)
                if response:
                    self._store_memory(
                        response,
                        source="assistant",
                        metadata={"stimulus": response_stimulus or stimulus, "source": source, **backend_metadata},
                    )
                    self._note_response(source=source, stimulus=stimulus, spontaneous=False)
                    self.focus_state.mode = "respond"
            else:
                self.focus_state.mode = "observe"
                self.replay.log(
                    "response_suppressed",
                    {
                        "source": source,
                        "stimulus": response_stimulus or stimulus,
                        "reason": inhibition_reason,
                        "salience": source_salience,
                        "tick": self.ganglion.t,
                    },
                )
        if source == "user":
            self.ganglion.tick()
            self._idle_counter = 0
        turn = AgentTurn(
            stimulus=stimulus,
            response=response,
            resonance=round(drive, 6),
            tick=self.ganglion.t,
            recalled=recalled,
            spontaneous=False,
            backend=backend_name,
            source=source,
            salience=round(source_salience, 6),
            mode=self.focus_state.mode,
            inhibition_reason=inhibition_reason,
            backend_metadata=backend_metadata,
        )
        self.last_turn = turn
        self.replay.log("turn", turn.to_dict())
        self._emit_voice(turn)
        if explicit and source == "user" and self.config.consolidation.auto_enabled:
            if self.consolidator.note_explicit_turn():
                report = self.consolidator.consolidate(force=False, reflect_only=False)
                self.last_consolidation_report = report
                self.replay.log("consolidation_auto", report.to_dict())
        if explicit and source == "user" and self.config.sleep.auto_enabled:
            if self.lifecycle.note_explicit_turn():
                sleep_report = self.lifecycle.sleep(force=False)
                self.last_sleep_report = sleep_report
                self.replay.log("sleep_auto", sleep_report.to_dict())
        return turn

    def _recall_context(self, stimulus: str, source: str = "user", explicit: bool = True, metadata: Optional[Dict[str, object]] = None) -> List[MemoryItem]:
        limit = self.config.agent.max_context_memories
        metadata = metadata or {}
        sensor_mode = (source in {"camera", "microphone"}) and not explicit
        has_transcript = bool(metadata.get("transcript"))
        working = self.memory.search(stimulus, k=max(1, limit // 2 + 1)) if stimulus else self.memory.top_memories(max(1, limit // 2 + 1))
        longterm = [entry.to_memory_item() for entry in (self.longterm.search(stimulus, k=max(1, limit // 3 + 1)) if stimulus else self.longterm.latest(limit=max(1, limit // 3 + 1)))]
        self_limit = 0 if sensor_mode else max(1, limit // 3)
        self_context = self.self_model.context_items(stimulus, limit=self_limit) if self_limit else []
        goal_context = self.goals.context_items(stimulus, limit=max(1, limit // 3)) if not (sensor_mode and not has_transcript) else []
        plan_context = self.plans.context_items(stimulus, limit=max(1, self.config.plans.context_limit)) if not (sensor_mode and not has_transcript) else []
        combined = list(goal_context) + list(plan_context) + list(self_context) + list(working) + longterm
        seen: set[str] = set()
        deduped: List[MemoryItem] = []
        for item in combined:
            if self._drop_context_item(item, sensor_mode=sensor_mode, has_transcript=has_transcript, spontaneous=(not stimulus.strip())):
                continue
            key = item.signature if hasattr(item, "signature") else item.text.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        if stimulus.strip():
            stimulus_lower = stimulus.lower()
            deduped.sort(
                key=lambda item: (
                    _goal_alignment_score(item, self.goals.focused(), stimulus_lower) + _plan_alignment_score(item, self.plans.focused(), stimulus_lower),
                    getattr(item, "priority", 0.0),
                    getattr(item, "importance", 0.0),
                    item.created_at_ms,
                ),
                reverse=True,
            )
        else:
            deduped.sort(key=lambda item: (getattr(item, "priority", 0.0), getattr(item, "importance", 0.0), item.created_at_ms), reverse=True)
        return deduped[:limit]

    def _effective_stimulus(self, stimulus: str, source: str, explicit: bool, metadata: Optional[Dict[str, object]]) -> str:
        if explicit:
            return stimulus
        metadata = metadata or {}
        if source == "microphone":
            transcript = str(metadata.get("transcript") or "").strip()
            if transcript:
                return transcript
        return stimulus

    def _drop_context_item(self, item: MemoryItem, sensor_mode: bool, has_transcript: bool, spontaneous: bool) -> bool:
        source = getattr(item, "source", "")
        text = (getattr(item, "text", "") or "").lower()
        metadata = getattr(item, "metadata", {}) or {}
        if source == 'microphone' and ('stt: sound only' in text or str(metadata.get('stt_state') or '') == 'sound-only'):
            return True
        if source.startswith("self/") or text.startswith("self/"):
            if "identity" in source or "style" in source or text.startswith("self/identity") or text.startswith("self/style"):
                return True if (sensor_mode or spontaneous) else False
            if sensor_mode and not has_transcript:
                return True
        if sensor_mode and not has_transcript and (source.startswith("goal/") or source.startswith("plan/")):
            return True
        if sensor_mode and not has_transcript and source == "assistant":
            return True
        return False

    def _store_memory(self, text: str, source: str, metadata: Dict[str, object]) -> None:
        goal_hits = self.goals.note_evidence(text, query=text)
        plan_hits = self.plans.note_evidence(text, query=text)
        merged_metadata = dict(metadata)
        if goal_hits:
            merged_metadata.setdefault("goal_ids", goal_hits)
        if plan_hits:
            merged_metadata.setdefault("plan_ids", plan_hits)
        item = self.memory.add(text, source=source, metadata=merged_metadata)
        self.replay.log(
            "memory_write",
            {
                "id": item.id,
                "source": source,
                "text": item.text,
                "priority": item.priority,
                "importance": item.importance,
                "novelty": item.novelty,
                "metadata": merged_metadata,
            },
        )

    def _seed_default_self_model(self) -> None:
        self.self_model.add_or_reinforce("identity", f"my name is {self.config.agent.name}", source="system", strength=0.92, evidence=["config.agent.name"], metadata={"seeded": True})
        self.self_model.add_or_reinforce("style", f"default response style: {self.config.agent.response_style}", source="system", strength=0.72, evidence=["config.agent.response_style"], metadata={"seeded": True})

    def _sync_goal_focus_anchor(self) -> None:
        focused = self.goals.focused()
        if focused is not None and focused.status in self.goals.ACTIVE_STATUSES:
            self.focus_state.goal_anchor = focused.title
            return
        focused_plan = self.plans.focused()
        if focused_plan is not None and focused_plan.status in self.plans.ACTIVE_STATUSES:
            self.focus_state.goal_anchor = focused_plan.title

    def _capture_phase8_signals(self, stimulus: str) -> None:
        for text in _extract_identity_statements(stimulus):
            if self.config.selfmodel.auto_capture_identity:
                self.capture_self_statement("identity", text, source="user", strength=0.88)
        for text in _extract_preference_statements(stimulus):
            if self.config.selfmodel.auto_capture_preferences:
                self.capture_self_statement("preference", text, source="user", strength=0.84)
        for text in _extract_constraint_statements(stimulus):
            if self.config.selfmodel.auto_capture_constraints:
                self.capture_self_statement("constraint", text, source="user", strength=0.82)
        captured_goal = None
        if self.config.goals.auto_capture_goals:
            for goal_text in _extract_goal_statements(stimulus):
                entry = self.goals.add_goal(
                    title=_split_goal_text(goal_text)[0],
                    detail=_split_goal_text(goal_text)[1],
                    priority=_goal_priority_from_text(goal_text, self.config.goals.default_priority),
                    focus=self.config.goals.auto_focus_new_goal,
                    evidence=[stimulus],
                    metadata={"captured": "auto"},
                )
                captured_goal = entry
                self.replay.log("goal_capture", {"id": entry.id, "title": entry.title, "detail": entry.detail})
        if self.config.plans.auto_capture_plans:
            step_texts = _extract_plan_steps(stimulus)
            should_capture_plan = bool(step_texts) or bool(_extract_plan_title(stimulus))
            if should_capture_plan:
                focused_goal = captured_goal or self.goals.focused()
                plan_title = _extract_plan_title(stimulus) or (focused_goal.title if focused_goal is not None else _split_goal_text(stimulus)[0] or "Execution plan")
                plan = self.plans.add_plan(
                    title=plan_title,
                    detail="",
                    goal_id=focused_goal.id if focused_goal is not None else "",
                    priority=(focused_goal.priority if focused_goal is not None else self.config.plans.default_priority),
                    focus=True,
                    action_intents=_extract_action_intents(stimulus),
                    evidence=[stimulus],
                    metadata={"captured": "auto"},
                )
                if self.config.plans.auto_capture_steps:
                    for step_text in step_texts:
                        step = self.plans.add_step(plan.id, step_text, evidence=[stimulus], metadata={"captured": "auto"})
                        if step is not None:
                            self.replay.log("plan_step_capture", {"plan_id": plan.id, "step_id": step.id, "title": step.title})
                self._sync_goal_progress_from_plan(plan, reason="auto-plan-capture")
                self.replay.log("plan_capture", {"id": plan.id, "title": plan.title, "goal_id": plan.goal_id, "step_count": len(plan.steps)})
        self._sync_goal_focus_anchor()

    def _capture_phase9_outcomes(self, stimulus: str) -> None:
        update = self.plans.note_outcome(stimulus)
        if update is None:
            return
        plan = self.plans.get(update.matched_plan_id)
        if plan is not None:
            self._sync_goal_progress_from_plan(plan, reason=f"outcome:{update.outcome_kind}")
        self._sync_goal_focus_anchor()
        self.replay.log("plan_outcome", update.to_dict())

    def _sync_goal_progress_from_plan(self, plan: PlanEntry, reason: str = "") -> None:
        if not self.config.plans.auto_update_goal_progress or not plan.goal_id:
            return
        goal = self.goals.get(plan.goal_id)
        if goal is None:
            return
        note = f"plan:{plan.title} [{reason}] progress={plan.progress:.2f}" if reason else f"plan:{plan.title} progress={plan.progress:.2f}"
        if plan.status == "done":
            self.goals.set_status(goal.id, status="done", progress=1.0, note=note)
        elif plan.status == "blocked":
            self.goals.set_status(goal.id, status="blocked", progress=max(goal.progress, min(plan.progress, 0.95)), note=note)
        else:
            updated = self.goals.set_status(goal.id, status="active", progress=max(goal.progress, min(plan.progress, 0.99)), note=note)
            if updated is None and plan.progress > goal.progress:
                goal.progress = max(goal.progress, min(plan.progress, 0.99))
                goal.updated_at_ms = _now_ms()
                goal.last_active_ms = goal.updated_at_ms
        self._sync_goal_focus_anchor()

    def _note_response(self, source: str, stimulus: str, spontaneous: bool) -> None:
        now = _now_ms()
        self._last_response_ms = now
        self._last_source_response_ms[source] = now
        if not spontaneous and source != "user":
            self._last_sensor_response_ms = now
        if spontaneous:
            self._last_spontaneous_ms = now
        normalized = _normalize_key(stimulus)
        if normalized:
            self._last_stimulus_seen_ms[normalized] = now

    def _update_focus(self, text: str, source: str, salience: float, explicit: bool) -> None:
        focus_text = _extract_focus(text)
        if not focus_text:
            return
        now = _now_ms()
        self.focus_state.last_updated_ms = now
        if explicit and source == "user":
            self.focus_state.goal_anchor = focus_text
        self.focus_state.active_focus = focus_text
        self.focus_state.recent.append(
            FocusEntry(
                text=focus_text,
                source=source,
                salience=round(max(salience, 0.05 if explicit else salience), 6),
                tick=self.ganglion.t,
                created_at_ms=now,
            )
        )
        self._trim_focus_entries(now)
        self._sync_goal_focus_anchor()

    def _trim_focus_entries(self, now_ms: int) -> None:
        decay_ms = int(self.config.agent.focus_decay_seconds * 1000)
        kept = [entry for entry in self.focus_state.recent if now_ms - entry.created_at_ms <= decay_ms]
        self.focus_state.recent = kept[-self.config.agent.focus_max_items :]

    def _response_allowed(self, source: str, explicit: bool, source_salience: float, stimulus: str, now_ms: int, metadata: Optional[Dict[str, object]] = None) -> tuple[bool, str]:
        if explicit:
            return True, ""
        metadata = metadata or {}
        transcript = str(metadata.get('transcript') or '').strip()
        transcript_ready = source == 'microphone' and bool(transcript)
        if source == "microphone" and not transcript_ready:
            state = str(metadata.get("stt_state") or "sound-only")
            if state == "transcript-rejected":
                return False, "microphone heard speech-like audio but no transcript was accepted"
            if state in {"speech-attempted", "accumulating"}:
                return False, "microphone is still evaluating speech; waiting for transcript"
            return False, "sound-only mic event; waiting for transcript"
        if (not transcript_ready) and source_salience < self.config.stability.minimum_response_salience:
            return False, "salience stayed below the minimum response threshold"
        if now_ms - self._last_sensor_response_ms < int(self.config.stability.sensor_response_cooldown_seconds * 1000):
            return False, "global sensor response cooldown active"
        last_for_source = self._last_source_response_ms.get(source, 0)
        if now_ms - last_for_source < int(self.config.stability.same_source_cooldown_seconds * 1000):
            return False, f"{source} cooldown active"
        normalized = _normalize_key(stimulus)
        if normalized:
            last_seen = self._last_stimulus_seen_ms.get(normalized, 0)
            if now_ms - last_seen < int(self.config.stability.repeated_stimulus_cooldown_seconds * 1000):
                return False, "repeated stimulus cooldown active"
        return True, ""

    def _live_speech_mode_active(self) -> bool:
        if not self.config.live.enabled:
            return False
        for status in self.sensor_hub.status():
            if status.source == 'microphone' and status.enabled and status.available:
                return True
        return False

    def _emit_voice(self, turn: AgentTurn) -> None:
        if not self.voice_enabled or not turn.response:
            return
        if turn.spontaneous and not self.config.tts.speak_spontaneous:
            return
        if turn.source in {"camera", "microphone"} and not self.config.tts.speak_sensor_responses:
            return
        if turn.source == "user" and not self.config.tts.speak_user_responses:
            return
        self.speaker.say(turn.response)
        self.replay.log("voice_output", {"text": turn.response, "source": turn.source, "spontaneous": turn.spontaneous})

    def _ingest_events(self, events: List[SegmentEvent]) -> float:
        if not events:
            return 0.0
        for event in events:
            self.ganglion.write_signature(event.signature)
        return self.resonance.score_many([event.signature for event in events], tick=self.ganglion.t)

    def _log_runtime_startup(self) -> None:
        self.replay.log(
            "startup",
            {
                "backend": getattr(self.backend, "name", type(self.backend).__name__),
                "diagnostics": self.diagnostics(),
            },
        )


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "do", "does", "for", "from", "get", "got", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "just", "like", "more", "my", "now", "of", "on", "or", "our", "so", "stuff", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "up", "use", "was", "we", "what", "when", "where", "which", "with", "would", "you", "your", "fren"
}


def _extract_focus(text: str) -> str:
    raw = " ".join(text.strip().split())
    if not raw:
        return ""
    lowered = raw.lower()
    patterns = [
        r"\b(?:about|regarding|focused on|centered on|working on|for)\s+([^,.!?]{4,90})",
        r"\b(?:design|build|make|wire|stabilize|improve|tune)\s+([^,.!?]{4,90})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            phrase = match.group(1).strip(" .,:;!?")
            if phrase:
                return phrase[:96]
    quoted = re.findall(r'"([^"]{3,96})"', raw)
    if quoted:
        return quoted[0].strip()
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-']+", raw)
    content = [token for token in tokens if token.lower() not in _STOPWORDS and len(token) > 2]
    if not content:
        return raw[:96]
    return " ".join(content[:6])[:96]


def _normalize_key(text: str) -> str:
    return " ".join(text.lower().split())


def _extract_identity_statements(text: str) -> List[str]:
    lowered = text.lower()
    out: List[str] = []
    patterns = [
        r"\bmy name is\s+([^,.!?]{2,64})",
        r"\bcall me\s+([^,.!?]{2,64})",
        r"\bi am\s+([^,.!?]{3,96})",
        r"\bi'm\s+([^,.!?]{3,96})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            phrase = _truncate_clause(match.group(0).strip(" .,:;!?"))
            if phrase and phrase not in out:
                out.append(phrase)
    return out[:3]


def _extract_preference_statements(text: str) -> List[str]:
    lowered = text.lower()
    out: List[str] = []
    patterns = [
        r"\bi prefer\s+([^,.!?]{4,120})",
        r"\bplease keep\s+([^,.!?]{4,120})",
        r"\bi want\s+([^,.!?]{4,120})",
        r"\bi like\s+([^,.!?]{4,120})",
        r"\bi'd like\s+([^,.!?]{4,120})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            phrase = _truncate_clause(match.group(0).strip(" .,:;!?"))
            if phrase and phrase not in out:
                out.append(phrase)
    return out[:4]


def _extract_constraint_statements(text: str) -> List[str]:
    lowered = text.lower()
    out: List[str] = []
    patterns = [
        r"\bdon't\s+([^,.!?]{4,120})",
        r"\bdo not\s+([^,.!?]{4,120})",
        r"\bavoid\s+([^,.!?]{4,120})",
        r"\bmust\s+([^,.!?]{4,120})",
        r"\bneed to\s+([^,.!?]{4,120})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, lowered):
            phrase = _truncate_clause(match.group(0).strip(" .,:;!?"))
            if phrase and phrase not in out:
                out.append(phrase)
    return out[:4]


def _extract_goal_statements(text: str) -> List[str]:
    lowered = text.lower().strip()
    out: List[str] = []
    leading_patterns = [
        r"^(?:next we should|we should|let's|lets|please|goal is to|the goal is to|we need to|need to)\s+(.+)$",
        r"^(?:phase\s+\d+\s*[:\-]?\s*)(.+)$",
    ]
    for pattern in leading_patterns:
        match = re.search(pattern, lowered)
        if match:
            phrase = _truncate_clause(match.group(1).strip(" .,:;!?"))
            if phrase:
                out.append(phrase)
    for match in re.finditer(r"\b(?:next we should|we should|let's|lets|goal is to|we need to|need to)\s+([^,.!?]{6,140})", lowered):
        phrase = _truncate_clause(match.group(1).strip(" .,:;!?"))
        if phrase and phrase not in out:
            out.append(phrase)
    filtered = []
    for phrase in out:
        if len(phrase.split()) >= 2:
            filtered.append(phrase)
    return filtered[:4]


def _split_goal_text(text: str) -> tuple[str, str]:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "", ""
    for sep in [":", " - ", " — ", ";"]:
        if sep in cleaned:
            head, tail = cleaned.split(sep, 1)
            return head.strip().capitalize(), tail.strip()
    words = cleaned.split()
    if len(words) <= 8:
        return cleaned.capitalize(), ""
    return " ".join(words[:8]).capitalize(), " ".join(words[8:])




def _truncate_clause(text: str) -> str:
    lowered = text.lower()
    separators = [" and next we should ", " and we should ", " but ", " while ", " so that "]
    cut = len(text)
    for sep in separators:
        idx = lowered.find(sep)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].strip(" .,:;!?")


def _extract_action_intents(text: str) -> List[str]:
    lowered = text.lower()
    verbs = [
        "build", "add", "fix", "test", "verify", "refactor", "wire", "document", "integrate", "train",
        "debug", "stabilize", "implement", "measure", "ship", "run", "design", "capture", "improve",
        "create", "update", "patch", "support", "harden", "tune", "profile", "promote", "sleep", "dream",
    ]
    out: List[str] = []
    for verb in verbs:
        if re.search(rf"\b{re.escape(verb)}\b", lowered):
            out.append(verb)
    return out[:6]


def _extract_plan_title(text: str) -> str:
    lowered = text.lower().strip()
    patterns = [
        r"\bplan(?: is)?(?: to)?\s+([^,.!?]{6,140})",
        r"\broadmap(?: is)?(?: to)?\s+([^,.!?]{6,140})",
        r"\bexecution plan(?: is)?(?: to)?\s+([^,.!?]{6,140})",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return _split_goal_text(_truncate_clause(match.group(1)))[0]
    if any(token in lowered for token in ["first ", "then ", "finally ", "next "]):
        return _split_goal_text(_truncate_clause(text))[0]
    return ""


def _extract_plan_steps(text: str) -> List[str]:
    lowered = text.lower()
    clauses: List[str] = []
    if any(token in lowered for token in ["first ", "then ", "finally ", "after that", "next "]):
        pattern = r"(?:first|then|next|after that|finally)\s+([^,.!?;]{4,120})"
        for match in re.finditer(pattern, lowered):
            clause = _truncate_clause(match.group(1).strip(" .,:;!?"))
            if clause and clause not in clauses:
                clauses.append(clause)
    if not clauses and ";" in text:
        for part in text.split(";"):
            clause = _truncate_clause(part.strip(" .,:;!?"))
            if len(clause.split()) >= 2:
                clauses.append(clause.lower())
    filtered: List[str] = []
    for clause in clauses:
        title = _split_goal_text(clause)[0]
        if title and title not in filtered:
            filtered.append(title)
    return filtered[:8]


def _goal_priority_from_text(text: str, default: float = 0.68) -> float:
    lowered = text.lower()
    priority = default
    if any(token in lowered for token in ["must", "critical", "urgent", "need to"]):
        priority += 0.16
    if any(token in lowered for token in ["maybe", "later", "eventually"]):
        priority -= 0.12
    return max(0.1, min(0.98, priority))


def _plan_alignment_score(item: MemoryItem, focused_plan: PlanEntry | None, stimulus_lower: str) -> float:
    score = 0.0
    source = getattr(item, "source", "")
    if source.startswith("plan/"):
        score += 0.48
    if focused_plan is not None:
        plan_text = focused_plan.combined_text().lower()
        overlap = _token_overlap(plan_text, (item.text or "").lower())
        score += 0.40 * overlap
        score += 0.18 * _token_overlap(plan_text, stimulus_lower)
        if item.metadata.get("plan_id") == focused_plan.id:
            score += 0.32
        plan_ids = item.metadata.get("plan_ids") or []
        if focused_plan.id in plan_ids:
            score += 0.24
        action_intents = item.metadata.get("action_intents") or []
        for intent in focused_plan.action_intents:
            if intent in stimulus_lower:
                score += 0.08
                break
        for intent in action_intents:
            if intent in stimulus_lower:
                score += 0.05
                break
    return round(score, 6)


def _goal_alignment_score(item: MemoryItem, focused_goal: GoalEntry | None, stimulus_lower: str) -> float:
    score = 0.0
    source = getattr(item, "source", "")
    if source.startswith("goal/"):
        score += 0.45
    if source.startswith("self/"):
        score += 0.24
    if focused_goal is not None:
        goal_text = focused_goal.combined_text().lower()
        overlap = _token_overlap(goal_text, (item.text or "").lower())
        score += 0.38 * overlap
        score += 0.18 * _token_overlap(goal_text, stimulus_lower)
        if item.metadata.get("goal_id") == focused_goal.id:
            score += 0.30
        goal_ids = item.metadata.get("goal_ids") or []
        if focused_goal.id in goal_ids:
            score += 0.24
    return round(score, 6)


def _token_overlap(a: str, b: str) -> float:
    aset = {token for token in re.findall(r"[a-z0-9][a-z0-9_\-']+", a.lower()) if token not in _STOPWORDS and len(token) > 2}
    bset = {token for token in re.findall(r"[a-z0-9][a-z0-9_\-']+", b.lower()) if token not in _STOPWORDS and len(token) > 2}
    if not aset or not bset:
        return 0.0
    return len(aset & bset) / max(len(aset), len(bset))


def _now_ms() -> int:
    return int(time.time() * 1000)


WonderBotConfig = WonderBotConfigAlias
