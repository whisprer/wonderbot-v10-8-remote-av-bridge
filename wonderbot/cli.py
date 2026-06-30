from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import warnings
from difflib import SequenceMatcher
from pathlib import Path

from .agent import AgentTurn, WonderBot
from .config import WonderBotConfig
from .execution import parse_kv_args
from .sensors.emotion_lite import (
    apply_multimodal_affect_to_text_and_metadata,
    extract_visual_affect_context,
)


class _DropKnownWhisperNoise(logging.Filter):
    _NEEDLES = (
        "return_token_timestamps is deprecated for WhisperFeatureExtractor",
        "Using custom `forced_decoder_ids` from the (generation) config",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(needle in message for needle in self._NEEDLES)


def _suppress_known_whisper_noise() -> None:
    """Suppress known harmless Whisper/Transformers warning spam.

    These messages are emitted during Whisper STT generation and clutter live-lite
    output. They do not indicate a bridge/STT failure.
    """
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    warnings.filterwarnings(
        "ignore",
        message=r".*return_token_timestamps.*WhisperFeatureExtractor.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*forced_decoder_ids.*deprecated.*task.*language.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The attention mask is not set.*",
    )

    filt = _DropKnownWhisperNoise()
    for logger_name in (
        "",
        "py.warnings",
        "transformers",
        "transformers.generation",
        "transformers.generation.utils",
        "transformers.models.whisper",
        "transformers.models.whisper.generation_whisper",
        "transformers.models.whisper.feature_extraction_whisper",
    ):
        logger = logging.getLogger(logger_name)
        if not any(isinstance(existing, _DropKnownWhisperNoise) for existing in logger.filters):
            logger.addFilter(filt)

    try:
        from transformers.utils import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except Exception:
        pass



_MULTIMODAL_AFFECT_ENABLED = True
_MULTIMODAL_AFFECT_VISUAL_TTL_SECONDS = 8.0
_MULTIMODAL_AFFECT_MIN_CONFIDENCE = 0.34
_MULTIMODAL_AFFECT_APPEND_TO_TEXT = True


def _configure_multimodal_affect_runtime(config: object | None) -> None:
    """Load multimodal affect fusion runtime knobs from config if present."""

    global _MULTIMODAL_AFFECT_ENABLED
    global _MULTIMODAL_AFFECT_VISUAL_TTL_SECONDS
    global _MULTIMODAL_AFFECT_MIN_CONFIDENCE
    global _MULTIMODAL_AFFECT_APPEND_TO_TEXT

    if config is None:
        return

    _MULTIMODAL_AFFECT_ENABLED = bool(getattr(config, "enabled", _MULTIMODAL_AFFECT_ENABLED))
    _MULTIMODAL_AFFECT_VISUAL_TTL_SECONDS = max(
        0.5,
        float(getattr(config, "visual_context_ttl_seconds", _MULTIMODAL_AFFECT_VISUAL_TTL_SECONDS)),
    )
    _MULTIMODAL_AFFECT_MIN_CONFIDENCE = max(
        0.0,
        min(1.0, float(getattr(config, "min_confidence", _MULTIMODAL_AFFECT_MIN_CONFIDENCE))),
    )
    _MULTIMODAL_AFFECT_APPEND_TO_TEXT = bool(
        getattr(config, "append_to_text", _MULTIMODAL_AFFECT_APPEND_TO_TEXT)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run WonderBot interactive CLI.")
    parser.add_argument("--config", default="configs/default.toml", help="Path to TOML config.")
    parser.add_argument("--backend", default=None, help="Override backend kind (lvtc or hf).")
    parser.add_argument("--hf-model", default=None, help="Override HuggingFace model name.")
    parser.add_argument("--live", action="store_true", help="Enable configured live sensor polling.")
    parser.add_argument("--camera", action="store_true", help="Enable camera adapter for this run.")
    parser.add_argument("--microphone", action="store_true", help="Enable microphone adapter for this run.")
    parser.add_argument("--caption", action="store_true", help="Enable caption enrichment for camera observations.")
    parser.add_argument("--stt", action="store_true", help="Enable speech-to-text enrichment for microphone observations.")
    parser.add_argument("--caption-model", default=None, help="Override image captioning model name.")
    parser.add_argument("--speech-model", default=None, help="Override speech transcription model name.")
    parser.add_argument("--tts", action="store_true", help="Enable voice output for this run.")
    parser.add_argument("--device", default=None, help="Set default runtime device (auto, cpu, cuda, cuda:0, cuda:1).")
    parser.add_argument("--speech-device", default=None, help="Override speech transcription device.")
    parser.add_argument("--caption-device", default=None, help="Override image captioning device.")
    parser.add_argument("--tts-device", default=None, help="Override TTS device.")
    parser.add_argument("--hf-device", default=None, help="Override HF text backend device.")
    parser.add_argument("--hf-device-map", default=None, help="Override HF text backend device_map (e.g. auto).")
    parser.add_argument("--diagnostics", action="store_true", help="Print runtime diagnostics at startup.")
    parser.add_argument("--live-lite", action="store_true", help="Start the safe direct sensor watch loop after startup.")
    parser.add_argument("--live-lite-cycles", default="forever", help="Live-lite cycles: integer count or forever.")
    parser.add_argument("--live-lite-interval", type=float, default=2.0, help="Seconds between live-lite sensor polls.")
    parser.add_argument("--live-lite-cooldown", type=float, default=8.0, help="Minimum seconds between live-lite backend calls.")
    parser.add_argument("--live-lite-exit", action="store_true", help="Exit after live-lite stops instead of returning to interactive CLI.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _suppress_known_whisper_noise()
    cfg = WonderBotConfig.load(args.config)
    _configure_multimodal_affect_runtime(getattr(cfg, "multimodal_affect", None))
    if args.backend is not None:
        cfg.backend.kind = args.backend
    if args.hf_model is not None:
        cfg.backend.hf_model = args.hf_model
    if args.live:
        cfg.live.enabled = True
    if args.camera:
        cfg.live.enabled = True
        cfg.camera.enabled = True
    if args.microphone:
        cfg.live.enabled = True
        cfg.microphone.enabled = True
    if args.caption:
        cfg.live.enabled = True
        cfg.camera.enabled = True
        cfg.caption.enabled = True
    if args.stt:
        cfg.live.enabled = True
        cfg.microphone.enabled = True
        cfg.speech.enabled = True
    if args.caption_model is not None:
        cfg.caption.model = args.caption_model
    if args.speech_model is not None:
        cfg.speech.model = args.speech_model
    if args.tts:
        cfg.tts.enabled = True
    if args.device is not None:
        cfg.runtime.default_device = args.device
    if args.speech_device is not None:
        cfg.runtime.speech_device = args.speech_device
    if args.caption_device is not None:
        cfg.runtime.caption_device = args.caption_device
    if args.tts_device is not None:
        cfg.runtime.tts_device = args.tts_device
    if args.hf_device is not None:
        cfg.runtime.hf_llm_device = args.hf_device
    if args.hf_device_map is not None:
        cfg.runtime.hf_llm_device_map = args.hf_device_map

    bot = WonderBot(cfg)
    print(f"[{cfg.agent.name}] ready. Type text or use /help.")
    if args.diagnostics:
        print(json.dumps(bot.diagnostics(), indent=2, ensure_ascii=False))

    if args.live_lite:
        live_lite_arg = f"{args.live_lite_cycles} {args.live_lite_interval} {args.live_lite_cooldown}"
        _handle_sense_watch(live_lite_arg, bot)
        if args.live_lite_exit:
            bot.close()
            print("State saved.")
            return 0
        print("[live-lite] returned to interactive CLI. Type /quit to exit.")

    try:
        while True:
            try:
                line = input("> ")
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break

            if not line.strip():
                turns = bot.idle_tick(1)
                _render_turns(turns)
                continue

            if line.startswith("/"):
                if _handle_command(line, bot):
                    continue
                break

            turn = bot.observe(line, source="user", explicit=True)
            _render_turn(turn)
    finally:
        bot.close()
        print("State saved.")
    return 0


def _handle_command(line: str, bot: WonderBot) -> bool:
    command, *rest = line.strip().split(maxsplit=1)
    arg = rest[0] if rest else ""
    if command == "/help":
        print(
            "/tick [n]  /sense  /sense-summary  /sense-ask  /sense-watch [cycles] [interval] [cooldown]  /sense-journal [n]  /sense-journal-clear  /watch [n]  /sensors  /diagnostics  /focus  /voice on|off  "
            "/state  /memory [n]  /stm [n]  /ltm [kind] [n]  /self [kind] [n]  /preferences  /goals [status] [n]  /goal ...  "
            "/plans [status] [n]  /plan ...  /next [n]  /queue [n]  /tools  /runs [n]  /act ...  "
            "/search <query>  /remember <query>  /consolidate  /reflect  /sleep  /dream [n]  /journal [kind] [n]  /tasks  /beliefs  /threads  /save  /quit"
        )
        return True
    if command == "/tick":
        count = int(arg) if arg else 1
        turns = bot.idle_tick(count)
        if not turns:
            print(f"[system] advanced {count} ticks.")
        _render_turns(turns)
        return True
    if command == "/sense":
        turns = bot.poll_sensors()
        if not turns:
            print("[system] no live sensor event crossed the salience threshold.")
        _render_turns(turns)
        return True
    if command == "/sense-summary":
        observations = bot.sensor_hub.poll()
        if not observations:
            print("[system] no live sensor event crossed the salience threshold.")
            return True
        observations, _ = _apply_multimodal_affect_to_observations(
            observations,
            now=time.time(),
            previous_visual_context=None,
        )

        camera_lines = []
        microphone_lines = []

        for obs in observations:
            source = getattr(obs, "source", "sensor")
            body = getattr(obs, "text", None) or str(obs)
            salience = getattr(obs, "salience", 0.0)
            print(f"[{source}] {body} (salience={salience:.2f})")

            source_lower = str(source).lower()
            body_lower = str(body).lower()

            if "camera" in source_lower or body_lower.startswith("camera "):
                camera_lines.append(str(body))
            elif "microphone" in source_lower or "mic" in source_lower or body_lower.startswith("microphone "):
                microphone_lines.append(str(body))

        summary_bits = []

        if camera_lines:
            summary_bits.append("remote camera observed " + camera_lines[-1])
        else:
            summary_bits.append("remote camera produced no new salient observation")

        if microphone_lines:
            summary_bits.append("remote microphone observed " + microphone_lines[-1])
        else:
            summary_bits.append("remote microphone produced no new salient observation")

        print("[sensor-summary] " + "; ".join(summary_bits) + ".")
        return True

    if command == "/sense-ask":
        observations = bot.sensor_hub.poll()
        if not observations:
            print("[system] no live sensor event crossed the salience threshold.")
            return True
        observations, _ = _apply_multimodal_affect_to_observations(
            observations,
            now=time.time(),
            previous_visual_context=None,
        )

        observation_lines = []

        for obs in observations:
            source = getattr(obs, "source", "sensor")
            body = getattr(obs, "text", None) or str(obs)
            salience = getattr(obs, "salience", 0.0)
            print(f"[{source}] {body} (salience={salience:.2f})")
            observation_lines.append(f"- [{source}] {body}")

        prompt = (
            "Summarize these live remote sensor observations in one short sentence. "
            "Use only the observations below. "
            "Do not invent temperature, weather, location, object identity, or environmental readings. "
            "If audio contains a transcript, mention the transcript briefly. "
            "If audio says transcriber disabled or sound-only, say audio was detected but not transcribed.\n\n"
            + "\n".join(observation_lines)
        )

        try:
            result = bot.backend.generate(prompt, [], "concise")
        except TypeError as exc:
            print(f"[system] backend direct-call signature mismatch: {exc}")
            return True

        answer = getattr(result, "text", None) or getattr(result, "content", None) or str(result)
        print(f"[hf-sensor] {answer.strip()}")
        return True

    if command == "/sense-watch":
        return _handle_sense_watch(arg, bot)

    if command == "/sense-journal":
        return _handle_sense_journal(arg)

    if command == "/sense-journal-clear":
        return _handle_sense_journal_clear(arg)

    if command == "/sensors":
        for status in bot.sensor_hub.status():
            state = "available" if status.available else "unavailable"
            enabled = "enabled" if status.enabled else "disabled"
            print(f"- [{status.source}] {enabled}, {state}: {status.detail}")
        speaker = bot.state_summary()["voice"]["status"]
        voice_state = "enabled" if bot.voice_enabled else "disabled"
        availability = "available" if speaker["available"] else "unavailable"
        print(f"- [voice] {voice_state}, {availability}: {speaker['detail']}")
        return True
    if command == "/diagnostics":
        print(json.dumps(bot.diagnostics(), indent=2, ensure_ascii=False))
        return True
    if command == "/focus":
        print(json.dumps(bot.state_summary()["focus"], indent=2, ensure_ascii=False))
        return True
    if command == "/voice":
        desired = arg.strip().lower()
        if desired not in {"on", "off"}:
            print("Usage: /voice on|off")
            return True
        enabled = bot.set_voice_enabled(desired == "on")
        if desired == "on" and not enabled:
            print("[system] voice output is unavailable in this environment.")
        else:
            print(f"[system] voice output {'enabled' if enabled else 'disabled'}.")
        return True
    if command == "/state":
        print(json.dumps(bot.state_summary(), indent=2, ensure_ascii=False))
        return True
    if command in {"/memory", "/stm"}:
        limit = int(arg) if arg else 10
        for item in bot.memory.top_memories(limit):
            print(f"- ({item.priority:.3f}) [{item.source}] {item.text}")
        return True
    if command == "/ltm":
        kind, limit = _kind_and_limit(arg, default_limit=10)
        entries = bot.longterm.latest(kind=kind, limit=limit)
        if not entries:
            print("[system] long-term memory is empty for that view.")
            return True
        for entry in entries:
            print(f"- ({entry.strength:.2f}) [{entry.kind}] {entry.text}")
        return True
    if command == "/self":
        kind, limit = _kind_and_limit(arg, default_limit=10)
        entries = bot.self_model.latest(kind=kind, limit=limit)
        if not entries:
            print("[system] self model is empty for that view.")
            return True
        for entry in entries:
            print(f"- ({entry.strength:.2f}) [{entry.kind}] {entry.text}")
        return True
    if command == "/preferences":
        entries = bot.self_model.latest(kind="preference", limit=12)
        if not entries:
            print("[system] no stored preferences yet.")
            return True
        for entry in entries:
            print(f"- ({entry.strength:.2f}) {entry.text}")
        return True
    if command == "/goals":
        status, limit = _kind_and_limit(arg, default_limit=10)
        entries = bot.goals.latest(status=status, limit=limit)
        if not entries:
            print("[system] no goals for that view.")
            return True
        focused = bot.goals.focused()
        for entry in entries:
            marker = "*" if focused and entry.id == focused.id else " "
            print(f"{marker} {entry.id[:8]} [{entry.status}] ({entry.priority:.2f}/{entry.progress:.2f}) {entry.title}" + (f" — {entry.detail}" if entry.detail else ""))
        return True
    if command == "/queue":
        limit = int(arg) if arg and arg.strip().isdigit() else 10
        entries = bot.goals.queue(limit=limit)
        if not entries:
            print("[system] no active work queue yet.")
            return True
        focused = bot.goals.focused()
        for entry in entries:
            marker = "*" if focused and entry.id == focused.id else " "
            print(f"{marker} {entry.id[:8]} [{entry.status}] ({entry.priority:.2f}/{entry.progress:.2f}) {entry.title}")
        return True
    if command == "/plans":
        status, limit = _kind_and_limit(arg, default_limit=10)
        entries = bot.plans.latest(status=status, limit=limit)
        if not entries:
            print("[system] no plans for that view.")
            return True
        focused = bot.plans.focused()
        for entry in entries:
            marker = "*" if focused and entry.id == focused.id else " "
            print(f"{marker} {entry.id[:8]} [{entry.status}] ({entry.priority:.2f}/{entry.progress:.2f}) {entry.title}" + (f" — goal {entry.goal_id[:8]}" if entry.goal_id else ""))
        return True
    if command == "/next":
        limit = int(arg) if arg and arg.strip().isdigit() else 8
        pairs = bot.plans.executable_steps(limit=limit)
        if not pairs:
            print("[system] no executable plan steps yet.")
            return True
        for plan, step in pairs:
            intent = f" [{step.action_intent}]" if step.action_intent else ""
            print(f"- {plan.id[:8]}/{step.id[:8]}{intent} {step.title} (plan: {plan.title})")
        return True
    if command == "/goal":
        return _handle_goal_command(arg, bot)
    if command == "/plan":
        return _handle_plan_command(arg, bot)
    if command == "/tools":
        for tool in bot.actions.list_tools():
            mode = "read-only" if tool.read_only else "mutating"
            aliases = f" aliases={','.join(tool.aliases)}" if tool.aliases else ""
            intents = f" intents={','.join(tool.intents)}" if tool.intents else ""
            print(f"- {tool.name} [{mode}] {tool.detail}{aliases}{intents}")
        return True
    if command == "/runs":
        limit = int(arg) if arg and arg.strip().isdigit() else 10
        runs = bot.actions.latest_runs(limit=limit)
        if not runs:
            print("[system] no tool runs yet.")
            return True
        for run in runs:
            mode = "dry" if run.dry_run else "commit"
            outcome = "ok" if run.success else "fail"
            target = f" {run.plan_id[:8]}/{run.step_id[:8]}" if run.plan_id and run.step_id else ""
            print(f"- {run.id[:8]} [{mode}/{outcome}] {run.tool_name}{target}: {run.summary}")
        return True
    if command == "/act":
        return _handle_act_command(arg, bot)
    if command == "/search":
        if not arg:
            print("Usage: /search your query")
            return True
        for item in bot.memory.search(arg, k=8):
            print(f"- ({item.priority:.3f}) [{item.source}] {item.text}")
        return True
    if command == "/remember":
        if not arg:
            print("Usage: /remember your query")
            return True
        entries = bot.longterm.search(arg, k=8)
        if not entries:
            print("[system] nothing in long-term memory matched strongly enough.")
            return True
        for entry in entries:
            print(f"- ({entry.strength:.2f}) [{entry.kind}] {entry.text}")
        return True
    if command == "/consolidate":
        report = bot.consolidate(force=True)
        _render_consolidation(report)
        return True
    if command == "/reflect":
        report = bot.reflect(force=True)
        _render_consolidation(report)
        return True
    if command == "/sleep":
        report = bot.sleep(force=True)
        _render_sleep(report)
        return True
    if command == "/dream":
        count = int(arg) if arg else 1
        for _ in range(max(1, count)):
            report = bot.dream(force=True)
            _render_sleep(report)
        return True
    if command == "/journal":
        kind, limit = _kind_and_limit(arg, default_limit=8)
        entries = bot.journal.latest(kind=kind, limit=limit)
        if not entries:
            print("[system] journal is empty for that view.")
            return True
        for entry in entries:
            print(f"- [{entry.kind}] ({entry.score:.2f}) {entry.text}")
        return True
    if command == "/tasks":
        for entry in bot.journal.latest(kind="task", limit=12):
            print(f"- ({entry.score:.2f}) {entry.text}")
        return True
    if command == "/beliefs":
        for entry in bot.journal.latest(kind="belief", limit=12):
            print(f"- ({entry.score:.2f}) {entry.text}")
        return True
    if command == "/threads":
        for entry in bot.journal.latest(kind="thread", limit=12):
            print(f"- ({entry.score:.2f}) {entry.text}")
        return True
    if command == "/save":
        bot.save()
        print("[system] state saved.")
        return True
    if command == "/quit":
        return False
    print(f"Unknown command: {command}. Use /help.")
    return True


def _live_lite_journal_path() -> Path:
    return Path("state") / "live_lite_events.jsonl"


def _journal_timestamp(ts: float | None = None) -> tuple[float, str]:
    if ts is None:
        ts = time.time()
    return ts, time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ts))


def _append_live_lite_journal(event: dict) -> None:
    """Append a structured live-lite event to state/live_lite_events.jsonl.

    This is intentionally a sidecar event log, not memory promotion. It lets us
    inspect what WonderBot sensed and why the backend did or did not run without
    feeding every event back into Qwen context.
    """
    path = _live_lite_journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    ts_unix, ts_local = _journal_timestamp()
    payload = {
        "ts_unix": ts_unix,
        "ts_local": ts_local,
        **event,
    }

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _read_live_lite_journal(limit: int = 20) -> list[dict]:
    path = _live_lite_journal_path()
    if not path.exists():
        return []

    limit = max(1, min(500, int(limit)))
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    events: list[dict] = []

    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            events.append({"kind": "journal_parse_error", "raw": line})

    return events


def _format_journal_event(event: dict) -> str:
    ts = event.get("ts_local", "<unknown-time>")
    kind = event.get("kind", "event")

    if kind == "sensor_observation":
        source = event.get("source", "sensor")
        salience = event.get("salience", 0.0)
        text = event.get("text", "")
        metadata = event.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        multimodal_label = metadata.get("multimodal_affect_label")
        multimodal_confidence = metadata.get("multimodal_affect_confidence")
        multimodal_hint = str(metadata.get("multimodal_affect_visual_hint") or "").strip()
        multimodal_changed = False
        if multimodal_label and metadata.get("multimodal_affect_should_report") and "Multimodal affect estimate:" not in str(text):
            source_phrase = "text/audio + expression-lite cue" if multimodal_hint else "text/audio + visual context"
            label_text = str(multimodal_label).replace('-', ' ')
            if bool(metadata.get("multimodal_affect_mixed_signals")):
                text = (
                    f"{text} Multimodal affect estimate: mixed cues, possibly {label_text} "
                    f"(confidence={float(multimodal_confidence or 0.0):.2f}; {source_phrase})."
                )
            else:
                text = (
                    f"{text} Multimodal affect estimate: possibly {label_text} "
                    f"(confidence={float(multimodal_confidence or 0.0):.2f}; {source_phrase})."
                )
            multimodal_changed = True
        affect_label = metadata.get("affect_label")
        affect_confidence = metadata.get("affect_confidence")
        if (
            not multimodal_changed
            and not metadata.get("multimodal_affect_should_report")
            and affect_label
            and metadata.get("affect_should_report")
            and "Affect estimate:" not in str(text)
            and "Multimodal affect estimate:" not in str(text)
        ):
            text = f"{text} Affect estimate: possibly {str(affect_label).replace('-', ' ')} (confidence={float(affect_confidence or 0.0):.2f})."
        backend_worthy = event.get("backend_worthy", False)
        reason = event.get("reject_reason")
        suffix = "backend-worthy" if backend_worthy else f"skipped: {reason or 'not backend-worthy'}"
        return f"{ts} [{source}] salience={salience:.2f} {suffix} :: {text}"

    if kind == "backend_summary":
        response = event.get("response", "")
        return f"{ts} [hf-sensor] {response}"

    if kind == "backend_skip":
        reason = event.get("reason", "unknown")
        return f"{ts} [sense-watch] backend skipped: {reason}"

    if kind == "poll_empty":
        return f"{ts} [sense-watch] poll {event.get('poll', '?')}: no salient sensor event"

    if kind == "watch_start":
        return (
            f"{ts} [sense-watch] start cycles={event.get('cycles')} "
            f"interval={event.get('interval')} cooldown={event.get('cooldown')}"
        )

    if kind == "watch_stop":
        return (
            f"{ts} [sense-watch] stop polls={event.get('polls')} "
            f"backend_summaries={event.get('backend_calls')} reason={event.get('reason')}"
        )

    return f"{ts} [{kind}] {json.dumps(event, ensure_ascii=False, sort_keys=True)}"


def _handle_sense_journal(arg: str) -> bool:
    arg = arg.strip()
    if arg in {"clear", "--clear"}:
        return _handle_sense_journal_clear("")

    if arg in {"help", "-h", "--help"}:
        print("Usage: /sense-journal [n]")
        print("Usage: /sense-journal-clear")
        return True

    limit = int(arg) if arg and arg.isdigit() else 20
    events = _read_live_lite_journal(limit)

    if not events:
        print("[sense-journal] no live-lite journal events yet.")
        return True

    print(f"[sense-journal] showing last {len(events)} event(s) from {_live_lite_journal_path()}")
    for event in events:
        print("- " + _format_journal_event(event))
    return True


def _handle_sense_journal_clear(arg: str) -> bool:
    arg = arg.strip().lower()
    if arg not in {"", "yes", "--yes"}:
        print("Usage: /sense-journal-clear")
        return True

    path = _live_lite_journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        archive = path.with_name(f"live_lite_events.{int(time.time())}.jsonl.bak")
        path.rename(archive)
        print(f"[sense-journal] archived previous journal to {archive}")
    else:
        print("[sense-journal] no existing journal to archive.")

    path.write_text("", encoding="utf-8")
    print(f"[sense-journal] cleared {path}")
    return True



def _apply_multimodal_affect_to_observations(
    observations: list,
    *,
    now: float,
    previous_visual_context,
) -> tuple[list, object | None]:
    """Fuse recent visual Expression-Lite cues into microphone Emotion-Lite observations.

    This function intentionally runs in the live-lite orchestration layer because
    camera and microphone observations are separate adapter outputs.
    """

    if not _MULTIMODAL_AFFECT_ENABLED or not observations:
        return observations, previous_visual_context

    current_visual_context = None
    for obs in observations:
        _source, _body, _salience, metadata = _sensor_observation_parts(obs)
        candidate = extract_visual_affect_context(metadata, observed_at=now)
        if candidate is not None:
            current_visual_context = candidate

    latest_visual_context = current_visual_context or previous_visual_context
    if (
        latest_visual_context is not None
        and hasattr(latest_visual_context, "is_fusable")
        and not latest_visual_context.is_fusable(now, _MULTIMODAL_AFFECT_VISUAL_TTL_SECONDS * 1.5)
    ):
        latest_visual_context = None

    visual_context_for_fusion = current_visual_context or latest_visual_context
    if (
        visual_context_for_fusion is not None
        and hasattr(visual_context_for_fusion, "is_fusable")
        and not visual_context_for_fusion.is_fusable(now, _MULTIMODAL_AFFECT_VISUAL_TTL_SECONDS)
    ):
        visual_context_for_fusion = None

    if visual_context_for_fusion is None:
        return observations, latest_visual_context

    for obs in observations:
        source, body, salience, metadata = _sensor_observation_parts(obs)
        if "microphone" not in source.lower():
            continue
        if not bool(metadata.get("emotion_lite")):
            continue
        updated_body, updated_metadata = apply_multimodal_affect_to_text_and_metadata(
            body,
            metadata,
            visual_context_for_fusion,
            now=now,
            max_visual_age_seconds=_MULTIMODAL_AFFECT_VISUAL_TTL_SECONDS,
            min_confidence=_MULTIMODAL_AFFECT_MIN_CONFIDENCE,
            append_to_text=_MULTIMODAL_AFFECT_APPEND_TO_TEXT,
        )
        try:
            obs.text = updated_body
            obs.metadata = updated_metadata
        except Exception:
            pass

    return observations, latest_visual_context


def _sensor_observation_parts(obs) -> tuple[str, str, float, dict[str, object]]:
    source = getattr(obs, "source", "sensor")
    body = getattr(obs, "text", None) or str(obs)
    salience = float(getattr(obs, "salience", 0.0) or 0.0)
    metadata = getattr(obs, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return str(source), str(body), salience, metadata


def _sensor_prompt_from_lines(observation_lines: list[str]) -> str:
    joined = "\n".join(observation_lines)
    has_speech = "microphone catches speech:" in joined.lower()

    if has_speech:
        return (
            "You are WonderBot in a live conversation with the human in front of you. "
            "Treat the microphone transcript as what the human just said to you. "
            "Reply directly to the human in first person, like a present conversational partner. "
            "Do not narrate telemetry. Do not say 'the microphone detected', 'the camera detects', "
            "'the transcript says', 'sensor observation', or 'live observations'. "
            "Never say generic assistant phrases such as 'How can I assist you today?' unless the human explicitly asks for help. "
            "Do not quote the transcript unless quoting is genuinely useful. "
            "Use camera context only if it helps the reply naturally. "
            "Keep the reply to one short sentence, maximum 18 words. "
            "If the transcript is unclear, say that briefly and ask for a repeat.\n\n"
            "Live observations:\n"
            + joined
        )

    return (
        "You are WonderBot noticing the world in real time. "
        "Make one brief natural comment about the visual change, not a sensor report. "
        "Do not say 'camera detects', 'sensor', 'telemetry', 'salience', 'texture', or 'scene change'. "
        "Do not invent objects, people, identity, mood, location, or activity. "
        "Maximum 12 words.\n\n"
        "Live observations:\n"
        + joined
    )

def _sensor_extract_transcript(body: str) -> str | None:
    match = re.search(r'microphone catches speech:\s*"([^"]*)"', body, flags=re.IGNORECASE)
    if not match:
        return None
    transcript = match.group(1).strip()
    return transcript or None


def _normalize_transcript(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _transcript_similarity(left: str, right: str) -> float:
    left_norm = _normalize_transcript(left)
    right_norm = _normalize_transcript(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        shorter = min(len(left_norm), len(right_norm))
        longer = max(len(left_norm), len(right_norm))
        if longer and shorter / longer >= 0.55:
            return 0.95
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _sensor_short_transcript_is_conversational(transcript: str) -> bool:
    """Return True for short transcripts that are likely meaningful user turns.

    This deliberately allows compact conversational phrases such as "yes",
    "no", "thank you", "that's funny", "they are adorable", and
    "what do you think?" while continuing to reject telemetry fragments,
    filler, and common self-echo/reporting debris.
    """
    raw = transcript.strip()
    norm = _normalize_transcript(raw)
    if not norm:
        return False

    words = norm.split()
    if len(words) > 6:
        return True

    exact_rejects = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "uh",
        "um",
        "erm",
        "hmm",
        "mm",
        "camera",
        "the camera",
        "microphone",
        "the microphone",
        "transcript",
        "the transcript",
        "sensor",
        "the sensor",
        "detected speech",
        "speech detected",
        "audio detected",
        "strong motion",
        "scene change",
        "lighting shift",
    }
    if norm in exact_rejects:
        return False

    reject_phrases = {
        "microphone detected",
        "microphone catches",
        "camera detects",
        "camera sees",
        "detected speech",
        "the transcript",
        "with the transcript",
        "sensor observation",
        "hf sensor",
        "salience",
        "frontend vad",
        "voice like banding",
        "mid lit scene",
        "sharp focus",
        "moderate texture",
        "busy texture",
        "dense texture",
        "visual state changed",
    }
    if any(phrase in norm for phrase in reject_phrases):
        return False

    accept_exact = {
        "yes",
        "yeah",
        "yep",
        "yup",
        "no",
        "nope",
        "nah",
        "ok",
        "okay",
        "sure",
        "please",
        "thanks",
        "thank you",
        "sorry",
        "hello",
        "hi",
        "hey",
        "stop",
        "wait",
        "listen",
        "look",
        "repeat that",
        "say that again",
        "come here",
        "talk to me",
    }
    if norm in accept_exact:
        return True

    if raw.endswith("?"):
        return True

    question_starts = {
        "what",
        "why",
        "who",
        "where",
        "when",
        "how",
        "can",
        "could",
        "would",
        "will",
        "do",
        "does",
        "did",
        "are",
        "is",
        "am",
        "should",
    }
    if words and words[0] in question_starts:
        return True

    conversational_words = {
        "i",
        "im",
        "ive",
        "id",
        "you",
        "youre",
        "youve",
        "we",
        "were",
        "thats",
        "that",
        "this",
        "they",
        "them",
        "he",
        "she",
        "it",
        "think",
        "feel",
        "mean",
        "meant",
        "want",
        "need",
        "know",
        "hear",
        "heard",
        "see",
        "look",
        "talk",
        "speak",
        "answer",
        "repeat",
        "help",
    }
    emotion_or_value_words = {
        "adorable",
        "cute",
        "funny",
        "amused",
        "weird",
        "strange",
        "cool",
        "good",
        "bad",
        "great",
        "brilliant",
        "amazing",
        "lovely",
        "nice",
        "wrong",
        "right",
        "better",
        "worse",
        "scary",
        "sad",
        "happy",
        "angry",
        "uncertain",
        "confused",
        "interesting",
    }

    word_set = set(words)
    if word_set & emotion_or_value_words:
        return True

    if len(words) >= 2 and word_set & conversational_words:
        return True

    imperative_starts = {
        "look",
        "listen",
        "wait",
        "stop",
        "come",
        "say",
        "tell",
        "answer",
        "repeat",
        "watch",
        "notice",
    }
    if words and words[0] in imperative_starts:
        return True

    return False


def _sensor_observation_backend_reject_reason(
    source: str,
    body: str,
    salience: float,
    metadata: dict[str, object] | None = None,
) -> str | None:
    """Return a short reason if an observation should not trigger Qwen."""
    source_lower = source.lower()
    body_lower = body.lower()
    metadata = metadata or {}

    if (
        "stt: sound only" in body_lower
        or "vad rejected" in body_lower
        or "transcriber disabled" in body_lower
    ):
        return "sound-only/VAD rejected"

    transcript = _sensor_extract_transcript(body)
    if transcript is not None:
        words = _normalize_transcript(transcript).split()

        if "transcript accepted (silence)" in body_lower:
            return "accepted-silence transcript"

        if len(words) < 4:
            if _sensor_short_transcript_is_conversational(transcript):
                if salience < 0.18:
                    return "low-salience short conversational transcript"
                return None
            return "short transcript"

        if salience < 0.30:
            return "low-salience transcript"

        return None

    if "transcript accepted" in body_lower or "microphone catches speech:" in body_lower:
        return None

    if "camera" in source_lower or body_lower.startswith("camera "):
        if bool(metadata.get("vision_lite")):
            backend_hint = bool(metadata.get("backend_hint"))
            visual_state_changed = bool(metadata.get("visual_state_changed"))
            scene_change_score = float(metadata.get("scene_change_score", 0.0) or 0.0)
            if backend_hint:
                return None
            if visual_state_changed:
                return "vision-lite state change below backend threshold"
            if scene_change_score < 0.22:
                return "low-salience vision-lite scene change"
            if salience < 0.35:
                return "low-salience vision-lite observation"
            return None
        if salience < 0.30:
            return "low-salience camera motion"
        return None

    if salience < 0.75:
        return "low-salience sensor event"

    return None


def _sensor_observation_is_backend_worthy(
    source: str,
    body: str,
    salience: float,
    metadata: dict[str, object] | None = None,
) -> bool:
    return _sensor_observation_backend_reject_reason(source, body, salience, metadata) is None


def _sense_watch_transcript_is_repeat(
    transcript: str,
    recent_transcripts: list[tuple[str, float]],
    now: float,
    ttl_seconds: float = 24.0,
    similarity_threshold: float = 0.82,
) -> tuple[bool, list[tuple[str, float]]]:
    transcript_norm = _normalize_transcript(transcript)
    if not transcript_norm:
        return True, recent_transcripts

    fresh_recent = [
        (old_norm, old_at)
        for old_norm, old_at in recent_transcripts
        if now - old_at <= ttl_seconds
    ]

    for old_norm, _old_at in fresh_recent:
        if _transcript_similarity(transcript_norm, old_norm) >= similarity_threshold:
            return True, fresh_recent

    fresh_recent.append((transcript_norm, now))
    return False, fresh_recent[-12:]


def _parse_sense_watch_args(arg: str) -> tuple[int | None, float, float] | None:
    parts = arg.split()
    cycles: int | None = 20
    interval = 2.0
    cooldown = 8.0

    if not parts:
        return cycles, interval, cooldown

    if parts[0].lower() in {"help", "-h", "--help"}:
        return None

    if parts[0].lower() in {"forever", "infinite", "inf", "loop", "0"}:
        cycles = None
    else:
        cycles = max(1, int(parts[0]))

    if len(parts) >= 2:
        interval = max(0.5, float(parts[1]))

    if len(parts) >= 3:
        cooldown = max(0.0, float(parts[2]))

    if len(parts) > 3:
        raise ValueError("too many arguments")

    return cycles, interval, cooldown


def _sense_watch_transcript_is_self_echo(
    transcript: str,
    last_spoken_text: str,
    now: float,
    last_spoken_at: float,
    ttl_seconds: float = 42.0,
    similarity_threshold: float = 0.38,
) -> bool:
    """Return True when STT appears to be hearing WonderBot's own recent TTS.

    Remote speaker playback means the Surface microphone can re-hear WonderBot.
    This matcher is intentionally forgiving: it catches exact echoes, partial
    fragments, and short phrase tails like "you today" from a longer spoken reply.
    """
    if not transcript or not last_spoken_text:
        return False
    if now - last_spoken_at > ttl_seconds:
        return False

    transcript_norm = _normalize_transcript(transcript)
    spoken_norm = _normalize_transcript(last_spoken_text)
    if not transcript_norm or not spoken_norm:
        return False

    if transcript_norm in spoken_norm or spoken_norm in transcript_norm:
        return True

    transcript_words_all = transcript_norm.split()
    spoken_words_all = spoken_norm.split()
    filler = {
        "a", "an", "the", "and", "or", "but", "to", "of", "in", "on",
        "for", "with", "that", "this", "it", "is", "are", "am", "be",
    }
    transcript_words = [w for w in transcript_words_all if w not in filler]
    spoken_words = [w for w in spoken_words_all if w not in filler]

    if transcript_words and spoken_words:
        spoken_set = set(spoken_words)
        transcript_set = set(transcript_words)
        overlap = len(transcript_set & spoken_set) / max(1, len(transcript_set))

        if len(transcript_words) <= 3 and len(transcript_set & spoken_set) >= 2:
            return True

        if overlap >= 0.50:
            return True

    # Consecutive bigram/trigram tail fragments are very common in speaker echo.
    spoken_joined = " ".join(spoken_words_all)
    for width in (3, 2):
        if len(transcript_words_all) >= width:
            for i in range(0, len(transcript_words_all) - width + 1):
                gram = " ".join(transcript_words_all[i:i + width])
                if gram and gram in spoken_joined:
                    return True

    assistant_stock_echoes = {
        "how can i assist you today",
        "can i assist you today",
        "assist you today",
        "youre welcome",
        "you are welcome",
        "could you repeat",
        "not sure i heard",
        "i heard that correctly",
        "i see you were just",
        "no worries",
        "glad were having a conversation",
    }
    if any(phrase in transcript_norm and phrase in spoken_norm for phrase in assistant_stock_echoes):
        return True

    return _transcript_similarity(transcript_norm, spoken_norm) >= similarity_threshold


def _handle_sense_watch(arg: str, bot: WonderBot) -> bool:
    try:
        parsed = _parse_sense_watch_args(arg)
    except ValueError:
        print("Usage: /sense-watch [cycles|forever] [interval_seconds] [cooldown_seconds]")
        return True

    if parsed is None:
        print("Usage: /sense-watch [cycles|forever] [interval_seconds] [cooldown_seconds]")
        print("Example: /sense-watch 30 2 8")
        print("Example: /sense-watch forever 2 8")
        return True

    cycles, interval, cooldown = parsed
    cycle_label = "forever" if cycles is None else str(cycles)
    print(
        f"[sense-watch] starting: cycles={cycle_label}, "
        f"interval={interval:.1f}s, backend_cooldown={cooldown:.1f}s"
    )
    print("[sense-watch] Ctrl+C stops watch mode and returns to the CLI.")
    _append_live_lite_journal({
        "kind": "watch_start",
        "cycles": cycle_label,
        "interval": interval,
        "cooldown": cooldown,
    })

    polls = 0
    backend_calls = 0
    last_backend_at = 0.0
    last_signature = ""
    recent_transcripts: list[tuple[str, float]] = []
    latest_visual_affect_context = None
    last_spoken_text = ""
    last_spoken_at = 0.0
    speaker_mute_seconds = 3.5

    try:
        while cycles is None or polls < cycles:
            polls += 1
            observations = bot.sensor_hub.poll()
            now = time.time()
            observations, latest_visual_affect_context = _apply_multimodal_affect_to_observations(
                observations,
                now=now,
                previous_visual_context=latest_visual_affect_context,
            )

            if not observations:
                print(f"[sense-watch] poll {polls}: no salient sensor event.")
                _append_live_lite_journal({
                    "kind": "poll_empty",
                    "poll": polls,
                })
            else:
                observation_lines = []
                backend_lines = []
                reject_reasons = []

                for obs in observations:
                    source, body, salience, metadata = _sensor_observation_parts(obs)
                    line = f"- [{source}] {body}"

                    # Half-duplex-lite: after WonderBot speaks through the Surface,
                    # the Surface microphone often hears the speaker playback.
                    # Suppress microphone observations briefly so he does not
                    # transcribe or display his own words as user speech.
                    if (
                        source == "microphone"
                        and last_spoken_at > 0.0
                        and now - last_spoken_at < speaker_mute_seconds
                    ):
                        remaining = speaker_mute_seconds - (now - last_spoken_at)
                        print(f"[sense-watch] skipped mic during speaker playback ({remaining:.1f}s remaining).")
                        _append_live_lite_journal({
                            "kind": "backend_skip",
                            "poll": polls,
                            "reason": "speaker playback mute",
                            "mute_remaining": remaining,
                            "lines": [line],
                        })
                        reject_reasons.append("speaker playback mute")
                        continue

                    print(f"[{source}] {body} (salience={salience:.2f})")
                    observation_lines.append(line)

                    reject_reason = _sensor_observation_backend_reject_reason(source, body, salience, metadata)
                    _append_live_lite_journal({
                        "kind": "sensor_observation",
                        "poll": polls,
                        "source": source,
                        "text": body,
                        "salience": salience,
                        "metadata": metadata,
                        "backend_worthy": reject_reason is None,
                        "reject_reason": reject_reason,
                    })

                    if reject_reason is not None:
                        reject_reasons.append(reject_reason)
                        continue

                    transcript = _sensor_extract_transcript(body)
                    if transcript is not None:
                        if _sense_watch_transcript_is_self_echo(
                            transcript,
                            last_spoken_text,
                            now,
                            last_spoken_at,
                        ):
                            reject_reasons.append("assistant self-echo")
                            _append_live_lite_journal({
                                "kind": "backend_skip",
                                "poll": polls,
                                "reason": "assistant self-echo",
                                "lines": [line],
                            })
                            continue

                        is_repeat, recent_transcripts = _sense_watch_transcript_is_repeat(
                            transcript,
                            recent_transcripts,
                            now,
                        )
                        if is_repeat:
                            reject_reasons.append("repeat/overlapping transcript")
                            _append_live_lite_journal({
                                "kind": "backend_skip",
                                "poll": polls,
                                "reason": "repeat/overlapping transcript",
                                "lines": [line],
                            })
                            continue

                    backend_lines.append(line)

                signature = "\n".join(backend_lines)

                if not backend_lines:
                    if reject_reasons:
                        reason_text = "; ".join(sorted(set(reject_reasons)))
                        print(f"[sense-watch] skipped backend: {reason_text}.")
                    else:
                        reason_text = "no backend-worthy observation"
                        print("[sense-watch] no backend-worthy observation; skipped backend call.")
                    _append_live_lite_journal({
                        "kind": "backend_skip",
                        "poll": polls,
                        "reason": reason_text,
                        "lines": observation_lines,
                    })
                elif signature == last_signature:
                    print("[sense-watch] duplicate backend-worthy observation; skipped backend call.")
                    _append_live_lite_journal({
                        "kind": "backend_skip",
                        "poll": polls,
                        "reason": "duplicate backend-worthy observation",
                        "lines": backend_lines,
                    })
                elif now - last_backend_at < cooldown:
                    remaining = cooldown - (now - last_backend_at)
                    print(f"[sense-watch] backend cooldown active ({remaining:.1f}s remaining); skipped backend call.")
                    _append_live_lite_journal({
                        "kind": "backend_skip",
                        "poll": polls,
                        "reason": "cooldown",
                        "cooldown_remaining": remaining,
                        "lines": backend_lines,
                    })
                else:
                    prompt = _sensor_prompt_from_lines(backend_lines)
                    try:
                        result = bot.backend.generate(prompt, [], "concise")
                    except TypeError as exc:
                        print(f"[system] backend direct-call signature mismatch: {exc}")
                        return True

                    answer = getattr(result, "text", None) or getattr(result, "content", None) or str(result)
                    answer_text = answer.strip()
                    print(f"[hf-sensor] {answer_text}")
                    last_spoken_text = answer_text
                    if getattr(bot, "voice_enabled", False):
                        try:
                            bot.speaker.say(answer_text)
                        except Exception as exc:
                            print(f"[voice] sensor speech failed: {exc}")
                        finally:
                            last_spoken_at = time.time()
                    else:
                        last_spoken_at = time.time()
                    backend_calls += 1
                    last_backend_at = time.time()
                    last_signature = signature
                    _append_live_lite_journal({
                        "kind": "backend_summary",
                        "poll": polls,
                        "lines": backend_lines,
                        "response": answer_text,
                        "backend_calls": backend_calls,
                    })

            if cycles is not None and polls >= cycles:
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[sense-watch] stopped by user.")
        stop_reason = "keyboard_interrupt"
    else:
        stop_reason = "completed"

    _append_live_lite_journal({
        "kind": "watch_stop",
        "polls": polls,
        "backend_calls": backend_calls,
        "reason": stop_reason,
    })
    print(f"[sense-watch] stopped after {polls} polls, {backend_calls} backend summaries.")
    return True


def _handle_goal_command(arg: str, bot: WonderBot) -> bool:
    parts = arg.split(maxsplit=2)
    if not parts:
        print("Usage: /goal add <text> | /goal done <id> | /goal block <id> [note] | /goal focus <id> | /goal progress <id> <0..1>")
        return True
    action = parts[0].lower()
    if action == "add":
        if len(parts) < 2:
            print("Usage: /goal add <text>")
            return True
        entry = bot.add_goal(parts[1] if len(parts) == 2 else parts[1] + " " + parts[2])
        print(f"[system] added goal {entry.id[:8]}: {entry.title}")
        return True
    if action == "done":
        if len(parts) < 2:
            print("Usage: /goal done <id>")
            return True
        entry = bot.set_goal_status(parts[1], status="done", progress=1.0)
        if entry is None:
            print("[system] no matching goal.")
        else:
            print(f"[system] marked goal {entry.id[:8]} done.")
        return True
    if action == "block":
        if len(parts) < 2:
            print("Usage: /goal block <id> [note]")
            return True
        note = parts[2] if len(parts) > 2 else ""
        entry = bot.set_goal_status(parts[1], status="blocked", note=note)
        if entry is None:
            print("[system] no matching goal.")
        else:
            print(f"[system] marked goal {entry.id[:8]} blocked.")
        return True
    if action == "focus":
        if len(parts) < 2:
            print("Usage: /goal focus <id>")
            return True
        entry = bot.focus_goal(parts[1])
        if entry is None:
            print("[system] no matching goal.")
        else:
            print(f"[system] focused goal {entry.id[:8]}: {entry.title}")
        return True
    if action == "progress":
        if len(parts) < 3:
            print("Usage: /goal progress <id> <0..1>")
            return True
        try:
            progress = float(parts[2])
        except ValueError:
            print("[system] progress must be a number between 0 and 1.")
            return True
        entry = bot.set_goal_status(parts[1], status="active", progress=progress)
        if entry is None:
            print("[system] no matching goal.")
        else:
            print(f"[system] updated goal {entry.id[:8]} progress to {entry.progress:.2f}.")
        return True
    print("Usage: /goal add <text> | /goal done <id> | /goal block <id> [note] | /goal focus <id> | /goal progress <id> <0..1>")
    return True


def _handle_plan_command(arg: str, bot: WonderBot) -> bool:
    parts = arg.split(maxsplit=4)
    if not parts or not parts[0]:
        print("Usage: /plan add <text> | /plan show <id> | /plan focus <id> | /plan done <id> | /plan block <id> [note] | /plan step add <plan_id> <text> | /plan step doing|done|block <plan_id> <step_id> [note] | /plan step depends <plan_id> <step_id> <dep_step_id>")
        return True
    action = parts[0].lower()
    if action == "add":
        if len(parts) < 2:
            print("Usage: /plan add <text>")
            return True
        entry = bot.add_plan(" ".join(parts[1:]))
        print(f"[system] added plan {entry.id[:8]}: {entry.title}")
        return True
    if action == "show":
        if len(parts) < 2:
            print("Usage: /plan show <id>")
            return True
        entry = bot.plans.get(parts[1])
        if entry is None:
            print("[system] no matching plan.")
            return True
        print(f"[{entry.id[:8]}] {entry.title} [{entry.status}] progress={entry.progress:.2f}" + (f" goal={entry.goal_id[:8]}" if entry.goal_id else ""))
        if entry.detail:
            print(f"  {entry.detail}")
        if entry.action_intents:
            print(f"  intents: {', '.join(entry.action_intents)}")
        if entry.steps:
            print("  [steps]")
            for step in sorted(entry.steps, key=lambda item: item.order):
                deps = ""
                if step.dependency_ids:
                    deps = " deps=" + ",".join(dep[:8] for dep in step.dependency_ids)
                blocker = f" blocker={step.blocker_note}" if step.blocker_note else ""
                intent = f" [{step.action_intent}]" if step.action_intent else ""
                print(f"  - {step.id[:8]} [{step.status}] ({step.progress:.2f}){intent} {step.title}{deps}{blocker}")
        else:
            print("  [steps] none yet")
        return True
    if action == "focus":
        if len(parts) < 2:
            print("Usage: /plan focus <id>")
            return True
        entry = bot.focus_plan(parts[1])
        if entry is None:
            print("[system] no matching plan.")
        else:
            print(f"[system] focused plan {entry.id[:8]}: {entry.title}")
        return True
    if action == "done":
        if len(parts) < 2:
            print("Usage: /plan done <id>")
            return True
        entry = bot.set_plan_status(parts[1], status="done")
        if entry is None:
            print("[system] no matching plan.")
        else:
            print(f"[system] marked plan {entry.id[:8]} done.")
        return True
    if action == "block":
        if len(parts) < 2:
            print("Usage: /plan block <id> [note]")
            return True
        note = " ".join(parts[2:]) if len(parts) > 2 else ""
        entry = bot.set_plan_status(parts[1], status="blocked", note=note)
        if entry is None:
            print("[system] no matching plan.")
        else:
            print(f"[system] marked plan {entry.id[:8]} blocked.")
        return True
    if action == "step":
        if len(parts) < 3:
            print("Usage: /plan step add <plan_id> <text> | /plan step doing|done|block <plan_id> <step_id> [note] | /plan step depends <plan_id> <step_id> <dep_step_id>")
            return True
        subaction = parts[1].lower()
        if subaction == "add":
            if len(parts) < 4:
                print("Usage: /plan step add <plan_id> <text>")
                return True
            plan, step = bot.add_plan_step(parts[2], " ".join(parts[3:]))
            if plan is None or step is None:
                print("[system] no matching plan.")
            else:
                print(f"[system] added step {step.id[:8]} to plan {plan.id[:8]}.")
            return True
        if subaction in {"doing", "done", "block"}:
            if len(parts) < 4:
                print("Usage: /plan step doing|done|block <plan_id> <step_id> [note]")
                return True
            note = parts[4] if len(parts) > 4 else ""
            status = {"doing": "doing", "done": "done", "block": "blocked"}[subaction]
            blocker_note = note if status == "blocked" else ""
            plan, step = bot.set_plan_step_status(parts[2], parts[3], status=status, note=note, blocker_note=blocker_note)
            if plan is None or step is None:
                print("[system] no matching plan/step.")
            else:
                print(f"[system] updated {plan.id[:8]}/{step.id[:8]} to {step.status}.")
            return True
        if subaction == "depends":
            if len(parts) < 5:
                print("Usage: /plan step depends <plan_id> <step_id> <dep_step_id>")
                return True
            plan, step = bot.add_plan_dependency(parts[2], parts[3], parts[4])
            if plan is None or step is None:
                print("[system] no matching plan/step.")
            else:
                print(f"[system] added dependency to {plan.id[:8]}/{step.id[:8]}.")
            return True
    print("Usage: /plan add <text> | /plan show <id> | /plan focus <id> | /plan done <id> | /plan block <id> [note] | /plan step add <plan_id> <text> | /plan step doing|done|block <plan_id> <step_id> [note] | /plan step depends <plan_id> <step_id> <dep_step_id>")
    return True


def _handle_act_command(arg: str, bot: WonderBot) -> bool:
    parts = arg.split(maxsplit=1)
    if not parts or not parts[0]:
        print("Usage: /act run <tool> [key=value ...] [--commit] | /act step <plan_id> <step_id> [--commit] | /act next [n] [--commit]")
        return True
    subaction = parts[0].lower()
    remainder = parts[1] if len(parts) > 1 else ""
    if subaction == "run":
        if not remainder:
            print("Usage: /act run <tool> [key=value ...] [--commit]")
            return True
        raw = remainder.strip()
        tokens = raw.split()
        commit = "--commit" in tokens
        raw = " ".join(token for token in tokens if token != "--commit")
        tool_name, _, arg_text = raw.partition(" ")
        args = parse_kv_args(arg_text)
        if "_" in args and not any(key in args for key in {"text", "query", "note"}):
            free = " ".join(str(item) for item in args.pop("_"))
            if tool_name in {"note", "speak", "goal_add", "plan_add"}:
                args["text"] = free
            elif tool_name in {"remember", "search_memory"}:
                args["query"] = free
        run = bot.run_tool(tool_name, args=args, dry_run=(not commit), source="cli")
        print(f"[action] {run.summary}")
        return True
    if subaction == "step":
        raw = remainder.strip()
        tokens = raw.split()
        commit = "--commit" in tokens
        bits = [token for token in tokens if token != "--commit"]
        if len(bits) < 2:
            print("Usage: /act step <plan_id> <step_id> [--commit]")
            return True
        run = bot.run_plan_step(bits[0], bits[1], dry_run=(not commit), source="cli-step")
        print(f"[action] {run.summary}")
        return True
    if subaction == "next":
        raw = remainder.strip()
        tokens = raw.split()
        commit = "--commit" in tokens
        bits = [token for token in tokens if token != "--commit"]
        limit = int(bits[0]) if bits and bits[0].isdigit() else 3
        pairs = bot.plans.executable_steps(limit=limit)
        if not pairs:
            print("[system] no executable plan steps yet.")
            return True
        if not commit:
            for plan, step in pairs:
                tool_name, inferred = bot.actions.resolve_step_tool(plan, step)
                print(f"- {plan.id[:8]}/{step.id[:8]} -> {tool_name} args={json.dumps(inferred, ensure_ascii=False)}")
            return True
        for plan, step in pairs:
            run = bot.run_plan_step(plan.id, step.id, dry_run=False, source="cli-next")
            print(f"[action] {plan.id[:8]}/{step.id[:8]} -> {run.summary}")
        return True
    print("Usage: /act run <tool> [key=value ...] [--commit] | /act step <plan_id> <step_id> [--commit] | /act next [n] [--commit]")
    return True


def _kind_and_limit(arg: str, default_limit: int = 10) -> tuple[str | None, int]:
    parts = arg.split()
    kind = None
    limit = default_limit
    if parts:
        if parts[0].isdigit():
            limit = int(parts[0])
        else:
            kind = parts[0]
            if len(parts) > 1 and parts[1].isdigit():
                limit = int(parts[1])
    return kind, limit


def _render_turns(turns: list[AgentTurn]) -> None:
    for turn in turns:
        _render_turn(turn)


def _render_turn(turn: AgentTurn) -> None:
    if turn.spontaneous:
        if turn.response:
            print(f"[{turn.backend}] {turn.response}")
        return
    if turn.source in {"camera", "microphone"}:
        print(f"[{turn.source}] {turn.stimulus} (salience={turn.salience:.2f})")
        if turn.response:
            print(f"[{turn.backend}] {turn.response}")
        else:
            detail = turn.inhibition_reason or "sensed and stored, but stayed grounded."
            print(f"[system] {detail}")
        return
    if turn.response:
        print(f"[{turn.backend}] {turn.response}")
    else:
        detail = turn.inhibition_reason or "registered, but nothing crossed the reaction threshold."
        print(f"[system] {detail}")


def _render_consolidation(report) -> None:
    if report.summary:
        print(f"[summary] {report.summary}")
    if report.tasks:
        print("[tasks]")
        for task in report.tasks:
            print(f"- {task}")
    if report.beliefs:
        print("[beliefs]")
        for belief in report.beliefs:
            print(f"- {belief}")
    if report.threads:
        print("[threads]")
        for thread in report.threads:
            print(f"- {thread}")
    if report.reflection:
        print(f"[reflection] {report.reflection}")
    if not any([report.summary, report.tasks, report.beliefs, report.threads, report.reflection]):
        print("[system] nothing substantial was ready for consolidation yet.")


def _render_sleep(report) -> None:
    if report.promoted_texts:
        print("[ltm]")
        for text in report.promoted_texts:
            print(f"- {text}")
    if report.dreams:
        print("[dreams]")
        for text in report.dreams:
            print(f"- {text}")
    if report.archived_count:
        print(f"[system] archived {report.archived_count} weak long-term entries.")
    if report.reinforced_existing:
        print(f"[system] reinforced {report.reinforced_existing} existing long-term entries.")
    if not any([report.promoted_texts, report.dreams, report.archived_count, report.reinforced_existing]):
        print("[system] nothing substantial was ready for sleep/dream processing yet.")


if __name__ == "__main__":
    raise SystemExit(main())
