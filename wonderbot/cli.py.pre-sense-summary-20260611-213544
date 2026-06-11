from __future__ import annotations

import argparse
import json
import time

from .agent import AgentTurn, WonderBot
from .config import WonderBotConfig
from .execution import parse_kv_args


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = WonderBotConfig.load(args.config)
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
            "/tick [n]  /sense  /watch [n]  /sensors  /diagnostics  /focus  /voice on|off  "
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
    if command == "/watch":
        count = int(arg) if arg else 10
        for i in range(max(1, count)):
            turns = bot.idle_tick(1)
            if not turns:
                print(f"[system] watch tick {i + 1}: no salient sensor event.")
            _render_turns(turns)
            time.sleep(max(0.0, bot.config.live.poll_interval_ms / 1000.0))
        return True
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
