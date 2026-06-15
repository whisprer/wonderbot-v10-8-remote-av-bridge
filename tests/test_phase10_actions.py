from __future__ import annotations

from pathlib import Path

from wonderbot.agent import WonderBot
from wonderbot.config import WonderBotConfig


def _config(tmp_path: Path) -> WonderBotConfig:
    cfg = WonderBotConfig.load(Path(__file__).resolve().parents[1] / 'configs' / 'default.toml')
    cfg.memory.path = str(tmp_path / 'memory.json')
    cfg.journal.path = str(tmp_path / 'journal.json')
    cfg.longterm.path = str(tmp_path / 'long_term_memory.json')
    cfg.selfmodel.path = str(tmp_path / 'self_model.json')
    cfg.goals.path = str(tmp_path / 'goals.json')
    cfg.plans.path = str(tmp_path / 'plans.json')
    cfg.execution.path = str(tmp_path / 'action_runs.json')
    cfg.logging.path = str(tmp_path / 'replay.jsonl')
    cfg.consolidation.auto_enabled = False
    cfg.sleep.auto_enabled = False
    cfg.tts.enabled = False
    return cfg


def test_phase10_dry_run_does_not_mutate_goals(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        run = bot.run_tool('goal_add', {'text': 'ship the tool execution layer'}, dry_run=True)
        assert run.success
        assert run.dry_run
        assert bot.goals.latest(limit=5) == []
        assert bot.actions.latest_runs(limit=1)[0].tool_name == 'goal_add'
    finally:
        bot.close()


def test_phase10_commit_tool_adds_goal_and_logs_run(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        run = bot.run_tool('goal_add', {'text': 'ship the tool execution layer'}, dry_run=False)
        assert run.success
        goals = bot.goals.latest(limit=5)
        assert goals
        assert goals[0].title == 'Ship the tool execution layer'
        assert bot.actions.latest_runs(limit=1)[0].dry_run is False
    finally:
        bot.close()


def test_phase10_execute_plan_step_marks_done_for_internal_tool(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        plan = bot.add_plan('run phase 10 action layer validation')
        _, step = bot.add_plan_step(plan.id, 'tool: sleep')
        assert step is not None
        run = bot.run_plan_step(plan.id, step.id, dry_run=False)
        refreshed = bot.plans.get(plan.id)
        assert run.success
        assert refreshed is not None
        updated = refreshed.steps[0]
        assert updated.status == 'done'
        assert bot.actions.latest_runs(limit=1)[0].step_id == updated.id
    finally:
        bot.close()
