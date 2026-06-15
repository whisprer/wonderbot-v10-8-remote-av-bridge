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
    cfg.logging.path = str(tmp_path / 'replay.jsonl')
    cfg.consolidation.auto_enabled = False
    cfg.sleep.auto_enabled = False
    cfg.tts.enabled = False
    return cfg


def test_phase9_executable_steps_respect_dependencies(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        plan = bot.add_plan('phase 9 planning and execution layer')
        _, step_one = bot.add_plan_step(plan.id, 'implement the plan store')
        _, step_two = bot.add_plan_step(plan.id, 'wire the CLI commands')
        assert step_one is not None and step_two is not None
        bot.add_plan_dependency(plan.id, step_two.id, step_one.id)

        executable = bot.plans.executable_steps(limit=5)
        assert executable
        assert executable[0][1].id == step_one.id
        assert all(step.id != step_two.id for _, step in executable[:1])

        bot.set_plan_step_status(plan.id, step_one.id, status='done', note='plan store implemented')
        executable_after = bot.plans.executable_steps(limit=5)
        assert executable_after
        assert any(step.id == step_two.id for _, step in executable_after)
    finally:
        bot.close()


def test_phase9_outcome_updates_plan_and_goal_progress(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        goal = bot.add_goal('stabilize camera caption compatibility')
        plan = bot.add_plan('stabilize camera caption compatibility', goal_id=goal.id)
        _, step = bot.add_plan_step(plan.id, 'patch the image caption pipeline task compatibility')
        assert step is not None

        bot.observe('the image caption pipeline task compatibility is fixed now and camera works', source='user', explicit=True)

        refreshed_plan = bot.plans.get(plan.id)
        refreshed_goal = bot.goals.get(goal.id)
        assert refreshed_plan is not None
        assert refreshed_goal is not None
        assert refreshed_plan.progress >= 0.99
        assert refreshed_plan.status == 'done'
        assert refreshed_goal.progress >= 0.99
        assert refreshed_goal.status == 'done'
    finally:
        bot.close()


def test_phase9_recall_context_includes_plan_items(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        goal = bot.add_goal('ship phase 9 planning and execution layer')
        plan = bot.add_plan('phase 9 execution plan', goal_id=goal.id)
        bot.add_plan_step(plan.id, 'implement the planning store')
        bot.add_plan_step(plan.id, 'wire command handlers for plans and steps')

        recalled = bot._recall_context('what should wonderbot do next for the planning layer and command handlers')
        assert recalled
        assert any(getattr(item, 'source', '').startswith('plan/') for item in recalled)
    finally:
        bot.close()
