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


def test_phase8_auto_captures_preferences_and_goals(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        bot.observe('i prefer coherence over flashy instability', source='user', explicit=True)
        bot.observe('next we should build a stable self model and a durable goal queue for wonderbot', source='user', explicit=True)

        preferences = bot.self_model.latest(kind='preference', limit=5)
        goals = bot.goals.queue(limit=5)

        assert preferences
        assert any('coherence over flashy instability' in entry.text.lower() for entry in preferences)
        assert goals
        assert any('stable self model' in entry.combined_text().lower() or 'goal queue' in entry.combined_text().lower() for entry in goals)
        assert bot.goals.focused() is not None
    finally:
        bot.close()


def test_phase8_recall_context_includes_goal_and_self_model_items(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        bot.capture_self_statement('preference', 'i prefer coherence over flashy instability', source='user', strength=0.9)
        bot.add_goal('stabilize wonderbot identity and goal persistence')

        recalled = bot._recall_context('how should wonderbot stay coherent while maintaining goals')
        assert recalled
        sources = {item.source for item in recalled}
        assert any(source.startswith('self/') for source in sources)
        assert any(source.startswith('goal/') for source in sources)
    finally:
        bot.close()


def test_phase8_goal_status_and_focus_commands_work_via_api(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        one = bot.add_goal('phase 8: stable self model')
        two = bot.add_goal('phase 8: persistent work queue')
        focused = bot.focus_goal(two.id[:8])
        assert focused is not None
        assert bot.goals.focused() is not None
        assert bot.goals.focused().id == two.id

        done = bot.set_goal_status(one.id[:8], status='done', progress=1.0)
        assert done is not None
        assert done.status == 'done'
        active_queue = bot.goals.queue(limit=10)
        assert all(entry.status in {'active', 'pending', 'blocked'} for entry in active_queue)
    finally:
        bot.close()
