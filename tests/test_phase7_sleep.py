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


def test_sleep_promotes_journal_and_memory(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        bot.observe('please stabilize wonderbot and preserve coherence over flashy instability', source='user', explicit=True)
        bot.observe('i prefer coherence over flashy instability', source='user', explicit=True)
        bot.observe('next we should add a sleep and dream cycle to long term memory', source='user', explicit=True)
        report = bot.consolidate(force=True)
        assert report.summary or report.tasks or report.beliefs

        sleep_report = bot.sleep(force=True)
        assert sleep_report.promoted_count >= 1
        assert bot.longterm.stats()['total'] >= 1
        remembered = bot.longterm.search('coherence stability', k=5)
        assert remembered
        assert any('coherence' in entry.text.lower() or 'stability' in entry.text.lower() for entry in remembered)
    finally:
        bot.close()


def test_dream_creates_synthesis_entries(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        bot.longterm.add_or_reinforce(
            'keep the system coherent and stable during live sensing',
            kind='journal/belief',
            strength=0.82,
            evidence=['coherence matters'],
        )
        bot.longterm.add_or_reinforce(
            'add a sleep and dream cycle that promotes durable knowledge',
            kind='journal/task',
            strength=0.79,
            evidence=['phase 7'],
        )
        report = bot.dream(force=True)
        assert report.dream_count >= 1
        assert any('dream synthesis' in text.lower() for text in report.dreams)
    finally:
        bot.close()


def test_recall_context_includes_longterm_entries(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        bot.longterm.add_or_reinforce(
            'coherence over flashy instability should guide wonderbot design',
            kind='journal/belief',
            strength=0.9,
            evidence=['user preference'],
        )
        recalled = bot._recall_context('how should wonderbot be designed for coherence')
        assert recalled
        assert any(getattr(item, 'source', '').startswith('ltm/') for item in recalled)
    finally:
        bot.close()
