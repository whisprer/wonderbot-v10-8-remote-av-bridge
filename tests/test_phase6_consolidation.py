from __future__ import annotations

from pathlib import Path

from wonderbot.agent import WonderBot
from wonderbot.config import WonderBotConfig


def _config(tmp_path: Path, auto_every_explicit_turns: int = 3) -> WonderBotConfig:
    config_path = tmp_path / 'config.toml'
    config_path.write_text(
        f'''
[agent]
name = "testbot"
response_style = "concise"
reaction_threshold = 0.0
spontaneous_interval = 99
max_context_memories = 4
focus_max_items = 4
focus_decay_seconds = 180.0

[codec]
dim = 64
ngram = 3
window_chars = 8
min_segment_chars = 3
cosine_drop = 0.10
lowercase = false
nfkc = true

[memory]
path = "{(tmp_path / 'memory.json').as_posix()}"
max_active_items = 64
protect_identity = true
importance_threshold = 0.2
min_novelty = 0.05

[journal]
path = "{(tmp_path / 'journal.json').as_posix()}"

[longterm]
path = "{(tmp_path / 'long_term_memory.json').as_posix()}"

[selfmodel]
path = "{(tmp_path / 'self_model.json').as_posix()}"

[goals]
path = "{(tmp_path / 'goals.json').as_posix()}"
auto_capture_goals = true
auto_focus_new_goal = true
default_priority = 0.68

[plans]
path = "{(tmp_path / 'plans.json').as_posix()}"
auto_capture_plans = true
auto_capture_steps = true
auto_update_goal_progress = true
default_priority = 0.66
context_limit = 3

[consolidation]
auto_enabled = true
auto_every_explicit_turns = {auto_every_explicit_turns}
summary_min_items = 2
summary_window_items = 12
max_summary_sentences = 3
task_limit = 4
belief_limit = 4
thread_limit = 4

[ganglion]
height = 4
width = 4
channels = 4
bleed = 0.02

[resonance]
sigma = 0.5
tau = 14.134725
alpha = 1.0
prime_count = 8

[backend]
kind = "lvtc"
hf_model = "distilgpt2"
max_new_tokens = 16
temperature = 0.8
delta_scale = 0.24
creative_depth = 1
anchor_pullback = 0.72
novelty_threshold = 0.01
drift_threshold = 0.80
repetition_threshold = 0.60
latency_budget_ms = 30

[live]
enabled = false
poll_interval_ms = 1
sensor_memory_threshold = 0.10
sensor_reaction_threshold = 0.18
sensor_reaction_gain = 1.2

[stability]
sensor_response_cooldown_seconds = 999.0
spontaneous_cooldown_seconds = 999.0
repeated_stimulus_cooldown_seconds = 999.0
same_source_cooldown_seconds = 999.0
minimum_response_salience = 0.18

[logging]
enabled = true
path = "{(tmp_path / 'replay.jsonl').as_posix()}"
flush_each_write = true
        '''.strip(),
        encoding='utf-8',
    )
    return WonderBotConfig.load(config_path)


def test_manual_consolidation_creates_summary_tasks_beliefs_and_threads(tmp_path: Path) -> None:
    cfg = _config(tmp_path, auto_every_explicit_turns=99)
    bot = WonderBot(cfg)
    bot.observe('please stabilize the live loop and add a replayable journal layer', source='user', explicit=True)
    bot.observe('i prefer a coherent creature over a flashy unstable one', source='user', explicit=True)
    bot.observe('what should we do next for phase 6 memory consolidation?', source='user', explicit=True)

    report = bot.consolidate(force=True)
    bot.close()

    assert report.summary
    assert report.tasks
    assert report.beliefs
    assert report.threads
    stats = bot.journal.stats()
    assert stats['by_kind']['summary'] >= 1
    assert stats['by_kind']['task'] >= 1
    assert stats['by_kind']['belief'] >= 1
    assert stats['by_kind']['thread'] >= 1


def test_auto_consolidation_runs_after_configured_number_of_explicit_turns(tmp_path: Path) -> None:
    cfg = _config(tmp_path, auto_every_explicit_turns=2)
    bot = WonderBot(cfg)
    bot.observe('please keep the focus on tokenizerless memory routing', source='user', explicit=True)
    assert bot.last_consolidation_report is None
    bot.observe('bring on phase 6 and make the journal useful', source='user', explicit=True)
    bot.close()

    assert bot.last_consolidation_report is not None
    assert bot.last_consolidation_report.summary
    assert bot.journal.latest(kind='summary', limit=1)


def test_reflection_uses_recent_replay_and_writes_reflection_entry(tmp_path: Path) -> None:
    cfg = _config(tmp_path, auto_every_explicit_turns=99)
    bot = WonderBot(cfg)
    bot.observe('henlo wonderbot', source='user', explicit=True)
    bot.observe('do you do more than notice stuff fren?', source='user', explicit=True)
    report = bot.reflect(force=True)
    bot.close()

    assert report.reflection
    reflections = bot.journal.latest(kind='reflection', limit=3)
    assert reflections
    assert 'recent loop' in reflections[0].text.lower() or 'recent turns' in reflections[0].text.lower()
