from pathlib import Path

from wonderbot.agent import WonderBot
from wonderbot.config import WonderBotConfig


def test_agent_observe_and_idle(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[agent]
name = "testbot"
response_style = "concise"
reaction_threshold = 0.0
spontaneous_interval = 2
max_context_memories = 4

[codec]
dim = 64
ngram = 3
window_chars = 8
min_segment_chars = 3
cosine_drop = 0.10
lowercase = false
nfkc = true

[memory]
path = "state/test_memory.json"
max_active_items = 32
protect_identity = true
importance_threshold = 0.2
min_novelty = 0.05

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
        """.strip(),
        encoding="utf-8",
    )
    cfg = WonderBotConfig.load(config_path)
    bot = WonderBot(cfg)
    turn = bot.observe("The system sees motion and starts reasoning.", source="user", explicit=True)
    assert turn.response is not None
    assert "noticed" not in turn.response.lower()
    idle_turns = bot.idle_tick(2)
    assert idle_turns


def test_agent_answers_capability_question(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[agent]
name = "testbot"
response_style = "concise"
reaction_threshold = 0.0
spontaneous_interval = 4
max_context_memories = 4

[codec]
dim = 64
ngram = 3
window_chars = 8
min_segment_chars = 3
cosine_drop = 0.10
lowercase = false
nfkc = true

[memory]
path = "state/test_memory.json"
max_active_items = 32
protect_identity = true
importance_threshold = 0.2
min_novelty = 0.05

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
        """.strip(),
        encoding="utf-8",
    )
    cfg = WonderBotConfig.load(config_path)
    bot = WonderBot(cfg)
    bot.observe("You are a continuously running system with memory.", source="user", explicit=True)
    turn = bot.observe("do you do more than notice stuff fren?", source="user", explicit=True)
    assert turn.response is not None
    lowered = turn.response.lower()
    assert "more than" in lowered or "event-coded" in lowered or "guarded imaginative" in lowered
