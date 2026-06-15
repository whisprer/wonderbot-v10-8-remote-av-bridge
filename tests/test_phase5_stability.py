from __future__ import annotations

import json
from pathlib import Path

from wonderbot.agent import WonderBot
from wonderbot.config import WonderBotConfig
from wonderbot.perception import _build_caption_pipeline
from wonderbot.sensors.base import SensorObservation, SensorStatus
from wonderbot.sensors.hub import SensorHub


class StaticSensor:
    def __init__(self, name: str, observations: list[SensorObservation]) -> None:
        self.name = name
        self._observations = list(observations)

    def poll(self):
        out = list(self._observations)
        self._observations.clear()
        return out

    def status(self):
        return SensorStatus(source=self.name, enabled=True, available=True, detail="ok")

    def close(self):
        return None



def _config(tmp_path: Path) -> WonderBotConfig:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'''
[agent]
name = "testbot"
response_style = "concise"
reaction_threshold = 0.0
spontaneous_interval = 2
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

[live]
enabled = true
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
        encoding="utf-8",
    )
    return WonderBotConfig.load(config_path)



def test_replay_log_is_written_and_contains_turns(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    bot = WonderBot(cfg)
    turn = bot.observe("please stabilize the live loop and keep a replay log", source="user", explicit=True)
    assert turn.response is not None
    bot.close()

    replay_path = tmp_path / "replay.jsonl"
    assert replay_path.exists()
    lines = [json.loads(line) for line in replay_path.read_text(encoding="utf-8") .splitlines() if line.strip()]
    kinds = {line["kind"] for line in lines}
    assert "startup" in kinds
    assert "memory_write" in kinds
    assert "turn" in kinds



def test_sensor_response_cooldown_prevents_chatter(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    hub = SensorHub(
        adapters=[
            StaticSensor(
                "camera",
                [
                    SensorObservation(source="camera", text="camera sees strong motion near the desk.", salience=0.7),
                    SensorObservation(source="camera", text="camera sees strong motion near the desk.", salience=0.72),
                ],
            )
        ]
    )
    bot = WonderBot(cfg, sensor_hub=hub)
    turns = bot.poll_sensors()
    assert len(turns) == 2
    assert turns[0].response is not None
    assert turns[1].response is None
    assert "cooldown" in turns[1].inhibition_reason.lower()



def test_focus_state_updates_from_user_thread(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    bot = WonderBot(cfg)
    bot.observe("please keep the focus on tokenizerless realtime memory routing", source="user", explicit=True)
    focus = bot.state_summary()["focus"]
    assert focus["active_focus"]
    assert "tokenizerless" in focus["active_focus"].lower() or "memory routing" in focus["active_focus"].lower()
    assert focus["goal_anchor"]



def test_caption_pipeline_tries_new_name_then_old_name() -> None:
    calls: list[str] = []

    def fake_pipeline(task: str, model: str):
        calls.append(task)
        if task == "image-text-to-text":
            return (task, model)
        raise KeyError(task)

    built = _build_caption_pipeline(fake_pipeline, "demo-model")
    assert built == ("image-text-to-text", "demo-model")
    assert calls == ["image-text-to-text"]
