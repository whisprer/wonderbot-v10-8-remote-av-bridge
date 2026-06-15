from pathlib import Path

from wonderbot.agent import WonderBot
from wonderbot.config import WonderBotConfig
from wonderbot.sensors import SensorHub, SensorObservation
from wonderbot.sensors.base import SensorStatus


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
        """
[agent]
name = "testbot"
response_style = "concise"
reaction_threshold = 0.0
spontaneous_interval = 3
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
path = "state/test_memory_live.json"
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
        """.strip(),
        encoding="utf-8",
    )
    return WonderBotConfig.load(config_path)



def test_sensor_observation_triggers_response(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    hub = SensorHub(
        adapters=[
            StaticSensor(
                "camera",
                [SensorObservation(source="camera", text="camera sees strong motion in a bright scene.", salience=0.6)],
            )
        ]
    )
    bot = WonderBot(cfg, sensor_hub=hub)
    turns = bot.poll_sensors()
    assert turns
    turn = turns[0]
    assert turn.source == "camera"
    assert turn.response is not None
    assert turn.salience >= 0.6



def test_idle_tick_includes_sensor_and_spontaneous_turns(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    hub = SensorHub(
        adapters=[
            StaticSensor(
                "microphone",
                [SensorObservation(source="microphone", text="microphone hears voice-like audio activity.", salience=0.55)],
            )
        ]
    )
    bot = WonderBot(cfg, sensor_hub=hub)
    turns = bot.idle_tick(3)
    assert any(turn.source == "microphone" for turn in turns)
    assert all(not turn.spontaneous for turn in turns)
