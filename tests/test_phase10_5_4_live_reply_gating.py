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
    config_path = tmp_path / 'config.toml'
    config_path.write_text(
        '''
[agent]
name = "testbot"
response_style = "concise"
reaction_threshold = 0.0
spontaneous_interval = 3
max_context_memories = 6

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
'''.strip(),
        encoding='utf-8',
    )
    return WonderBotConfig.load(config_path)


def test_microphone_sound_only_event_stays_grounded(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    hub = SensorHub(
        adapters=[
            StaticSensor(
                'microphone',
                [SensorObservation(
                    source='microphone',
                    text='microphone hears a sharp transient with voice-like banding. STT: sound only.',
                    salience=1.0,
                    metadata={'stt_state': 'sound-only', 'stt_detail': 'below transcript threshold'},
                )],
            )
        ]
    )
    bot = WonderBot(cfg, sensor_hub=hub)
    turns = bot.poll_sensors()
    assert turns
    turn = turns[0]
    assert turn.response is None
    assert 'waiting for transcript' in turn.inhibition_reason.lower()
    assert bot.memory.stats()['total'] == 0


def test_sensor_context_filters_self_identity_without_transcript(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    bot = WonderBot(cfg, sensor_hub=SensorHub(adapters=[]))
    recalled = bot._recall_context(
        'microphone hears a sharp transient with voice-like banding. STT: sound only.',
        source='microphone',
        explicit=False,
        metadata={'stt_state': 'sound-only'},
    )
    texts = [item.text.lower() for item in recalled]
    assert all(not text.startswith('self/identity:') for text in texts)
    assert all(not text.startswith('self/style:') for text in texts)


def test_microphone_transcript_can_trigger_real_reply(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    hub = SensorHub(
        adapters=[
            StaticSensor(
                'microphone',
                [SensorObservation(
                    source='microphone',
                    text='microphone hears voice-like audio activity and catches speech: "hello wonderbot". STT: transcript accepted.',
                    salience=0.9,
                    metadata={'transcript': 'hello wonderbot', 'stt_state': 'transcript-accepted', 'memory_eligible': True},
                )],
            )
        ]
    )
    bot = WonderBot(cfg, sensor_hub=hub)
    turns = bot.poll_sensors()
    assert turns
    turn = turns[0]
    assert turn.response is not None
    assert 'my name is wonderbot' not in turn.response.lower()
