import math
from pathlib import Path

from wonderbot.agent import WonderBot
from wonderbot.config import WonderBotConfig
from wonderbot.sensors.base import SensorObservation, SensorStatus
from wonderbot.sensors.camera import OpenCVCameraAdapter
from wonderbot.sensors.microphone import SoundDeviceMicrophoneAdapter
from wonderbot.sensors.hub import SensorHub


class FakeCaptioner:
    model_name = "fake-captioner"

    def __init__(self, text: str) -> None:
        self.text = text

    def caption(self, image):
        class Result:
            def __init__(self, text: str):
                self.text = text
        return Result(self.text)


class FakeTranscriber:
    model_name = "fake-transcriber"

    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, audio, sample_rate: int):
        class Result:
            def __init__(self, text: str):
                self.text = text
        return Result(self.text)


class FakeCV2:
    COLOR_BGR2GRAY = 0

    @staticmethod
    def cvtColor(frame, code):
        return frame.mean(axis=2).astype("uint8")

    @staticmethod
    def absdiff(a, b):
        import numpy as np
        return np.abs(a.astype("int16") - b.astype("int16")).astype("uint8")


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
path = "state/test_memory_phase4.json"
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



def test_camera_adapter_adds_caption_when_available():
    import numpy as np

    adapter = object.__new__(OpenCVCameraAdapter)
    adapter._cv2 = FakeCV2()
    adapter._np = np
    adapter.motion_threshold = 0.08
    adapter.brightness_threshold = 0.05
    adapter.min_salience = 0.12
    adapter.captioner = FakeCaptioner("a desk with a monitor")
    adapter.caption_interval_seconds = 0.0
    adapter.caption_salience_threshold = 0.2
    adapter.caption_min_chars = 4
    adapter._prev_gray = np.zeros((4, 4), dtype="uint8")
    adapter._prev_brightness = 0.0
    adapter._last_text = ""
    adapter._last_caption = ""
    adapter._last_caption_at = 0.0
    adapter.read_frame = lambda: np.ones((4, 4, 3), dtype="uint8") * 255

    observations = adapter.poll()
    assert observations
    text = observations[0].text
    assert "Scene impression: a desk with a monitor." in text
    assert observations[0].metadata["caption"] == "a desk with a monitor"



def test_microphone_adapter_adds_transcript_when_available():
    import numpy as np

    adapter = object.__new__(SoundDeviceMicrophoneAdapter)
    adapter._np = np
    adapter.sample_rate = 16000
    adapter.channels = 1
    adapter.window_seconds = 0.35
    adapter.rms_threshold = 0.02
    adapter.peak_threshold = 0.05
    adapter.min_salience = 0.10
    adapter.transcriber = FakeTranscriber("henlo wonderbot")
    adapter.transcript_salience_threshold = 0.2
    adapter.transcript_min_chars = 4
    adapter.transcript_cooldown_seconds = 0.0
    adapter._prev_rms = 0.0
    adapter._prev_zcr = 0.0
    adapter._last_text = ""
    adapter._last_transcript = ""
    adapter._last_transcript_at = 0.0
    t = np.arange(int(adapter.sample_rate * adapter.window_seconds), dtype="float32") / adapter.sample_rate
    sample = (0.2 * np.sin(2 * math.pi * 440.0 * t)).astype("float32")
    adapter.record = lambda seconds=None: sample.reshape(-1, 1)

    observations = adapter.poll()
    assert observations == []
    adapter.silence_endpoint_seconds = 0.0
    observations = adapter.poll()
    assert observations
    text = observations[0].text
    assert 'catches speech: "henlo wonderbot"' in text
    assert observations[0].metadata["transcript"] == "henlo wonderbot"



def test_sensor_observation_with_transcript_reaches_memory_and_response(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    hub = SensorHub(
        adapters=[
            StaticSensor(
                "microphone",
                [
                    SensorObservation(
                        source="microphone",
                        text='microphone hears voice-like audio activity and catches speech: "henlo wonderbot".',
                        salience=0.72,
                        metadata={"transcript": "henlo wonderbot"},
                    )
                ],
            )
        ]
    )
    bot = WonderBot(cfg, sensor_hub=hub)
    turns = bot.poll_sensors()
    assert turns
    assert turns[0].response is not None
    stored = bot.memory.search("henlo wonderbot", k=3)
    assert any("henlo wonderbot" in item.text for item in stored)
