from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

from wonderbot.agent import WonderBot
from wonderbot.config import WonderBotConfig
from wonderbot.sensors.microphone import SoundDeviceMicrophoneAdapter


class FakeTranscriberSeq:
    model_name = "fake-transcriber-seq"

    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, audio, sample_rate: int):
        class Result:
            def __init__(self, text: str):
                self.text = text
        if not self.texts:
            return Result("")
        return Result(self.texts.pop(0))


class FakeSoundDeviceStream:
    class _Stream:
        def __init__(self, owner, **kwargs):
            self.owner = owner
            self.kwargs = kwargs

        def start(self):
            self.owner.started = True
            return self

        def stop(self):
            self.owner.stopped = True

        def close(self):
            self.owner.closed = True

    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.closed = False

    def check_input_settings(self, device=None, samplerate=None, channels=None):
        return None

    def query_devices(self, device=None, kind=None):
        return {"max_input_channels": 2, "default_samplerate": 48000.0}

    def InputStream(self, **kwargs):
        return self._Stream(self, **kwargs)


def _config(tmp_path: Path) -> WonderBotConfig:
    config_path = tmp_path / 'config.toml'
    config_path.write_text(
        '''
[agent]
name = "testbot"
response_style = "concise"
reaction_threshold = 0.0
spontaneous_interval = 50
max_context_memories = 6

[memory]
path = "state/test_memory_live.json"
max_active_items = 32
protect_identity = true
importance_threshold = 0.2
min_novelty = 0.05

[live]
enabled = true
poll_interval_ms = 1
sensor_memory_threshold = 0.10
sensor_reaction_threshold = 0.18
sensor_reaction_gain = 1.2

[microphone]
transcript_reply_min_words = 2
store_sound_only_events = false

[stability]
minimum_response_salience = 0.8
        '''.strip(),
        encoding='utf-8',
    )
    return WonderBotConfig.load(config_path)


def test_microphone_transcript_bypasses_salience_threshold(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    bot = WonderBot(cfg)
    obs = type('Obs', (), {
        'source': 'microphone',
        'text': 'microphone catches speech: "hello wonderbot can you hear me". STT: transcript accepted (silence).',
        'salience': 0.05,
        'metadata': {'transcript': 'hello wonderbot can you hear me', 'stt_state': 'transcript-accepted', 'memory_eligible': True},
    })()
    turn = bot.observe_sensor(obs)
    assert turn.response is not None
    assert 'unfinished thread' not in turn.response.lower()


def test_utterance_fragments_are_joined_and_emitted_after_silence(monkeypatch):
    fake_sd = FakeSoundDeviceStream()
    monkeypatch.setitem(sys.modules, 'sounddevice', fake_sd)
    adapter = SoundDeviceMicrophoneAdapter(
        sample_rate=48000,
        channels=1,
        window_seconds=0.35,
        min_salience=0.01,
        rms_threshold=0.001,
        peak_threshold=0.005,
        transcriber=FakeTranscriberSeq(['hello wonderbot', 'can you hear me clearly']),
        transcript_salience_threshold=0.01,
        transcript_min_chars=1,
        transcript_cooldown_seconds=0.0,
        rolling_seconds=6.0,
        transcript_window_seconds=3.0,
        startup_grace_seconds=0.0,
        silence_endpoint_seconds=0.01,
        transcript_reply_min_words=2,
    )
    t = np.arange(int(adapter.sample_rate * 2.0), dtype='float32') / adapter.sample_rate
    sample = (0.04 * np.sin(2 * math.pi * 220.0 * t)).astype('float32')
    adapter.record = lambda seconds=None: sample
    adapter._prepare_signal = lambda x: np.asarray(x, dtype='float32')
    adapter._frontend_vad = lambda audio, sample_rate: {'speech_like': True, 'detail': 'mode=test'}

    first = adapter.poll()
    second = adapter.poll()
    time.sleep(0.2)
    final = adapter.poll()
    assert first == []
    assert second == []
    assert final
    assert 'hello wonderbot can you hear me clearly' in final[0].text.lower()
