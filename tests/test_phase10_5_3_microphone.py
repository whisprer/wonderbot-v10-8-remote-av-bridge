from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np

from wonderbot.sensors.microphone import SoundDeviceMicrophoneAdapter


class FakeTranscriber:
    model_name = "fake-stream-transcriber"

    def transcribe(self, audio, sample_rate: int):
        class Result:
            def __init__(self, text: str):
                self.text = text
        return Result(f"heard at {sample_rate}hz")


class FakeSoundDevice:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.closed = False

    class _Stream:
        def __init__(self, owner, *, samplerate, channels, dtype, device, latency, callback):
            self.owner = owner
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self.device = device
            self.latency = latency
            self.callback = callback

        def start(self):
            self.owner.started = True
            sr = int(self.samplerate)
            t = np.arange(int(sr * 2.0), dtype="float32") / sr
            chunk = (0.045 * np.sin(2 * math.pi * 1200.0 * t)).astype("float32").reshape(-1, 1)
            self.callback(chunk, len(chunk), None, None)
            return self

        def stop(self):
            self.owner.stopped = True

        def close(self):
            self.owner.closed = True

    def check_input_settings(self, device=None, samplerate=None, channels=None):
        if int(samplerate) == 16000:
            raise RuntimeError("invalid sample rate")
        return None

    def query_devices(self, device=None, kind=None):
        return {
            "max_input_channels": 2,
            "default_samplerate": 48000.0,
        }

    def InputStream(self, **kwargs):
        return self._Stream(self, **kwargs)


def test_microphone_stream_falls_back_to_device_default_rate(monkeypatch):
    fake_sd = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    # numpy already available; adapter imports it dynamically
    adapter = SoundDeviceMicrophoneAdapter(
        sample_rate=16000,
        channels=1,
        window_seconds=0.35,
        min_salience=0.01,
        rms_threshold=0.001,
        peak_threshold=0.005,
        transcriber=FakeTranscriber(),
        transcript_salience_threshold=0.01,
        transcript_min_chars=1,
        transcript_cooldown_seconds=0.0,
        device="Microphone Array (Realtek(R) Audio), Windows WASAPI",
        rolling_seconds=3.0,
        transcript_window_seconds=1.2,
        startup_grace_seconds=0.0,
        silence_endpoint_seconds=0.1,
    )

    observations = adapter.poll()
    assert fake_sd.started is True
    assert adapter.status().available is True
    assert "48000" in adapter.status().detail
    assert observations == []
    import time
    time.sleep(0.2)
    observations = adapter.poll()
    assert observations
    assert observations[0].metadata["sample_rate"] == 48000
    assert "catches speech" in observations[0].text
    assert "heard at 48000hz" in observations[0].text

    adapter.close()
    assert fake_sd.stopped is True
    assert fake_sd.closed is True
