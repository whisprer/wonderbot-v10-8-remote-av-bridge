from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import numpy as np

from wonderbot.sensors.microphone import SoundDeviceMicrophoneAdapter


class FakeTranscriber:
    model_name = "fake-transcriber"

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio, sample_rate: int):
        self.calls += 1

        class Result:
            def __init__(self, text: str):
                self.text = text

        return Result("hello wonderbot")


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)
        self.model = SimpleNamespace(generation_config=SimpleNamespace(is_multilingual=False, lang_to_id=None))

    def __call__(self, payload, **kwargs):
        self.calls.append((payload, kwargs))
        return {"text": "henlo from pipeline"}


def test_hf_speech_transcriber_english_only_avoids_language_kwargs(monkeypatch):
    fake_pipeline = FakePipeline()

    def pipeline(task, model):
        assert task == "automatic-speech-recognition"
        assert model == "distil-whisper/distil-small.en"
        return fake_pipeline

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(pipeline=pipeline))
    import wonderbot.perception as perception
    importlib.reload(perception)

    transcriber = perception.HFSpeechTranscriber("distil-whisper/distil-small.en", language="en")
    audio = np.linspace(-1.0, 1.0, num=48000, dtype="float32")
    result = transcriber.transcribe(audio, sample_rate=48000)
    assert result is not None
    assert result.text == "henlo from pipeline"
    assert fake_pipeline.calls
    payload, kwargs = fake_pipeline.calls[0]
    assert payload["sampling_rate"] == 16000
    assert kwargs == {}


def test_frontend_vad_blocks_transcription_when_rejected():
    adapter = object.__new__(SoundDeviceMicrophoneAdapter)
    adapter._np = np
    adapter.sample_rate = 48000
    adapter.window_seconds = 0.8
    adapter.transcriber = FakeTranscriber()
    adapter.transcript_salience_threshold = 0.08
    adapter.transcript_min_chars = 1
    adapter.transcript_cooldown_seconds = 0.0
    adapter._last_transcript = ""
    adapter._last_transcript_at = 0.0
    adapter._last_duplicate_transcript_at = 0.0
    adapter.duplicate_transcript_cooldown_seconds = 1.25
    adapter._last_transcribe_attempt_at = 0.0
    adapter._resolved_sample_rate = 48000
    adapter.record = lambda seconds=None: np.ones((48000,), dtype="float32") * 0.01
    adapter._prepare_signal = lambda x: np.asarray(x, dtype="float32")
    adapter._frontend_vad = lambda audio, sample_rate: {"speech_like": False, "detail": "mode=torchaudio, voiced=0.00s, ratio=0.00"}

    info = adapter._maybe_transcribe(salience=0.9, event="a sharp transient", zcr=0.08)
    assert info["state"] == "sound-only"
    assert "frontend VAD rejected" in str(info["detail"])
    assert adapter.transcriber.calls == 0


def test_frontend_vad_allows_transcription_even_for_sharp_transient():
    adapter = object.__new__(SoundDeviceMicrophoneAdapter)
    adapter._np = np
    adapter.sample_rate = 48000
    adapter.window_seconds = 0.8
    adapter.transcriber = FakeTranscriber()
    adapter.transcript_salience_threshold = 0.08
    adapter.transcript_min_chars = 1
    adapter.transcript_cooldown_seconds = 0.0
    adapter._last_transcript = ""
    adapter._last_transcript_at = 0.0
    adapter._last_duplicate_transcript_at = 0.0
    adapter.duplicate_transcript_cooldown_seconds = 1.25
    adapter._last_transcribe_attempt_at = 0.0
    adapter._resolved_sample_rate = 48000
    t = np.arange(int(adapter.sample_rate * 2.5), dtype="float32") / adapter.sample_rate
    sample = (0.05 * np.sin(2 * np.pi * 220.0 * t)).astype("float32")
    adapter.record = lambda seconds=None: sample
    adapter._prepare_signal = lambda x: np.asarray(x, dtype="float32")
    adapter._frontend_vad = lambda audio, sample_rate: {"speech_like": True, "detail": "mode=torchaudio, voiced=1.20s, ratio=0.34"}

    info = adapter._maybe_transcribe(salience=1.0, event="a sharp transient", zcr=0.08)
    assert info["state"] == "transcript-accepted"
    assert info["transcript"] == "hello wonderbot"
    assert adapter.transcriber.calls == 1
