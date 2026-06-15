from __future__ import annotations

import json
from pathlib import Path

from wonderbot.config import TTSConfig
from wonderbot.tts import HFTTSSpeaker, OpenAITTSSpeaker, SpeakerStatus, build_speaker


class FakeSpeaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def status(self) -> SpeakerStatus:
        return SpeakerStatus(enabled=True, available=True, detail='fake fallback active', voice_name='fake', engine='fake')

    def close(self) -> None:
        return None


class DummyResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False



def test_build_speaker_falls_back_when_hf_unavailable(monkeypatch) -> None:
    fake = FakeSpeaker()
    monkeypatch.setattr('wonderbot.tts._try_build_pyttsx3', lambda config: fake)
    monkeypatch.setattr(
        'wonderbot.tts._build_hf_synthesizer',
        lambda config: (_ for _ in ()).throw(__import__('wonderbot.tts').tts.TTSUnavailableError('hf broken')),
    )

    speaker = build_speaker(TTSConfig(enabled=True, engine='hf', fallback_engine='pyttsx3'))
    status = speaker.status()
    assert status.available is True
    assert status.engine == 'pyttsx3-fallback'
    assert 'preferred HF voice unavailable' in status.detail



def test_hf_tts_speaker_synthesizes_and_plays(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_synthesizer(text: str):
        seen['text'] = text
        return [0.0, 0.25, -0.25, 0.0], 16000

    def fake_write_audio(path: Path, audio, sample_rate: int) -> None:
        seen['path_exists_before_write'] = path.exists()
        path.write_bytes(b'RIFFfakewavdata')
        seen['sample_rate'] = sample_rate
        seen['audio'] = list(audio)

    def fake_player(path: Path, fmt: str, backend: str) -> str:
        seen['path_exists_during_play'] = path.exists()
        seen['played_bytes'] = path.read_bytes()
        seen['fmt'] = fmt
        seen['backend'] = backend
        return backend

    monkeypatch.setattr('wonderbot.tts._resolve_playback_backend', lambda fmt, backend: 'fake-player')
    monkeypatch.setattr('wonderbot.tts._write_audio_file', fake_write_audio)

    speaker = HFTTSSpeaker(
        TTSConfig(enabled=True, engine='hf', hf_model='microsoft/speecht5_tts', hf_speaker_id=7306),
        synthesizer=fake_synthesizer,
        player=fake_player,
    )
    speaker.say('henlo wonderbot')

    assert seen['text'] == 'henlo wonderbot'
    assert seen['sample_rate'] == 16000
    assert seen['audio'] == [0.0, 0.25, -0.25, 0.0]
    assert seen['fmt'] == 'wav'
    assert seen['backend'] == 'fake-player'
    assert seen['played_bytes'] == b'RIFFfakewavdata'
    assert seen['path_exists_before_write'] is True
    assert seen['path_exists_during_play'] is True



def test_openai_tts_speaker_posts_expected_payload_and_plays_audio(monkeypatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    seen: dict[str, object] = {}

    def fake_opener(request, timeout=0):
        seen['url'] = request.full_url
        seen['timeout'] = timeout
        seen['headers'] = dict(request.headers)
        seen['payload'] = json.loads(request.data.decode('utf-8'))
        return DummyResponse(b'RIFFfakewavdata')

    def fake_player(path: Path, fmt: str, backend: str) -> str:
        seen['path_exists_during_play'] = path.exists()
        seen['played_bytes'] = path.read_bytes()
        seen['fmt'] = fmt
        seen['backend'] = backend
        return backend

    monkeypatch.setattr('wonderbot.tts._resolve_playback_backend', lambda fmt, backend: 'fake-player')

    speaker = OpenAITTSSpeaker(
        TTSConfig(
            enabled=True,
            engine='openai',
            openai_model='gpt-4o-mini-tts',
            openai_voice='sage',
            openai_response_format='wav',
            openai_speed=1.1,
            openai_timeout_seconds=12.5,
        ),
        opener=fake_opener,
        player=fake_player,
    )
    speaker.say('henlo wonderbot')

    assert seen['url'] == 'https://api.openai.com/v1/audio/speech'
    assert seen['timeout'] == 12.5
    assert seen['payload'] == {
        'model': 'gpt-4o-mini-tts',
        'voice': 'sage',
        'input': 'henlo wonderbot',
        'response_format': 'wav',
        'speed': 1.1,
    }
    assert seen['fmt'] == 'wav'
    assert seen['backend'] == 'fake-player'
    assert seen['played_bytes'] == b'RIFFfakewavdata'
    assert seen['path_exists_during_play'] is True
