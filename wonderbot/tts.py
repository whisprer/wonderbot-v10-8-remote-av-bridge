from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Optional, Protocol
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import TTSConfig
from .hardware import choose_component_device


class TTSUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class SpeakerStatus:
    enabled: bool
    available: bool
    detail: str
    voice_name: str = ""
    engine: str = ""


class Speaker(Protocol):
    def say(self, text: str) -> None:
        ...

    def status(self) -> SpeakerStatus:
        ...

    def close(self) -> None:
        ...


class NullSpeaker:
    def __init__(self, enabled: bool, detail: str) -> None:
        self._status = SpeakerStatus(enabled=enabled, available=False, detail=detail, voice_name="", engine="none")

    def say(self, text: str) -> None:
        return None

    def status(self) -> SpeakerStatus:
        return self._status

    def close(self) -> None:
        return None


class DelegatingSpeaker:
    def __init__(self, delegate: Speaker, detail: str, engine: str) -> None:
        self._delegate = delegate
        base = delegate.status()
        self._status = SpeakerStatus(
            enabled=base.enabled,
            available=base.available,
            detail=detail,
            voice_name=base.voice_name,
            engine=engine,
        )

    def say(self, text: str) -> None:
        self._delegate.say(text)

    def status(self) -> SpeakerStatus:
        return self._status

    def close(self) -> None:
        self._delegate.close()


class Pyttsx3Speaker:
    def __init__(self, config: TTSConfig) -> None:
        try:
            import pyttsx3
        except ImportError as exc:
            raise TTSUnavailableError('pyttsx3 is not installed. Install with: pip install -e .[voice]') from exc
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', int(config.rate))
        self._engine.setProperty('volume', float(config.volume))
        selected_voice = ""
        voice_filter = config.voice_contains.strip().lower()
        if voice_filter:
            for voice in self._engine.getProperty('voices') or []:
                name = str(getattr(voice, 'name', ''))
                if voice_filter in name.lower():
                    self._engine.setProperty('voice', getattr(voice, 'id'))
                    selected_voice = name
                    break
        if not selected_voice:
            current = self._engine.getProperty('voice')
            for voice in self._engine.getProperty('voices') or []:
                if getattr(voice, 'id', None) == current:
                    selected_voice = str(getattr(voice, 'name', ''))
                    break
        self._status = SpeakerStatus(
            enabled=True,
            available=True,
            detail='pyttsx3 voice output active',
            voice_name=selected_voice,
            engine='pyttsx3',
        )

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._engine.say(text)
        self._engine.runAndWait()

    def status(self) -> SpeakerStatus:
        return self._status

    def close(self) -> None:
        try:
            self._engine.stop()
        except Exception:
            pass


class HFTTSSpeaker:
    def __init__(
        self,
        config: TTSConfig,
        *,
        fallback: Speaker | None = None,
        synthesizer: Callable[[str], tuple[object, int]] | None = None,
        player: Callable[[Path, str, str], str] | None = None,
    ) -> None:
        self._config = config
        self._fallback = fallback
        self._synthesizer = synthesizer or _build_hf_synthesizer(config)
        self._player = player or _play_audio_file
        self._resolved_backend = _resolve_playback_backend('wav', config.playback_backend)
        resolved = _resolve_tts_device(config)
        self._status = SpeakerStatus(
            enabled=True,
            available=True,
            detail=f'HF TTS active ({config.hf_model} on {resolved}, playback={self._resolved_backend})',
            voice_name=_hf_voice_name(config),
            engine='hf',
        )

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            audio, sample_rate = self._synthesizer(text)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as handle:
                temp_path = Path(handle.name)
            try:
                _write_audio_file(temp_path, audio, sample_rate)
                self._player(temp_path, 'wav', self._resolved_backend)
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            if self._fallback is not None:
                self._fallback.say(text)
                return
            raise

    def status(self) -> SpeakerStatus:
        return self._status

    def close(self) -> None:
        if self._fallback is not None:
            self._fallback.close()


class OpenAITTSSpeaker:
    def __init__(
        self,
        config: TTSConfig,
        *,
        fallback: Speaker | None = None,
        opener: Callable[..., object] | None = None,
        player: Callable[[Path, str, str], str] | None = None,
    ) -> None:
        api_key = os.getenv(config.openai_api_key_env, '').strip()
        if not api_key:
            raise TTSUnavailableError(f'{config.openai_api_key_env} is not set')
        fmt = config.openai_response_format.strip().lower() or 'wav'
        if fmt not in {'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'}:
            raise TTSUnavailableError(f'unsupported OpenAI response format: {fmt}')
        self._config = config
        self._api_key = api_key
        self._base_url = (os.getenv('OPENAI_BASE_URL') or config.openai_base_url or 'https://api.openai.com/v1').rstrip('/')
        self._fmt = fmt
        self._fallback = fallback
        self._opener = opener or urlrequest.urlopen
        self._player = player or _play_audio_file
        self._resolved_backend = _resolve_playback_backend(fmt, config.playback_backend)
        self._status = SpeakerStatus(
            enabled=True,
            available=True,
            detail=f'OpenAI TTS active ({config.openai_model}/{config.openai_voice}, {fmt}, playback={self._resolved_backend})',
            voice_name=config.openai_voice,
            engine='openai',
        )

    def _request_audio(self, text: str) -> bytes:
        payload = {
            'model': self._config.openai_model,
            'voice': self._config.openai_voice,
            'input': text,
            'response_format': self._fmt,
            'speed': float(self._config.openai_speed),
        }
        data = json.dumps(payload).encode('utf-8')
        req = urlrequest.Request(
            f'{self._base_url}/audio/speech',
            data=data,
            headers={
                'Authorization': f'Bearer {self._api_key}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        timeout = float(self._config.openai_timeout_seconds)
        try:
            with self._opener(req, timeout=timeout) as response:
                return response.read()
        except urlerror.HTTPError as exc:
            try:
                detail = exc.read().decode('utf-8', errors='replace')
            except Exception:
                detail = str(exc)
            raise RuntimeError(f'OpenAI TTS request failed: {detail}') from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f'OpenAI TTS network error: {exc}') from exc

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            audio = self._request_audio(text)
            suffix = f'.{self._fmt}'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(audio)
                temp_path = Path(handle.name)
            try:
                self._player(temp_path, self._fmt, self._resolved_backend)
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            if self._fallback is not None:
                self._fallback.say(text)
                return
            raise

    def status(self) -> SpeakerStatus:
        return self._status

    def close(self) -> None:
        if self._fallback is not None:
            self._fallback.close()




def _resolve_tts_device(config: TTSConfig) -> str:
    resolved = choose_component_device(config.hf_device or config.device, 'auto')
    return resolved.resolved

def _hf_voice_name(config: TTSConfig) -> str:
    model_name = (config.hf_model or '').lower()
    if 'speecht5' in model_name:
        return f'{config.hf_model}#{config.hf_speaker_id}'
    return config.hf_model


def _build_hf_synthesizer(config: TTSConfig) -> Callable[[str], tuple[object, int]]:
    model_name = (config.hf_model or '').strip() or 'facebook/mms-tts-eng'
    model_name_lower = model_name.lower()
    if 'speecht5' in model_name_lower:
        return _build_speecht5_synthesizer(config)
    if 'mms-tts' in model_name_lower or 'vits' in model_name_lower:
        return _build_vits_synthesizer(config)
    return _build_transformers_tts_synthesizer(config)





def _build_vits_synthesizer(config: TTSConfig) -> Callable[[str], tuple[object, int]]:
    try:
        import torch
        from transformers import AutoTokenizer, VitsModel
    except ImportError as exc:
        raise TTSUnavailableError(
            'VITS/MMS dependencies are missing. Install with: pip install -e .[hf-voice]'
        ) from exc

    model_name = config.hf_model or 'facebook/mms-tts-eng'
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = VitsModel.from_pretrained(model_name)
    except Exception as exc:
        raise TTSUnavailableError(f'failed to load VITS/MMS assets: {exc}') from exc

    device = _resolve_tts_device(config)
    model = model.to(device)
    sampling_rate = int(getattr(model.config, 'sampling_rate', config.hf_sample_rate or 16000))

    def synthesize(text: str) -> tuple[object, int]:
        inputs = tokenizer(text, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model(**inputs).waveform
        audio = output.detach().cpu().numpy().squeeze()
        return audio, sampling_rate

    return synthesize


def _build_transformers_tts_synthesizer(config: TTSConfig) -> Callable[[str], tuple[object, int]]:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise TTSUnavailableError('transformers is not installed. Install with: pip install -e .[hf-voice]') from exc

    errors: list[str] = []
    pipe = None
    for task in ('text-to-speech', 'text-to-audio'):
        try:
            pipe = pipeline(task, model=config.hf_model, device=choose_component_device(config.hf_device or config.device, 'auto').pipeline_device)
            break
        except Exception as exc:  # pragma: no cover - depends on local transformers version
            errors.append(f'{task}: {exc}')
    if pipe is None:
        joined = '; '.join(errors) if errors else 'no supported transformers text-to-speech pipeline found'
        raise TTSUnavailableError(f'could not build HF TTS pipeline for {config.hf_model}: {joined}')

    def synthesize(text: str) -> tuple[object, int]:
        result = pipe(text)
        audio = result.get('audio') if isinstance(result, dict) else result
        sample_rate = config.hf_sample_rate
        if isinstance(result, dict):
            sample_rate = int(result.get('sampling_rate') or result.get('sample_rate') or sample_rate)
        return audio, sample_rate

    return synthesize



def _build_speecht5_synthesizer(config: TTSConfig) -> Callable[[str], tuple[object, int]]:
    try:
        import torch
        from datasets import load_dataset
        from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
    except ImportError as exc:
        raise TTSUnavailableError(
            'SpeechT5 dependencies are missing. Install with: pip install -e .[hf-voice]'
        ) from exc

    model_name = config.hf_model or 'microsoft/speecht5_tts'
    vocoder_name = config.hf_vocoder_model or 'microsoft/speecht5_hifigan'
    embeddings_source = config.hf_speaker_embeddings_source or 'Matthijs/cmu-arctic-xvectors'
    try:
        processor = SpeechT5Processor.from_pretrained(model_name)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
        vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name)
        dataset = load_dataset(embeddings_source, split='validation')
    except Exception as exc:
        raise TTSUnavailableError(f'failed to load SpeechT5 assets: {exc}') from exc

    speaker_id = max(0, min(int(config.hf_speaker_id), len(dataset) - 1))
    speaker_embeddings = torch.tensor(dataset[speaker_id]['xvector']).unsqueeze(0)
    device = _resolve_tts_device(config)
    model = model.to(device)
    vocoder = vocoder.to(device)
    speaker_embeddings = speaker_embeddings.to(device)

    def synthesize(text: str) -> tuple[object, int]:
        inputs = processor(text=text, return_tensors='pt')
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            audio = model.generate_speech(
                inputs['input_ids'],
                speaker_embeddings,
                vocoder=vocoder,
            )
        audio = audio.detach().cpu().numpy()
        return audio, int(config.hf_sample_rate)

    return synthesize



def _write_audio_file(path: Path, audio: object, sample_rate: int) -> None:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise TTSUnavailableError('soundfile and numpy are required for HF TTS playback. Install with: pip install -e .[hf-voice]') from exc

    array = audio
    if hasattr(array, 'detach'):
        array = array.detach()
    if hasattr(array, 'cpu'):
        array = array.cpu()
    if hasattr(array, 'numpy'):
        array = array.numpy()
    array = np.asarray(array, dtype='float32').squeeze()
    sf.write(str(path), array, int(sample_rate))



def _resolve_playback_backend(fmt: str, backend: str) -> str:
    choice = (backend or 'auto').strip().lower()
    candidates = [choice] if choice != 'auto' else ['winsound', 'sounddevice', 'afplay', 'aplay', 'pw-play']
    for candidate in candidates:
        if candidate == 'winsound' and sys.platform.startswith('win') and fmt == 'wav':
            return 'winsound'
        if candidate == 'sounddevice':
            try:
                import sounddevice  # noqa: F401
                import soundfile  # noqa: F401
                return 'sounddevice'
            except Exception:
                continue
        if candidate in {'afplay', 'aplay', 'pw-play'} and shutil.which(candidate):
            return candidate
    raise TTSUnavailableError(
        'no audio playback backend is available for TTS; use wav on Windows, install sounddevice+soundfile, or rely on pyttsx3 fallback'
    )



def _play_audio_file(path: Path, fmt: str, backend: str) -> str:
    if backend == 'winsound':
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return backend
    if backend == 'sounddevice':
        import sounddevice as sd
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype='float32')
        sd.play(data, sample_rate)
        sd.wait()
        return backend
    if backend in {'afplay', 'aplay', 'pw-play'}:
        subprocess.run([backend, str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return backend
    raise TTSUnavailableError(f'unsupported playback backend: {backend}')



def _try_build_pyttsx3(config: TTSConfig) -> Speaker | None:
    try:
        return Pyttsx3Speaker(config)
    except TTSUnavailableError:
        return None



def build_speaker(config: TTSConfig) -> Speaker:
    if not config.enabled:
        return NullSpeaker(enabled=False, detail='voice output disabled in config')
    engine = (config.engine or 'hf').strip().lower()
    fallback_engine = (config.fallback_engine or '').strip().lower()

    fallback: Speaker | None = None
    if fallback_engine == 'pyttsx3' and engine != 'pyttsx3':
        fallback = _try_build_pyttsx3(config)

    if engine in {'hf', 'hf_tts', 'huggingface'}:
        try:
            return HFTTSSpeaker(config, fallback=fallback)
        except TTSUnavailableError as exc:
            if fallback is not None:
                return DelegatingSpeaker(
                    fallback,
                    detail=f'{fallback.status().detail}; preferred HF voice unavailable: {exc}',
                    engine='pyttsx3-fallback',
                )
            return NullSpeaker(enabled=True, detail=f'voice output unavailable: {exc}')
    if engine in {'openai', 'openai_tts'}:
        try:
            return OpenAITTSSpeaker(config, fallback=fallback)
        except TTSUnavailableError as exc:
            if fallback is not None:
                return DelegatingSpeaker(
                    fallback,
                    detail=f'{fallback.status().detail}; preferred OpenAI voice unavailable: {exc}',
                    engine='pyttsx3-fallback',
                )
            return NullSpeaker(enabled=True, detail=f'voice output unavailable: {exc}')
    if engine == 'pyttsx3':
        try:
            return Pyttsx3Speaker(config)
        except TTSUnavailableError as exc:
            return NullSpeaker(enabled=True, detail=f'voice output unavailable: {exc}')
    return NullSpeaker(enabled=True, detail=f'unknown tts engine: {config.engine}')
