from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from typing import List

from .base import SensorObservation, SensorStatus
from .camera import _brightness_phrase, _clean_generated_text, _normalize_text as _normalize_camera_text
from .microphone import _clean_transcript, _normalize_text as _normalize_mic_text, _suffix_prefix_overlap
from ..perception import ImageCaptioner, SpeechTranscriber


class RemoteBridgeUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class CameraMetrics:
    motion: float
    brightness_delta: float
    brightness: float
    contrast: float


class RemoteCameraAdapter:
    name = 'camera'

    def __init__(
        self,
        base_url: str,
        api_token: str = '',
        request_timeout_seconds: float = 5.0,
        motion_threshold: float = 0.08,
        brightness_threshold: float = 0.05,
        min_salience: float = 0.12,
        captioner: ImageCaptioner | None = None,
        caption_interval_seconds: float = 3.0,
        caption_salience_threshold: float = 0.22,
        caption_min_chars: int = 12,
        verify_tls: bool = True,
    ) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            import httpx
        except ImportError as exc:
            raise RemoteBridgeUnavailableError('remote bridge camera requires opencv-python, numpy, and httpx') from exc
        self._cv2 = cv2
        self._np = np
        self._httpx = httpx
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.request_timeout_seconds = request_timeout_seconds
        self.motion_threshold = motion_threshold
        self.brightness_threshold = brightness_threshold
        self.min_salience = min_salience
        self.captioner = captioner
        self.caption_interval_seconds = caption_interval_seconds
        self.caption_salience_threshold = caption_salience_threshold
        self.caption_min_chars = caption_min_chars
        self._client = httpx.Client(timeout=request_timeout_seconds, verify=verify_tls)
        self._prev_gray = None
        self._prev_brightness = None
        self._last_text = ''
        self._last_caption = ''
        self._last_caption_at = 0.0
        self._last_timestamp_ms = 0

    def _headers(self):
        return {'X-Bridge-Token': self.api_token} if self.api_token else {}

    def read_frame(self):
        response = self._client.get(f'{self.base_url}/api/camera/latest.jpg', headers=self._headers())
        if response.status_code != 200:
            raise RemoteBridgeUnavailableError(f'remote camera unavailable: HTTP {response.status_code}')
        payload = response.content
        if not payload:
            raise RemoteBridgeUnavailableError('remote camera returned empty frame payload')
        array = self._np.frombuffer(payload, dtype=self._np.uint8)
        frame = self._cv2.imdecode(array, self._cv2.IMREAD_COLOR)
        if frame is None:
            raise RemoteBridgeUnavailableError('remote camera frame decode failed')
        self._last_timestamp_ms = int(response.headers.get('X-Timestamp-Ms', '0') or '0')
        return frame

    def poll(self) -> List[SensorObservation]:
        frame = self.read_frame()
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        metrics = self._analyze(gray)
        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_brightness = metrics.brightness
            return []
        salience = min(1.0, max(metrics.motion * 4.2, metrics.brightness_delta * 3.6, max(0.0, metrics.contrast - 0.18) * 0.8))
        self._prev_gray = gray
        self._prev_brightness = metrics.brightness
        if salience < self.min_salience:
            return []
        motion_phrase = 'strong motion' if metrics.motion >= self.motion_threshold * 2.0 else 'noticeable motion'
        if metrics.motion < self.motion_threshold:
            motion_phrase = 'subtle motion'
        light_phrase = 'lighting shift' if metrics.brightness_delta >= self.brightness_threshold else 'stable lighting'
        brightness_phrase = _brightness_phrase(metrics.brightness)
        texture_phrase = 'busy visual texture' if metrics.contrast >= 0.38 else 'simple visual texture'
        text = f'camera sees {motion_phrase} with {light_phrase} in a {brightness_phrase} and {texture_phrase}.'
        metadata = {
            'motion': round(metrics.motion, 6),
            'brightness_delta': round(metrics.brightness_delta, 6),
            'brightness': round(metrics.brightness, 6),
            'contrast': round(metrics.contrast, 6),
            'remote_timestamp_ms': self._last_timestamp_ms,
            'remote_bridge': True,
        }
        caption = self._maybe_caption(frame, salience)
        if caption:
            text = f'{text} Scene impression: {caption}.'
            metadata['caption'] = caption
        if text == self._last_text and salience < 0.45:
            return []
        self._last_text = text
        return [SensorObservation(source=self.name, text=text, salience=round(salience, 6), metadata=metadata)]

    def status(self) -> SensorStatus:
        detail = f'remote camera adapter active ({self.base_url})'
        if self.captioner is not None:
            detail += f'; captioning via {getattr(self.captioner, "model_name", "captioner")}'
        return SensorStatus(source=self.name, enabled=True, available=True, detail=detail)

    def close(self) -> None:
        self._client.close()

    def _analyze(self, gray) -> CameraMetrics:
        brightness = float(gray.mean()) / 255.0
        contrast = min(1.0, float(gray.std()) / 96.0)
        motion = 0.0
        brightness_delta = 0.0
        if self._prev_gray is not None:
            diff = self._cv2.absdiff(gray, self._prev_gray)
            motion = float(diff.mean()) / 255.0
            brightness_delta = abs(brightness - float(self._prev_brightness or 0.0))
        return CameraMetrics(motion=motion, brightness_delta=brightness_delta, brightness=brightness, contrast=contrast)

    def _maybe_caption(self, frame, salience: float) -> str | None:
        if self.captioner is None or salience < self.caption_salience_threshold:
            return None
        now = time.monotonic()
        if self._last_caption and (now - self._last_caption_at) < self.caption_interval_seconds:
            return None
        try:
            result = self.captioner.caption(frame)
        except Exception:
            return None
        if result is None:
            return None
        caption = _clean_generated_text(result.text)
        if len(caption) < self.caption_min_chars:
            return None
        if _normalize_camera_text(caption) == _normalize_camera_text(self._last_caption):
            return None
        self._last_caption = caption
        self._last_caption_at = now
        return caption


class RemoteMicrophoneAdapter:
    name = 'microphone'

    def __init__(
        self,
        base_url: str,
        api_token: str = '',
        request_timeout_seconds: float = 5.0,
        sample_rate: int = 16000,
        channels: int = 1,
        window_seconds: float = 0.35,
        rms_threshold: float = 0.03,
        peak_threshold: float = 0.12,
        min_salience: float = 0.10,
        transcriber: SpeechTranscriber | None = None,
        transcript_salience_threshold: float = 0.22,
        transcript_min_chars: int = 1,
        transcript_cooldown_seconds: float = 0.20,
        rolling_seconds: float = 4.0,
        transcript_window_seconds: float = 3.0,
        preamp_gain: float = 2.5,
        agc_target_rms: float = 0.05,
        agc_max_gain: float = 4.0,
        vad_enabled: bool = True,
        vad_trigger_level: float = 5.5,
        vad_min_voiced_seconds: float = 0.18,
        vad_min_voiced_ratio: float = 0.08,
        silence_endpoint_seconds: float = 0.65,
        utterance_max_seconds: float = 8.0,
        transcript_reply_min_words: int = 2,
        store_sound_only_events: bool = False,
        verify_tls: bool = True,
    ) -> None:
        try:
            import httpx
            import numpy as np
            import soundfile as sf
        except ImportError as exc:
            raise RemoteBridgeUnavailableError('remote bridge microphone requires httpx, numpy, and soundfile') from exc
        self._httpx = httpx
        self._np = np
        self._sf = sf
        self.base_url = base_url.rstrip('/')
        self.api_token = api_token
        self.request_timeout_seconds = request_timeout_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_seconds = window_seconds
        self.rms_threshold = rms_threshold
        self.peak_threshold = peak_threshold
        self.min_salience = min_salience
        self.transcriber = transcriber
        self.transcript_salience_threshold = transcript_salience_threshold
        self.transcript_min_chars = transcript_min_chars
        self.transcript_cooldown_seconds = transcript_cooldown_seconds
        self.rolling_seconds = max(1.0, float(rolling_seconds))
        self.transcript_window_seconds = max(self.window_seconds, float(transcript_window_seconds))
        self.preamp_gain = max(0.0, float(preamp_gain))
        self.agc_target_rms = max(0.0, float(agc_target_rms))
        self.agc_max_gain = max(1.0, float(agc_max_gain))
        self.vad_enabled = bool(vad_enabled)
        self.vad_trigger_level = float(vad_trigger_level)
        self.vad_min_voiced_seconds = max(0.01, float(vad_min_voiced_seconds))
        self.vad_min_voiced_ratio = max(0.0, float(vad_min_voiced_ratio))
        self.silence_endpoint_seconds = max(0.15, float(silence_endpoint_seconds))
        self.utterance_max_seconds = max(self.silence_endpoint_seconds, float(utterance_max_seconds))
        self.transcript_reply_min_words = max(1, int(transcript_reply_min_words))
        self.store_sound_only_events = bool(store_sound_only_events)
        self._prev_rms = 0.0
        self._prev_zcr = 0.0
        self._last_text = ''
        self._last_transcript = ''
        self._last_transcript_at = 0.0
        self._last_duplicate_transcript_at = 0.0
        self.duplicate_transcript_cooldown_seconds = 1.25
        self._last_transcribe_attempt_at = 0.0
        self._utterance_fragments: list[str] = []
        self._utterance_started_at = 0.0
        self._utterance_last_fragment_at = 0.0
        self._utterance_last_salience = 0.0
        self._utterance_sample_rate = int(sample_rate)
        self._utterance_device = 'remote-bridge'
        self._resolved_sample_rate = int(sample_rate)
        self._resolved_device = 'remote-bridge'
        self._client = httpx.Client(timeout=request_timeout_seconds, verify=verify_tls)

    def _headers(self):
        return {'X-Bridge-Token': self.api_token} if self.api_token else {}

    def record(self, seconds: float | None = None):
        duration = self.window_seconds if seconds is None else float(seconds)
        response = self._client.get(f'{self.base_url}/api/audio/window.wav', params={'seconds': duration}, headers=self._headers())
        if response.status_code != 200:
            raise RemoteBridgeUnavailableError(f'remote microphone unavailable: HTTP {response.status_code}')
        array, sr = self._sf.read(io.BytesIO(response.content), dtype='float32', always_2d=False)
        if getattr(array, 'ndim', 1) > 1:
            array = self._np.asarray(array, dtype='float32').mean(axis=1)
        self._resolved_sample_rate = int(sr or self.sample_rate)
        return self._np.asarray(array, dtype='float32').reshape(-1)

    def poll(self) -> List[SensorObservation]:
        finalized = self._finalize_utterance_if_ready()
        if finalized is not None:
            return [finalized]
        mono = self.record()
        resolved_rate = int(getattr(self, '_resolved_sample_rate', self.sample_rate))
        min_frames = max(4, int(resolved_rate * min(self.window_seconds, 0.08)))
        if mono.size < min_frames:
            return []
        analysis = self._prepare_signal(mono)
        if analysis.size == 0:
            return []
        rms = float(math.sqrt(float((analysis * analysis).mean())))
        peak = float(self._np.max(self._np.abs(analysis)))
        signs = analysis[:-1] * analysis[1:]
        zcr = float((signs < 0).mean()) if signs.size else 0.0
        delta_rms = abs(rms - self._prev_rms)
        delta_zcr = abs(zcr - self._prev_zcr)
        salience = min(1.0, max(rms * 6.0, peak * 4.0, delta_rms * 8.0, delta_zcr * 2.2))
        self._prev_rms = rms
        self._prev_zcr = zcr
        if salience < self.min_salience:
            return []
        texture = 'voice-like banding' if 0.03 <= zcr <= 0.22 else ('noisy texture' if zcr > 0.22 else 'low-frequency texture')
        event = 'sharp transient' if peak >= self.peak_threshold * 1.6 else 'faint audio change'
        transcript_info = self._maybe_transcribe(salience, event, zcr)
        metadata = {
            'rms': round(rms, 6),
            'peak': round(peak, 6),
            'zcr': round(zcr, 6),
            'sample_rate': resolved_rate,
            'device': 'remote-bridge',
            'stt_state': transcript_info['state'],
            'stt_detail': transcript_info['detail'],
            'remote_bridge': True,
        }
        if 'vad_detail' in transcript_info:
            metadata['vad_detail'] = transcript_info['vad_detail']
        if transcript_info['transcript']:
            transcript = str(transcript_info['transcript'])
            metadata['transcript_fragment'] = transcript
            metadata['memory_eligible'] = False
            self._append_utterance_fragment(transcript, salience=salience, sample_rate=resolved_rate)
            if self._utterance_should_flush_immediately(transcript):
                finalized = self._finalize_utterance(reason='punctuation')
                if finalized is not None:
                    return [finalized]
            return []
        elif transcript_info['state'] == 'transcript-rejected':
            detail = str(transcript_info.get('detail', 'rejected'))
            text = f'microphone hears {event} with {texture}. STT: transcript rejected ({detail}).'
            metadata['memory_eligible'] = False
        elif transcript_info['state'] == 'speech-attempted':
            text = f'microphone hears {event} with {texture}. STT: speech attempted.'
            metadata['memory_eligible'] = False
        else:
            detail = str(transcript_info.get('detail') or '').strip()
            if detail and detail not in {'no speech attempt', 'below transcript salience threshold'}:
                text = f'microphone hears {event} with {texture}. STT: sound only ({detail}).'
            else:
                text = f'microphone hears {event} with {texture}. STT: sound only.'
            metadata['memory_eligible'] = bool(self.store_sound_only_events)
        if text == self._last_text and salience < 0.45:
            return []
        self._last_text = text
        return [SensorObservation(source=self.name, text=text, salience=round(salience, 6), metadata=metadata)]

    def status(self) -> SensorStatus:
        detail = f'remote microphone adapter active ({self.base_url} @ {self._resolved_sample_rate} Hz)'
        if self.transcriber is not None:
            detail += f'; speech transcription via {getattr(self.transcriber, "model_name", "transcriber")}'
        if self.vad_enabled:
            detail += '; frontend VAD enabled'
        return SensorStatus(source=self.name, enabled=True, available=True, detail=detail)

    def close(self) -> None:
        self._client.close()

    def _prepare_signal(self, mono):
        data = self._np.asarray(mono, dtype='float32').reshape(-1)
        if data.size == 0:
            return data
        if self.preamp_gain != 1.0:
            data = data * self.preamp_gain
        rms = float(math.sqrt(float((data * data).mean()))) if data.size else 0.0
        if rms > 1e-8 and self.agc_target_rms > 0.0:
            auto_gain = min(self.agc_max_gain, self.agc_target_rms / rms)
            data = data * auto_gain
        data = data - float(data.mean())
        data = self._np.tanh(data)
        return data.astype('float32', copy=False)

    def _maybe_transcribe(self, salience: float, event: str, zcr: float) -> dict[str, object]:
        info: dict[str, object] = {'state': 'sound-only', 'detail': 'no speech attempt', 'transcript': None}
        if self.transcriber is None:
            info['detail'] = 'transcriber disabled'
            return info
        if salience < self.transcript_salience_threshold:
            info['detail'] = 'below transcript salience threshold'
            return info
        now = time.monotonic()
        if (not self._utterance_fragments) and self._last_transcript and (now - self._last_transcript_at) < self.transcript_cooldown_seconds:
            info['state'] = 'speech-attempted'
            info['detail'] = 'transcript cooldown active'
            return info
        if (now - self._last_transcribe_attempt_at) < min(0.25, self.transcript_cooldown_seconds):
            info['state'] = 'speech-attempted'
            info['detail'] = 'attempt throttled'
            return info
        transcript_audio = self._prepare_signal(self.record(self.transcript_window_seconds))
        resolved_rate = int(getattr(self, '_resolved_sample_rate', self.sample_rate))
        min_frames = max(8, int(resolved_rate * min(self.transcript_window_seconds, 0.35)))
        if transcript_audio.size < min_frames:
            info['state'] = 'speech-attempted'
            info['detail'] = 'transcript window too short'
            return info
        vad_info = self._frontend_vad(transcript_audio, resolved_rate)
        info['vad_detail'] = str(vad_info.get('detail') or '')
        if not bool(vad_info.get('speech_like')):
            info['detail'] = f"frontend VAD rejected: {vad_info.get('detail', 'no voiced segment')}"
            return info
        self._last_transcribe_attempt_at = now
        info['state'] = 'speech-attempted'
        info['detail'] = f"speech attempted after VAD accepted: {vad_info.get('detail', 'speech-like segment')}"
        try:
            result = self.transcriber.transcribe(transcript_audio, sample_rate=resolved_rate)
        except Exception as exc:
            info['state'] = 'transcript-rejected'
            info['detail'] = f'transcriber error: {exc}'
            return info
        if result is None:
            info['state'] = 'transcript-rejected'
            info['detail'] = 'transcriber returned no result'
            return info
        text = _clean_transcript(result.text)
        if len(text) < self.transcript_min_chars:
            info['state'] = 'transcript-rejected'
            info['detail'] = 'transcript too short'
            return info
        if (not self._utterance_fragments) and _normalize_mic_text(text) == _normalize_mic_text(self._last_transcript):
            if (now - self._last_duplicate_transcript_at) < self.duplicate_transcript_cooldown_seconds:
                info['state'] = 'transcript-rejected'
                info['detail'] = 'duplicate transcript cooldown'
                return info
            self._last_duplicate_transcript_at = now
        info['state'] = 'transcript-accepted'
        info['detail'] = 'transcript accepted'
        info['transcript'] = text
        return info

    def _append_utterance_fragment(self, transcript: str, salience: float, sample_rate: int) -> None:
        transcript = _clean_transcript(transcript)
        if not transcript:
            return
        now = time.monotonic()
        if self._utterance_fragments and (now - self._utterance_last_fragment_at) > self.silence_endpoint_seconds:
            self._reset_utterance()
        if not self._utterance_fragments:
            self._utterance_started_at = now
        self._utterance_last_fragment_at = now
        self._utterance_last_salience = max(float(salience), float(self._utterance_last_salience))
        self._utterance_sample_rate = int(sample_rate)
        fragment = transcript.strip()
        if self._utterance_fragments:
            prev = self._utterance_fragments[-1]
            if _normalize_mic_text(prev) == _normalize_mic_text(fragment):
                return
            overlap = _suffix_prefix_overlap(prev, fragment)
            if overlap > 0:
                words = fragment.split()
                fragment = ' '.join(words[overlap:])
        if fragment:
            self._utterance_fragments.append(fragment)

    def _utterance_should_flush_immediately(self, transcript: str) -> bool:
        cleaned = transcript.strip()
        if not cleaned:
            return False
        if cleaned.endswith(('.', '?', '!')) and len(cleaned.split()) >= self.transcript_reply_min_words:
            return True
        if self._utterance_started_at and (time.monotonic() - self._utterance_started_at) >= self.utterance_max_seconds:
            return True
        return False

    def _finalize_utterance_if_ready(self):
        if not self._utterance_fragments:
            return None
        if (time.monotonic() - self._utterance_last_fragment_at) < self.silence_endpoint_seconds:
            return None
        return self._finalize_utterance(reason='silence')

    def _finalize_utterance(self, reason: str = 'silence'):
        transcript = _clean_transcript(' '.join(self._utterance_fragments))
        salience = round(float(self._utterance_last_salience), 6)
        sample_rate = int(self._utterance_sample_rate)
        self._reset_utterance()
        if not transcript or len(transcript.split()) < self.transcript_reply_min_words:
            return None
        text = f'microphone catches speech: "{transcript}". STT: transcript accepted ({reason}).'
        metadata = {
            'transcript': transcript,
            'stt_state': 'transcript-accepted',
            'stt_detail': f'transcript accepted ({reason})',
            'sample_rate': sample_rate,
            'device': 'remote-bridge',
            'memory_eligible': True,
            'utterance_final': True,
            'remote_bridge': True,
        }
        self._last_transcript = transcript
        self._last_transcript_at = time.monotonic()
        self._last_text = text
        return SensorObservation(source=self.name, text=text, salience=salience, metadata=metadata)

    def _reset_utterance(self) -> None:
        self._utterance_fragments = []
        self._utterance_started_at = 0.0
        self._utterance_last_fragment_at = 0.0
        self._utterance_last_salience = 0.0

    def _frontend_vad(self, audio, sample_rate: int) -> dict[str, object]:
        data = self._np.asarray(audio, dtype='float32').reshape(-1)
        if not self.vad_enabled:
            return {'speech_like': True, 'detail': 'frontend VAD disabled', 'mode': 'disabled'}
        if data.size == 0:
            return {'speech_like': False, 'detail': 'empty transcript window', 'mode': 'empty'}
        try:
            import torch
            import torchaudio.functional as AF
            waveform = torch.from_numpy(data.copy())
            trimmed = AF.vad(waveform, sample_rate=sample_rate, trigger_level=self.vad_trigger_level)
            if int(trimmed.numel()) > 0:
                trimmed = torch.flip(trimmed, dims=[0])
                trimmed = AF.vad(trimmed, sample_rate=sample_rate, trigger_level=self.vad_trigger_level)
                trimmed = torch.flip(trimmed, dims=[0])
            voiced_samples = int(trimmed.numel())
            voiced_seconds = voiced_samples / max(1, sample_rate)
            voiced_ratio = voiced_samples / max(1, int(data.size))
            speech_like = bool(voiced_seconds >= self.vad_min_voiced_seconds and voiced_ratio >= self.vad_min_voiced_ratio)
            detail = f'mode=torchaudio, voiced={voiced_seconds:.2f}s, ratio={voiced_ratio:.2f}, trigger={self.vad_trigger_level:.1f}'
            return {'speech_like': speech_like, 'detail': detail, 'mode': 'torchaudio'}
        except Exception as exc:
            return {'speech_like': True, 'detail': f'fallback pass-through: {exc}', 'mode': 'fallback'}
