from __future__ import annotations

from collections import deque
from typing import List
import math
import threading
import time

from .base import SensorObservation, SensorStatus
from ..perception import SpeechTranscriber


class MicrophoneUnavailableError(RuntimeError):
    pass


class SoundDeviceMicrophoneAdapter:
    name = "microphone"

    def __init__(
        self,
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
        device: str = '',
        latency: str = 'high',
        rolling_seconds: float = 4.0,
        transcript_window_seconds: float = 1.8,
        preamp_gain: float = 1.0,
        agc_target_rms: float = 0.08,
        agc_max_gain: float = 8.0,
        startup_grace_seconds: float = 0.40,
        vad_enabled: bool = True,
        vad_trigger_level: float = 5.5,
        vad_min_voiced_seconds: float = 0.18,
        vad_min_voiced_ratio: float = 0.08,
        silence_endpoint_seconds: float = 0.65,
        utterance_max_seconds: float = 8.0,
        transcript_reply_min_words: int = 2,
        store_sound_only_events: bool = False,
    ) -> None:
        try:
            import sounddevice as sd  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise MicrophoneUnavailableError(
                "sounddevice and numpy are not installed. Install with: pip install -e .[audio]"
            ) from exc
        self._sd = sd
        self._np = np
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
        self.device = device.strip()
        self.latency = latency
        self.rolling_seconds = max(1.0, float(rolling_seconds))
        self.transcript_window_seconds = max(self.window_seconds, float(transcript_window_seconds))
        self.preamp_gain = max(0.0, float(preamp_gain))
        self.agc_target_rms = max(0.0, float(agc_target_rms))
        self.agc_max_gain = max(1.0, float(agc_max_gain))
        self.startup_grace_seconds = max(0.0, float(startup_grace_seconds))
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
        self._last_text = ""
        self._last_transcript = ""
        self._last_transcript_detail = ""
        self._last_transcript_at = 0.0
        self._last_duplicate_transcript_at = 0.0
        self.duplicate_transcript_cooldown_seconds = 1.25
        self._last_transcribe_attempt_at = 0.0
        self._utterance_fragments: list[str] = []
        self._utterance_started_at = 0.0
        self._utterance_last_fragment_at = 0.0
        self._utterance_last_salience = 0.0
        self._utterance_device = ''
        self._utterance_sample_rate = int(sample_rate)
        self._lock = threading.Lock()
        self._chunks: deque = deque()
        self._buffered_frames = 0
        self._stream = None
        self._stream_started_at = 0.0
        self._stream_error: str | None = None
        self._resolved_sample_rate = int(sample_rate)
        self._resolved_channels = int(channels)
        self._resolved_device = self.device
        self._buffer_frames = max(1, int(self.rolling_seconds * max(1, self.sample_rate)))
        self._ensure_stream_started()

    def _ensure_stream_started(self) -> None:
        if getattr(self, '_stream', None) is not None:
            return
        if getattr(self, '_stream_error', None) is not None:
            raise MicrophoneUnavailableError(self._stream_error)
        if not hasattr(self, '_sd'):
            return
        try:
            device, sample_rate, channels = self._resolve_stream_params()
            self._resolved_device = device or ''
            self._resolved_sample_rate = sample_rate
            self._resolved_channels = channels
            self._buffer_frames = max(1, int(self.rolling_seconds * self._resolved_sample_rate))
            self._stream = self._sd.InputStream(
                samplerate=self._resolved_sample_rate,
                channels=self._resolved_channels,
                dtype='float32',
                device=device or None,
                latency=self.latency,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._stream_started_at = time.monotonic()
        except Exception as exc:  # pragma: no cover - exercised through status / runtime behavior
            self._stream_error = f'failed to open microphone stream: {exc}'
            raise MicrophoneUnavailableError(self._stream_error) from exc

    def _resolve_stream_params(self) -> tuple[str, int, int]:
        device = self.device or ''
        channels = max(1, int(self.channels))
        preferred = max(1, int(self.sample_rate))
        try:
            self._sd.check_input_settings(device=device or None, samplerate=preferred, channels=channels)
            return device, preferred, channels
        except Exception:
            info = self._sd.query_devices(device or None, 'input')
            max_input = int(info.get('max_input_channels') or channels or 1)
            channels = min(channels, max_input) if max_input > 0 else channels
            default_sr = int(round(float(info.get('default_samplerate') or preferred)))
            try:
                self._sd.check_input_settings(device=device or None, samplerate=default_sr, channels=channels)
                return device, default_sr, channels
            except Exception as exc:
                raise MicrophoneUnavailableError(
                    f'input device {device or "default"!r} rejected sample rates {preferred} and {default_sr}: {exc}'
                ) from exc

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - callback path
        _ = frames, time_info
        if status:
            self._stream_error = str(status)
        mono = self._to_mono(indata)
        if mono.size == 0:
            return
        mono = mono.copy()
        with self._lock:
            self._chunks.append(mono)
            self._buffered_frames += int(mono.size)
            while self._buffered_frames > self._buffer_frames and self._chunks:
                removed = self._chunks.popleft()
                self._buffered_frames -= int(removed.size)

    def _snapshot_recent(self, seconds: float):
        frame_count = max(1, int(seconds * max(1, int(getattr(self, '_resolved_sample_rate', self.sample_rate)))))
        with self._lock:
            if not self._chunks:
                return self._np.zeros((0,), dtype='float32')
            chunks = list(self._chunks)
        data = self._np.concatenate(chunks).astype('float32', copy=False)
        if data.size <= frame_count:
            return data
        return data[-frame_count:]

    def record(self, seconds: float | None = None):
        self._ensure_stream_started()
        duration = self.window_seconds if seconds is None else float(seconds)
        return self._snapshot_recent(duration)

    def poll(self) -> List[SensorObservation]:
        self._ensure_stream_started()
        if getattr(self, '_stream_error', None) is not None:
            raise MicrophoneUnavailableError(self._stream_error)
        if getattr(self, '_stream_started_at', 0.0) and (time.monotonic() - self._stream_started_at) < getattr(self, 'startup_grace_seconds', 0.0):
            return []
        finalized = self._finalize_utterance_if_ready()
        if finalized is not None:
            return [finalized]
        mono = self.record()
        mono = self._to_mono(mono)
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

        texture = "voice-like banding" if 0.03 <= zcr <= 0.22 else ("noisy texture" if zcr > 0.22 else "low-frequency texture")
        if peak >= self.peak_threshold * 1.6:
            event = "a sharp transient"
        elif rms >= self.rms_threshold and 0.03 <= zcr <= 0.22:
            event = "voice-like audio activity"
        elif rms >= self.rms_threshold:
            event = "rising ambient audio"
        else:
            event = "a faint audio change"
        text = f"microphone hears {event} with {texture}."
        metadata = {
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "zcr": round(zcr, 6),
            "delta_rms": round(delta_rms, 6),
            "sample_rate": int(getattr(self, '_resolved_sample_rate', self.sample_rate)),
            "device": getattr(self, '_resolved_device', '') or 'default',
        }
        transcript_info = self._maybe_transcribe(salience, event, zcr)
        metadata["stt_state"] = transcript_info["state"]
        metadata["stt_detail"] = transcript_info["detail"]
        if "vad_detail" in transcript_info:
            metadata["vad_detail"] = transcript_info["vad_detail"]
        if transcript_info["transcript"]:
            transcript = str(transcript_info["transcript"])
            metadata["transcript_fragment"] = transcript
            metadata["memory_eligible"] = False
            self._append_utterance_fragment(transcript, salience=salience, sample_rate=resolved_rate)
            if self._utterance_should_flush_immediately(transcript):
                finalized = self._finalize_utterance(reason="punctuation")
                if finalized is not None:
                    return [finalized]
            return []
        elif transcript_info["state"] == "transcript-rejected":
            detail = str(transcript_info.get("detail", "rejected"))
            text = f"microphone hears {event} with {texture}. STT: transcript rejected ({detail})."
            metadata["memory_eligible"] = False
        elif transcript_info["state"] == "speech-attempted":
            text = f"microphone hears {event} with {texture}. STT: speech attempted."
            metadata["memory_eligible"] = False
        else:
            detail = str(transcript_info.get('detail') or '').strip()
            if detail and detail not in {"no speech attempt", "below transcript salience threshold"}:
                text = f"microphone hears {event} with {texture}. STT: sound only ({detail})."
            else:
                text = f"microphone hears {event} with {texture}. STT: sound only."
            metadata["memory_eligible"] = bool(self.store_sound_only_events)
        if text == self._last_text and salience < 0.45:
            return []
        self._last_text = text
        return [
            SensorObservation(
                source=self.name,
                text=text,
                salience=round(salience, 6),
                metadata=metadata,
            )
        ]

    def status(self) -> SensorStatus:
        if getattr(self, '_stream_error', None) is not None:
            return SensorStatus(source=self.name, enabled=True, available=False, detail=self._stream_error)
        resolved_device = getattr(self, '_resolved_device', '') or 'default'
        resolved_rate = int(getattr(self, '_resolved_sample_rate', self.sample_rate))
        detail = (
            f"microphone stream active ({resolved_device} @ {resolved_rate} Hz, "
            f"rolling={float(getattr(self, 'rolling_seconds', self.window_seconds)):.1f}s, window={self.window_seconds:.2f}s)"
        )
        if self.transcriber is not None:
            detail += f"; speech transcription via {getattr(self.transcriber, 'model_name', 'transcriber')}"
        if getattr(self, 'vad_enabled', False):
            detail += f"; frontend VAD enabled"
        return SensorStatus(source=self.name, enabled=True, available=True, detail=detail)

    def close(self) -> None:
        stream = getattr(self, '_stream', None)
        self._stream = None
        if stream is None:
            return None
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        return None

    def _prepare_signal(self, mono):
        data = self._np.asarray(mono, dtype='float32').reshape(-1)
        if data.size == 0:
            return data
        preamp_gain = float(getattr(self, 'preamp_gain', 1.0))
        agc_target_rms = float(getattr(self, 'agc_target_rms', 0.0))
        agc_max_gain = float(getattr(self, 'agc_max_gain', 8.0))
        if preamp_gain != 1.0:
            data = data * preamp_gain
        rms = float(math.sqrt(float((data * data).mean()))) if data.size else 0.0
        if rms > 1e-8 and agc_target_rms > 0.0:
            auto_gain = min(agc_max_gain, agc_target_rms / rms)
            data = data * auto_gain
        data = data - float(data.mean())
        data = self._np.tanh(data)
        return data.astype('float32', copy=False)

    def _to_mono(self, sample):
        data = self._np.asarray(sample, dtype='float32')
        if data.ndim == 1:
            return data
        if data.ndim == 2:
            return data.mean(axis=1)
        return data.reshape(-1)

    def _maybe_transcribe(self, salience: float, event: str, zcr: float) -> dict[str, object]:
        info: dict[str, object] = {"state": "sound-only", "detail": "no speech attempt", "transcript": None}
        if self.transcriber is None:
            info["detail"] = "transcriber disabled"
            return info
        if salience < self.transcript_salience_threshold:
            info["detail"] = "below transcript salience threshold"
            return info
        now = time.monotonic()
        if (not getattr(self, '_utterance_fragments', [])) and self._last_transcript and (now - self._last_transcript_at) < self.transcript_cooldown_seconds:
            info["state"] = "speech-attempted"
            info["detail"] = "transcript cooldown active"
            return info
        if (now - getattr(self, '_last_transcribe_attempt_at', 0.0)) < min(0.25, self.transcript_cooldown_seconds):
            info["state"] = "speech-attempted"
            info["detail"] = "attempt throttled"
            return info
        transcript_window_seconds = float(getattr(self, 'transcript_window_seconds', max(self.window_seconds, 1.8)))
        transcript_audio = self.record(transcript_window_seconds)
        transcript_audio = self._prepare_signal(transcript_audio)
        resolved_rate = int(getattr(self, '_resolved_sample_rate', self.sample_rate))
        min_frames = max(8, int(resolved_rate * min(transcript_window_seconds, 0.35)))
        if transcript_audio.size < min_frames:
            info["state"] = "speech-attempted"
            info["detail"] = "transcript window too short"
            return info
        vad_info = self._frontend_vad(transcript_audio, resolved_rate)
        info["vad_detail"] = str(vad_info.get('detail') or '')
        if not bool(vad_info.get('speech_like')):
            info["detail"] = f"frontend VAD rejected: {vad_info.get('detail', 'no voiced segment')}"
            return info
        self._last_transcribe_attempt_at = now
        info["state"] = "speech-attempted"
        info["detail"] = f"speech attempted after VAD accepted: {vad_info.get('detail', 'speech-like segment')}"
        try:
            result = self.transcriber.transcribe(transcript_audio, sample_rate=resolved_rate)
        except Exception as exc:
            info["state"] = "transcript-rejected"
            info["detail"] = f"transcriber error: {exc}"
            return info
        if result is None:
            info["state"] = "transcript-rejected"
            info["detail"] = "transcriber returned no result"
            return info
        text = _clean_transcript(result.text)
        if len(text) < self.transcript_min_chars:
            info["state"] = "transcript-rejected"
            info["detail"] = "transcript too short"
            return info
        if (not getattr(self, '_utterance_fragments', [])) and _normalize_text(text) == _normalize_text(self._last_transcript):
            if (now - getattr(self, '_last_duplicate_transcript_at', 0.0)) < getattr(self, 'duplicate_transcript_cooldown_seconds', 1.25):
                info["state"] = "transcript-rejected"
                info["detail"] = "duplicate transcript cooldown"
                return info
            self._last_duplicate_transcript_at = now
        info["state"] = "transcript-accepted"
        info["detail"] = "transcript accepted"
        info["transcript"] = text
        return info

    def _append_utterance_fragment(self, transcript: str, salience: float, sample_rate: int) -> None:
        transcript = _clean_transcript(transcript)
        if not transcript:
            return
        now = time.monotonic()
        if not hasattr(self, '_utterance_fragments'):
            self._reset_utterance()
        if getattr(self, '_utterance_fragments', []) and (now - getattr(self, '_utterance_last_fragment_at', 0.0)) > getattr(self, 'silence_endpoint_seconds', 0.65):
            self._reset_utterance()
        if not getattr(self, '_utterance_fragments', []):
            self._utterance_started_at = now
        self._utterance_last_fragment_at = now
        self._utterance_last_salience = max(float(salience), float(getattr(self, '_utterance_last_salience', 0.0)))
        self._utterance_sample_rate = int(sample_rate)
        self._utterance_device = getattr(self, '_resolved_device', '') or 'default'
        fragment = transcript.strip()
        if not fragment:
            return
        if self._utterance_fragments:
            prev = self._utterance_fragments[-1]
            if _normalize_text(prev) == _normalize_text(fragment):
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
        if cleaned.endswith(('.', '?', '!')) and len(cleaned.split()) >= getattr(self, 'transcript_reply_min_words', 1):
            return True
        if getattr(self, '_utterance_started_at', 0.0) and (time.monotonic() - getattr(self, '_utterance_started_at', 0.0)) >= getattr(self, 'utterance_max_seconds', 8.0):
            return True
        return False

    def _finalize_utterance_if_ready(self):
        if not getattr(self, '_utterance_fragments', []):
            return None
        now = time.monotonic()
        if (now - self._utterance_last_fragment_at) < self.silence_endpoint_seconds:
            return None
        return self._finalize_utterance(reason='silence')

    def _finalize_utterance(self, reason: str = 'silence'):
        transcript = _clean_transcript(' '.join(self._utterance_fragments))
        salience = round(float(getattr(self, '_utterance_last_salience', 0.0)), 6)
        sample_rate = int(getattr(self, '_utterance_sample_rate', getattr(self, '_resolved_sample_rate', self.sample_rate)))
        device = getattr(self, '_utterance_device', '') or getattr(self, '_resolved_device', '') or 'default'
        self._reset_utterance()
        if not transcript or len(transcript.split()) < getattr(self, 'transcript_reply_min_words', 1):
            return None
        text = f'microphone catches speech: "{transcript}". STT: transcript accepted ({reason}).'
        metadata = {
            'transcript': transcript,
            'stt_state': 'transcript-accepted',
            'stt_detail': f'transcript accepted ({reason})',
            'sample_rate': sample_rate,
            'device': device,
            'memory_eligible': True,
            'utterance_final': True,
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
        self._utterance_device = ''

    def _frontend_vad(self, audio, sample_rate: int) -> dict[str, object]:
        if not getattr(self, 'vad_enabled', True):
            return {"speech_like": True, "detail": "frontend VAD disabled", "mode": "disabled"}
        data = self._np.asarray(audio, dtype='float32').reshape(-1)
        if data.size == 0:
            return {"speech_like": False, "detail": "empty transcript window", "mode": "empty"}
        try:
            import torch
            import torchaudio.functional as AF

            waveform = torch.from_numpy(data.copy())
            trigger_level = float(getattr(self, 'vad_trigger_level', 5.5))
            trimmed = AF.vad(waveform, sample_rate=sample_rate, trigger_level=trigger_level)
            if int(trimmed.numel()) > 0:
                trimmed = torch.flip(trimmed, dims=[0])
                trimmed = AF.vad(trimmed, sample_rate=sample_rate, trigger_level=trigger_level)
                trimmed = torch.flip(trimmed, dims=[0])
            voiced_samples = int(trimmed.numel())
            voiced_seconds = voiced_samples / max(1, sample_rate)
            voiced_ratio = voiced_samples / max(1, int(data.size))
            speech_like = bool(
                voiced_seconds >= float(getattr(self, 'vad_min_voiced_seconds', 0.18))
                and voiced_ratio >= float(getattr(self, 'vad_min_voiced_ratio', 0.08))
            )
            detail = f"mode=torchaudio, voiced={voiced_seconds:.2f}s, ratio={voiced_ratio:.2f}, trigger={trigger_level:.1f}"
            return {
                "speech_like": speech_like,
                "detail": detail,
                "mode": "torchaudio",
                "voiced_seconds": voiced_seconds,
                "voiced_ratio": voiced_ratio,
            }
        except Exception as exc:
            return self._fallback_vad(data, sample_rate, reason=f"fallback after torchaudio failure: {exc}")

    def _fallback_vad(self, data, sample_rate: int, reason: str = "heuristic") -> dict[str, object]:
        frame_size = max(1, int(sample_rate * 0.02))
        usable = int(data.size // frame_size) * frame_size
        if usable <= 0:
            return {"speech_like": False, "detail": f"{reason}; no complete frames", "mode": "heuristic"}
        frames = data[:usable].reshape(-1, frame_size)
        rms = self._np.sqrt(self._np.mean(frames * frames, axis=1))
        zcr = self._np.mean(frames[:, :-1] * frames[:, 1:] < 0, axis=1) if frame_size > 1 else self._np.zeros((frames.shape[0],), dtype='float32')
        voice_mask = (rms >= max(float(getattr(self, 'rms_threshold', 0.03)) * 0.55, 0.0025)) & (zcr >= 0.02) & (zcr <= 0.25)
        voiced_seconds = float(voice_mask.sum() * frame_size) / max(1, sample_rate)
        voiced_ratio = float(voice_mask.mean()) if voice_mask.size else 0.0
        speech_like = bool(
            voiced_seconds >= float(getattr(self, 'vad_min_voiced_seconds', 0.18))
            and voiced_ratio >= float(getattr(self, 'vad_min_voiced_ratio', 0.08))
        )
        detail = f"mode=heuristic, voiced={voiced_seconds:.2f}s, ratio={voiced_ratio:.2f}, reason={reason}"
        return {
            "speech_like": speech_like,
            "detail": detail,
            "mode": "heuristic",
            "voiced_seconds": voiced_seconds,
            "voiced_ratio": voiced_ratio,
        }


def _clean_transcript(text: str) -> str:
    return " ".join(str(text).strip().split())



def _normalize_text(text: str) -> str:
    return _clean_transcript(text).lower()


def _suffix_prefix_overlap(previous: str, current: str) -> int:
    prev_words = previous.lower().split()
    curr_words = current.lower().split()
    limit = min(len(prev_words), len(curr_words), 6)
    best = 0
    for size in range(1, limit + 1):
        if prev_words[-size:] == curr_words[:size]:
            best = size
    return best
