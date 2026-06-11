from __future__ import annotations

from typing import List, Sequence

from .base import SensorAdapter, SensorObservation, SensorStatus
from .camera import CameraUnavailableError, OpenCVCameraAdapter
from .microphone import MicrophoneUnavailableError, SoundDeviceMicrophoneAdapter
from .remote_bridge import RemoteBridgeUnavailableError, RemoteCameraAdapter, RemoteMicrophoneAdapter
from ..config import WonderBotConfig
from ..perception import PerceptionUnavailableError, build_image_captioner, build_speech_transcriber


class SensorHub:
    def __init__(self, adapters: Sequence[SensorAdapter] | None = None, statuses: Sequence[SensorStatus] | None = None) -> None:
        self.adapters = list(adapters or [])
        self._statuses = list(statuses or [])

    def poll(self) -> List[SensorObservation]:
        observations: List[SensorObservation] = []
        updated_statuses: List[SensorStatus] = []
        for adapter in self.adapters:
            try:
                observations.extend(adapter.poll())
                updated_statuses.append(adapter.status())
            except Exception as exc:
                updated_statuses.append(SensorStatus(source=adapter.name, enabled=True, available=False, detail=str(exc)))
        if updated_statuses:
            self._statuses = updated_statuses
        return observations

    def status(self) -> List[SensorStatus]:
        if self._statuses:
            return list(self._statuses)
        return [adapter.status() for adapter in self.adapters]

    def close(self) -> None:
        for adapter in self.adapters:
            try:
                adapter.close()
            except Exception:
                pass



def build_sensor_hub(config: WonderBotConfig) -> SensorHub:
    adapters: List[SensorAdapter] = []
    statuses: List[SensorStatus] = []

    captioner = None
    caption_detail = "captioning disabled"
    if config.caption.enabled:
        try:
            captioner = build_image_captioner(
                config.caption.model,
                max_new_tokens=config.caption.max_new_tokens,
                device_spec=(config.runtime.caption_device or config.runtime.default_device),
            )
            resolved = getattr(captioner, 'resolved_device', 'cpu')
            caption_detail = f"captioning active ({config.caption.model} on {resolved})"
        except PerceptionUnavailableError as exc:
            caption_detail = f"captioning unavailable: {exc}"

    speech_transcriber = None
    speech_detail = "speech transcription disabled"
    if config.speech.enabled:
        try:
            speech_transcriber = build_speech_transcriber(
                config.speech.model,
                language=config.speech.language,
                device_spec=(config.runtime.speech_device or config.runtime.default_device),
            )
            resolved = getattr(speech_transcriber, 'resolved_device', 'cpu')
            speech_detail = f"speech transcription active ({config.speech.model} on {resolved})"
        except PerceptionUnavailableError as exc:
            speech_detail = f"speech transcription unavailable: {exc}"

    if config.camera.enabled:
        try:
            if config.remote_bridge.enabled and config.remote_bridge.camera_enabled:
                adapters.append(
                    RemoteCameraAdapter(
                        base_url=config.remote_bridge.base_url,
                        api_token=config.remote_bridge.api_token,
                        request_timeout_seconds=config.remote_bridge.request_timeout_seconds,
                        motion_threshold=config.camera.motion_threshold,
                        brightness_threshold=config.camera.brightness_threshold,
                        min_salience=config.camera.min_salience,
                        captioner=captioner,
                        caption_interval_seconds=config.caption.interval_seconds,
                        caption_salience_threshold=config.caption.salience_threshold,
                        caption_min_chars=config.caption.min_chars,
                        verify_tls=config.remote_bridge.verify_tls,
                    )
                )
                statuses.append(SensorStatus(source="camera", enabled=True, available=True, detail=f"remote camera adapter active via {config.remote_bridge.base_url}; {caption_detail}"))
            else:
                adapters.append(
                    OpenCVCameraAdapter(
                        index=config.camera.index,
                        width=config.camera.width,
                        height=config.camera.height,
                        motion_threshold=config.camera.motion_threshold,
                        brightness_threshold=config.camera.brightness_threshold,
                        min_salience=config.camera.min_salience,
                        captioner=captioner,
                        caption_interval_seconds=config.caption.interval_seconds,
                        caption_salience_threshold=config.caption.salience_threshold,
                        caption_min_chars=config.caption.min_chars,
                    )
                )
                statuses.append(SensorStatus(source="camera", enabled=True, available=True, detail=f"camera adapter active; {caption_detail}"))
        except (CameraUnavailableError, RemoteBridgeUnavailableError) as exc:
            statuses.append(SensorStatus(source="camera", enabled=True, available=False, detail=str(exc)))
    else:
        statuses.append(SensorStatus(source="camera", enabled=False, available=False, detail="camera disabled in config"))

    if config.microphone.enabled:
        try:
            if config.remote_bridge.enabled and config.remote_bridge.microphone_enabled:
                adapters.append(
                    RemoteMicrophoneAdapter(
                        base_url=config.remote_bridge.base_url,
                        api_token=config.remote_bridge.api_token,
                        request_timeout_seconds=config.remote_bridge.request_timeout_seconds,
                        sample_rate=config.microphone.sample_rate,
                        channels=config.microphone.channels,
                        window_seconds=config.microphone.window_seconds,
                        rms_threshold=config.microphone.rms_threshold,
                        peak_threshold=config.microphone.peak_threshold,
                        min_salience=config.microphone.min_salience,
                        transcriber=speech_transcriber,
                        transcript_salience_threshold=config.speech.salience_threshold,
                        transcript_min_chars=config.speech.min_chars,
                        transcript_cooldown_seconds=config.speech.cooldown_seconds,
                        rolling_seconds=config.microphone.rolling_seconds,
                        transcript_window_seconds=config.microphone.transcript_window_seconds,
                        preamp_gain=config.microphone.preamp_gain,
                        agc_target_rms=config.microphone.agc_target_rms,
                        agc_max_gain=config.microphone.agc_max_gain,
                        vad_enabled=config.microphone.vad_enabled,
                        vad_trigger_level=config.microphone.vad_trigger_level,
                        vad_min_voiced_seconds=config.microphone.vad_min_voiced_seconds,
                        vad_min_voiced_ratio=config.microphone.vad_min_voiced_ratio,
                        silence_endpoint_seconds=config.microphone.silence_endpoint_seconds,
                        utterance_max_seconds=config.microphone.utterance_max_seconds,
                        transcript_reply_min_words=config.microphone.transcript_reply_min_words,
                        store_sound_only_events=config.microphone.store_sound_only_events,
                        verify_tls=config.remote_bridge.verify_tls,
                    )
                )
                statuses.append(SensorStatus(source="microphone", enabled=True, available=True, detail=f"remote microphone adapter active via {config.remote_bridge.base_url}; {speech_detail}"))
            else:
                adapters.append(
                    SoundDeviceMicrophoneAdapter(
                        sample_rate=config.microphone.sample_rate,
                        channels=config.microphone.channels,
                        window_seconds=config.microphone.window_seconds,
                        rms_threshold=config.microphone.rms_threshold,
                        peak_threshold=config.microphone.peak_threshold,
                        min_salience=config.microphone.min_salience,
                        transcriber=speech_transcriber,
                        transcript_salience_threshold=config.speech.salience_threshold,
                        transcript_min_chars=config.speech.min_chars,
                        transcript_cooldown_seconds=config.speech.cooldown_seconds,
                        device=config.microphone.device,
                        latency=config.microphone.latency,
                        rolling_seconds=config.microphone.rolling_seconds,
                        transcript_window_seconds=config.microphone.transcript_window_seconds,
                        preamp_gain=config.microphone.preamp_gain,
                        agc_target_rms=config.microphone.agc_target_rms,
                        agc_max_gain=config.microphone.agc_max_gain,
                        vad_enabled=config.microphone.vad_enabled,
                        vad_trigger_level=config.microphone.vad_trigger_level,
                        vad_min_voiced_seconds=config.microphone.vad_min_voiced_seconds,
                        vad_min_voiced_ratio=config.microphone.vad_min_voiced_ratio,
                        silence_endpoint_seconds=config.microphone.silence_endpoint_seconds,
                        utterance_max_seconds=config.microphone.utterance_max_seconds,
                        transcript_reply_min_words=config.microphone.transcript_reply_min_words,
                        store_sound_only_events=config.microphone.store_sound_only_events,
                    )
                )
                statuses.append(SensorStatus(source="microphone", enabled=True, available=True, detail=f"microphone adapter active; {speech_detail}"))
        except (MicrophoneUnavailableError, RemoteBridgeUnavailableError) as exc:
            statuses.append(SensorStatus(source="microphone", enabled=True, available=False, detail=str(exc)))
    else:
        statuses.append(SensorStatus(source="microphone", enabled=False, available=False, detail="microphone disabled in config"))

    return SensorHub(adapters=adapters, statuses=statuses)
