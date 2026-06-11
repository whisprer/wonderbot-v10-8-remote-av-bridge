from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

from .hardware import choose_component_device


class PerceptionUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class CaptionResult:
    text: str
    model_name: str
    latency_ms: int


@dataclass(slots=True)
class TranscriptResult:
    text: str
    model_name: str
    latency_ms: int


class ImageCaptioner(Protocol):
    model_name: str

    def caption(self, image: Any) -> CaptionResult | None:
        ...


class SpeechTranscriber(Protocol):
    model_name: str

    def transcribe(self, audio: Any, sample_rate: int) -> TranscriptResult | None:
        ...


class HFImageCaptioner:
    def __init__(self, model_name: str, max_new_tokens: int = 24, device_spec: str = "auto") -> None:
        try:
            from transformers import pipeline
            from PIL import Image
            import numpy as np
        except ImportError as exc:
            raise PerceptionUnavailableError(
                'Captioning requires transformers, torch, pillow, and numpy. Install with: pip install -e .[multimodal]'
            ) from exc
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._device = choose_component_device(device_spec, "auto")
        self.resolved_device = self._device.resolved
        self._pipeline = _build_caption_pipeline(pipeline, model_name, self._device.pipeline_device)
        self._Image = Image
        self._np = np

    def caption(self, image: Any) -> CaptionResult | None:
        pil_image = self._coerce_image(image)
        start = time.perf_counter()
        raw = self._pipeline(pil_image, max_new_tokens=self.max_new_tokens)
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = _extract_generated_text(raw)
        text = _normalize_caption(text)
        if not text:
            return None
        return CaptionResult(text=text, model_name=self.model_name, latency_ms=latency_ms)

    def _coerce_image(self, image: Any):
        if hasattr(image, 'mode') and hasattr(image, 'size'):
            return image
        array = self._np.asarray(image)
        if array.ndim == 2:
            return self._Image.fromarray(array)
        if array.ndim == 3 and array.shape[-1] == 3:
            rgb = array[..., ::-1]
            return self._Image.fromarray(rgb)
        raise ValueError('Unsupported image shape for captioning.')


class HFSpeechTranscriber:
    def __init__(self, model_name: str, language: str = 'en', device_spec: str = "auto") -> None:
        try:
            from transformers import pipeline
            import numpy as np
        except ImportError as exc:
            raise PerceptionUnavailableError(
                'Speech transcription requires transformers, torch, and numpy. Install with: pip install -e .[multimodal]'
            ) from exc
        self.model_name = model_name
        self.language = language.strip() if language else ''
        self._device = choose_component_device(device_spec, "auto")
        self.resolved_device = self._device.resolved
        self._pipeline = _build_asr_pipeline(pipeline, model_name, self._device.pipeline_device)
        self._np = np
        feature_extractor = getattr(self._pipeline, 'feature_extractor', None)
        self._target_sample_rate = int(getattr(feature_extractor, 'sampling_rate', 16000) or 16000)
        generation_config = getattr(getattr(self._pipeline, 'model', None), 'generation_config', None)
        is_multilingual = getattr(generation_config, 'is_multilingual', None)
        lang_to_id = getattr(generation_config, 'lang_to_id', None)
        self._english_only = bool(
            model_name.endswith('.en')
            or (is_multilingual is False)
            or (lang_to_id is None and model_name.endswith('.en'))
        )

    def transcribe(self, audio: Any, sample_rate: int) -> TranscriptResult | None:
        array = self._np.asarray(audio, dtype='float32').reshape(-1)
        if array.size == 0:
            return None
        prepared = self._resample_if_needed(array, sample_rate)
        start = time.perf_counter()
        kwargs = {}
        if self.language and not self._english_only:
            kwargs = {'generate_kwargs': {'language': self.language, 'task': 'transcribe'}}
        raw = self._pipeline({'array': prepared, 'sampling_rate': self._target_sample_rate}, **kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = _extract_transcript_text(raw)
        text = _normalize_transcript(text)
        if not text:
            return None
        return TranscriptResult(text=text, model_name=self.model_name, latency_ms=latency_ms)

    def _resample_if_needed(self, array, sample_rate: int):
        if int(sample_rate) == int(self._target_sample_rate):
            return array.astype('float32', copy=False)
        try:
            import torch
            import torchaudio.functional as AF

            tensor = torch.from_numpy(array.astype('float32', copy=False))
            resampled = AF.resample(tensor, int(sample_rate), int(self._target_sample_rate))
            return resampled.detach().cpu().numpy().astype('float32', copy=False)
        except Exception:
            return _numpy_resample(array, int(sample_rate), int(self._target_sample_rate), self._np)


class NullImageCaptioner:
    model_name = 'none'

    def caption(self, image: Any) -> CaptionResult | None:
        return None


class NullSpeechTranscriber:
    model_name = 'none'

    def transcribe(self, audio: Any, sample_rate: int) -> TranscriptResult | None:
        return None



def build_image_captioner(model_name: str, max_new_tokens: int = 24, device_spec: str = "auto") -> ImageCaptioner:
    return HFImageCaptioner(model_name=model_name, max_new_tokens=max_new_tokens, device_spec=device_spec)



def build_speech_transcriber(model_name: str, language: str = 'en', device_spec: str = "auto") -> SpeechTranscriber:
    return HFSpeechTranscriber(model_name=model_name, language=language, device_spec=device_spec)



def _build_asr_pipeline(pipeline, model_name: str, pipeline_device: int):
    try:
        return pipeline('automatic-speech-recognition', model=model_name, device=pipeline_device)
    except TypeError:
        return pipeline('automatic-speech-recognition', model=model_name)


def _build_caption_pipeline(pipeline, model_name: str, pipeline_device: int = -1):
    tasks = ['image-text-to-text', 'image-to-text']
    errors: list[str] = []
    for task in tasks:
        try:
            try:
                return pipeline(task, model=model_name, device=pipeline_device)
            except TypeError:
                return pipeline(task, model=model_name)
        except KeyError as exc:
            errors.append(f'{task}: {exc}')
    raise PerceptionUnavailableError('No compatible image caption pipeline task was available. Tried: ' + '; '.join(errors))



def _extract_generated_text(raw: Any) -> str:
    if isinstance(raw, list) and raw:
        item = raw[0]
        if isinstance(item, dict):
            return str(item.get('generated_text') or item.get('caption') or item.get('generated_texts') or '')
        return str(item)
    if isinstance(raw, dict):
        return str(raw.get('generated_text') or raw.get('caption') or raw.get('generated_texts') or '')
    return str(raw or '')



def _extract_transcript_text(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get('text') or '')
    if isinstance(raw, list) and raw:
        item = raw[0]
        if isinstance(item, dict):
            return str(item.get('text') or '')
        return str(item)
    return str(raw or '')



def _normalize_caption(text: str) -> str:
    text = ' '.join(text.strip().split())
    lowered = text.lower()
    prefixes = ['a photo of ', 'an image of ', 'a picture of ']
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip(' .')



def _normalize_transcript(text: str) -> str:
    text = ' '.join(text.strip().split())
    return text.strip()



def _numpy_resample(array, source_rate: int, target_rate: int, np_module):
    if source_rate <= 0 or target_rate <= 0 or array.size == 0:
        return array.astype('float32', copy=False)
    if source_rate == target_rate:
        return array.astype('float32', copy=False)
    duration = float(array.size) / float(source_rate)
    target_size = max(1, int(round(duration * float(target_rate))))
    source_positions = np_module.linspace(0.0, duration, num=array.size, endpoint=False)
    target_positions = np_module.linspace(0.0, duration, num=target_size, endpoint=False)
    resampled = np_module.interp(target_positions, source_positions, array).astype('float32')
    return resampled
