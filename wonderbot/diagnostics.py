from __future__ import annotations

from importlib import metadata
import platform
from typing import Any, Dict

from .hardware import runtime_hardware_summary


def _safe_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_runtime_diagnostics(
    *,
    config: Any,
    backend_name: str,
    sensor_statuses: list[dict[str, Any]],
    speaker_status: dict[str, Any],
    replay_status: dict[str, Any],
    focus_state: dict[str, Any],
    memory_stats: dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> dict[str, Any]:
    packages = {
        'python': platform.python_version(),
        'transformers': _safe_version('transformers'),
        'torch': _safe_version('torch'),
        'opencv-python': _safe_version('opencv-python'),
        'sounddevice': _safe_version('sounddevice'),
        'soundfile': _safe_version('soundfile'),
        'pillow': _safe_version('pillow'),
        'pyttsx3': _safe_version('pyttsx3'),
        'datasets': _safe_version('datasets'),
        'sentencepiece': _safe_version('sentencepiece'),
        'numpy': _safe_version('numpy'),
    }
    payload = {
        'agent': config.agent.name,
        'backend': backend_name,
        'runtime': {
            'default_device': config.runtime.default_device,
            'speech_device': config.runtime.speech_device,
            'caption_device': config.runtime.caption_device,
            'tts_device': config.runtime.tts_device,
            'hf_llm_device': config.runtime.hf_llm_device,
            'hf_llm_device_map': config.runtime.hf_llm_device_map,
            'hf_llm_torch_dtype': config.runtime.hf_llm_torch_dtype,
            'hf_llm_load_in_4bit': config.runtime.hf_llm_load_in_4bit,
            'hf_llm_quant_type': config.runtime.hf_llm_quant_type,
            'hf_llm_compute_dtype': config.runtime.hf_llm_compute_dtype,
            'hf_llm_double_quant': config.runtime.hf_llm_double_quant,
            'offload_dir': config.runtime.offload_dir,
        },
        'hardware': runtime_hardware_summary(),
        'packages': packages,
        'sensors': sensor_statuses,
        'speaker': speaker_status,
        'replay': replay_status,
        'focus': focus_state,
        'memory': memory_stats,
    }
    if extra:
        payload.update(extra)
    return payload
