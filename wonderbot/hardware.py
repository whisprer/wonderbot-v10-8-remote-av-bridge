from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_DEVICE_PREFIXES = ('cpu', 'cuda', 'mps')


@dataclass(slots=True)
class ResolvedDevice:
    requested: str
    resolved: str
    available: bool
    reason: str
    pipeline_device: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            'requested': self.requested,
            'resolved': self.resolved,
            'available': self.available,
            'reason': self.reason,
            'pipeline_device': self.pipeline_device,
        }



def normalize_device_spec(spec: str | None, *, fallback: str = 'auto') -> str:
    cleaned = (spec or '').strip().lower()
    if not cleaned:
        cleaned = fallback
    if cleaned == 'gpu':
        cleaned = 'cuda'
    if cleaned == 'cuda:auto':
        cleaned = 'cuda'
    if cleaned.startswith('cuda:'):
        suffix = cleaned.split(':', 1)[1]
        if suffix.isdigit():
            return cleaned
        return fallback
    if cleaned in {'auto', 'cpu', 'cuda', 'mps'}:
        return cleaned
    return fallback



def resolve_device(spec: str | None, *, fallback: str = 'auto') -> ResolvedDevice:
    requested = normalize_device_spec(spec, fallback=fallback)
    try:
        import torch
    except Exception:
        return ResolvedDevice(requested=requested, resolved='cpu', available=False, reason='torch unavailable', pipeline_device=-1)

    if requested == 'auto':
        if torch.cuda.is_available():
            return ResolvedDevice(requested='auto', resolved='cuda:0', available=True, reason='auto-selected first CUDA device', pipeline_device=0)
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            return ResolvedDevice(requested='auto', resolved='mps', available=True, reason='auto-selected MPS device', pipeline_device=-1)
        return ResolvedDevice(requested='auto', resolved='cpu', available=True, reason='no accelerator available; using CPU', pipeline_device=-1)

    if requested == 'cpu':
        return ResolvedDevice(requested='cpu', resolved='cpu', available=True, reason='CPU selected explicitly', pipeline_device=-1)

    if requested == 'mps':
        mps = getattr(torch.backends, 'mps', None)
        if mps is not None and mps.is_available():
            return ResolvedDevice(requested='mps', resolved='mps', available=True, reason='MPS available', pipeline_device=-1)
        return ResolvedDevice(requested='mps', resolved='cpu', available=False, reason='MPS requested but unavailable; falling back to CPU', pipeline_device=-1)

    if requested == 'cuda':
        if torch.cuda.is_available():
            return ResolvedDevice(requested='cuda', resolved='cuda:0', available=True, reason='CUDA selected explicitly', pipeline_device=0)
        return ResolvedDevice(requested='cuda', resolved='cpu', available=False, reason='CUDA requested but unavailable; falling back to CPU', pipeline_device=-1)

    if requested.startswith('cuda:'):
        try:
            index = int(requested.split(':', 1)[1])
        except ValueError:
            return ResolvedDevice(requested=requested, resolved='cpu', available=False, reason='invalid CUDA device index; falling back to CPU', pipeline_device=-1)
        if torch.cuda.is_available() and 0 <= index < torch.cuda.device_count():
            return ResolvedDevice(requested=requested, resolved=requested, available=True, reason=f'CUDA device {index} selected explicitly', pipeline_device=index)
        return ResolvedDevice(requested=requested, resolved='cpu', available=False, reason=f'CUDA device {index} unavailable; falling back to CPU', pipeline_device=-1)

    return ResolvedDevice(requested=requested, resolved='cpu', available=False, reason='unsupported device spec; falling back to CPU', pipeline_device=-1)



def choose_component_device(component_spec: str | None, default_spec: str | None) -> ResolvedDevice:
    component = normalize_device_spec(component_spec, fallback='')
    if component:
        return resolve_device(component, fallback='auto')
    return resolve_device(default_spec or 'auto', fallback='auto')



def ensure_offload_dir(path: str | Path) -> str:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return str(target)



def parse_torch_dtype(spec: str | None):
    cleaned = (spec or 'auto').strip().lower()
    try:
        import torch
    except Exception:
        return None
    mapping = {
        'auto': 'auto',
        'float16': torch.float16,
        'fp16': torch.float16,
        'half': torch.float16,
        'bfloat16': getattr(torch, 'bfloat16', None),
        'bf16': getattr(torch, 'bfloat16', None),
        'float32': torch.float32,
        'fp32': torch.float32,
    }
    return mapping.get(cleaned, 'auto')



def runtime_hardware_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {
        'cuda_available': False,
        'cuda_device_count': 0,
        'devices': [],
    }
    try:
        import torch
    except Exception as exc:
        summary['detail'] = f'torch unavailable: {exc}'
        return summary

    cuda_available = bool(torch.cuda.is_available())
    summary['cuda_available'] = cuda_available
    summary['cuda_device_count'] = int(torch.cuda.device_count() if cuda_available else 0)
    if cuda_available:
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    'index': index,
                    'name': props.name,
                    'total_memory_mb': int(props.total_memory // (1024 * 1024)),
                    'major': int(props.major),
                    'minor': int(props.minor),
                }
            )
        summary['devices'] = devices
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None:
        summary['mps_available'] = bool(mps.is_available())
    return summary
