from wonderbot.hardware import normalize_device_spec, resolve_device


def test_normalize_device_spec_accepts_cuda_index() -> None:
    assert normalize_device_spec('CUDA:1') == 'cuda:1'


def test_resolve_device_cpu_explicit() -> None:
    resolved = resolve_device('cpu')
    assert resolved.resolved == 'cpu'
    assert resolved.pipeline_device == -1
