from __future__ import annotations

import json

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"torch unavailable: {exc}")

payload = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [],
}
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        payload["devices"].append(
            {
                "index": index,
                "name": props.name,
                "total_memory_mb": int(props.total_memory // (1024 * 1024)),
                "major": int(props.major),
                "minor": int(props.minor),
            }
        )
print(json.dumps(payload, indent=2))
