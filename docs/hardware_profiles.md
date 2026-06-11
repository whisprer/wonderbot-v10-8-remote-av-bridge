# Hardware/device profiles

WonderBot has a runtime device layer. The important knobs are:

- `runtime.default_device`
- `runtime.speech_device`
- `runtime.caption_device`
- `runtime.tts_device`
- `runtime.hf_llm_device`
- `runtime.hf_llm_device_map`
- `runtime.hf_llm_torch_dtype`
- `runtime.hf_llm_load_in_4bit`
- `runtime.hf_llm_quant_type`
- `runtime.hf_llm_compute_dtype`
- `runtime.offload_dir`

Supported device specs are:

- `auto`
- `cpu`
- `cuda`
- `cuda:0`, `cuda:1`, ...
- `mps`

## How to use

Run with a specific profile:

```powershell
py -3.11 -m wonderbot.cli --config configs/profiles/current-box-cpu.toml --diagnostics
```

Override per run without editing files:

```powershell
py -3.11 -m wonderbot.cli --device auto --speech-device cuda:0 --tts-device cpu --diagnostics
```

## Strategy

### Current machine

- CPU-only: use `configs/profiles/current-box-cpu.toml`
- Quadro P1000 tactical use: use `configs/profiles/current-box-p1000.toml` and keep only ASR on GPU first

### Dual Tesla P40 Ubuntu server

Start with `configs/profiles/dual-p40-server-qwen14b.toml`.

Recommended first routing:

- HF LLM -> both P40s through `device_map = "auto"`
- ASR -> `cuda:0`
- captioning -> CPU initially
- TTS -> CPU initially
- durable state/cache/offload -> weights SSD

There is also an experimental stretch profile:

- `configs/profiles/dual-p40-server-qwen32b-experimental.toml`

Only move there after the 14B profile is stable.

## Notes

- Device diagnostics show CUDA availability, device count, and detected GPU names.
- Captioning, ASR, HF TTS, and HF text backends honor the runtime device layer.
- The custom `lvtc` backend remains CPU-oriented by design; the device routing matters most for HF-backed components.
- The P40 pair is useful, but it is **not** one flat 48 GB memory pool. Use `device_map="auto"` and offload thoughtfully.
