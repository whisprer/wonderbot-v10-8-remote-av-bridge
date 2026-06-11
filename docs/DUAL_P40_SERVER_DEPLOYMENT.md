# Dual Tesla P40 server deployment

This repo now includes a first-class deployment path for an Ubuntu 22.04 server with:

- 2 × Tesla P40 (24 GB each)
- dual Xeon CPUs
- 128 GB RAM
- dedicated weights SSD

## Why this layout

WonderBot's custom shell stays in place. The thing we are swapping is the **inner language backend**:

- keep the event codec / memory / journal / sleep / dream / planning / action shell
- replace the lightweight `lvtc` talker with a serious HF model

The recommended first model is **Qwen/Qwen3-14B** in 4-bit via bitsandbytes.

## Install strategy

1. Mount the weights SSD at `/srv/weights`.
2. Clone the repo to `/srv/wonderbot`.
3. Run `scripts/server/bootstrap_dual_p40_ubuntu22.sh`.
4. Verify CUDA visibility with `python scripts/server/verify_dual_p40.py`.
5. Start WonderBot with the dual-P40 profile.

## Profiles

### Stable first profile

- `configs/profiles/dual-p40-server-qwen14b.toml`
- HF backend
- `Qwen/Qwen3-14B`
- `device_map = "auto"`
- 4-bit NF4 quantization
- speech routed to `cuda:0`
- TTS on CPU

### Experimental bigger profile

- `configs/profiles/dual-p40-server-qwen32b-experimental.toml`
- `Qwen/Qwen3-32B-AWQ`
- only move here after the 14B profile is stable

## Notes

- The weights SSD is treated as the canonical home for cache, offload, and durable state.
- The P40s do **not** act like one flat 48 GB pool. Use `device_map="auto"` and expect dispatch/offload behavior.
- Keep camera/captioning disabled during early speech-turn tuning.
- The next software priority after migration is the speech-turn repair pass.


## Container option

If the server already has Docker and the NVIDIA Container Toolkit configured, you can use the included `Dockerfile` and `docker-compose.yml` instead of a host venv. The container keeps the WonderBot runtime isolated while mounting `/srv/weights/wonderbot` for cache, offload, workspace, and durable state. See `docs/CONTAINER_DEPLOYMENT.md`.
