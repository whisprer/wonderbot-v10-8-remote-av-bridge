# Docker / NVIDIA container deployment

This repo can run either:

- directly on the host with a Python venv, or
- inside Docker with the NVIDIA Container Toolkit.

The container path is useful when you want reproducible installs and fast rollback.
The host keeps the NVIDIA driver, Docker, and the mounted weights SSD. The container
holds the WonderBot runtime stack.

## Layout assumptions

- repo checked out somewhere convenient on the server
- weights SSD mounted at `/srv/weights`
- persistent WonderBot data stored at `/srv/weights/wonderbot`

## Quick start

```bash
cp .env.example .env
# edit .env if your mount paths differ

docker compose build
# sanity-check GPU visibility
COMPOSE_PROFILES=diag docker compose run --rm wonderbot-diagnostics
# start the main container

docker compose up -d wonderbot
# follow logs

docker compose logs -f wonderbot
```

## Interactive CLI

```bash
docker compose run --rm wonderbot wonderbot --config /app/configs/profiles/dual-p40-server-qwen14b.toml --diagnostics
```

## Suggested first model

Start with the dual-P40 14B profile:

- `configs/profiles/dual-p40-server-qwen14b.toml`
- Qwen 14B in 4-bit
- `device_map = "auto"`
- state/cache/offload on the weights SSD

Use the 32B profile only after the 14B path is stable.

## Notes

- The mounted `/srv/weights/wonderbot` directory is the canonical home for:
  - HF cache
  - offload directory
  - state files
  - optional workspace
- This image intentionally keeps TTS on CPU in the default server profile.
- If you want to pin visible GPUs, set `NVIDIA_VISIBLE_DEVICES` in `.env`.
- If a model is gated, set `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` in `.env`.


## Networking

No host port is exposed by default because the current default runtime is the interactive CLI, not an HTTP API service. If you later add a web/UI or API layer, expose only that service port and keep the core agent internal.
