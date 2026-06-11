# WonderBot Remote AV Bridge

This bridge lets a desktop or laptop with the *real* webcam and microphone stream camera frames and microphone audio to a headless/server-hosted WonderBot instance.

## Pieces

- **Server bridge**: runs near WonderBot and exposes HTTP ingest + latest media endpoints.
- **Desktop bridge client**: runs on the local machine with the webcam and microphone.
- **Remote sensor adapters**: let WonderBot poll the bridge instead of local `/dev/video0` and `/dev/snd`.

## Install extras

### Server / container side

```bash
pip install -e .[bridge,multimodal,hf,hf-voice]
```

### Desktop / client side

```bash
pip install -e .[bridge]
```

## Start the server bridge

```bash
export WONDERBOT_BRIDGE_TOKEN='change-me'
python -m wonderbot.bridge server --host 0.0.0.0 --port 8765
```

## Start the desktop bridge client

```bash
set WONDERBOT_BRIDGE_TOKEN=change-me
python -m wonderbot.bridge client \
  --server-url http://SERVER_IP:8765 \
  --camera-index 0 \
  --sample-rate 48000 \
  --channels 1 \
  --source-name desktop-bridge
```

On Linux/macOS use `export` instead of `set`.

## Use the remote-bridge WonderBot profile

```bash
python -m wonderbot.cli --config configs/profiles/dual-p40-server-qwen14b-remote-bridge.toml --diagnostics
```

## Security note

The bridge is intentionally simple and token-protected, but it is not a full zero-trust media gateway. Keep it on your LAN or behind your own tunnel/VPN.
