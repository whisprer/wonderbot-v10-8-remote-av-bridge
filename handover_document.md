# WonderBot Remote AV + STT + Qwen Recovery Handover

## Purpose

This handover documents the current known-good state of the `wonderbot-v10_8-remote-av-bridge` project on server `woflserv1`, after a long recovery/debugging session.

The main objective was to get WonderBot running on the Linux server with:

* Qwen/Qwen3-14B as the HF backend on dual Tesla P40 GPUs.
* Remote Windows camera/audio bridged into the server.
* WonderBot sensor hub consuming remote camera/mic observations.
* CPU Whisper STT working from remote mic audio.
* Qwen able to summarize remote sensor observations without crashing/OOM.

The current state has reached that milestone in a controlled/manual CLI workflow.

---

## Critical Rule for the Next Assistant

Do **not** blindly patch `wonderbot/llm_backends.py`.

Do **not** restore old backups unless deliberately rolling back to the latest known-good checkpoint.

Do **not** enable full `--live` yet.

Do **not** enable captioning/BLIP yet.

Do **not** enable TTS yet.

Do **not** use Docker.

Do **not** use local `/dev/video0` or local PortAudio mic paths on the server.

Do **not** reinstall bitsandbytes or re-enable 4-bit quantization.

The working path is:

```text
Windows desktop camera/mic
→ WonderBot bridge client
→ Linux bridge server
→ WonderBot remote sensor adapters
→ CPU Whisper STT
→ Qwen HF backend
→ manual CLI commands: /sensors, /sense-summary, /sense-ask
```

---

## Server Paths

Repo/worktree path:

```bash
/srv/wonderbot-v10_8-remote-av-bridge
```

Python venv:

```bash
/home/wofl/.venvs/wb-bridge
```

Environment file:

```bash
/srv/wonderbot-env.sh
```

Important checkpoint directory:

```bash
/srv/wonderbot-checkpoints/av-stt-qwen-working
```

Current likely working profile:

```bash
/srv/wonderbot-v10_8-remote-av-bridge/configs/profiles/dual-p40-server-qwen14b-remote-bridge-fp16-stt-tiny-cpu-sensitive.toml
```

Earlier working no-STT profile:

```bash
/srv/wonderbot-v10_8-remote-av-bridge/configs/profiles/dual-p40-server-qwen14b-remote-bridge-fp16-no-enrich-lowmem.toml
```

---

## Known-Good Environment Setup

Always start server terminals like this:

```bash
cd /srv/wonderbot-v10_8-remote-av-bridge
source /home/wofl/.venvs/wb-bridge/bin/activate
source /srv/wonderbot-env.sh
```

`/srv/wonderbot-env.sh` should contain at least:

```bash
export TMPDIR=/srv/weights/tmp
export TEMP=/srv/weights/tmp
export TMP=/srv/weights/tmp

export HF_HOME=/srv/weights/huggingface
export HF_HUB_CACHE=/srv/weights/huggingface/hub
export HF_XET_CACHE=/srv/weights/huggingface/xet
export HF_ASSETS_CACHE=/srv/weights/huggingface/assets
export HF_HUB_DISABLE_XET=1

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export WONDERBOT_BRIDGE_TOKEN=change-me
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Important: `/tmp` did not exist earlier and was fixed. It should exist with sticky bit:

```bash
ls -ld /tmp
```

Expected shape:

```text
drwxrwxrwt ... /tmp
```

---

## Hardware / Torch Notes

Server has dual Tesla P40 GPUs.

P40 is Pascal / compute capability 6.1. Modern Torch/CUDA wheels may fail with:

```text
CUDA error: no kernel image is available for execution on the device
```

The working fix was to install a P40-compatible PyTorch CUDA 11.8 build:

```bash
python -m pip uninstall -y torch torchvision torchaudio bitsandbytes triton

python -m pip install \
  torch==2.5.1+cu118 \
  torchvision==0.20.1+cu118 \
  torchaudio==2.5.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

Do not upgrade Torch casually. Newer Torch may break P40 kernels.

Do not re-enable bitsandbytes. It was part of the original failure path.

---

## Bridge Server

Run this on the Linux server:

```bash
cd /srv/wonderbot-v10_8-remote-av-bridge
source /home/wofl/.venvs/wb-bridge/bin/activate
source /srv/wonderbot-env.sh

python -m wonderbot.bridge server \
  --host 0.0.0.0 \
  --port 8765 \
  --token change-me
```

Expected:

```text
Uvicorn running on http://0.0.0.0:8765
```

Health endpoint:

```bash
curl -s http://127.0.0.1:8765/health | python -m json.tool
```

Known-good bridge health looked like:

```json
{
  "ok": true,
  "camera": {
    "available": true,
    "width": 640,
    "height": 360,
    "source": "desktop-bridge"
  },
  "audio": {
    "available": true,
    "sample_rate": 48000,
    "channels": 1,
    "source": "desktop-bridge"
  }
}
```

---

## Windows Bridge Client

On Windows PowerShell, using the server LAN IP `192.168.1.191`:

```powershell
py -3.11 -m wonderbot.bridge client `
  --server-url "http://192.168.1.191:8765" `
  --token "change-me" `
  --camera-index 0 `
  --sample-rate 48000 `
  --channels 1 `
  --source-name desktop-bridge
```

Important corrections discovered:

* Client option is `--token`, **not** `--bridge-token`.
* Server checks `X-Bridge-Token`.
* The profile token is `change-me`.
* A typo like `192.1768.1.191` obviously breaks the client.

---

## Bridge Payloads Verified

Direct payload fetch worked:

```bash
curl -fS \
  -H "X-Bridge-Token: change-me" \
  "http://127.0.0.1:8765/api/camera/latest.jpg" \
  -o /srv/weights/tmp/wonderbot-bridge-test/wonderbot-latest.jpg

curl -fS \
  -H "X-Bridge-Token: change-me" \
  "http://127.0.0.1:8765/api/audio/window.wav?seconds=3" \
  -o /srv/weights/tmp/wonderbot-bridge-test/wonderbot-window.wav
```

Known-good results:

```text
JPEG image data ... 640x360
RIFF ... WAVE audio, IEEE Float, mono 48000 Hz
```

Python’s built-in `wave` module cannot parse the float WAV because it is IEEE float format tag `3`; that is not a bridge failure.

---

## Qwen Smoke Test Result

A standalone smoke test succeeded:

```text
OK: Qwen FP16 non-bnb smoke test completed
```

That proved Qwen can load and generate in FP16 without bitsandbytes on the P40 setup.

Important: the smoke test used safer model load settings including:

```python
torch_dtype=torch.float16
device_map="auto"
max_memory={0: "18GiB", 1: "18GiB", "cpu": "96GiB"}
offload_folder="/srv/weights/offload"
low_cpu_mem_usage=True
attn_implementation="eager"
```

The full WonderBot backend initially did not use eager attention and repeatedly OOMed via:

```text
transformers/integrations/sdpa_attention.py
torch.nn.functional.scaled_dot_product_attention
```

This was eventually fixed by patching the exact active HF loader block.

---

## Important `llm_backends.py` State

Focused inspection showed the active HF loader lives around `wonderbot/llm_backends.py` line ~249 onward.

Relevant active structure:

```python
class HFBackend:
    ...
    def __init__(...):
        ...
        model_kwargs = {'trust_remote_code': self.trust_remote_code}
        ...
        if config.hf_device_map:
            model_kwargs['device_map'] = config.hf_device_map
            model_kwargs['offload_folder'] = ...
            model_kwargs['low_cpu_mem_usage'] = True
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
```

This block was patched so the device-map branch forces:

```python
model_kwargs['offload_folder'] = "/srv/weights/offload"
model_kwargs['low_cpu_mem_usage'] = True
model_kwargs['attn_implementation'] = 'eager'
model_kwargs['max_memory'] = {0: '14GiB', 1: '14GiB', 'cpu': '112GiB'}
```

This was the key fix that allowed `/sense-ask` to work without hitting SDPA OOM.

Also, Qwen visible `<think>` output was disabled by patching all `apply_chat_template(...)` calls to include:

```python
enable_thinking=False
```

Do not remove this; otherwise Qwen spends the whole token budget narrating its reasoning and never answers.

Current known-good text test:

```text
reply with exactly two words: backend alive
```

Expected:

```text
[hf] backend alive
```

---

## Working CLI Launch

Use this command for the current best working manual AV+STT+Qwen workflow:

```bash
cd /srv/wonderbot-v10_8-remote-av-bridge
source /home/wofl/.venvs/wb-bridge/bin/activate
source /srv/wonderbot-env.sh

python -m wonderbot.cli \
  --config /srv/wonderbot-v10_8-remote-av-bridge/configs/profiles/dual-p40-server-qwen14b-remote-bridge-fp16-stt-tiny-cpu-sensitive.toml \
  --backend hf \
  --hf-device-map auto \
  --diagnostics
```

Do **not** add `--live` yet.

---

## Working CLI Commands

### `/sensors`

Known-good output:

```text
- [camera] enabled, available: remote camera adapter active via http://127.0.0.1:8765; captioning disabled
- [microphone] enabled, available: remote microphone adapter active via http://127.0.0.1:8765; speech transcription active (openai/whisper-tiny.en on cpu)
- [voice] disabled, unavailable: voice output disabled in config
```

### `/sense-summary`

This command was patched to use `bot.sensor_hub.poll()` directly, not `bot.poll_sensors()`.

Purpose:

* Poll remote sensors.
* Print raw sensor observations.
* Produce deterministic, non-Qwen summary.
* Avoid state save, cooldown, Qwen response, memory expansion, and OOM.

Known-good output:

```text
[microphone] microphone catches speech: "this. Okay, here we go.". STT: transcript accepted (punctuation). (salience=1.00)
[sensor-summary] remote camera produced no new salient observation; remote microphone observed microphone catches speech: "this. Okay, here we go.". STT: transcript accepted (punctuation)..
```

Minor issue: duplicate punctuation / wording is ugly, but functional.

### `/sense-ask`

This command was patched to:

* Poll `bot.sensor_hub.poll()` directly.
* Print raw sensor observations.
* Build compact observation text.
* Call `bot.backend.generate(prompt, [], "concise")`.
* Print `[hf-sensor] ...`.

Known-good output:

```text
[camera] camera sees subtle motion with stable lighting in a dim scene and busy visual texture. (salience=0.31)
[microphone] microphone catches speech: "another short phrase on run this.". STT: transcript accepted (punctuation). (salience=0.11)
[hf-sensor] The camera detects subtle motion in a dim, visually complex scene with stable lighting, while the microphone captures speech: "another short phrase on run this."
```

This is the key working milestone.

---

## STT Status

CPU Whisper STT works.

The working model is:

```toml
[speech]
enabled = true
model = "openai/whisper-tiny.en"
language = "en"
```

Runtime should force CPU:

```toml
[runtime]
speech_device = "cpu"
```

This avoids stealing VRAM from Qwen.

A forced direct STT probe succeeded with:

```text
TRANSCRIPT: 'Do I have to say something one bot testing one two three?'
latency_ms=509
```

The actual sensor-hub STT probe also succeeded repeatedly with transcript-accepted events.

Working profile:

```bash
configs/profiles/dual-p40-server-qwen14b-remote-bridge-fp16-stt-tiny-cpu-sensitive.toml
```

Important: STT transcript quality is imperfect. It sometimes produces cursed/stitched phrases due to rolling windows, VAD thresholds, and Whisper tiny. This is now a tuning problem, not an integration problem.

---

## Known Bad / Avoid

### Do not use full live loop yet

Avoid:

```bash
--live
```

Full `--live` previously caused OOM or context bloat. Manual commands work. Next work should be done manually first.

### Do not enable captioning yet

Captioning tried to load:

```text
Salesforce/blip-image-captioning-base
```

and failed because `transformers` refused `.bin` weight loading with Torch `<2.6` due to the torch.load vulnerability guard. Upgrading Torch is risky because P40 compatibility depends on the current Torch 2.5.1+cu118 build.

Captioning remains disabled.

### Do not enable TTS yet

Voice output is disabled:

```text
[voice] disabled, unavailable: voice output disabled in config
```

Do not enable until AV/STT/Qwen path has been checkpointed and backed up.

### Do not use `bot.poll_sensors()` for the custom summary commands

`bot.poll_sensors()` can create `AgentTurn`s, save state, invoke Qwen automatically, and trigger cooldown/global state behavior.

For low-risk CLI sensor commands, use:

```python
bot.sensor_hub.poll()
```

directly.

---

## Known Mistakes During Session

These are important so the next assistant avoids repeating them.

### 1. `sudo cat > file` does not work

Use:

```bash
sudo tee /path/file >/dev/null <<'EOF'
...
EOF
```

Redirection happens in the user shell, not under sudo.

### 2. Do not run `sudo python3` for venv scripts

This bypasses the venv and misses packages like Pillow.

Use:

```bash
source /home/wofl/.venvs/wb-bridge/bin/activate
python script.py
```

### 3. `/tmp` was missing

It was fixed. If curl output to `/tmp` fails again, check:

```bash
ls -ld /tmp
df -h /tmp /srv/weights
```

### 4. Do not patch by guessing backend signatures

Actual backend signature is:

```python
def generate(self, stimulus: str, memories: List[MemoryItem], style: str, spontaneous: bool = False) -> BackendResult
```

Correct direct call shape:

```python
bot.backend.generate(prompt, [], "concise")
```

### 5. Do not insert runtime attention patch inside `with inference_mode()` incorrectly

A bad patch caused:

```text
IndentationError: expected an indented block after 'with' statement
```

If this happens, restore from:

```bash
/srv/wonderbot-checkpoints/current-good
```

or:

```bash
/srv/wonderbot-checkpoints/av-stt-qwen-working
```

---

## Checkpoint Directories

Most important checkpoint:

```bash
/srv/wonderbot-checkpoints/av-stt-qwen-working
```

It should contain:

```text
llm_backends.py
cli.py
profile.toml
STATUS.txt
```

Earlier checkpoint:

```bash
/srv/wonderbot-checkpoints/av-qwen-working
```

Current-good checkpoint:

```bash
/srv/wonderbot-checkpoints/current-good
```

If things break, restore from `av-stt-qwen-working` first, not older backups.

---

## Git / Backup Status

Important: `/srv/wonderbot-v10_8-remote-av-bridge` was discovered to **not be a Git repo** initially:

```text
fatal: not a git repository (or any of the parent directories): .git
```

A safe backup procedure was proposed:

1. Create local tarball backup under:

```bash
/srv/wonderbot-backups
```

2. Initialise Git if needed:

```bash
git init
git config user.name "wofl"
git config user.email "phineaskfreak@yahoo.co.uk"
```

3. Add a `.gitignore` excluding venvs, caches, models, weights, offload folders, media, etc.

4. Commit code/config/checkpoint docs.

5. Push to GitHub with `gh repo create ...` if GitHub CLI auth is available.

It is not confirmed in this handover whether the final GitHub push completed. Verify first with:

```bash
cd /srv/wonderbot-v10_8-remote-av-bridge
git status
git remote -v
git log --oneline --decorate -5
```

If still not a Git repo, initialise only after preserving a tarball backup.

---

## Suggested Next Steps After Fresh Start

Do these in order.

### Step 1 — Verify current working manual stack

Start bridge server, Windows client, then launch CLI with STT profile:

```bash
python -m wonderbot.cli \
  --config /srv/wonderbot-v10_8-remote-av-bridge/configs/profiles/dual-p40-server-qwen14b-remote-bridge-fp16-stt-tiny-cpu-sensitive.toml \
  --backend hf \
  --hf-device-map auto \
  --diagnostics
```

Then test:

```text
/sensors
/sense-summary
/sense-ask
```

Expected: remote camera/mic active, STT transcript, Qwen `[hf-sensor]` answer.

### Step 2 — Checkpoint to GitHub

Before modifying any more code, get a GitHub commit/tag done.

### Step 3 — Cosmetic cleanup only

Safe cleanup targets:

* `/sense-summary` duplicate punctuation.
* `/sense-ask` wording.
* Maybe suppress repeated Whisper deprecation warnings.
* Prevent accidental `/sense-summary/sense-ask` typed as one command from doing anything confusing.

Do not touch Qwen loader, CUDA, or remote bridge.

### Step 4 — Tune STT quality

Current STT works but transcript quality is imperfect.

Likely tuning areas:

* `microphone.transcript_window_seconds`
* `microphone.window_seconds`
* VAD threshold values
* rolling window behavior
* maybe use `openai/whisper-base.en` on CPU if latency tolerable
* maybe later replace HF pipeline with faster-whisper/whisper.cpp

### Step 5 — Only later attempt full `--live`

Manual commands are stable. Full `--live` previously caused OOM/context bloat.

Before enabling `--live`, implement a strict compact live event path that does not keep stuffing Qwen with memory-expanded sensor turns.

---

## Current Victory Statement

As of this handover, the project has successfully reached:

```text
Remote Windows AV bridge
→ Linux WonderBot sensor hub
→ CPU Whisper STT
→ Qwen/Qwen3-14B HF backend
→ grounded sensor answer via /sense-ask
```

Known-good final observed output:

```text
[hf-sensor] The camera detects subtle motion in a dim, visually complex scene with stable lighting, while the microphone captures speech: "another short phrase on run this."
```

That is the safe baseline. Protect it.
