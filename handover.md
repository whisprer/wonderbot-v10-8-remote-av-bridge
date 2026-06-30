\# WonderBot Live Conversation + Surface Voice Bridge Handover



\*\*Date:\*\* 2026-06-30

\*\*Host:\*\* `woflserv1`

\*\*Current server IP:\*\* `192.168.1.190`

\*\*Project path:\*\* `/srv/wonderbot-v10\_8-remote-av-bridge`

\*\*Surface repo path:\*\* `C:\\github\\wonderbot-v10\_8-remote-av-bridge`



\---



\## 0. Current Victory State



WonderBot has reached a working “live conversational” milestone:



```text

Surface camera/mic

→ Surface bridge client

→ Linux bridge server

→ WonderBot sensor hub

→ CPU Whisper STT

→ Qwen/Qwen3-14B HF backend

→ conversational /sense-watch reply

→ HF TTS generated on Linux

→ TTS routed back through bridge

→ Surface speakers

→ post-speech mic mute prevents self-echo loops

```



This is \*\*not full `--live` mode\*\*.

The safe working mode is still:



```text

WonderBot CLI + /sense-watch forever 1 4

```



The previous handover’s core rule still applies: \*\*do not enable full `--live`, BLIP captioning, Docker, bitsandbytes, or random Torch/CUDA upgrades.\*\* The safe baseline was originally remote AV bridge → CPU Whisper STT → Qwen `/sense-ask`. That baseline has now been extended to live conversational voice.



\---



\## 1. Do Not Touch These Unless Deliberately Working On Them



Do \*\*not\*\* casually edit:



```text

wonderbot/llm\_backends.py

Torch / CUDA / transformers versions

P40 Qwen loader settings

bitsandbytes / 4-bit quantization

BLIP captioning

full --live mode

```



The critical Qwen/P40 loader fix remains:



```python

attn\_implementation = "eager"

max\_memory = {0: "14GiB", 1: "14GiB", "cpu": "112GiB"}

low\_cpu\_mem\_usage = True

offload\_folder = "/srv/weights/offload"

enable\_thinking = False

```



Qwen backend sanity test previously passed:



```text

reply with exactly two words: backend alive

\[hf] backend alive

```



\---



\## 2. Current Known-Good Runtime Files



Important server files modified during this session:



```text

wonderbot/bridge.py

wonderbot/tts.py

wonderbot/config.py

wonderbot/cli.py

configs/profiles/dual-p40-server-qwen14b-remote-bridge-fp16-stt-tiny-cpu-sensitive.toml

launch-kit/

```



Important Surface-side file:



```text

C:\\github\\wonderbot-v10\_8-remote-av-bridge\\launch-kit\\surface-start-wonderbot-bridge.ps1

```



Important profile addition:



```toml

playback\_backend = "bridge"

```



This makes Linux-generated TTS route back through the bridge instead of trying dead Linux ALSA playback.



\---



\## 3. Git / Backup State



The user committed and pushed several working milestones.



At minimum, the following milestones were committed/tagged or intended to be committed/tagged:



```text

Surface AV bridge + STT + Qwen /sense-ask working

Surface remote TTS playback bridge working

Conversational /sense-watch working with self-echo suppression

Short meaningful phrases allowed in /sense-watch

Speaker mute window working

Launch kit working

```



Before doing new work, verify:



```bash

cd /srv/wonderbot-v10\_8-remote-av-bridge



git status --short

git log --oneline --decorate -10

git remote -v

```



Expected remote shape:



```text

https://github.com/whisprer/wonderbot-v10-8-remote-av-bridge.git

```



If untracked backup folders remain, they are probably safe to leave local and \*\*should not be committed\*\*:



```text

patch-backup-\*

```



\---



\## 4. Working Startup Ritual — Launch Kit



\### Server terminal 1: start bridge server



```bash

cd /srv/wonderbot-v10\_8-remote-av-bridge

./launch-kit/server-start-bridge.sh

```



Expected:



```text

Bridge server health OK: http://127.0.0.1:8765/health

```



If needed, stop it:



```bash

./launch-kit/server-stop-bridge.sh

```



Check whether anything is listening manually:



```bash

ss -ltnp | grep ':8765' || echo "No bridge server listening on 8765"

```



\---



\### Surface PowerShell: start Surface bridge client



```powershell

cd C:\\github\\wonderbot-v10\_8-remote-av-bridge

.\\launch-kit\\surface-start-wonderbot-bridge.ps1

```



This should start the client with:



```text

\--server-url http://192.168.1.190:8765

\--token change-me

\--camera-index 0

\--sample-rate 48000

\--channels 1

\--source-name surface-bridge

\--tts-playback

```



The Surface launcher was fixed after an earlier bad version used Bash heredoc syntax inside PowerShell. The working version uses a PowerShell here-string piped into Python.



If PowerShell warns the script came from the internet:



```powershell

Unblock-File .\\launch-kit\\surface-start-wonderbot-bridge.ps1

```



Optional health-only check:



```powershell

.\\launch-kit\\surface-start-wonderbot-bridge.ps1 -HealthOnly

```



Optional without speaker playback:



```powershell

.\\launch-kit\\surface-start-wonderbot-bridge.ps1 -NoTtsPlayback

```



\---



\### Server terminal 2: start WonderBot CLI



```bash

cd /srv/wonderbot-v10\_8-remote-av-bridge

./launch-kit/server-start-cli.sh

```



This launches WonderBot CLI with:



```text

\--backend hf

\--hf-device-map auto

\--tts

\--tts-device cpu

\--diagnostics

```



At the `>` prompt, first check sensors:



```text

/sensors

```



Expected good shape:



```text

\- \[camera] enabled, available: remote camera adapter active ...

\- \[microphone] enabled, available: remote microphone adapter active ...

\- \[voice] enabled, available: HF TTS active ... playback=bridge

```



Then run live conversation mode:



```text

/sense-watch forever 1 4

```



Stop with:



```text

Ctrl+C

```



Quit CLI with:



```text

/quit

```



\---



\## 5. Working Behaviour Confirmed



\### `/sense-ask` working



A successful `/sense-ask` looked like:



```text

\[camera] camera sees strong motion; major scene change...

\[hf-sensor] The camera detected strong motion...

```



\### Speech/STT working



A successful STT event looked like:



```text

\[microphone] microphone catches speech: "Wonder but one two three this is the server". STT: transcript accepted

```



\### Voice bridge working



The Linux server initially failed ALSA playback:



```text

ALSA lib ... Unknown PCM default

aplay: audio open error

```



That was solved by routing TTS back through the bridge to the Surface. WonderBot then audibly spoke from the Surface speaker in a silly “A-Team narrator” style voice.



\### Conversational sense-watch working



Earlier, `/sense-watch` merely narrated reports like:



```text

The microphone detected speech...

```



It was patched to respond conversationally. A good line looked like:



```text

\[hf-sensor] I see, you were just reading what I displayed—no worries, I'm glad we're having a conversation.

```



\### Short phrase gate working



Short meaningful phrases like:



```text

Thank you.

```



now trigger responses.



\### Speaker mute window working



Because the Surface speaker plays WonderBot’s TTS, the Surface mic hears WonderBot’s own voice. The current fix is \*\*half-duplex-lite\*\*:



```text

after WonderBot speaks, skip microphone observations briefly

```



Expected log shape:



```text

\[sense-watch] skipped mic during speaker playback (6.4s remaining).

```



This prevents self-echo spirals.



\---



\## 6. Current Limitations



WonderBot is working, but not polished.



\### Slow-ish response



Current speed is limited by:



```text

Qwen3-14B generation

CPU TTS

/sense-watch polling interval

backend cooldown

post-speech mic mute window

```



Useful current live command:



```text

/sense-watch forever 1 4

```



If too slow, tune carefully. Do not remove echo suppression outright.



\### Voice quality



Current voice is functional but goofy. It uses:



```text

facebook/mms-tts-eng on CPU

```



The user described it as sounding like “the narrator from the A-Team.”



Future target: better voice, likely Piper, OpenAI TTS, or another local TTS option routed through the same bridge playback path.



\### STT quality



Whisper tiny sometimes hallucinates or repeats phrases:



```text

"I'm going to take a look at it..." repeated many times

```



This is a tuning problem, not a dead pipeline.



Future work:



```text

better VAD

different Whisper model

faster-whisper / whisper.cpp

rolling window cleanup

half-duplex timing improvements

```



\### Echo handling



Current post-speech mute works, but it is crude. Better future option:



```text

full half-duplex state machine:

LISTENING → THINKING → SPEAKING → MUTE\_DECAY → LISTENING

```



\---



\## 7. Important Commands



\### Check bridge health from server



```bash

curl -s http://127.0.0.1:8765/health | python -m json.tool

```



Expected source names:



```text

surface-bridge

```



\### Check process on port 8765



```bash

ss -ltnp | grep ':8765' || echo "No bridge server listening on 8765"

```



\### Compile check



```bash

cd /srv/wonderbot-v10\_8-remote-av-bridge

source /home/wofl/.venvs/wb-bridge/bin/activate



python -m py\_compile \\

&#x20; wonderbot/cli.py \\

&#x20; wonderbot/bridge.py \\

&#x20; wonderbot/tts.py \\

&#x20; wonderbot/config.py \\

&#x20; wonderbot/llm\_backends.py

```



\### Basic backend sanity in CLI



At `>` prompt:



```text

reply with exactly two words: backend alive

```



Expected:



```text

\[hf] backend alive

```



\---



\## 8. If Things Break



\### If `/sensors` says connection refused



Bridge server is not running.



Start:



```bash

./launch-kit/server-start-bridge.sh

```



\### If Surface client says `--tts-playback` unrecognized



Surface repo is not patched or using wrong branch/file.



Check:



```powershell

cd C:\\github\\wonderbot-v10\_8-remote-av-bridge

Select-String -Path .\\wonderbot\\bridge.py -Pattern "tts-playback"

```



\### If Linux TTS tries ALSA / `aplay` and fails



Profile or TTS config may not be routing to bridge. Check profile contains:



```toml

playback\_backend = "bridge"

```



\### If WonderBot reports his own speech



Speaker mute window may not be loaded. Restart CLI after patch changes. Check `wonderbot/cli.py` contains:



```python

speaker\_mute\_seconds

```



\### If Qwen OOMs or SDPA errors return



Do \*\*not\*\* upgrade Torch. Do \*\*not\*\* re-enable bitsandbytes.



Check `wonderbot/llm\_backends.py` still forces:



```python

attn\_implementation = "eager"

max\_memory = {0: "14GiB", 1: "14GiB", "cpu": "112GiB"}

```



\---



\## 9. Suggested Next Work



Recommended order:



\### 1. Stabilize launch kit



Make sure all launch scripts are committed and executable:



```bash

chmod +x launch-kit/\*.sh

git status --short

```



\### 2. Add a one-command server status dashboard



Possible command:



```bash

./launch-kit/server-status.sh

```



Should show:



```text

bridge PID

bridge health

latest Surface source

git status

CLI reminder

```



\### 3. Improve response speed



Targets:



```text

shorter Qwen max output for sensor replies

shorter TTS utterances

reduce mute window carefully

reduce cooldown carefully

```



\### 4. Improve voice



Keep bridge playback architecture. Swap TTS backend later.



\### 5. Improve STT



Focus on hallucination reduction and cleaner phrase segmentation.



\### 6. Build proper state machine



Future design:



```text

LISTENING

THINKING

SPEAKING

POST\_SPEECH\_MUTE

LISTENING

```



This will be better than ad hoc cooldown/mute checks.



\---



\## 10. Current Best Mental Model



WonderBot is no longer just “Qwen with sensors.”



Current state is:



```text

a live conversational agent with:

\- remote eyes

\- remote ears

\- STT

\- vision-lite

\- affect-lite

\- conversational Qwen responses

\- server-side TTS

\- Surface speaker playback

\- self-echo suppression

\- scripted launch procedure

```



The magic is working. Protect it.



