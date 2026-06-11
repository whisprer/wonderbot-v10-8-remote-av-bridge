# wonderbot-v10.7

A clean consolidation base for the archived LLM/agent experiments, now extended through **phase 10 planning, execution, controlled tool/action runs, local HF TTS, and a server-ready runtime/device layer for dual-GPU deployment**.

This repo is deliberately **not** another fragile wrapper around a standard tokenizer pretending to be tokenizerless.
Instead it separates the system into clear layers:

1. **Event codec** — resonant segmentation + lossless byte encoding + feature signatures.
2. **Memory** — append-only, priority-ranked, searchable, non-destructive.
3. **Ganglion** — a clocked CA bus that gives the agent a continuously evolving internal substrate.
4. **LLM backend** — swappable. The default backend is a no-dependency local LVTC-style backend with a grounded path plus a guarded imagination branch; optional HuggingFace support can be enabled later.
5. **Live perception** — optional camera and microphone adapters, plus optional caption/STT enrichers.
6. **Stability shell** — focus state, anti-chatter cooldowns, replay logging, diagnostics, and optional TTS.
7. **Journal layer** — summaries, extracted tasks, beliefs, unfinished threads, and replay-based reflection.
8. **Memory lifecycle** — long-term promotion, sleep consolidation, dream/rehearsal synthesis, and decay/archiving.
9. **Self-model + goals** — durable preferences/constraints/identity, persistent work queue, focused goal anchor, and recall weighted by what the agent is actively trying to do.
10. **Planning + execution** — multi-step plans, dependencies, blockers, action intents, executable next-step selection, and progress updates driven by reported outcomes.
11. **Voice upgrade** — local HF TTS as the preferred voice path with pyttsx3 fallback and OpenAI TTS kept optional.

That split is the point: the old projects drifted because the “new tokenizer” was treated as if it could be dropped into a pretrained LM without retraining the representational contract. This base stops doing that.

## What phase 6 adds

- **Journal store** in `state/journal.json`.
- **Consolidation engine** that turns recent episodic memories into:
  - summaries
  - tasks
  - beliefs/preferences
  - unfinished threads
- **Replay-based reflection** that comments on imagination use, suppressions, and live-sensor influence.
- **Auto-consolidation** after a configurable number of explicit user turns.
- New CLI commands:
  - `/consolidate`
  - `/reflect`
  - `/journal [kind] [n]`
  - `/tasks`
  - `/beliefs`
  - `/threads`


## What phase 7 adds

- **Long-term memory store** in `state/long_term_memory.json`.
- **Sleep cycle** that promotes durable summaries/tasks/beliefs/threads into long-term memory.
- **Dream cycle** that synthesizes adjacent long-term strands without making imagination the default response mode.
- **Mixed recall**: normal context retrieval now blends short-term working memory with long-term memory hits.
- New CLI commands:
  - `/sleep`
  - `/dream`
  - `/ltm [kind] [n]`
  - `/remember <query>`
  - `/stm [n]`


## What phase 8 adds

- **Self-model store** in `state/self_model.json`.
- **Goal store** in `state/goals.json`.
- Automatic capture of:
  - identity cues
  - preferences
  - constraints
  - explicit or implied goals
- **Focused goal anchor** that persists across turns.
- **Goal-weighted recall**, so context is selected by active work as well as by similarity.
- New CLI commands:
  - `/self [kind] [n]`
  - `/preferences`
  - `/goals [status] [n]`
  - `/goal add <text>`
  - `/goal done <id>`
  - `/goal block <id> [note]`
  - `/goal focus <id>`
  - `/goal progress <id> <0..1>`
  - `/queue [n]`


## What phase 9 adds

- **Plan store** in `state/plans.json`.
- **Multi-step plans** linked to goals when useful.
- **Step dependencies** and blocker notes.
- **Action intents** captured from plan and step language.
- **Executable next-step selection** so the agent can surface what is actually unblocked.
- **Outcome-driven progress updates**: when you report that something was fixed, implemented, blocked, or in progress, the matching plan step and linked goal can update automatically.
- New CLI commands:
  - `/plans [status] [n]`
  - `/plan add <text>`
  - `/plan show <id>`
  - `/plan focus <id>`
  - `/plan done <id>`
  - `/plan block <id> [note]`
  - `/plan step add <plan_id> <text>`
  - `/plan step doing|done|block <plan_id> <step_id> [note]`
  - `/plan step depends <plan_id> <step_id> <dep_step_id>`
  - `/next [n]`



## What phase 10 adds

- **Action/tool registry** in `wonderbot/execution.py`.
- **Execution log** in `state/action_runs.json`.
- Controlled tool runs with **dry-run by default** and explicit commit mode.
- **Plan-step execution** that can resolve step intents into safe internal actions.
- New CLI commands:
  - `/tools`
  - `/runs [n]`
  - `/act run <tool> [key=value ...] [--commit]`
  - `/act step <plan_id> <step_id> [--commit]`
  - `/act next [n] [--commit]`

## What phase 10.5.2 adds

- **Local HF TTS** as the preferred voice backend using `facebook/mms-tts-eng`.
- Default local speaker embedding set via the CMU Arctic xvector set.
- Existing **pyttsx3** path kept as fallback when local HF dependencies or model assets are unavailable.
- **OpenAI TTS** kept available as an optional engine rather than the default.
- Configurable voice settings in `[tts]`, including HF model selection, optional OpenAI engine settings, and playback backend.


## What this repo does now

- Runs immediately with **no third-party dependencies**.
- Uses a **local LVTC-style controlled-imagination backend** by default.
- Supports an interactive, always-on-ish CLI agent that forms and searches memory continuously.
- Uses a **replacement tokenizer architecture where it is actually sound**: segmentation, salience, memory, and internal event coding.
- Keeps the LLM backend abstract so you can stay lightweight now, plug in HF later, or replace the backend entirely with a future native event-stream model.
- Supports **optional live camera and microphone sensing**.
- Supports **optional image captioning and speech transcription enrichment** behind the same sensor contract.
- Supports **optional local or API-backed voice output** without making voice a hard dependency.
- Supports **durable journaled consolidation**, so the system can accumulate structured takeaways instead of only raw turns.
- Supports a **memory lifecycle** where durable knowledge can survive beyond the recent episode stream.

## What this repo does not pretend to do

- It does **not** claim that a pretrained LM has become tokenizerless.
- It does **not** require camera, mic, Whisper, BLIP, or TTS just to boot.
- It does **not** destroy memory entries when consolidating.
- It does **not** force imagination into every turn.

## Quick start

```bash
py -3.11 -m wonderbot.cli
```

Or after install:

```bash
py -3.11 -m pip install -e .
wonderbot
```

## What phase 10.7 adds

- **Server-ready HF backend controls** for a serious inner language model.
- **4-bit HF loading support** (bitsandbytes) for Pascal-class deployment targets.
- **Per-component runtime routing** that stays portable between CPU, tactical single-GPU, and dual-GPU servers.
- **Dual-P40 deployment assets**, including:
  - `configs/profiles/dual-p40-server-qwen14b.toml`
  - `configs/profiles/dual-p40-server-qwen32b-experimental.toml`
  - `scripts/server/bootstrap_dual_p40_ubuntu22.sh`
  - `scripts/server/verify_dual_p40.py`
  - `docs/DUAL_P40_SERVER_DEPLOYMENT.md`
- **HF backend chat-template support** so modern chat models such as Qwen3 can be used as the inner talker without throwing away WonderBot's custom shell.

## Live sensing

Sensor-only live mode:

```bash
py -3.11 -m pip install -e .[live]
py -3.11 -m wonderbot.cli --live --camera --microphone
```

Phase 4/5/6/10.5 live mode with captioning, STT, and upgraded TTS:

```bash
py -3.11 -m pip install -e .[live-full,hf-voice,voice,dev]
py -3.11 -m wonderbot.cli --live --camera --microphone --caption --stt --tts --diagnostics
```

## Useful commands

- `/sensors` — show adapter and voice availability
- `/sense` — poll sensors once immediately
- `/watch 20` — run 20 live polling ticks with the configured interval
- `/memory 10` — inspect what actually got stored
- `/stm 10` — inspect short-term working memory explicitly
- `/ltm 10` — inspect long-term memory
- `/remember coherence` — search long-term memory
- `/focus` — inspect the current active focus and goal anchor
- `/self preference 10` — inspect stored self-model facets
- `/preferences` — inspect stored preferences only
- `/goals active 10` — inspect persistent goals
- `/goal add improve live sensor coherence` — add a goal manually
- `/goal focus <id>` — focus a goal explicitly
- `/plans active 10` — inspect active plans
- `/plan add phase 9 execution plan` — add a plan manually
- `/plan show <id>` — inspect a plan and its steps
- `/plan step add <plan_id> wire command handlers` — add a step
- `/next 8` — inspect currently executable next steps
- `/tools` — list built-in action tools and their intent mapping
- `/runs 10` — inspect recent tool/action executions
- `/act run diagnostics` — dry-run or run a specific built-in tool
- `/act step <plan_id> <step_id> --commit` — execute a specific plan step
- `/act next 3` — preview the next executable steps and their mapped tools
- `/queue 10` — inspect the active work queue
- `/diagnostics` — dump runtime diagnostics
- `/voice on` / `/voice off` — enable or disable speaking for the current run
- `/consolidate` — synthesize a session summary + tasks + beliefs + threads
- `/sleep` — promote durable knowledge into long-term memory and decay weak entries
- `/dream` — generate dream/rehearsal synthesis entries only
- `/reflect` — run the replay-based reflection pass only
- `/journal summary 5` — inspect the latest summary entries
- `/tasks`, `/beliefs`, `/threads` — inspect structured takeaways

## Replay log

Every significant runtime event is appended as JSONL to `state/replay.jsonl`, including:

- startup diagnostics
- sensor observations
- dropped sensor events
- memory writes
- response suppressions
- generated turns
- spontaneous responses
- voice output events
- journal summaries/tasks/beliefs/threads/reflections

That gives you a clean basis for later replay, tuning, and behavior audits.

## Optional HuggingFace backend

```bash
py -3.11 -m pip install -e .[hf]
py -3.11 -m wonderbot.cli --backend hf --hf-model distilgpt2
```

Note: the HF backend still uses its own tokenizer internally. That is intentional. The **agent contract** is event-coded text and memory; the backend is only one possible renderer.

## Repo layout

```text
wonderbot/
  agent.py
  cli.py
  config.py
  consolidation.py
  diagnostics.py
  event_codec.py
  ganglion.py
  journal.py
  llm_backends.py
  memory.py
  planner.py
  perception.py
  replay.py
  resonance.py
  tts.py
  sensors/
configs/
  default.toml
docs/
  ARCHITECTURE.md
  CONSOLIDATION_NOTES.md
  LEGACY_MAP.md
  SLEEP_NOTES.md
scripts/
  seed_from_legacy.py
tests/
```


## Dual-P40 server quick start

```bash
bash scripts/server/bootstrap_dual_p40_ubuntu22.sh
source /srv/wonderbot/.venv/bin/activate
wonderbot --config configs/profiles/dual-p40-server-qwen14b.toml --diagnostics
```

That profile keeps WonderBot's memory/sleep/dream/planning shell intact while swapping the lightweight local talker for a serious HF backend.
