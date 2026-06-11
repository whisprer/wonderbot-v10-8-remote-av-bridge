# Phase 7: long-term memory, sleep, and dream cycle

Phase 7 adds a **memory lifecycle** on top of the phase 6 journal layer.

## Why this exists

Before phase 7, WonderBot could:
- sense and react
- write episodic memory
- consolidate recent episodes into journal summaries, tasks, beliefs, threads, and reflections

But it still lacked a distinction between:
- **what happened recently**
- **what should still matter later**

That distinction is the point of phase 7.

## New layers

### 1. Long-term memory store

`state/long_term_memory.json`

This store keeps durable entries with:
- kind
- strength
- use count
- last-accessed time
- evidence
- source ids

It is not just another dump of raw turns. It is the place where durable takeaways live.

### 2. Sleep cycle

`WonderBot.sleep()` and `/sleep`

The sleep pass:
- promotes strong journal entries into long-term memory
- promotes especially strong episodic memories when useful
- reinforces existing long-term entries rather than duplicating them
- decays weak long-term entries and archives those that no longer justify staying active

### 3. Dream cycle

`WonderBot.dream()` and `/dream`

The dream pass creates guarded synthetic links between adjacent long-term entries.
It is meant to resemble rehearsal / associative recombination, not hallucination as a default mode.

## Retrieval policy

Normal context recall now mixes:
- short-term working memory
- long-term memory hits

That means the backend can answer from the current thread while still being nudged by durable beliefs, tasks, and prior distilled knowledge.

## Commands

- `/sleep` — run the promotion + decay pass
- `/dream` — run the synthetic adjacency pass only
- `/ltm [kind] [n]` — inspect long-term memory
- `/remember <query>` — search long-term memory
- `/stm [n]` — inspect short-term working memory explicitly

## Design stance

This still does **not** pretend the agent is a trained native tokenizerless model.
The shell, event codec, memory system, and live loop are custom.
Optional HF caption / STT / text backends remain optional enrichers.
