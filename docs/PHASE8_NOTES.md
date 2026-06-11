# Phase 8 notes

Phase 8 adds a persistent self-model and explicit goals on top of the phase 7 sleep/dream lifecycle.

## Core idea

The agent should not retrieve context only by semantic similarity. It should also retrieve by:

- current focused goal
- durable preferences and constraints
- persistent identity cues
- unfinished work

That makes replies feel more continuous and intentional.

## New state files

- `state/self_model.json`
- `state/goals.json`

## New behaviors

- auto-capture `i prefer ...`, `please keep ...`, `avoid ...`, `my name is ...`, `call me ...`, `we should ...`, `next we should ...`, `need to ...`
- focused goal anchor survives across turns
- active goals get converted into pseudo-memory context items during recall
- self-model facets get converted into pseudo-memory context items during recall
- memory writes are back-linked to matching goals when relevant

## Limits

The extraction rules are intentionally heuristic and conservative. They are there to create a stable explicit state layer, not to pretend full semantic planning has been solved.
