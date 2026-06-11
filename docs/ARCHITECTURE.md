# Architecture

## Guiding correction

The old projects conflated three different things:

- a representation problem,
- a continuous-agent problem,
- and a backend generation problem.

This repo unbundles them.

## Layers

### 1. Event codec

The event codec replaces the old “pretend tokenizer replacement” with a clearer contract:

- raw text in
- resonant segmentation
- lossless byte IDs when exact recovery matters
- hashed feature vectors for search and salience
- signatures for routing into the ganglion and resonance field

### 2. Memory

Memory is append-only and non-destructive.
Consolidation archives low-priority active memories when the store grows too large, but protected identity-like items stay active.

### 3. Ganglion / CA bus

The ganglion is a continuously ticking substrate. Event signatures are written into the bus as compact patches. The bus then shifts, updates, and bleeds between channels each tick.

That gives you a living internal state that is not reducible to “wait for prompt -> call model -> exit”.

### 4. Resonance field

Resonance is used as a ranking / gating / internal pressure signal rather than empty naming.

### 5. Backend

The backend is downstream, not upstream.
That is the crucial correction.

- Echo backend: guaranteed local, dependency-free startup.
- HF backend: optional renderer using a normal tokenizer internally.
- Future native backend: event-stream model trained against the codec contract.

## Why this is a better base

Because it gives you something that boots now while preserving the path to a truly native event-coded or tokenizerless-ish model later.
