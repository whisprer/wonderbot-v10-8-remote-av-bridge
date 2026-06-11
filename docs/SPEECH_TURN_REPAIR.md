# Speech-turn repair (10.5.7)

This pass changes live speech handling in four important ways:

1. Microphone transcript fragments are accumulated into an utterance and only emitted after silence or punctuation.
2. Accepted microphone transcripts bypass raw acoustic salience gating and can trigger replies directly.
3. Sound-only microphone events are quarantined from memory/retrieval by default.
4. Spontaneous reflection is suppressed while live microphone mode is active, to avoid filler interrupting speech turns.

The intent is to make live speech behavior depend on actual transcript text rather than raw microphone spikes.
