# Phase 10.5.3 microphone/STT repair

This pass replaces the blocking `sounddevice.rec(...); sounddevice.wait()` microphone loop with a persistent `InputStream` and a rolling audio buffer.

Highlights:
- explicit microphone device selection (`[microphone].device`)
- automatic sample-rate fallback to the input device default if the configured rate is rejected
- non-blocking rolling audio capture
- separate short analysis window vs longer transcript window
- optional software preamp + AGC before salience and STT
- clearer microphone diagnostics with resolved device and sample rate
