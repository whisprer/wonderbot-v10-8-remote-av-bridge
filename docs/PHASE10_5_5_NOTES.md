# Phase 10.5.5 STT Fix

- default STT model changed to distil-whisper/distil-small.en
- transcript window increased to 3.0s
- transcript minimum chars lowered to 1
- duplicate transcript rejection changed to cooldown-based
- microphone logs now include transcript rejection detail
- Whisper task forced to transcribe when language is set
