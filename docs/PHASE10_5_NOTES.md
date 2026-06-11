# phase 10.5.2 notes

Preferred voice path: local HF TTS with `facebook/mms-tts-eng` as the default model and `pyttsx3` kept as the safety fallback.

This pass keeps OpenAI TTS available as an optional engine, but the default config no longer depends on API credits or network availability.

Included changes:
- HF TTS speaker path in `wonderbot/tts.py`
- SpeechT5-specific synthesizer with CMU Arctic xvector speaker embeddings
- Generic transformers TTS pipeline fallback for alternate HF models
- diagnostics now report `datasets` and `sentencepiece` availability
- new optional dependency group: `hf-voice`

Install for local HF voice:

```bash
py -3.11 -m pip install -e .[live-full,hf-voice,voice,dev]
py -3.11 -m wonderbot.cli --live --camera --microphone --caption --stt --tts --diagnostics
```

If SpeechT5 setup proves heavier than desired, you can switch to another HF TTS model in `configs/default.toml` under `[tts].hf_model` while keeping the same local voice path.
