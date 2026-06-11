# Consolidation notes

## Kept in spirit

- `resonant-llm` → salvage event segmentation and continuous-agent intent.
- `riemann-resonance-llm` → salvage resonance as a core control idea rather than just a style label.
- `woflchess` / `claude's-neural-chess` → salvage ganglion + clocked substrate.
- `wofl-brain` → salvage conceptual framing.

## Explicitly corrected

1. The replacement tokenizer is no longer treated as a direct drop-in LM tokenizer.
2. The agent can run without forcing camera/mic/TTS stacks at boot.
3. Memory remains append-only and non-destructive.
4. The backend is swappable and downstream of event coding.

## Near-term roadmap

1. Add real sensor adapters for camera captioning and STT behind optional extras.
2. Add a learned embedding adapter from event signatures into a frozen pretrained LM.
3. Add a native event-stream trainer so the codec becomes the actual model contract.
4. Add GUI once the core loop is stable.
