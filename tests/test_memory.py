from pathlib import Path

from wonderbot.event_codec import EventCodec
from wonderbot.memory import MemoryStore


def test_memory_add_and_search(tmp_path: Path) -> None:
    codec = EventCodec()
    store = MemoryStore(codec=codec, path=str(tmp_path / "memory.json"), max_active_items=8)
    store.add("Riemann resonance matters here.", source="user")
    store.add("Tokenizer replacement should govern memory, not fake LM alignment.", source="assistant")
    results = store.search("resonance", k=2)
    assert results
    assert "resonance" in results[0].text.lower()


def test_memory_persists(tmp_path: Path) -> None:
    codec = EventCodec()
    path = tmp_path / "memory.json"
    store = MemoryStore(codec=codec, path=str(path))
    store.add("Persistent memory item.", source="user")
    store.save()
    clone = MemoryStore(codec=codec, path=str(path))
    assert clone.items
    assert clone.items[0].text == "Persistent memory item."
