from wonderbot.event_codec import EventCodec


def test_lossless_roundtrip() -> None:
    codec = EventCodec()
    text = "camera input + memory loop ✅"
    encoded = codec.encode_lossless(text)
    decoded = codec.decode_lossless(encoded)
    assert decoded == text


def test_segment_text_returns_nonempty_chunks() -> None:
    codec = EventCodec(min_segment_chars=3)
    text = "The camera sees motion. The mic hears a click. Memory priority rises."
    segments = codec.segment_text(text)
    assert segments
    assert all(segment.strip() for segment in segments)


def test_analyze_text_produces_signatures() -> None:
    codec = EventCodec()
    events = codec.analyze_text("Real time memory with phase changes.")
    assert events
    assert all(event.signature for event in events)
