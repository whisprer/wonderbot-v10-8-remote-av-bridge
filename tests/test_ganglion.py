from wonderbot.ganglion import Ganglion


def test_ganglion_ticks_and_accepts_signature() -> None:
    ganglion = Ganglion(height=4, width=4, channels=4, bleed=0.01)
    ganglion.write_signature("0123456789abcdef0123456789abcdef")
    before = ganglion.state_summary().mean_value
    ganglion.tick(3)
    after = ganglion.state_summary().mean_value
    assert ganglion.t == 3
    assert after >= 0.0
    assert before >= 0.0
