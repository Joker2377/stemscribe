"""Self-check for note_continuity's counters.  Run in envs/tools: python test_note_continuity.py"""
from note_continuity import part_switches, segmentation


def note(pitch: int, start: float, end: float) -> dict:
    """A note dict."""
    return {"pitch": pitch, "start": start, "end": end}


def demo() -> None:
    """A fragmented note, a glued repeat, a short and a long match; a part that switches track once."""
    ref = [note(60, 0.0, 1.0),                          # we split it at 0.5
           note(62, 2.0, 2.4), note(62, 2.5, 3.0),     # repeated: we hold one note through 2.5
           note(64, 4.0, 5.0), note(65, 6.0, 6.2)]     # matched 0.3x (short) / 3x (long)
    est = [note(60, 0.0, 0.5), note(60, 0.5, 1.0), note(62, 2.0, 3.0), note(64, 4.0, 4.3), note(65, 6.0, 6.6)]
    c = segmentation(ref, est)
    assert (c["n_long"], c["split"], c["n_repeat"], c["merge"]) == (5, 1, 1, 1), c
    assert (c["short"], c["long"]) == (1, 2), c  # the glued 62 is also too long
    part = [note(60, 0.0, 0.5), note(62, 0.5, 1.0), note(64, 1.0, 1.5), note(65, 9.0, 9.5)]
    pos = {(60, 0.0): 1.0, (62, 0.5): 1.0, (64, 1.0): 0.0, (65, 9.0): 1.0}  # last pair is a new phrase
    assert part_switches(part, [part], pos) == (1.0, 2)
    print("ok")


if __name__ == "__main__":
    demo()
