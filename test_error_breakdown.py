"""Self-check for error_breakdown's classification.  Run in envs/tools: python test_error_breakdown.py"""
from error_breakdown import aligned_refs, breakdown


def note(pitch: int, start: float, end: float = None) -> dict:
    """A note dict; default length 0.5 s."""
    return {"pitch": pitch, "start": start, "end": start + 0.5 if end is None else end}


def demo() -> None:
    """One note of every kind lands in its own bucket; the octave-transposed reference is aligned first."""
    refs = {"guitar": [note(60, 0.0, 2.0), note(62, 3.0), note(64, 4.0), note(65, 5.0), note(67, 6.0)],
            "piano": [note(70, 7.0)], "bass": [note(28, 0.0)]}  # bass MIDI an octave below the sound
    ests = {"guitar": [note(60, 0.0), note(74, 3.0), note(70, 7.0), note(60, 1.0), note(50, 9.0)],
            "piano": [note(64, 4.0)], "bass": [note(40, 0.0)], "drums": []}
    rows = {r["stem"]: r for r in breakdown(aligned_refs(refs, ests), ests)}
    g = rows["guitar"]
    assert (g["extra_octave"], g["extra_leak"], g["extra_inside"], g["extra_other"]) == (1, 1, 1, 1), g
    assert (g["missed_octave"], g["missed_elsewhere"], g["missed_lost"]) == (1, 1, 2), g
    assert rows["bass"]["extra_other"] == rows["bass"]["missed_lost"] == 0, rows["bass"]
    print("ok")


if __name__ == "__main__":
    demo()
