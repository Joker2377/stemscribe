"""Self-check for window_stitch."""
from window_stitch import combine, segment_index


def note(pitch: int, start: float, end: float) -> dict:
    """Note dict as transcribe.py writes it."""
    return {"pitch": pitch, "start": start, "end": end, "instrument": "voice"}


def test_segments() -> None:
    """Centres: A owns [0, 1.25) and [1.25, 3.75), B owns [3.75, 6.25), A [6.25, 8.75) ..."""
    assert [segment_index(t) for t in (0.0, 1.3, 3.8, 6.3)] == [-1, 0, 1, 2]


def test_long_note_joined() -> None:
    """A note both passes hear across several centres comes back whole."""
    out = combine([note(60, 1.0, 9.0)], [note(60, 2.6, 9.0)])
    assert len(out) == 1 and (out[0]["start"], out[0]["end"]) == (1.0, 9.0), out


def test_margin_artifacts_dropped() -> None:
    """B's false onset at its window start (2.5 s, inside A's centre) is dropped; A's note stays."""
    out = combine([note(60, 1.5, 3.0)], [note(60, 2.5, 3.0), note(64, 2.5, 3.0)])
    assert [(n["pitch"], n["start"], n["end"]) for n in out] == [(60, 1.5, 3.0)], out


def test_stuck_note_not_carried() -> None:
    """A note one pass holds on into the other pass's centre ends at the edge if that pass disagrees."""
    out = combine([note(69, 2.0, 9.0)], [note(71, 4.0, 6.0)])
    assert [(n["pitch"], n["start"], n["end"]) for n in out] == [(69, 2.0, 3.75), (71, 4.0, 6.0), ], out


def test_bad_window_replaced() -> None:
    """A runaway B window (2.5-7.5 s) gives its centre (3.75-6.25 s) to A's windows 0-5 and 5-10."""
    spam = [note(80, 4.0 + i * 0.01, 4.5) for i in range(50)]
    out = combine([note(71, 4.0, 6.0)], spam, bad={2.5})
    assert [(n["pitch"], n["start"], n["end"]) for n in out] == [(71, 4.0, 6.0)], out
