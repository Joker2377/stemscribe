"""Alignment check for merge.py: tempo change 90 -> 140 BPM must not drift.

Run: envs/tools/Scripts/python test_merge.py
"""
import tempfile
from pathlib import Path

import mido
import numpy as np

from merge import TPQ, write_midi


def synth_beats() -> tuple[list[float], list[float]]:
    """32 beats at 90 BPM then 64 at 140 BPM, starting at 0.37 s, with a 4/4 downbeat grid."""
    gaps = [60 / 90] * 32 + [60 / 140] * 64
    beats = list(0.37 + np.concatenate([[0], np.cumsum(gaps)]))
    return beats, beats[::4]


def note_on_seconds(path: Path) -> tuple[list[float], list[int]]:
    """Read note_on times back in seconds (mido integrates tempo events) and in ticks."""
    secs, now = [], 0.0
    for msg in mido.MidiFile(path):
        now += msg.time
        if msg.type == "note_on":
            secs.append(now)
    ticks, tick = [], 0
    for msg in mido.MidiFile(path).tracks[1]:
        tick += msg.time
        if msg.type == "note_on":
            ticks.append(tick)
    return secs, ticks


def test_tempo_change_alignment() -> None:
    """Notes on every beat plus off-beat notes come back within 1 ms, downbeats on bar lines."""
    beats, downbeats = synth_beats()
    starts = sorted(beats + [b + 0.123 for b in beats] + [beats[-1] + 3.0])  # off-beat + past last beat
    notes = [{"pitch": 60, "start": t, "end": t + 0.05, "instrument": "acoustic_piano"} for t in starts]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.mid"
        write_midi(path, {"piano": notes}, beats, downbeats)
        secs, ticks = note_on_seconds(path)
    err = np.abs(np.array(secs) - np.array(starts)).max()
    assert err < 0.001, f"max drift {err * 1000:.2f} ms"
    downbeat_ticks = [ticks[starts.index(d)] for d in downbeats]
    assert all(t % (4 * TPQ) == 0 for t in downbeat_ticks), "downbeats off the bar grid"
    print(f"ok: max drift {err * 1000:.3f} ms over {len(starts)} notes")


if __name__ == "__main__":
    test_tempo_change_alignment()
