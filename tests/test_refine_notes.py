"""Self-check for refine_notes silence cutting and crowded-note trimming."""
import numpy as np

from refine_notes import HOP, POLY_MAX, refine

SR = 44100


def test_note_cut_at_silence() -> None:
    """A note held across a silent gap ends where the gap starts; notes starting in it are dropped."""
    fps = SR / HOP
    power = np.ones((64, int(10 * fps)))
    power[:, int(4 * fps):int(8 * fps)] = 0.0  # silent 4-8 s
    notes = [{"pitch": 69, "start": 1.0, "end": 9.0}, {"pitch": 60, "start": 1.0, "end": 3.0},
             {"pitch": 62, "start": 3.9, "end": 4.1},  # short dip into silence (< SILENCE_MIN_S) stays
             {"pitch": 45, "start": 5.0, "end": 5.17}, {"pitch": 45, "start": 6.0, "end": 7.5}]  # in silence: dropped
    out = refine(notes, power, SR, 0.0, None)
    assert len(out) == 3, out
    held, sound, short = out
    assert abs(held["end"] - 4.0) < 2 * HOP / SR, held
    assert sound["end"] == 3.0 and short["end"] == 4.1, (sound, short)


def test_crowded_notes_trimmed() -> None:
    """With crowd_trim, notes in a > POLY_MAX pile-up end where their (absent) harmonics fade; sparse notes keep
    their length even though their harmonics are absent too."""
    fps = SR / HOP
    power = np.full((1025, int(10 * fps)), 1e-6)
    power[0] = 1.0  # sound everywhere, so no silence cut
    power[1:, int(0.5 * fps):int(0.6 * fps)] = 1.0  # every note's harmonics: a 100 ms attack, then gone
    power[1:, int(5.0 * fps):int(5.1 * fps)] = 1.0
    sparse = [{"pitch": 60, "start": 0.5, "end": 2.0}]
    pile = [{"pitch": 40 + i, "start": 5.0, "end": 9.0} for i in range(POLY_MAX + 1)]
    out = refine(sparse + pile, power, SR, 0.0, None, crowd_trim=True)
    assert out[0]["end"] == 2.0, out[0]
    assert all(n["end"] < 5.2 for n in out[1:]), out[1:]
