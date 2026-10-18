"""Stitch two half-offset transcription passes into one note list, keeping only each window's centre.

Pass A runs MuScriptor windows at 0, W, 2W, ... s; pass B at W/2, 3W/2, ... s (no tie forcing in either).
The song is cut into W/2-long segments at W/4 + k*W/2; each segment is the centre of exactly one window,
alternating A, B, A, ... (the first W/4 s belongs to A's first window). A note is clipped to the
segments its pass owns; pieces cut at a segment edge are joined to a same-pitch piece on the other
side of that edge, and a piece cut at its start with nothing to join is dropped: its onset lies in
the other pass's centre, and that pass did not hear it.  A window that ran away (no EOS within the token
limit: demo piano 77.5-82.5 s looped into 1626 notes) hands its centre to the other pass's two windows
around it, whose margins cover it.  Stdlib only.
"""
import math

WINDOW_S = 5.0  # MuScriptor chunk length
EPS = 1e-6


def segment_index(t: float, window: float = WINDOW_S) -> int:
    """Index of the centre segment containing time t (-1 = the leading quarter window, owned by pass A)."""
    return math.floor((t - window / 4) / (window / 2))


def centre_window(seg: int, window: float = WINDOW_S) -> tuple[bool, float]:
    """(is pass B, window start time) of the window whose centre is this segment."""
    if seg >= 0 and seg % 2:
        return True, window / 2 + (seg - 1) // 2 * window
    return False, max(0, seg) // 2 * window


def covering_windows(seg: int, window: float = WINDOW_S) -> list[float]:
    """Start times of the other pass's windows whose margins cover this segment (none for the leading one)."""
    _, start = centre_window(seg, window)
    # ponytail: assumes both covering windows exist; at the song's end the later one may not (rare: last window bad)
    cover = [start - window / 2, start + window / 2]
    return cover if seg >= 0 and cover[0] >= 0 else []


def owned_by(seg: int, pass_b: bool, bad: set[float] = frozenset(), window: float = WINDOW_S) -> bool:
    """Whether pass B (else A) supplies this segment: the window centred on it, unless that window is bad
    and the other pass's covering windows are not."""
    centre_b, start = centre_window(seg, window)
    cover = covering_windows(seg, window)
    if round(start, 3) in bad and cover and not any(round(w, 3) in bad for w in cover):
        centre_b = not centre_b
    return centre_b == pass_b


def segment_span(seg: int, window: float = WINDOW_S) -> tuple[float, float]:
    """Start and end time of a centre segment (segment -1 starts at 0)."""
    a = window / 4 + seg * window / 2
    return max(0.0, a), a + window / 2


def clip_to_owned(notes: list[dict], pass_b: bool, bad: set[float] = frozenset(),
                  window: float = WINDOW_S) -> list[dict]:
    """Pieces of each note inside the segments its pass supplies, flagged where they were cut."""
    pieces = []
    for n in notes:
        for seg in range(segment_index(n["start"], window), segment_index(n["end"] - EPS, window) + 1):
            if not owned_by(seg, pass_b, bad, window):
                continue
            a, b = segment_span(seg, window)
            start, end = max(n["start"], a), min(n["end"], b)
            if end - start > EPS:
                pieces.append({**n, "start": start, "end": end,
                               "cut_start": n["start"] < a - EPS, "cut_end": n["end"] > b + EPS})
    return pieces


def stitch(pieces: list[dict]) -> list[dict]:
    """Join pieces cut at the same edge (same pitch); drop start-cut pieces with no partner."""
    open_ends: dict[tuple[int, float], dict] = {}  # (pitch, edge time) -> note still waiting to be continued
    out = []
    for p in sorted(pieces, key=lambda p: (p["start"], p["pitch"])):
        if p["cut_start"]:
            prev = open_ends.pop((p["pitch"], round(p["start"], 4)), None)
            if prev is None:
                continue
            prev["end"] = p["end"]
            note = prev
        else:
            note = {k: v for k, v in p.items() if k not in ("cut_start", "cut_end")}
            out.append(note)
        if p["cut_end"]:
            open_ends[(p["pitch"], round(p["end"], 4))] = note
    return sorted(out, key=lambda n: (n["start"], n["pitch"]))


def combine(pass_a: list[dict], pass_b: list[dict], bad: set[float] = frozenset(),
            window: float = WINDOW_S) -> list[dict]:
    """One note list from the two passes (times in song seconds); bad = start times of runaway windows."""
    bad = {round(w, 3) for w in bad}
    return stitch(clip_to_owned(pass_a, False, bad, window) + clip_to_owned(pass_b, True, bad, window))
