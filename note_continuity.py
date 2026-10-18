"""How well do our notes hold together?  Note-segmentation and guitar-assignment continuity against Slakh MIDI.

Runs in envs/tools.  Usage: note_continuity.py <variant> [--split=<suffix>] <slakh_track_dir>...   (oracle_* reads work/oracle)
Pitched stems (reference moved by its best octave, like eval_midi):
- split:  share of reference notes (>= MIN_REF_S long) that we re-trigger inside (fragmented note)
- merge:  share of repeated reference notes (same pitch, gap < REPEAT_GAP_S) that we hold through without
          a new onset (two notes glued into one)
- short / long: share of onset-matched notes whose length is < 0.5x / > 2x the reference
Guitar (<variant>_<suffix>, --split=<suffix>, default modelassign): switch = mean |track change| between
consecutive notes of the same reference guitar part (0 = a part stays on one track, 0.5 = random).
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from error_breakdown import aligned_refs, has, onset_index, work_of
from eval_guitar_split import labelled_notes, load_notes, owner, reference_guitars
from eval_midi import ONSET_TOL, estimate, reference

STEMS = ("guitar", "piano", "bass")
MIN_REF_S = 0.15     # shorter reference notes have no room to fragment
REPEAT_GAP_S = 0.3   # same-pitch notes closer than this are a repeated note, not a new phrase
PHRASE_GAP_S = 1.0   # consecutive guitar-part notes further apart than this start a new phrase (not scored)


def by_pitch(notes: list[dict]) -> dict[int, list[dict]]:
    """{pitch: notes sorted by start}."""
    out = defaultdict(list)
    for n in sorted(notes, key=lambda n: n["start"]):
        out[n["pitch"]].append(n)
    return out


def is_split(ref: dict, est_starts: np.ndarray) -> bool:
    """Whether one of our onsets of this pitch falls well inside the reference note."""
    return bool(((est_starts > ref["start"] + ONSET_TOL) & (est_starts < ref["end"] - ONSET_TOL)).any())


def is_merged(second: dict, est: list[dict], est_ix: dict) -> bool:
    """Whether a repeated reference note's onset is held through by an earlier note of ours (no re-attack)."""
    p, t = second["pitch"], second["start"]
    held = any(n["start"] < t - ONSET_TOL and n["end"] > t + ONSET_TOL for n in est)
    return held and not has(est_ix, p, t)


def length_ratios(ref: list[dict], est: list[dict]) -> np.ndarray:
    """est / ref duration of each reference note that has an onset match (nearest one)."""
    est_p = by_pitch(est)
    out = []
    for r in ref:
        cands = [n for n in est_p.get(r["pitch"], []) if abs(n["start"] - r["start"]) <= ONSET_TOL]
        if cands and r["end"] > r["start"]:
            n = min(cands, key=lambda n: abs(n["start"] - r["start"]))
            out.append((n["end"] - n["start"]) / (r["end"] - r["start"]))
    return np.array(out)


def segmentation(ref: list[dict], est: list[dict]) -> dict:
    """Counts for split / merge / short / long of one stem."""
    est_p, est_ix = by_pitch(est), onset_index(est)
    c = dict.fromkeys(["n_long", "split", "n_repeat", "merge", "n_matched", "short", "long"], 0)
    for p, refs in by_pitch(ref).items():
        starts = np.array([n["start"] for n in est_p.get(p, [])])
        for r in refs:
            if r["end"] - r["start"] >= MIN_REF_S:
                c["n_long"] += 1
                c["split"] += is_split(r, starts)
        for a, b in zip(refs, refs[1:]):
            if b["start"] - a["end"] < REPEAT_GAP_S:
                c["n_repeat"] += 1
                c["merge"] += is_merged(b, est_p.get(p, []), est_ix)
    ratios = length_ratios(ref, est)
    c["n_matched"], c["short"], c["long"] = len(ratios), int((ratios < 0.5).sum()), int((ratios > 2).sum())
    return c


def part_switches(part: list[dict], refs: list[list[dict]], pos: dict) -> tuple[float, int]:
    """(sum of |track change|, pairs) over consecutive scored notes of one reference guitar part."""
    seq = sorted((n["start"], pos[k]) for n in part if (k := (n["pitch"], round(n["start"], 2))) in pos)
    pairs = [(a, b) for (ta, a), (tb, b) in zip(seq, seq[1:]) if tb - ta < PHRASE_GAP_S]
    return sum(abs(a - b) for a, b in pairs), len(pairs)


def guitar_switch(track: Path, work: Path, variant: str, suffix: str = "modelassign") -> tuple[float, int]:
    """Track changes within each reference guitar part of <variant>_<suffix>, summed over parts."""
    g1, g2 = (load_notes(work, f"{variant}_{suffix}", f"guitar_{k}") for k in (1, 2))
    return switches(reference_guitars(track), g1, g2)


def switches(refs: list[list[dict]], g1: list[dict], g2: list[dict]) -> tuple[float, int]:
    """(sum of |track change|, pairs) for one split against the reference guitar parts (0, 0 if < 2 parts)."""
    if len(refs) < 2:
        return 0.0, 0
    pos = {}  # (ref pitch, ref onset) -> our track position of the estimate matching it
    for n, p in labelled_notes(g1, g2):
        o = owner(n, refs)
        if o is not None:
            r = min((r for r in refs[o] if r["pitch"] == n["pitch"]), key=lambda r: abs(r["start"] - n["start"]))
            pos[(r["pitch"], round(r["start"], 2))] = p
    s, k = zip(*(part_switches(part, refs, pos) for part in refs))
    return sum(s), sum(k)


def main(variant: str, tracks: list[Path], suffix: str = "modelassign") -> None:
    """Print summed shares over tracks."""
    total = defaultdict(lambda: defaultdict(float))
    sw, pairs = 0.0, 0
    for track in tracks:
        work = work_of(track, variant)
        ests = {s: estimate(work, variant, s) for s in STEMS}
        refs = aligned_refs({s: n for s, n in reference(track).items() if s in STEMS}, ests)
        for stem, ref in refs.items():
            for k, v in segmentation(ref, ests[stem]).items():
                total[stem][k] += v
        if not variant.startswith("oracle"):
            s, k = guitar_switch(track, work, variant, suffix)
            sw, pairs = sw + s, pairs + k
    print(f"{variant}  ({len(tracks)} tracks)")
    print(f"{'stem':7s}{'split':>8s}{'merge':>8s}{'short':>8s}{'long':>8s}   (n_long / n_repeat / n_matched)")
    for stem, t in total.items():
        print(f"{stem:7s}{t['split'] / max(t['n_long'], 1):8.3f}{t['merge'] / max(t['n_repeat'], 1):8.3f}"
              f"{t['short'] / max(t['n_matched'], 1):8.3f}{t['long'] / max(t['n_matched'], 1):8.3f}"
              f"   ({int(t['n_long'])} / {int(t['n_repeat'])} / {int(t['n_matched'])})")
    if pairs:
        print(f"guitar part switch {sw / pairs:.3f}  ({pairs} consecutive pairs, 0 = stays on one track, 0.5 = random)")


if __name__ == "__main__":
    args = sys.argv[2:]
    split = [a.split("=", 1)[1] for a in args if a.startswith("--split=")]
    main(sys.argv[1], [Path(a) for a in args if not a.startswith("--")], *split)
