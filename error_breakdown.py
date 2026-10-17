"""Where do the transcription errors come from?  Classifies every note of one variant against Slakh's MIDI.

Runs in envs/tools.  Usage: error_breakdown.py <variant> <slakh_track_dir>...   (variant oracle_* reads work/oracle/<track>)
For each stem (guitar / piano / bass / drums):
- extra notes (false positives): octave (right onset, pitch +-12), leak (matches another class's reference:
  separation put it in the wrong stem), inside (starts while a reference note of that pitch still sounds:
  a re-triggered fragment), other (nothing there)
- missed notes (false negatives): elsewhere (some other stem's estimate, incl. other / vocals, has it),
  octave (our stem has it +-12), lost (no stem has it)
Matching = same pitch and onset within ONSET_TOL, not one-to-one (a quick diagnostic, not a metric).
Like eval_midi, each reference stem is first moved by its best whole octave (transposing Slakh patches);
drums are matched by onset only.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from eval_midi import OCTAVE_FIXES, ONSET_TOL, estimate, reference
from run import WORK, song_id

STEMS = ("guitar", "piano", "bass", "drums")
EXTRA_STEMS = ("other", "vocals", "lead_vocals", "backing_vocals")  # no reference, but missed notes may land there
FIELDS = ["stem", "n_est", "extra_octave", "extra_leak", "extra_inside", "extra_other",
          "n_ref", "missed_elsewhere", "missed_octave", "missed_lost"]


def onset_index(notes: list[dict]) -> dict[int, np.ndarray]:
    """{pitch: sorted onsets} for fast lookup."""
    out = defaultdict(list)
    for n in notes:
        out[n["pitch"]].append(n["start"])
    return {p: np.sort(v) for p, v in out.items()}


def has(index: dict[int, np.ndarray], pitch: int, t: float) -> bool:
    """Whether a note of this pitch starts within ONSET_TOL of t."""
    v = index.get(pitch)
    if v is None:
        return False
    i = np.searchsorted(v, t)
    return any(abs(v[j] - t) <= ONSET_TOL for j in (i - 1, i) if 0 <= j < len(v))


def sounding(notes: list[dict], pitch: int, t: float) -> bool:
    """Whether a note of this pitch is held at time t (started more than ONSET_TOL before)."""
    return any(n["pitch"] == pitch and n["start"] + ONSET_TOL < t < n["end"] for n in notes)


def has_octave(index: dict[int, np.ndarray], pitch: int, t: float) -> bool:
    """has() an octave up or down."""
    return has(index, pitch + 12, t) or has(index, pitch - 12, t)


def aligned_refs(refs: dict[str, list[dict]], ests: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """References with each pitched stem moved by the octave that matches our notes best; drum pitches dropped."""
    out = {}
    for stem, ref in refs.items():
        if stem == "drums":
            out[stem] = [{**n, "pitch": 0} for n in ref]
            continue
        ix = onset_index(ests.get(stem, []))
        fix = max(OCTAVE_FIXES, key=lambda f: sum(has(ix, n["pitch"] + f, n["start"]) for n in ref))
        out[stem] = [{**n, "pitch": n["pitch"] + fix} for n in ref]
    return out


def breakdown(refs: dict[str, list[dict]], ests: dict[str, list[dict]]) -> list[dict]:
    """One row of error counts per stem."""
    ref_ix = {s: onset_index(n) for s, n in refs.items()}
    est_ix = {s: onset_index(n) for s, n in ests.items()}
    rows = []
    for stem in STEMS:
        if stem not in refs:
            continue
        own_ref, own_est = ref_ix[stem], est_ix.get(stem, {})
        other_refs = [ix for s, ix in ref_ix.items() if s not in (stem, "drums")]
        other_ests = [ix for s, ix in est_ix.items() if s not in (stem, "drums")]
        row = dict.fromkeys(FIELDS, 0) | {"stem": stem}
        for n in ests.get(stem, []):
            p, t = n["pitch"], n["start"]
            row["n_est"] += 1
            if has(own_ref, p, t):
                continue
            key = ("extra_octave" if has_octave(own_ref, p, t)
                   else "extra_leak" if any(has(ix, p, t) for ix in other_refs)
                   else "extra_inside" if sounding(refs[stem], p, t) else "extra_other")
            row[key] += 1
        for n in refs[stem]:
            p, t = n["pitch"], n["start"]
            row["n_ref"] += 1
            if has(own_est, p, t):
                continue
            key = ("missed_octave" if has_octave(own_est, p, t)
                   else "missed_elsewhere" if any(has(ix, p, t) for ix in other_ests) else "missed_lost")
            row[key] += 1
        rows.append(row)
    return rows


def work_of(track: Path, variant: str) -> Path:
    """Pipeline work dir of a Slakh track (oracle runs live under work/oracle/<track>)."""
    return WORK / "oracle" / track.name if variant.startswith("oracle") else WORK / song_id(track / "mix.flac")


def main(variant: str, tracks: list[Path]) -> None:
    """Print the summed breakdown over tracks as shares of n_est / n_ref."""
    total = defaultdict(lambda: defaultdict(int))
    for track in tracks:
        work = work_of(track, variant)
        ests = {s: estimate(work, variant, s) for s in STEMS + EXTRA_STEMS}
        ests["drums"] = [{**n, "pitch": 0} for n in ests["drums"]]
        refs = aligned_refs(reference(track), ests)
        for row in breakdown(refs, ests):
            for k in FIELDS[1:]:
                total[row["stem"]][k] += row[k]
    print(f"{variant}  ({len(tracks)} tracks; extra = share of our notes, missed = share of reference notes)")
    print(f"{'stem':7s}{'octave':>8s}{'leak':>8s}{'inside':>8s}{'other':>8s}  |"
          f"{'elsewh':>8s}{'octave':>8s}{'lost':>8s}")
    for stem, t in total.items():
        e, r = max(t["n_est"], 1), max(t["n_ref"], 1)
        print(f"{stem:7s}{t['extra_octave'] / e:8.3f}{t['extra_leak'] / e:8.3f}{t['extra_inside'] / e:8.3f}"
              f"{t['extra_other'] / e:8.3f}  |"
              f"{t['missed_elsewhere'] / r:8.3f}{t['missed_octave'] / r:8.3f}{t['missed_lost'] / r:8.3f}")


if __name__ == "__main__":
    main(sys.argv[1], [Path(a) for a in sys.argv[2:]])
