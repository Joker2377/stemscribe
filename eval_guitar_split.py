"""Score a two-guitar split (guitar_1 / guitar_2 note lists) against a Slakh track's per-guitar MIDI,
or a MoisesDB song's per-track transcriptions (transcribe_moises.py; MoisesDB has no MIDI).

Runs in envs/tools.  Usage: eval_guitar_split.py <slakh_track_or_moises_song_dir> <work_dir> <midi_variant> [csv]
Each estimated guitar note is matched (same pitch, onset within 50 ms) to the reference guitar stems.
Notes matching exactly one stem are scored; the reference stems are split into two groups in every
possible way and the best grouping (and track order) is kept, since our outputs have no fixed identity.
balanced_acc: mean over the two groups of the per-note credit (1 = only on the right track,
0.5 = on both tracks, 0 = only on the wrong one). Putting everything on one track, on both, or at
random all give 0.5. With many guitars the best-grouping search makes the score a bit optimistic.
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from eval_midi import ONSET_TOL, midi_notes
from merge import drop_zero_length
from refine_notes import LAG_S

FIELDS = ["track", "variant", "n_ref_guitars", "n_scored", "share_both", "balanced_acc"]


def moises_guitars(song: Path) -> list[list[dict]]:
    """Transcribed notes of every MoisesDB guitar track, lag-shifted like our refined notes."""
    lag = LAG_S["guitar"]
    return [[{**n, "start": n["start"] - lag, "end": n["end"] - lag} for n in json.loads(f.read_text())["notes"]]
            for f in sorted((song / "guitar").glob("*.json"))]


def reference_guitars(track: Path) -> list[list[dict]]:
    """Notes of every rendered Slakh guitar stem that has MIDI (or every MoisesDB guitar track), one list each."""
    if (track / "data.json").exists():
        return moises_guitars(track)
    meta = yaml.safe_load((track / "metadata.yaml").read_text())["stems"]
    return [midi_notes(track / "MIDI" / f"{sid}.mid") for sid, s in meta.items()
            if s["inst_class"] == "Guitar" and s.get("audio_rendered") and (track / "MIDI" / f"{sid}.mid").exists()]


def owner(note: dict, refs: list[list[dict]]) -> int | None:
    """Index of the only reference stem with a note of this pitch within ONSET_TOL, else None."""
    hits = [i for i, ref in enumerate(refs)
            if any(r["pitch"] == note["pitch"] and abs(r["start"] - note["start"]) <= ONSET_TOL for r in ref)]
    return hits[0] if len(hits) == 1 else None


def labelled_notes(g1: list[dict], g2: list[dict]) -> list[tuple[dict, float]]:
    """Every distinct estimated note with its track position: 1.0 = guitar_1 only, 0.0 = guitar_2 only, 0.5 = both."""
    key = lambda n: (n["pitch"], round(n["start"], 4))
    k1, k2 = {key(n) for n in g1}, {key(n) for n in g2}
    notes = {key(n): n for n in g1 + g2}
    return [(n, 0.5 if k in k1 and k in k2 else float(k in k1)) for k, n in notes.items()]


def balanced_accuracy(owners: np.ndarray, pos: np.ndarray, n_refs: int) -> float:
    """Best balanced accuracy over every 2-group split of the reference stems and both track orders."""
    best = 0.0
    for r in range(1, n_refs):
        for group_a in itertools.combinations(range(n_refs), r):
            in_a = np.isin(owners, group_a)
            if in_a.all() or not in_a.any():
                continue
            for credit in (pos, 1 - pos):  # group A on guitar_1, or on guitar_2
                best = max(best, (credit[in_a].mean() + (1 - credit[~in_a]).mean()) / 2)
    return float(best)


def score(refs: list[list[dict]], g1: list[dict], g2: list[dict]) -> dict:
    """n_scored, share_both and balanced_acc for one split; NaN accuracy if nothing can be scored."""
    owned = [(owner(n, refs), p) for n, p in labelled_notes(drop_zero_length(g1), drop_zero_length(g2))]
    owned = [(o, p) for o, p in owned if o is not None]
    if len({o for o, _ in owned}) < 2:  # fewer than two guitars actually heard: nothing to split
        return {"n_scored": len(owned), "share_both": float("nan"), "balanced_acc": float("nan")}
    owners, pos = np.array([o for o, _ in owned]), np.array([p for _, p in owned])
    return {"n_scored": len(owned), "share_both": float((pos == 0.5).mean()),
            "balanced_acc": balanced_accuracy(owners, pos, len(refs))}


def load_notes(work: Path, variant: str, name: str) -> list[dict]:
    """Our notes for guitar_1 / guitar_2 ([] if missing)."""
    path = work / "02_midi" / variant / f"{name}.json"
    return json.loads(path.read_text())["notes"] if path.exists() else []


def main(track: Path, work: Path, variant: str, out_csv: Path) -> None:
    """Score one track and append a CSV row."""
    refs = reference_guitars(track)
    row = {"track": track.name, "variant": variant, "n_ref_guitars": len(refs),
           **score(refs, load_notes(work, variant, "guitar_1"), load_notes(work, variant, "guitar_2"))}
    print(f"{track.name} {variant} guitars {len(refs)}  scored {row['n_scored']}  "
          f"both {row['share_both']:.2f}  balanced_acc {row['balanced_acc']:.3f}", flush=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3],
         Path(sys.argv[4]) if len(sys.argv) > 4 else Path("work/eval/guitar_split.csv"))
