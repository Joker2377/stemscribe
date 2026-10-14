"""Score one transcription variant against a Slakh track's ground-truth MIDI.

Runs in envs/tools.  Usage: eval_midi.py <slakh_track_dir> <work_dir> <variant> [csv]
Pitched stems: mir_eval note F1, onset only and onset+offset (50 ms onset tolerance).
Drums: onset F1 ignoring which drum. Rows are appended to csv (default eval/results.csv).
"""
import csv
import json
import sys
from pathlib import Path

import mido
import numpy as np
import yaml
from mir_eval.onset import f_measure as onset_f
from mir_eval.transcription import precision_recall_f1_overlap as note_prf
from mir_eval.util import midi_to_hz

from merge import drop_zero_length

# Slakh inst_class -> our stem name
CLASS_TO_STEM = {"Guitar": "guitar", "Piano": "piano", "Bass": "bass", "Drums": "drums"}
ONSET_TOL = 0.05
FIELDS = ["track", "variant", "stem", "n_ref", "n_est", "octave_fix", "precision", "recall", "f1", "f1_offset"]
# Slakh MIDI is not always at sounding pitch: some patches transpose (e.g. classic_bass.nkm plays an
# octave below its MIDI). Ref pitches are shifted by whichever of these gives the best onset F1.
OCTAVE_FIXES = (0, -12, 12)


def midi_notes(path: Path) -> list[dict]:
    """Notes of a MIDI file in seconds (mido applies the tempo map when iterating)."""
    notes, t, open_ = [], 0.0, {}
    for msg in mido.MidiFile(path):
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            open_.setdefault((msg.channel, msg.note), t)
        elif msg.type in ("note_off", "note_on"):
            start = open_.pop((msg.channel, msg.note), None)
            if start is not None:
                notes.append({"pitch": msg.note, "start": start, "end": t})
    return notes


def reference(track: Path) -> dict[str, list[dict]]:
    """Ground-truth notes per our stem name, merging all Slakh stems of the same class."""
    meta = yaml.safe_load((track / "metadata.yaml").read_text())["stems"]
    ref: dict[str, list[dict]] = {}
    for sid, s in meta.items():
        stem = CLASS_TO_STEM.get(s["inst_class"])
        mid = track / "MIDI" / f"{sid}.mid"
        if stem and s.get("audio_rendered") and mid.exists():
            ref.setdefault(stem, []).extend(midi_notes(mid))
    return ref


def as_arrays(notes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(intervals, pitch_hz) arrays for mir_eval; zero-length notes dropped."""
    notes = drop_zero_length(notes)
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    return (np.array([[n["start"], n["end"]] for n in notes]),
            midi_to_hz(np.array([n["pitch"] for n in notes], dtype=float)))


def score_pitched(ref: list[dict], est: list[dict]) -> dict:
    """Note precision / recall / F1 (onset only) plus F1 with offsets."""
    ri, rp = as_arrays(ref)
    ei, ep = as_arrays(est)
    p, r, f, _ = note_prf(ri, rp, ei, ep, onset_tolerance=ONSET_TOL, offset_ratio=None)
    f_off = note_prf(ri, rp, ei, ep, onset_tolerance=ONSET_TOL)[2]
    return {"precision": p, "recall": r, "f1": f, "f1_offset": f_off}


def score_pitched_octave(ref: list[dict], est: list[dict]) -> dict:
    """score_pitched with the reference moved by the best whole-octave transposition."""
    best = None
    for fix in OCTAVE_FIXES:
        scores = {"octave_fix": fix, **score_pitched([{**n, "pitch": n["pitch"] + fix} for n in ref], est)}
        if best is None or scores["f1"] > best["f1"]:
            best = scores
    return best


def unique_onsets(notes: list[dict]) -> np.ndarray:
    """Sorted onset times with hits closer than 10 ms merged (several drums on one hit)."""
    return np.unique(np.round([n["start"] for n in notes], 2))


def score_drums(ref: list[dict], est: list[dict]) -> dict:
    """Onset F1 ignoring drum type; offsets are meaningless for drums."""
    f, p, r = onset_f(unique_onsets(ref), unique_onsets(est), window=ONSET_TOL)
    return {"octave_fix": 0, "precision": p, "recall": r, "f1": f, "f1_offset": float("nan")}


def estimate(work: Path, variant: str, stem: str) -> list[dict]:
    """Our transcribed notes for one stem ([] if the stem was skipped as silent)."""
    path = work / "02_midi" / variant / f"{stem}.json"
    return json.loads(path.read_text())["notes"] if path.exists() else []


def main(track: Path, work: Path, variant: str, out_csv: Path) -> None:
    """Score every reference stem and append one CSV row per stem."""
    rows = []
    for stem, ref in reference(track).items():
        est = estimate(work, variant, stem)
        scores = (score_drums if stem == "drums" else score_pitched_octave)(ref, est)
        rows.append({"track": track.name, "variant": variant, "stem": stem,
                     "n_ref": len(ref), "n_est": len(est), **scores})
        print(f"{track.name} {variant:22s} {stem:7s} F1 {scores['f1']:.3f}  "
              f"P {scores['precision']:.3f} R {scores['recall']:.3f}", flush=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3],
         Path(sys.argv[4]) if len(sys.argv) > 4 else Path("eval/results.csv"))
