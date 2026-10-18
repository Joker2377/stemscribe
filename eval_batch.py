"""Run the full pipeline on the Slakh eval tracks and score it.

Usage (any Python): python eval_batch.py [slakh_dir=dataset/slakh] [track_list]
track_list: file of track names inside slakh_dir (default: every track folder with a mix.flac).
Per track: run.py (all stages, cached) -> eval_midi on the dereverbed, refined transcription
plus eval_guitar_split on the final guitar_1 / guitar_2 split (per-guitar Slakh MIDI).
Scores: work/eval/results.csv, work/eval/guitar_split.csv (rewritten each run) + mean tables.
"""
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from run import ROOT, WORK, final_variant, run_stage, song_id

SEP = "bs_roformer_sw"
VARIANT = f"{SEP}_dry_win_refined"
OUT_CSV = WORK / "eval" / "results.csv"
SPLIT_CSV = WORK / "eval" / "guitar_split.csv"


def process(track: Path) -> Path:
    """Run the whole pipeline on one track's mix; return its work dir."""
    mix = track / "mix.flac"
    subprocess.run([sys.executable, str(ROOT / "run.py"), str(mix), "--sep", SEP], check=True)
    return WORK / song_id(mix)


def summarize(path: Path) -> None:
    """Print mean precision / recall / onset F1 / onset+offset F1 per stem."""
    acc = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            acc[row["stem"]].append([float(row[k]) for k in ("precision", "recall", "f1", "f1_offset")])
    print(f"\n{'stem':7s}{'P':>8s}{'R':>8s}{'F1':>8s}{'F1+off':>8s}   (n={len(next(iter(acc.values())))})")
    for stem, vals in sorted(acc.items()):
        means = [sum(c) / len(c) for c in zip(*vals)]
        print(f"{stem:7s}" + "".join(f"{m:8.3f}" for m in means))


def summarize_split(path: Path) -> None:
    """Print the mean balanced accuracy over tracks that have at least two scored guitars."""
    with open(path) as f:
        accs = [float(r["balanced_acc"]) for r in csv.DictReader(f) if r["balanced_acc"] != "nan"]
    print(f"\nguitar split balanced_acc {sum(accs) / max(len(accs), 1):.3f}  (n={len(accs)}, 0.5 = no split)")


def main(slakh: Path, track_list: Path | None) -> None:
    """Process and score every track folder under slakh/ (or the ones named in track_list)."""
    OUT_CSV.unlink(missing_ok=True)
    SPLIT_CSV.unlink(missing_ok=True)
    tracks = ([slakh / n for n in track_list.read_text().split()] if track_list
              else sorted(p for p in slakh.iterdir() if (p / "mix.flac").exists()))
    for track in tracks:
        print(f"=== {track.name}", flush=True)
        work = process(track)
        run_stage("tools", "eval_midi.py", [str(track), str(work), VARIANT, str(OUT_CSV)])
        run_stage("tools", "eval_guitar_split.py", [str(track), str(work), final_variant(SEP), str(SPLIT_CSV)])
    summarize(OUT_CSV)
    summarize_split(SPLIT_CSV)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dataset" / "slakh",
         Path(sys.argv[2]) if len(sys.argv) > 2 else None)
