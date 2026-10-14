"""Run the full pipeline on the Slakh eval tracks and score it.

Usage (any Python): python eval_batch.py [slakh_dir=dataset/slakh]
Per track: run.py (all stages, cached) -> eval_midi on the dereverbed transcription
(the noteassign split is not scored: Slakh has no guitar_1 / guitar_2 ground truth).
Scores: work/eval/results.csv (rewritten each run) + a mean table printed at the end.
"""
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from run import ROOT, WORK, run_stage, song_id

SEP = "bs_roformer_sw"
VARIANT = f"{SEP}_dry"
OUT_CSV = WORK / "eval" / "results.csv"


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


def main(slakh: Path) -> None:
    """Process and score every track folder under slakh/."""
    OUT_CSV.unlink(missing_ok=True)
    for track in sorted(p for p in slakh.iterdir() if (p / "mix.flac").exists()):
        print(f"=== {track.name}", flush=True)
        work = process(track)
        run_stage("tools", "eval_midi.py", [str(track), str(work), VARIANT, str(OUT_CSV)])
    summarize(OUT_CSV)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dataset" / "slakh")
