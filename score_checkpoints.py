"""Pick a guitar-split checkpoint by the real task: split separated Slakh validation stems, assign notes, score.

Usage (any Python): python score_checkpoints.py [checkpoint_dir=envs/models/guitar_split] [tracks=slakh_val_tracks.txt] [csv]
tracks: Slakh validation track names, or MoisesDB song dirs (work/exp/moises_val.txt).
Needs each track already run through run.py (Slakh) or prep_moises_val.py (MoisesDB).
Per checkpoint (step_*.pt, then best.pt): split_guitars_model -> assign_guitars model -> eval_guitar_split.
Writes work/eval/checkpoints.csv (one row per checkpoint and track) and prints the mean balanced_acc per checkpoint.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

from run import ROOT, WORK, run_stage, song_id

SPLIT = ROOT / "dataset" / "slakh_flac" / "validation"
VARIANT, REFINED = "bs_roformer_sw_dry", "bs_roformer_sw_dry_refined"
OUT_CSV = WORK / "eval" / "checkpoints.csv"


def work_dir(track: Path) -> Path:
    """Pipeline work dir of a Slakh track (run.py, keyed by the mix hash) or a MoisesDB song (prep_moises_val.py)."""
    return WORK / "moises" / track.name if (track / "data.json").exists() else WORK / song_id(track / "mix.flac")


def score_track(ckpt: Path, track: Path, out_csv: Path) -> None:
    """Split one track's guitar with ckpt, assign the notes and append its score row to out_csv."""
    w = str(work_dir(track))
    run_stage("separator", "split_guitars_model.py", [w, VARIANT, str(ckpt)])
    run_stage("tools", "assign_guitars.py", [w, VARIANT, REFINED, "model"])
    run_stage("tools", "eval_guitar_split.py", [str(track), w, f"{REFINED}_modelassign", str(out_csv)])


def score_checkpoint(ckpt: Path, tracks: list[Path], out_csv: Path) -> list[dict]:
    """Rows (checkpoint, track, n_scored, balanced_acc) for every track."""
    tmp = out_csv.with_suffix(".tmp.csv")
    tmp.unlink(missing_ok=True)
    for track in tracks:
        score_track(ckpt, track, tmp)
    with open(tmp) as f:
        rows = [{"checkpoint": ckpt.name, "track": r["track"], "n_scored": r["n_scored"],
                 "balanced_acc": r["balanced_acc"]} for r in csv.DictReader(f)]
    tmp.unlink()
    return rows


def main(ckpt_dir: Path, track_list: Path, out_csv: Path) -> None:
    """Score every snapshot (and best.pt) and print the mean per checkpoint."""
    tracks = [SPLIT / name for name in track_list.read_text().split()]  # absolute dirs pass through unchanged
    ckpts = sorted(ckpt_dir.glob("step_*.pt")) + [p for p in [ckpt_dir / "best.pt"] if p.exists()]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    means = defaultdict(list)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, ["checkpoint", "track", "n_scored", "balanced_acc"])
        w.writeheader()
        for ckpt in ckpts:
            for row in score_checkpoint(ckpt, tracks, out_csv):
                w.writerow(row)
                if row["balanced_acc"] != "nan":
                    means[ckpt.name].append(float(row["balanced_acc"]))
            f.flush()
            accs = means[ckpt.name]
            print(f"{ckpt.name:20s} balanced_acc {sum(accs) / max(len(accs), 1):.3f}  (n={len(accs)})", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "envs" / "models" / "guitar_split",
         Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "slakh_val_tracks.txt",
         Path(sys.argv[3]) if len(sys.argv) > 3 else OUT_CSV)
