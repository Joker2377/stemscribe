"""Oracle run: transcribe a Slakh track's true per-class stems (no separation), to split pipeline error into
"separation" (pipeline vs oracle) and "transcription" (oracle vs ground truth).

Runs in envs/tools.  Usage: oracle_stems.py <slakh_track_dir> [out_csv=work/eval/oracle.csv]
Sums the rendered stems of each class (Guitar / Piano / Bass / Drums) into
work/oracle/<track>/01_stems/oracle/<stem>.wav, then transcribe.py -> refine_notes.py -> eval_midi.py
(variant oracle_refined).  Cached per stage like run.py.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from eval_midi import CLASS_TO_STEM
from run import WORK, run_stage

VARIANT = "oracle"


def class_stems(track: Path) -> dict[str, list[Path]]:
    """{our stem name: rendered Slakh stem files of that class}."""
    meta = yaml.safe_load((track / "metadata.yaml").read_text())["stems"]
    out: dict[str, list[Path]] = {}
    for sid, s in meta.items():
        f = track / "stems" / f"{sid}.flac"
        if s["inst_class"] in CLASS_TO_STEM and s.get("audio_rendered") and f.exists():
            out.setdefault(CLASS_TO_STEM[s["inst_class"]], []).append(f)
    return out


def write_stems(track: Path, dst: Path) -> None:
    """Sum each class's stems into dst/<stem>.wav (stereo, like separated stems)."""
    dst.mkdir(parents=True, exist_ok=True)
    for stem, files in class_stems(track).items():
        audio = [sf.read(f, dtype="float32", always_2d=True) for f in files]
        n = max(len(a) for a, _ in audio)
        mix = sum(np.pad(a, ((0, n - len(a)), (0, 0))) for a, _ in audio)
        sf.write(dst / f"{stem}.wav", np.repeat(mix, 2, 1) if mix.shape[1] == 1 else mix, audio[0][1],
                 subtype="FLOAT")


def main(track: Path, out_csv: Path) -> None:
    """Oracle stems -> notes -> scores for one track."""
    work = WORK / "oracle" / track.name
    stems = work / "01_stems" / VARIANT
    if not stems.exists():
        write_stems(track, stems)
    if not (work / "02_midi" / f"{VARIANT}.done.json").exists():
        run_stage("muscriptor", "transcribe.py", [str(work), VARIANT])
    run_stage("tools", "refine_notes.py", [str(work), VARIANT])
    run_stage("tools", "eval_midi.py", [str(track), str(work), f"{VARIANT}_refined", str(out_csv)])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else WORK / "eval" / "oracle.csv")
