"""Put the held-out MoisesDB songs through the pipeline up to refined guitar notes, for score_checkpoints.py.

Runs in envs/tools.  Usage: prep_moises_val.py [data=dataset/moisesdb_flac]
Per validation song: work/moises/<id>/01_stems/bs_roformer_sw_dry/guitar.wav (= sw_guitar_dry.flac from
prep_moises_guitar.py) -> transcribe.py -> refine_notes.py.  Writes the song list to VAL_LIST.  Resumable.
"""
import sys
from pathlib import Path

import soundfile as sf

from run import ROOT, WORK, run_stage

VAL_LIST = ROOT / "work" / "exp" / "moises_val.txt"
VAL_EVERY_NTH = 9  # ~14 of 129 songs held out
VARIANT = "bs_roformer_sw_dry"


def val_songs(data: Path) -> list[Path]:
    """Held-out songs: every VAL_EVERY_NTH song dir sorted by id (the training script uses the same rule)."""
    return sorted(s for s in data.iterdir() if (s / "data.json").exists())[::VAL_EVERY_NTH]


def prep_song(song: Path) -> None:
    """Guitar stem -> transcription -> refined notes in work/moises/<id>/."""
    work = WORK / "moises" / song.name
    stems = work / "01_stems" / VARIANT
    if (work / "02_midi" / f"{VARIANT}_refined" / "guitar.json").exists():
        return
    stems.mkdir(parents=True, exist_ok=True)
    audio, sr = sf.read(song / "sw_guitar_dry.flac", dtype="float32")
    sf.write(stems / "guitar.wav", audio, sr, subtype="FLOAT")
    run_stage("muscriptor", "transcribe.py", [str(work), VARIANT])
    run_stage("tools", "refine_notes.py", [str(work), VARIANT])


def main(data: Path) -> None:
    """Prep every validation song and write VAL_LIST."""
    songs = val_songs(data)
    VAL_LIST.parent.mkdir(parents=True, exist_ok=True)
    VAL_LIST.write_text("\n".join(str(s.resolve()) for s in songs) + "\n")
    for i, song in enumerate(songs):
        prep_song(song)
        print(f"{i + 1}/{len(songs)} {song.name}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dataset" / "moisesdb_flac")
