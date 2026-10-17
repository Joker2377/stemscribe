"""Make the real model input for every MoisesDB song: the pipeline's guitar stem of the song's own mix.

Runs in envs/separator.  Usage: prep_moises_guitar.py [data=dataset/moisesdb_flac]
Per song: mix = sum of every track -> BS-Roformer-SW -> guitar -> MDX23C dereverb (same as run.py's
separate + dereverb stages) -> <song>/sw_guitar_dry.flac.  Models are loaded once.  Resumable.
"""
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from audio_separator.separator import Separator

from dereverb import DRY_NAMES, MODELS
from separate_uvr import MODEL_DIR

SW = "BS-Roformer-SW.ckpt"
OUT_NAME = "sw_guitar_dry.flac"


def load_separator(model_file: str, out_dir: str) -> Separator:
    """audio-separator with one model loaded, writing into out_dir, never rescaling."""
    sep = Separator(model_file_dir=str(MODEL_DIR), output_dir=out_dir, normalization_threshold=1.0)
    sep.load_model(model_file)
    return sep


def song_mix(song: Path) -> tuple[np.ndarray, int]:
    """(samples, 2) sum of every track of a song."""
    mix, sr = None, 44100
    for f in song.glob("*/*.flac"):
        x, sr = sf.read(f, dtype="float32", always_2d=True)
        x = np.repeat(x, 2, 1) if x.shape[1] == 1 else x
        if mix is None:
            mix = x
        else:
            n = max(len(mix), len(x))
            mix = np.pad(mix, ((0, n - len(mix)), (0, 0))) + np.pad(x, ((0, n - len(x)), (0, 0)))
    return mix, sr


def last_tag(path: Path) -> str:
    """Innermost-model stem tag: 'x_(guitar)_SW_(dry)_MDX.wav' -> 'dry' (chained outputs keep every tag)."""
    return re.findall(r"_\(([^)]+)\)", path.name)[-1].lower()


def pick(sep: Separator, wav: Path, names: set[str]) -> Path:
    """Run sep on wav and return the output file whose stem name is in names (others deleted)."""
    files = [Path(sep.output_dir) / Path(f).name for f in sep.separate(str(wav))]
    keep = next(f for f in files if last_tag(f) in names)
    for f in files:
        if f != keep:
            f.unlink()
    return keep


def prep_song(song: Path, sw: Separator, dry: Separator, tmp: Path) -> None:
    """Write song/OUT_NAME: dereverbed SW guitar stem of the song's mix."""
    mix, sr = song_mix(song)
    wav = tmp / f"{song.name}.wav"
    sf.write(wav, mix, sr, subtype="FLOAT")
    guitar = pick(sw, wav, {"guitar"})
    out = pick(dry, guitar, DRY_NAMES)
    audio, sr = sf.read(out, dtype="float32")
    sf.write(song / OUT_NAME, audio, sr, subtype="PCM_24")
    for f in (wav, guitar, out):
        f.unlink()


def main(data: Path) -> None:
    """Prep every song that has no OUT_NAME yet."""
    songs = [s for s in sorted(data.iterdir()) if (s / "data.json").exists() and not (s / OUT_NAME).exists()]
    with tempfile.TemporaryDirectory() as tmp:
        sw, dry = load_separator(SW, tmp), load_separator(MODELS["guitar"], tmp)
        for i, song in enumerate(songs):
            prep_song(song, sw, dry, Path(tmp))
            print(f"{i + 1}/{len(songs)} {song.name}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/moisesdb_flac"))
