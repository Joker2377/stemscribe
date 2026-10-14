"""Stage 2 (audio-separator): run any UVR/Roformer model into 01_stems/<name>/.

Runs in envs/separator.  Usage: separate_uvr.py <work_dir> <model_file> <name>
e.g. separate_uvr.py work/abc BS-Roformer-SW.ckpt bs_roformer_sw
"""
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from audio_separator.separator import Separator

from stems_io import write_stems

MODEL_DIR = Path(__file__).parent / "envs" / "models"


def stem_name(path: Path) -> str:
    """'master_(Guitar)_BS-Roformer-SW.wav' -> 'guitar'."""
    return re.search(r"_\(([^)]+)\)", path.name).group(1).lower()


def separate(master: Path, model_file: str) -> dict[str, np.ndarray]:
    """Run the model on master.wav and return {stem: (samples, 2) float32}."""
    with tempfile.TemporaryDirectory() as tmp:
        sep = Separator(model_file_dir=str(MODEL_DIR), output_dir=tmp,
                        normalization_threshold=1.0)  # 1.0 = never rescale, keeps stems summable
        sep.load_model(model_file)
        files = [Path(tmp) / Path(f).name for f in sep.separate(str(master))]
        return {stem_name(f): sf.read(f, dtype="float32")[0] for f in files}


def main(work: Path, model_file: str, name: str) -> None:
    """Separate master.wav and write stems under 01_stems/<name>/."""
    master = work / "00_prep" / "master.wav"
    audio, sr = sf.read(master, dtype="float32")
    write_stems(work / "01_stems" / name, audio, separate(master, model_file), sr,
                {"model": model_file})


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
