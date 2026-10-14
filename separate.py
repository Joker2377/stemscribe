"""Stage 2 (demucs): one-shot htdemucs_6s separation into 01_stems/htdemucs_6s/.

Runs in envs/demucs.  Usage: separate.py <work_dir>
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from stems_io import write_stems

MODEL = "htdemucs_6s"
OVERLAP = 0.25  # keep fixed across experiments (see init_plan: segment seams)


def separate(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Run demucs on (samples, 2) audio and return {source_name: (samples, 2)}."""
    model = get_model(MODEL)
    assert sr == model.samplerate, f"expected {model.samplerate} Hz, got {sr}"
    wav = torch.from_numpy(audio.T.copy())
    # Same input standardisation demucs' own CLI applies.
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std()
    with torch.no_grad():
        out = apply_model(model, ((wav - mean) / std)[None], device="cuda",
                          overlap=OVERLAP, split=True, progress=True)[0]
    out = out * std + mean
    return {name: out[i].numpy().T for i, name in enumerate(model.sources)}


def main(work: Path) -> None:
    """Separate master.wav and write stems under 01_stems/htdemucs_6s/."""
    audio, sr = sf.read(work / "00_prep" / "master.wav", dtype="float32")
    write_stems(work / "01_stems" / MODEL, audio, separate(audio, sr), sr,
                {"model": MODEL, "overlap": OVERLAP})


if __name__ == "__main__":
    main(Path(sys.argv[1]))
