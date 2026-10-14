"""Stage 3.5: split guitar MIDI into guitar_1 / guitar_2 by each note's stereo position.

Runs in envs/tools.  Usage: assign_guitars.py <work_dir> <stem_variant> [threshold]
Reads 01_stems/<v>/guitar.wav + 02_midi/<v>/guitar.json, writes a new midi variant
02_midi/<v>_noteassign/ with guitar_1.json (left), guitar_2.json (right); other stems hard-linked.
A note whose L/R balance is within +-threshold goes to both tracks (both guitars playing it).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

NFFT = 4096
HOP = 1024
HARMONICS = 6
ATTACK_S = 0.3  # only look at the start of each note; long sustains collect bleed from later notes
DEFAULT_THRESHOLD = 0.2


def stereo_power(wav: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Power spectrograms of left and right channel, plus sample rate."""
    audio, sr = sf.read(wav, dtype="float32")
    _, _, sl = stft(audio[:, 0], sr, nperseg=NFFT, noverlap=NFFT - HOP)
    _, _, sr_ = stft(audio[:, 1], sr, nperseg=NFFT, noverlap=NFFT - HOP)
    return np.abs(sl) ** 2, np.abs(sr_) ** 2, sr


def harmonic_bins(pitch: int, sr: int, n_bins: int) -> np.ndarray:
    """STFT bin indices (+-1 neighbour) of the first HARMONICS partials of a MIDI pitch."""
    f0 = 440.0 * 2 ** ((pitch - 69) / 12)
    centers = np.round(f0 * np.arange(1, HARMONICS + 1) * NFFT / sr).astype(int)
    bins = np.unique(np.concatenate([centers - 1, centers, centers + 1]))
    return bins[(bins >= 0) & (bins < n_bins)]


def note_pan(note: dict, pl: np.ndarray, pr: np.ndarray, sr: int) -> float:
    """(R - L) / (R + L) energy at the note's harmonics during its attack; -1 = hard left."""
    f0 = int(note["start"] * sr / HOP)
    f1 = max(f0 + 1, int(min(note["end"], note["start"] + ATTACK_S) * sr / HOP) + 1)
    bins = harmonic_bins(note["pitch"], sr, pl.shape[0])
    el, er = pl[bins, f0:f1].sum(), pr[bins, f0:f1].sum()
    return float((er - el) / (er + el + 1e-12))


def assign(notes: list[dict], pans: list[float], threshold: float) -> tuple[list[dict], list[dict]]:
    """Left-leaning notes -> guitar_1, right-leaning -> guitar_2, centred -> both."""
    g1 = [n for n, p in zip(notes, pans) if p <= threshold]
    g2 = [n for n, p in zip(notes, pans) if p >= -threshold]
    return g1, g2


def link_other_stems(src: Path, dst: Path) -> None:
    """Hard-link every note json except guitar into the new variant folder."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.json"):
        if not f.stem.startswith("guitar") and not (dst / f.name).exists():
            os.link(f, dst / f.name)


def main(work: Path, variant: str, threshold: float) -> None:
    """Assign each guitar note to guitar_1 / guitar_2 and write the new midi variant."""
    src = work / "02_midi" / variant
    dst = work / "02_midi" / f"{variant}_noteassign"
    link_other_stems(src, dst)
    data = json.loads((src / "guitar.json").read_text())
    pl, pr, sr = stereo_power(work / "01_stems" / variant / "guitar.wav")
    pans = [note_pan(n, pl, pr, sr) for n in data["notes"]]
    g1, g2 = assign(data["notes"], pans, threshold)
    for name, notes in (("guitar_1", g1), ("guitar_2", g2)):
        (dst / f"{name}.json").write_text(json.dumps({**data, "notes": notes, "split": "noteassign",
                                                      "threshold": threshold}))
    hist = np.histogram(pans, bins=10, range=(-1, 1))[0]
    both = sum(abs(p) < threshold for p in pans)
    print(f"notes {len(pans)}  guitar_1 {len(g1)}  guitar_2 {len(g2)}  both {both}")
    print("pan histogram L..R:", " ".join(str(h) for h in hist))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_THRESHOLD)
