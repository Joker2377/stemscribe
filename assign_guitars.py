"""Stage 3.5: split guitar MIDI into guitar_1 / guitar_2 by comparing each note's energy in two signals.

Runs in envs/tools.  Usage: assign_guitars.py <work_dir> <stem_variant> <midi_variant> [pan|model] [threshold]
- pan:   left vs right channel of 01_stems/<stem_variant>/guitar.wav -> 02_midi/<midi_variant>_noteassign/
- model: the trained splitter's 01_stems/<stem_variant>_gmodel/guitar_{1,2}.wav (higher / lower part)
         -> 02_midi/<midi_variant>_modelassign/  (Slakh balanced_acc 0.76 vs 0.50 for pan)
- smooth: model, then each note's balance is averaged with its neighbours, weighted by closeness in time and
          pitch (same voice = near in both; McLeod & Steedman's voice-separation principle)
          -> 02_midi/<midi_variant>_smoothassign/
The full guitar is transcribed once (02_midi/<midi_variant>/guitar.json); only the assignment uses the two
signals, so bleed between them cannot add notes. A note whose balance is within +-threshold goes to both
tracks; other stems are hard-linked.
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
SMOOTH_T_S = 0.25     # time scale of the neighbour weighting (Slakh val + test grid, see notes/roadmap.md)
SMOOTH_PITCH = 12.0   # pitch scale in semitones (wide: mostly "notes played together share a track")
SMOOTH_THRESHOLD = 0.1  # smoothed balances are smaller, so a narrower "both" band


OUT_SUFFIX = {"pan": "noteassign", "model": "modelassign", "smooth": "smoothassign"}


def power(x: np.ndarray, sr: int) -> np.ndarray:
    """Power spectrogram of a mono signal."""
    return np.abs(stft(x, sr, nperseg=NFFT, noverlap=NFFT - HOP)[2]) ** 2


def stereo_power(wav: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Power spectrograms of left and right channel, plus sample rate."""
    audio, sr = sf.read(wav, dtype="float32")
    return power(audio[:, 0], sr), power(audio[:, 1], sr), sr


def model_power(stems: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Power spectrograms of the model's guitar_1 / guitar_2 (channels averaged), plus sample rate."""
    (a, sr), (b, _) = (sf.read(stems / f"guitar_{k}.wav", dtype="float32") for k in (1, 2))
    return power(a.mean(1), sr), power(b.mean(1), sr), sr


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


def smooth_pans(notes: list[dict], pans: list[float], sigma_t: float = SMOOTH_T_S,
                sigma_p: float = SMOOTH_PITCH) -> list[float]:
    """Gaussian-weighted mean of the balances of notes near in time (onset) and pitch, the note itself included."""
    t = np.array([n["start"] for n in notes])
    pitch = np.array([n["pitch"] for n in notes], dtype=float)
    pans_a = np.array(pans)
    order = np.argsort(t)
    ts = t[order]
    out = np.empty(len(notes))
    for i in range(len(notes)):  # ponytail: per-note window loop, fine for a few thousand notes
        lo, hi = np.searchsorted(ts, [t[i] - 3 * sigma_t, t[i] + 3 * sigma_t])
        j = order[lo:hi]
        w = np.exp(-0.5 * (((t[j] - t[i]) / sigma_t) ** 2 + ((pitch[j] - pitch[i]) / sigma_p) ** 2))
        out[i] = (w * pans_a[j]).sum() / w.sum()
    return out.tolist()


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


def main(work: Path, variant: str, midi_variant: str, mode: str, threshold: float) -> None:
    """Assign each guitar note to guitar_1 / guitar_2 and write the new midi variant."""
    src = work / "02_midi" / midi_variant
    dst = work / "02_midi" / f"{midi_variant}_{OUT_SUFFIX[mode]}"
    link_other_stems(src, dst)
    data = json.loads((src / "guitar.json").read_text())
    pl, pr, sr = (stereo_power(work / "01_stems" / variant / "guitar.wav") if mode == "pan"
                  else model_power(work / "01_stems" / f"{variant}_gmodel"))
    pans = [note_pan(n, pl, pr, sr) for n in data["notes"]]
    if mode == "smooth":
        pans = smooth_pans(data["notes"], pans)
    g1, g2 = assign(data["notes"], pans, threshold)
    for name, notes in (("guitar_1", g1), ("guitar_2", g2)):
        (dst / f"{name}.json").write_text(json.dumps({**data, "notes": notes, "split": mode,
                                                      "threshold": threshold}))
    hist = np.histogram(pans, bins=10, range=(-1, 1))[0]
    both = sum(abs(p) < threshold for p in pans)
    print(f"notes {len(pans)}  guitar_1 {len(g1)}  guitar_2 {len(g2)}  both {both}")
    print("pan histogram L..R:", " ".join(str(h) for h in hist))


if __name__ == "__main__":
    mode = sys.argv[4] if len(sys.argv) > 4 else "pan"
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3], mode,
         float(sys.argv[5]) if len(sys.argv) > 5 else SMOOTH_THRESHOLD if mode == "smooth" else DEFAULT_THRESHOLD)
