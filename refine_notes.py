"""Stage 3.2: fix MuScriptor's constant onset lag and over-long note tails.

Runs in envs/tools.  Usage: refine_notes.py <work_dir> <stem_variant>
Reads 01_stems/<v>/<stem>.wav + 02_midi/<v>/<stem>.json, writes 02_midi/<v>_refined/<stem>.json.
- lag: every note of a stem moves earlier by LAG_S[stem] (measured on Slakh, eval_midi median onset error)
- tail: a pitched note ends where the energy at its harmonics has fallen DROP_DB below its attack peak
- blips: notes shorter than MIN_NOTE_S are dropped (raw MuScriptor output; ~10% of bass notes are < 20 ms)
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

from assign_guitars import HARMONICS, NFFT, HOP

LAG_S = {"guitar": 0.023, "bass": 0.0145, "piano": 0.0044}
# Tuned on 7 Slakh tracks (onset+offset F1: guitar 0.139 -> 0.318, bass 0.471 -> 0.725, plateau 10-15 dB).
# Piano is left alone: trimming lowered its F1 (sustain pedal keeps notes ringing). Unlisted stems keep their ends.
DROP_DB = {"guitar": 12.0, "bass": 12.0}
for _g in ("guitar_1", "guitar_2"):  # model-split guitars get the guitar settings
    LAG_S[_g], DROP_DB[_g] = LAG_S["guitar"], DROP_DB["guitar"]
# Slakh F1, 10 test / 12 validation tracks: bass .808 -> .844 / .882 -> .884, piano .708 -> .710 / .759 -> .780,
# guitar unchanged. Only the stems that were measured get it.
MIN_NOTE_S = {s: 0.04 for s in LAG_S}
PEAK_S = 0.1  # the attack peak is searched in the first 100 ms of a note
MIN_LEN_S = 0.05


def mono_power(wav: Path) -> tuple[np.ndarray, int]:
    """Power spectrogram of the mid (L+R) signal, plus sample rate."""
    audio, sr = sf.read(wav, dtype="float32")
    mono = audio.mean(1) if audio.ndim == 2 else audio
    _, _, z = stft(mono, sr, nperseg=NFFT, noverlap=NFFT - HOP)
    return np.abs(z) ** 2, sr


def harmonic_energy(pitch: int, power: np.ndarray, sr: int) -> np.ndarray:
    """Per-frame energy at the first HARMONICS partials of a MIDI pitch (nearest bin each)."""
    f0 = 440.0 * 2 ** ((pitch - 69) / 12)
    bins = np.round(f0 * np.arange(1, HARMONICS + 1) * NFFT / sr).astype(int)
    return power[bins[bins < power.shape[0]]].sum(0)


def trimmed_end(note: dict, energy: np.ndarray, sr: int, drop_db: float) -> float:
    """First time after the attack peak where energy is drop_db below the peak (never later than note end)."""
    f0, f1 = int(note["start"] * sr / HOP), int(note["end"] * sr / HOP) + 1
    seg = energy[f0:f1]
    if len(seg) < 2:
        return note["end"]
    peak_i = int(np.argmax(seg[:max(1, int(PEAK_S * sr / HOP))]))
    below = np.nonzero(seg[peak_i:] < seg[peak_i] * 10 ** (-drop_db / 10))[0]
    if len(below) == 0:
        return note["end"]
    end = (f0 + peak_i + below[0]) * HOP / sr
    return float(min(note["end"], max(end, note["start"] + MIN_LEN_S)))


def refine(notes: list[dict], power: np.ndarray | None, sr: int, lag: float, drop_db: float | None,
           min_len: float = 0.0) -> list[dict]:
    """Drop notes shorter than min_len, shift the rest earlier by lag and (if drop_db) trim their tails
    against the stem's power spectrogram."""
    cache: dict[int, np.ndarray] = {}
    out = []
    for n in notes:
        if n["end"] - n["start"] < min_len:
            continue
        if drop_db is not None and power is not None:
            if n["pitch"] not in cache:
                cache[n["pitch"]] = harmonic_energy(n["pitch"], power, sr)
            n = {**n, "end": trimmed_end(n, cache[n["pitch"]], sr, drop_db)}
        out.append({**n, "start": max(0.0, n["start"] - lag), "end": max(0.0, n["end"] - lag)})
    return out


def main(work: Path, variant: str) -> None:
    """Refine every stem json of 02_midi/<variant>/ into 02_midi/<variant>_refined/."""
    src, dst = work / "02_midi" / variant, work / "02_midi" / f"{variant}_refined"
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("*.json")):
        stem, data = path.stem, json.loads(path.read_text())
        drop = DROP_DB.get(stem)
        power, sr = mono_power(work / "01_stems" / variant / f"{stem}.wav") if drop else (None, 0)
        min_len = MIN_NOTE_S.get(stem, 0.0)
        data["notes"] = refine(data["notes"], power, sr, LAG_S.get(stem, 0.0), drop, min_len)
        data["refine"] = {"lag_s": LAG_S.get(stem, 0.0), "drop_db": drop, "min_note_s": min_len}
        (dst / path.name).write_text(json.dumps(data))
        print(f"{stem}: lag {LAG_S.get(stem, 0.0) * 1000:.1f} ms, tail drop {drop} dB", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
