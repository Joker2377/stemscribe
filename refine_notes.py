"""Stage 3.2: fix MuScriptor's constant onset lag and over-long note tails.

Runs in envs/tools.  Usage: refine_notes.py <work_dir> <stem_variant> [midi_variant=stem_variant]
Reads 01_stems/<v>/<stem>.wav + 02_midi/<m>/<stem>.json, writes 02_midi/<m>_refined/<stem>.json.
- lag: every note of a stem moves earlier by LAG_S[stem] (measured on Slakh, eval_midi median onset error)
- tail: a pitched note ends where the energy at its harmonics has fallen DROP_DB below its attack peak
- blips: notes shorter than MIN_NOTE_S are dropped (raw MuScriptor output; ~10% of bass notes are < 20 ms)
- silence: every stem's notes end where the stem goes silent (MuScriptor gets no note-off from silence:
  demo lead vocal held one A4 196.5 -> 230.3 s across a 30 s gap the karaoke split left empty); notes that
  start in silence are dropped (demo bass: 197 repeated A2s over a silent 134-169 s)
- crowding: stems without a tail trim (piano, vocals, ...) still get one, at POLY_DROP_DB, for notes that overlap
  a moment with more than POLY_MAX notes sounding (MuScriptor sometimes keeps re-tying dozens of notes chunk
  after chunk: demo piano 33-48 s had 40-49 at once, pitches 19-104)
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
# Slakh piano F1+off test / val: 90 dB .385 -> .387 / .328 -> .326, other stems unchanged; 60 dB cut real quiet
# piano tails (val .311).
SILENCE_DB = 90.0     # a frame is silent when its power is this far below the stem's loudest frame
SILENCE_MIN_S = 0.3   # a note is cut at the first silent run at least this long
# Real Slakh piano peaks at 4-8 notes at once; 3 of 22 tracks hit 34-59 (the same failure). Scores unchanged
# (the crowded notes match nothing), while trimming every piano note at 20-30 dB lowers test F1+off .387 -> .34-.36.
# A real onset is audible: notes whose first SILENCE_MIN_S averages ONSET_QUIET_DB below the loudest frame go
# (demo bass 135-169 s sits on -77..-100 dB leakage, so "all frames silent" alone misses them).
ONSET_QUIET_DB = 60.0
POLY_MAX = 16
POLY_DROP_DB = 20.0


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


def silent_frames(power: np.ndarray) -> np.ndarray:
    """Per-frame flag: total power SILENCE_DB below the loudest frame of the stem."""
    frame = power.sum(0)
    return frame <= frame.max() * 10 ** (-SILENCE_DB / 10)


def silence_end(note: dict, silent: np.ndarray, sr: int) -> float:
    """Start of the first silent run (>= SILENCE_MIN_S) inside the note, else the note's own end."""
    f0, f1 = int(note["start"] * sr / HOP), int(note["end"] * sr / HOP) + 1
    k = max(1, int(np.ceil(SILENCE_MIN_S * sr / HOP)))
    seg = silent[f0:f1].astype(int)
    if len(seg) < k:
        return note["end"]
    runs = np.nonzero(np.convolve(seg, np.ones(k, int), "valid") == k)[0]
    if len(runs) == 0:
        return note["end"]
    return float(min(note["end"], max((f0 + runs[0]) * HOP / sr, note["start"] + MIN_LEN_S)))


def crowded_spans(notes: list[dict], max_poly: int) -> list[tuple[float, float]]:
    """Time spans where more than max_poly notes sound at once."""
    events = sorted([(n["start"], 1) for n in notes] + [(n["end"], -1) for n in notes])
    spans, count, since = [], 0, None
    for t, step in events:
        count += step
        if count > max_poly and since is None:
            since = t
        elif count <= max_poly and since is not None:
            spans.append((since, t))
            since = None
    return spans


def overlaps(note: dict, spans: list[tuple[float, float]]) -> bool:
    """Whether the note overlaps any of the spans."""
    return any(a < note["end"] and note["start"] < b for a, b in spans)


def frame_level_db(power: np.ndarray) -> np.ndarray:
    """Per-frame total power in dB relative to the stem's loudest frame."""
    frame = power.sum(0)
    return 10 * np.log10(frame / max(frame.max(), 1e-30) + 1e-30)


def starts_in_silence(note: dict, level_db: np.ndarray, sr: int) -> bool:
    """Whether the note's first SILENCE_MIN_S (or whole length, if shorter) averages ONSET_QUIET_DB below peak."""
    f0 = int(note["start"] * sr / HOP)
    f1 = max(f0 + 1, min(int(note["end"] * sr / HOP), f0 + int(np.ceil(SILENCE_MIN_S * sr / HOP))))
    seg = level_db[f0:f1]
    return len(seg) > 0 and 10 * np.log10(np.mean(10 ** (seg / 10))) < -ONSET_QUIET_DB


def refine(notes: list[dict], power: np.ndarray, sr: int, lag: float, drop_db: float | None,
           min_len: float = 0.0, crowd_trim: bool = False) -> list[dict]:
    """Drop notes shorter than min_len or starting in stem silence, cut notes at silence, trim tails against the stem's power
    spectrogram (all notes at drop_db; else, with crowd_trim, crowded notes at POLY_DROP_DB), shift by lag."""
    cache: dict[int, np.ndarray] = {}
    silent, level_db = silent_frames(power), frame_level_db(power)
    kept = [{**n, "end": silence_end(n, silent, sr)} for n in notes
            if n["end"] - n["start"] >= min_len and not starts_in_silence(n, level_db, sr)]
    crowded = crowded_spans(kept, POLY_MAX) if crowd_trim and drop_db is None else []
    out = []
    for n in kept:
        db = drop_db if drop_db is not None else (POLY_DROP_DB if overlaps(n, crowded) else None)
        if db is not None:
            if n["pitch"] not in cache:
                cache[n["pitch"]] = harmonic_energy(n["pitch"], power, sr)
            n = {**n, "end": trimmed_end(n, cache[n["pitch"]], sr, db)}
        out.append({**n, "start": max(0.0, n["start"] - lag), "end": max(0.0, n["end"] - lag)})
    return out


def main(work: Path, variant: str, midi_variant: str) -> None:
    """Refine every stem json of 02_midi/<midi_variant>/ into 02_midi/<midi_variant>_refined/,
    measuring against the audio of 01_stems/<variant>/."""
    src, dst = work / "02_midi" / midi_variant, work / "02_midi" / f"{midi_variant}_refined"
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.glob("*.json")):
        stem, data = path.stem, json.loads(path.read_text())
        drop = DROP_DB.get(stem)
        power, sr = mono_power(work / "01_stems" / variant / f"{stem}.wav")
        min_len = MIN_NOTE_S.get(stem, 0.0)
        data["notes"] = refine(data["notes"], power, sr, LAG_S.get(stem, 0.0), drop, min_len,
                               crowd_trim=stem != "drums")
        data["refine"] = {"lag_s": LAG_S.get(stem, 0.0), "drop_db": drop, "min_note_s": min_len,
                          "silence_db": SILENCE_DB, "poly_max": POLY_MAX}
        (dst / path.name).write_text(json.dumps(data))
        print(f"{stem}: lag {LAG_S.get(stem, 0.0) * 1000:.1f} ms, tail drop {drop} dB", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else sys.argv[2])
