"""Train the two-guitar splitter on real multitrack recordings only (MoisesDB), same model and loss as
train_guitar_split.py (Slakh).

Runs in envs/separator.  Usage: train_guitar_split_moises.py [hours=12] [--resume | --init=<checkpoint>]
Needs: prep_moisesdb.py (FLAC tracks), prep_moises_guitar.py (sw_guitar_dry.flac), transcribe_moises.py
(per-track notes, standing in for MIDI).
Groups: a song's guitar tracks sorted by mean pitch and cut at the biggest pitch gap (fixed per song, so the
same input never gets two different targets).  Output 1 = the group with the higher mean pitch in the crop,
within MIN_PITCH_GAP the louder one (as for Slakh).
Input, per example (one random channel, the model is mono):
- REAL_P: the song's own pipeline guitar stem (mix -> BS-Roformer-SW -> dereverb): the exact input at run time
- else: the two groups summed + quiet bleed from the song's other tracks, sometimes compressed / limited
Targets are always the clean groups.  Every SEG crop is put at a random level, input and targets alike.
Validation songs: prep_moises_val.val_songs (also scored on the real task by score_checkpoints.py).
Output: envs/models/guitar_split_moises/, work/train_guitar_moises/log.csv
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pedalboard as pb
import soundfile as sf
import torch
from torch.utils.data import IterableDataset, get_worker_info

from prep_moises_val import val_songs
from train_guitar_split import (ACTIVE_DB, BLEED_DB, BOTH_ACTIVE_P, LIMIT_P, MIN_PITCH_GAP, ROOT, SEG, SR, VAL_N,
                                at_level, rms_db, train)

DATA = ROOT / "dataset" / "moisesdb_flac"
CKPT = ROOT / "envs" / "models" / "guitar_split_moises"
LOG = ROOT / "work" / "train_guitar_moises" / "log.csv"
SW_NAME = "sw_guitar_dry.flac"
REAL_P = 0.6
BLEED_P = 0.7
NO_BLEED = ("guitar", "drums", "percussion")  # the guitar stem has little of these


def notes_array(f: Path) -> np.ndarray:
    """(N, 3) (start s, end s, pitch) of a track's transcription (transcribe_moises.py)."""
    notes = json.loads(f.with_suffix(".json").read_text())["notes"]
    return np.array([(n["start"], n["end"], n["pitch"]) for n in notes]).reshape(-1, 3)


def mean_pitch(notes: list[np.ndarray], t0: float = 0.0, t1: float = float("inf")) -> float:
    """Duration-weighted mean pitch of a group's notes inside [t0, t1); NaN if it plays nothing there."""
    n = np.concatenate(notes)
    w = np.clip(np.minimum(n[:, 1], t1) - np.maximum(n[:, 0], t0), 0, None)
    return float((w * n[:, 2]).sum() / w.sum()) if w.sum() > 0 else float("nan")


def pitch_groups(tracks: list[Path]) -> list[list[Path]] | None:
    """Tracks with notes, sorted by mean pitch and cut at the biggest gap into [low, high]; None if < 2."""
    pitched = sorted((mean_pitch([notes_array(f)]), f) for f in tracks if len(notes_array(f)))
    if len(pitched) < 2:
        return None
    cut = 1 + int(np.argmax(np.diff([p for p, _ in pitched])))
    return [[f for _, f in pitched[:cut]], [f for _, f in pitched[cut:]]]


def song_entry(song: Path) -> dict | None:
    """{groups, notes, others, sw, frames} of one song, or None if it can't give a pair."""
    meta = json.loads((song / "data.json").read_text(encoding="utf-8"))
    tracks = {s["stemName"]: [song / s["stemName"] / f"{t['id']}.flac" for t in s["tracks"]] for s in meta["stems"]}
    groups = pitch_groups(tracks.get("guitar", []))
    if groups is None or not (song / SW_NAME).exists():
        return None
    others = [f for stem, fs in tracks.items() if stem not in NO_BLEED for f in fs]
    return {"groups": groups, "notes": [[notes_array(f) for f in g] for g in groups], "others": others,
            "sw": song / SW_NAME, "frames": sf.info(song / SW_NAME).frames}


def build_index() -> tuple[list[dict], list[dict]]:
    """(train songs, validation songs)."""
    songs = sorted(s for s in DATA.iterdir() if (s / "data.json").exists())
    val = set(val_songs(DATA))
    entries = {s: song_entry(s) for s in songs}
    pick = lambda keep: [e for s, e in entries.items() if e is not None and keep(s)]
    return pick(lambda s: s not in val), pick(lambda s: s in val)


def read_channel(files: list[Path], start: int, ch: int) -> np.ndarray:
    """Sum of SEG samples of one channel from start of every file (zero padded at the end)."""
    out = np.zeros(SEG, np.float32)
    for f in files:
        x, _ = sf.read(f, start=start, frames=SEG, dtype="float32", always_2d=True)
        out[:len(x)] += x[:, min(ch, x.shape[1] - 1)]
    return out


def ordered(entry: dict, a: np.ndarray, b: np.ndarray, start: int) -> np.ndarray:
    """(2, SEG) targets with the higher group of this crop first (louder first when the pitches are close)."""
    t0, t1 = start / SR, (start + SEG) / SR
    pa, pb_ = (mean_pitch(n, t0, t1) for n in entry["notes"])
    if np.isnan(pa) or np.isnan(pb_):
        pa, pb_ = (mean_pitch(n) for n in entry["notes"])
    gap = pa - pb_
    first = gap > 0 if abs(gap) >= MIN_PITCH_GAP else rms_db(a) >= rms_db(b)
    return np.stack([a, b] if first else [b, a])


def synthetic_input(entry: dict, y: np.ndarray, start: int, ch: int, rng: random.Random) -> np.ndarray:
    """Groups summed + sometimes quiet bleed from 1-3 other tracks + sometimes compression / limiting."""
    mix = y.sum(0)
    if entry["others"] and rng.random() < BLEED_P:
        bleed = read_channel(rng.sample(entry["others"], min(len(entry["others"]), rng.randint(1, 3))), start, ch)
        if np.any(bleed):
            mix = mix + at_level(bleed, float(np.sqrt(np.mean(mix ** 2))) * 10 ** (rng.uniform(*BLEED_DB) / 20))
    if rng.random() < LIMIT_P:
        mix = pb.Pedalboard([pb.Compressor(rng.uniform(-30, -15), rng.uniform(2, 6)),
                             pb.Limiter(rng.uniform(-6, -1))])(mix, SR, reset=True)
    return mix


def sample_example(entry: dict, rng: random.Random, need_both: bool | None = None,
                   real: bool | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """(input (SEG,), targets (2, SEG)) at a random level, or None if no sounding crop was found."""
    need_both = rng.random() < BOTH_ACTIVE_P if need_both is None else need_both
    real = rng.random() < REAL_P if real is None else real
    ch = rng.randint(0, 1)
    for _ in range(10):
        start = rng.randint(0, max(0, entry["frames"] - SEG))
        a, b = (read_channel(g, start, ch) for g in entry["groups"])
        levels = (rms_db(a), rms_db(b))
        if (min(levels) if need_both else max(levels)) <= ACTIVE_DB:
            continue
        y = ordered(entry, a, b, start)
        mix = read_channel([entry["sw"]], start, ch) if real else synthetic_input(entry, y, start, ch, rng)
        gain = 10 ** (rng.uniform(-30, -12) / 20) / max(float(np.sqrt(np.mean(mix ** 2))), 1e-6)
        return (mix * gain).astype(np.float32), (y * gain).astype(np.float32)
    return None


class MoisesPairs(IterableDataset):
    """Endless stream of (input, targets) training examples."""

    def __init__(self, index: list[dict]):
        """index: training songs from build_index."""
        self.index = index

    def __iter__(self):
        """Yield (1, SEG) input and (2, 1, SEG) targets forever."""
        info = get_worker_info()
        rng = random.Random(time.time_ns() + (info.id if info else 0))
        while True:
            ex = sample_example(rng.choice(self.index), rng)
            if ex is not None:
                yield torch.from_numpy(ex[0][None]), torch.from_numpy(ex[1][:, None])


def fixed_val_set(index: list[dict], real: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """VAL_N seeded examples (both groups sounding): (N, 1, SEG) inputs, (N, 2, 1, SEG) targets."""
    rng, mixes, ys = random.Random(0), [], []
    while len(ys) < VAL_N:
        ex = sample_example(rng.choice(index), rng, need_both=True, real=real)
        if ex is not None:
            mixes.append(ex[0])
            ys.append(ex[1])
    return torch.from_numpy(np.stack(mixes))[:, None], torch.from_numpy(np.stack(ys))[:, :, None]


def main(hours: float, resume: bool, init: Path | None) -> None:
    """Train on MoisesDB: real-input and synthetic-input validation sets."""
    train_idx, val_idx = build_index()
    print(f"train songs {len(train_idx)}  validation songs {len(val_idx)}", flush=True)
    val_sets = {"real": fixed_val_set(val_idx, True), "synth": fixed_val_set(val_idx, False)}
    train(MoisesPairs(train_idx), val_sets, CKPT, LOG, hours, resume, init)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    init = next((Path(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--init=")), None)
    main(float(args[0]) if args else 12.0, "--resume" in sys.argv, init)
