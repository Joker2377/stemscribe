"""Train a mono Mel-Roformer that splits one guitar signal into two guitars by role (fixed order).

Runs in envs/separator.  Usage: train_guitar_split.py [hours=3] [--resume | --init=<checkpoint>]
Data: dataset/slakh_flac/{train,validation}/<track>/stems/Sxx.flac (fetch_slakh_full.py).
Per example: either split a track's guitar patches into two groups (different tone), or split the
stems of one patch into two groups (same tone: forces the model to use what each part plays), sum each
group, take a random 8 s crop where both groups sound, and train on (group A + group B) -> (high, low):
output 1 is the group with the higher mean MIDI pitch in the crop, output 2 the lower one (fixed order;
PIT got stuck at "each output = half the mix" for three runs); within MIN_PITCH_GAP the louder group goes first.
The input is made to look like a separated guitar stem: per-group tone (EQ / reverb / drive, pedalboard),
quiet bleed from the track's other instruments, and mix compression / limiting; targets stay the clean groups.
Model-only snapshots every SNAPSHOT_EVERY steps are kept for picking a checkpoint on real separated stems.
Output: envs/models/guitar_split/{last,best}.pt, work/train_guitar/log.csv
"""
import csv
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # one-off OOMs near the 16 GB limit

import numpy as np
import soundfile as sf
import torch
import mido
import pedalboard as pb
import yaml
from functools import lru_cache
from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "slakh_flac"
CKPT = ROOT / "envs" / "models" / "guitar_split"
LOG = ROOT / "work" / "train_guitar" / "log.csv"
SR = 44100
SEG = 8 * SR  # long enough to hear each part's phrase / role
ACTIVE_DB = -40.0  # crop RMS above this counts as "sounding"
BOTH_ACTIVE_P = 0.8  # rest of the crops may have one group silent (solo passages)
SAME_TONE_P = 0.5  # share of pairs cut from one patch: tone can't tell them apart, the part must
MODEL_CFG = dict(dim=96, depth=4, num_stems=2, time_transformer_depth=1, freq_transformer_depth=1,
                 num_bands=40, stereo=False)
BATCH = 2
LR = 3e-4
VAL_N = 50  # per validation set (different tone / same tone)
VAL_EVERY = 1000
MIN_PITCH_GAP = 2.0  # semitones; closer pairs have no clear "higher part" and are ordered by loudness
BLEED_P = 0.7  # share of examples with other instruments leaking into the input
BLEED_DB = (-30.0, -12.0)  # bleed level relative to the guitars
REVERB_P, DRIVE_P, LIMIT_P = 0.3, 0.15, 0.5
SNAPSHOT_EVERY = 5000
SNR_TAU = 10 ** (-30 / 10)  # error floor = -30 dB of the mixture power, keeps silent references finite


@lru_cache(maxsize=4096)
def midi_notes(f: Path) -> np.ndarray:
    """(N, 3) array of (start s, end s, pitch) for the Slakh MIDI matching a stem file stems/Sxx.flac."""
    mid = f.parent.parent / "MIDI" / f"{f.stem}.mid"
    if not mid.exists():
        return np.zeros((0, 3))
    t, on, notes = 0.0, {}, []
    for msg in mido.MidiFile(mid):  # iterating a MidiFile yields delta times in seconds (tempo map applied)
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on.setdefault(msg.note, t)
        elif msg.type in ("note_off", "note_on") and msg.note in on:
            notes.append((on.pop(msg.note), t, msg.note))
    return np.array(notes).reshape(-1, 3)


def mean_pitch(files: list[Path], t0: float = 0.0, t1: float = float("inf")) -> float:
    """Duration-weighted mean MIDI pitch of a group's notes inside [t0, t1); NaN if it plays nothing there."""
    n = np.concatenate([midi_notes(f) for f in files])
    w = np.clip(np.minimum(n[:, 1], t1) - np.maximum(n[:, 0], t0), 0, None)
    return float((w * n[:, 2]).sum() / w.sum()) if w.sum() > 0 else float("nan")


def pitch_gap(groups: list[list[Path]], start: int) -> float:
    """Mean pitch of groups[0] minus groups[1] (semitones) in the crop starting at sample `start`.
    A group silent in the crop falls back to its mean pitch over the whole song; NaN if still unknown."""
    t0, t1 = start / SR, (start + SEG) / SR
    a, b = (mean_pitch(g, t0, t1) for g in groups)
    if np.isnan(a) or np.isnan(b):
        a, b = (mean_pitch(g) for g in groups)
    return a - b


@lru_cache(maxsize=4096)
def other_stems(track: Path) -> tuple[Path, ...]:
    """Rendered non-guitar, non-drum stem files of a track (the instruments that bleed into a guitar stem)."""
    stems = yaml.safe_load((track / "metadata.yaml").read_text())["stems"]
    files = (track / "stems" / f"{sid}.flac" for sid, s in stems.items()
             if s["inst_class"] not in ("Guitar", "Drums") and s.get("audio_rendered"))
    return tuple(f for f in files if f.exists())


def guitar_patches(track: Path) -> dict[str, list[Path]]:
    """{patch name: [guitar stem files]} for stems that are on disk."""
    stems = yaml.safe_load((track / "metadata.yaml").read_text())["stems"]
    out: dict[str, list[Path]] = {}
    for sid, s in stems.items():
        f = track / "stems" / f"{sid}.flac"
        if s["inst_class"] == "Guitar" and s.get("audio_rendered") and f.exists():
            out.setdefault(s["plugin_name"].split("/")[-1], []).append(f)
    return out


def can_split(patches: dict[str, list[Path]], same_tone: bool) -> bool:
    """Whether a track can give a pair of the requested kind."""
    return any(len(f) >= 2 for f in patches.values()) if same_tone else len(patches) >= 2


def build_index(split: str) -> list[dict[str, list[Path]]]:
    """Every track of a split that can give a different-tone or a same-tone guitar pair."""
    tracks = [guitar_patches(t) for t in sorted((DATA / split).glob("Track*"))]
    return [p for p in tracks if can_split(p, False) or can_split(p, True)]


def split_groups(patches: dict[str, list[Path]], same_tone: bool, rng: random.Random) -> list[list[Path]]:
    """Two non-empty groups of stem files: two parts on one patch, or patches split into two sets."""
    if same_tone:
        files = list(rng.choice([f for f in patches.values() if len(f) >= 2]))
        rng.shuffle(files)
        cut = rng.randint(1, len(files) - 1)
        return [files[:cut], files[cut:]]
    names = list(patches)
    rng.shuffle(names)
    cut = rng.randint(1, len(names) - 1)
    return [sum((patches[n] for n in names[:cut]), []), sum((patches[n] for n in names[cut:]), [])]


def rms_db(x: np.ndarray) -> float:
    """RMS level in dBFS."""
    return 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-12))


def read_sum(files: list[Path], start: int) -> np.ndarray:
    """Sum of SEG samples from start of every file (zero padded at the end)."""
    out = np.zeros(SEG, np.float32)
    for f in files:
        x, _ = sf.read(f, start=start, frames=SEG, dtype="float32")
        out[:len(x)] += x
    return out


def sample_example(patches: dict[str, list[Path]], rng: random.Random, need_both: bool | None = None,
                   same_tone: bool | None = None) -> np.ndarray | None:
    """(2, SEG) targets for one random pair and crop, or None if the track can't give it / no sounding crop.
    need_both: both groups must sound (None = with probability BOTH_ACTIVE_P).
    same_tone: pair from one patch (None = with probability SAME_TONE_P, falling back to what exists)."""
    if same_tone is None:
        same_tone = rng.random() < SAME_TONE_P
        if not can_split(patches, same_tone):
            same_tone = not same_tone
    if not can_split(patches, same_tone):
        return None
    groups = split_groups(patches, same_tone, rng)
    length = min(sf.info(f).frames for g in groups for f in g)
    if need_both is None:
        need_both = rng.random() < BOTH_ACTIVE_P
    for _ in range(10):
        start = rng.randint(0, max(0, length - SEG))
        a, b = read_sum(groups[0], start), read_sum(groups[1], start)
        levels = (rms_db(a), rms_db(b))
        if (min(levels) if need_both else max(levels)) <= ACTIVE_DB:
            continue
        gap = pitch_gap(groups, start)
        first = gap > 0 if abs(gap) >= MIN_PITCH_GAP else levels[0] >= levels[1]  # NaN gap -> loudness
        others = other_stems(groups[0][0].parent.parent)
        n_bleed = min(len(others), rng.randint(1, 3))
        bleed = read_sum(rng.sample(others, n_bleed), start) if others else np.zeros(SEG, np.float32)
        return np.stack([a, b] if first else [b, a]), bleed, same_tone
    return None


def tone_chain(rng: random.Random) -> pb.Pedalboard:
    """Random EQ, sometimes a little drive and room reverb: different mixing / amp choices for one guitar group."""
    fx = [pb.HighpassFilter(rng.uniform(40, 200)), pb.LowpassFilter(rng.uniform(4000, 16000)),
          pb.PeakFilter(rng.uniform(300, 4000), rng.uniform(-6, 6), rng.uniform(0.7, 2.0))]
    if rng.random() < DRIVE_P:
        fx.append(pb.Distortion(rng.uniform(3, 15)))
    if rng.random() < REVERB_P:
        fx.append(pb.Reverb(room_size=rng.uniform(0.1, 0.5), wet_level=rng.uniform(0.05, 0.25), dry_level=1.0))
    return pb.Pedalboard(fx)


def at_level(x: np.ndarray, rms: float) -> np.ndarray:
    """x scaled to the given RMS."""
    return x * (rms / max(float(np.sqrt(np.mean(x ** 2))), 1e-9))


def augment(example: tuple, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """(mixture (SEG,), targets (2, SEG)) from sample_example's output.
    Targets: each group through its tone chain (one shared chain for same-tone pairs), +-6 dB gain each.
    Mixture: targets + bleed, set to -30..-12 dBFS RMS, sometimes compressed and limited (loud masters)."""
    y, bleed, same_tone = example
    chains = [tone_chain(rng)] * 2 if same_tone else [tone_chain(rng), tone_chain(rng)]
    y = np.stack([c(g, SR, reset=True) for c, g in zip(chains, y)]).astype(np.float32)
    y = y * np.array([[10 ** (rng.uniform(-6, 6) / 20)] for _ in range(2)], np.float32)
    mix = y.sum(0)
    if rng.random() < BLEED_P and np.any(bleed):
        mix = mix + at_level(bleed, float(np.sqrt(np.mean(mix ** 2))) * 10 ** (rng.uniform(*BLEED_DB) / 20))
    gain = 10 ** (rng.uniform(-30, -12) / 20) / max(float(np.sqrt(np.mean(mix ** 2))), 1e-6)
    mix, y = mix * gain, y * gain
    if rng.random() < LIMIT_P:
        mix = pb.Pedalboard([pb.Compressor(rng.uniform(-30, -15), rng.uniform(2, 6)),
                             pb.Limiter(rng.uniform(-6, -1))])(mix, SR, reset=True)
    return mix.astype(np.float32), y


class GuitarPairs(IterableDataset):
    """Endless stream of (mixture, targets) training examples."""

    def __init__(self, index: list[dict[str, list[Path]]]):
        """index: output of build_index."""
        self.index = index

    def __iter__(self):
        """Yield (1, SEG) mixture and (2, 1, SEG) targets forever."""
        info = get_worker_info()
        rng = random.Random(time.time_ns() + (info.id if info else 0))
        while True:
            ex = sample_example(rng.choice(self.index), rng)
            if ex is not None:
                mix, y = augment(ex, rng)
                yield torch.from_numpy(mix[None]), torch.from_numpy(y[:, None])


def fixed_val_set(index: list[dict[str, list[Path]]], same_tone: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """VAL_N seeded examples from the validation split: (N, 1, SEG) mixtures, (N, 2, 1, SEG) targets.
    Both guitars sound in every example, so SI-SDR is well defined."""
    rng, mixes, ys = random.Random(0), [], []
    index = [p for p in index if can_split(p, same_tone)]
    while len(ys) < VAL_N:
        ex = sample_example(rng.choice(index), rng, need_both=True, same_tone=same_tone)
        if ex is not None:
            mix, y = augment(ex, rng)
            mixes.append(mix)
            ys.append(y)
    return torch.from_numpy(np.stack(mixes))[:, None], torch.from_numpy(np.stack(ys))[:, :, None]


def thresholded_snr(est: torch.Tensor, ref: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Negative SNR with an error floor tied to the mixture (MixIT), finite for silent refs. (B,1,T) -> (B,)."""
    err = ((ref - est) ** 2).sum(dim=(1, 2)) + SNR_TAU * (mix ** 2).sum(dim=(1, 2))
    return 10 * torch.log10(err + 1e-8) - 10 * torch.log10((ref ** 2).sum(dim=(1, 2)) + 1e-8)


def fixed_loss(est: torch.Tensor, ref: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Mean over batch and outputs of output i vs target i (high, low). est/ref: (B, 2, 1, T), mix: (B, 1, T)."""
    return (thresholded_snr(est[:, 0], ref[:, 0], mix) + thresholded_snr(est[:, 1], ref[:, 1], mix)).mean() / 2


def si_sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Scale-invariant SDR in dB over the last axis."""
    a = (est * ref).sum(-1, keepdim=True) / ((ref ** 2).sum(-1, keepdim=True) + 1e-8)
    t = a * ref
    return 10 * torch.log10((t ** 2).sum(-1) / (((est - t) ** 2).sum(-1) + 1e-8) + 1e-8)


def fixed_si_sdr(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Per-example mean SI-SDR of output i vs target i. (B, 2, 1, T) -> (B,)."""
    return ((si_sdr(est[:, 0], ref[:, 0]) + si_sdr(est[:, 1], ref[:, 1])) / 2).squeeze(-1)


def trim(est: torch.Tensor, ref: torch.Tensor, mix: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Cut est / ref / mix to a common length (the iSTFT can return a few samples less)."""
    n = min(est.shape[-1], ref.shape[-1])
    return est[..., :n], ref[..., :n], mix[..., :n]


@torch.no_grad()
def validate(model: torch.nn.Module, mix: torch.Tensor, ref: torch.Tensor, dev: str) -> tuple[float, float]:
    """(fixed-order loss, SI-SDR improvement over using the mixture as both outputs) on the fixed set."""
    model.eval()
    losses, gains = [], []
    for i in range(0, len(mix), BATCH):
        m, r = mix[i:i + BATCH].to(dev), ref[i:i + BATCH].to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16):
            est = model(m)
        est, r, m = trim(est.float(), r, m)
        losses.append(fixed_loss(est, r, m).item())
        base = fixed_si_sdr(m[:, None].expand_as(est), r)
        gains.append((fixed_si_sdr(est, r) - base).mean().item())
    model.train()
    return float(np.mean(losses)), float(np.mean(gains))


def log_row(row: dict, log: Path = LOG) -> None:
    """Append one row to the training log csv."""
    log.parent.mkdir(parents=True, exist_ok=True)
    new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def train(data: IterableDataset, val_sets: dict, ckpt: Path, log: Path, hours: float, resume: bool,
          init: Path | None) -> None:
    """Train on data until the time budget runs out, validating on val_sets ({name: (mix, ref)}) and
    checkpointing into ckpt every VAL_EVERY steps (best.pt = best mean SI-SDR gain, step_*.pt snapshots)."""
    dev = "cuda"
    model = MelBandRoformer(**MODEL_CFG).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    step, best = 0, -1e9
    ckpt.mkdir(parents=True, exist_ok=True)
    if resume and (ckpt / "last.pt").exists():
        ck = torch.load(ckpt / "last.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
    elif init is not None:  # warm start: weights only, fresh optimizer and step count
        model.load_state_dict(torch.load(init, map_location=dev)["model"])
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M  start step {step}", flush=True)
    loader = DataLoader(data, batch_size=BATCH, num_workers=4, persistent_workers=True)
    end, t0, run = time.time() + hours * 3600, time.time(), []
    for mix, ref in loader:
        mix, ref = mix.to(dev), ref.to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16):
            est = model(mix)
        est, ref, mix = trim(est.float(), ref, mix)
        loss = fixed_loss(est, ref, mix)  # no mixture consistency: the input also holds bleed
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        run.append(loss.item())
        if step % VAL_EVERY == 0 or time.time() > end:
            scores = {k: validate(model, m, r, dev) for k, (m, r) in val_sets.items()}
            gain = float(np.mean([g for _, g in scores.values()]))
            best = max(best, gain)
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                     "cfg": MODEL_CFG}
            torch.save(state, ckpt / "last.pt")
            if gain >= best:
                torch.save(state, ckpt / "best.pt")
            if step % SNAPSHOT_EVERY == 0:
                torch.save({"model": model.state_dict(), "cfg": MODEL_CFG, "step": step},
                           ckpt / f"step_{step:06d}.pt")
            row = {"step": step, "train_loss": round(float(np.mean(run)), 4),
                   **{f"val_loss_{k}": round(v[0], 4) for k, v in scores.items()},
                   **{f"gain_db_{k}": round(v[1], 2) for k, v in scores.items()},
                   "hours": round((time.time() - t0) / 3600, 2)}
            log_row(row, log)
            print(row, flush=True)
            run = []
            if time.time() > end:
                break


def main(hours: float, resume: bool, init: Path | None) -> None:
    """Train on Slakh: different-tone and same-tone validation sets."""
    train_idx, val_idx = build_index("train"), build_index("validation")
    print(f"train tracks {len(train_idx)}  validation tracks {len(val_idx)}", flush=True)
    val_sets = {"diff": fixed_val_set(val_idx, False), "same": fixed_val_set(val_idx, True)}
    train(GuitarPairs(train_idx), val_sets, CKPT, LOG, hours, resume, init)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    init = next((Path(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--init=")), None)
    main(float(args[0]) if args else 3.0, "--resume" in sys.argv, init)
