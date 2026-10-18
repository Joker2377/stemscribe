"""Stage 2.7: split the guitar stem into two guitars with the trained model (train_guitar_split.py).

Runs in envs/separator.  Usage: split_guitars_model.py <work_dir> <stem_variant> [checkpoint]
Writes 01_stems/<variant>_gmodel/guitar_1.wav (higher part) and guitar_2.wav (lower part).
Outputs are made to sum to the input (mixture consistency, as run 5 was trained).
The model is mono: left and right are split separately (the fixed high/low order keeps them consistent),
in 8 s windows with 50% overlap, cross-faded with a Hann window.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer

SEG = 8 * 44100  # chunk length the model was trained on (training/train_guitar_split.py SEG)

# Run 5 + mixture consistency beat run 6 (Slakh balanced_acc test 0.756 vs 0.700, val 0.676 vs 0.669)
MODEL = Path(__file__).resolve().parents[1] / "envs" / "models" / "guitar_split_run5_gap" / "best.pt"  # frozen copy; training writes guitar_split/

HOP = SEG // 2
BATCH = 8


def load_model(path: Path, dev: str) -> torch.nn.Module:
    """Rebuild the model from a training checkpoint."""
    ck = torch.load(path, map_location=dev)
    model = MelBandRoformer(**ck["cfg"]).to(dev).eval()
    model.load_state_dict(ck["model"])
    return model


def mixture_consistent(est: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    """Share the residual (mix - sum of outputs) equally, so outputs add up to the input.
    est: (B, 2, 1, T), mix: (B, 1, T)."""
    return est + (mix[:, None] - est.sum(1, keepdim=True)) / est.shape[1]


def windows(x: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Cut a 1-D signal into (N, SEG) overlapping windows (zero padded); return windows and their starts."""
    starts = list(range(0, max(1, len(x) - HOP), HOP))
    pad = np.zeros(starts[-1] + SEG, np.float32)
    pad[:len(x)] = x
    return np.stack([pad[s:s + SEG] for s in starts]), starts


@torch.no_grad()
def split_channel(model: torch.nn.Module, x: np.ndarray, dev: str) -> np.ndarray:
    """(2, len(x)) high / low parts of one mono channel via windowed overlap-add."""
    wins, starts = windows(x)
    out = np.zeros((2, starts[-1] + SEG), np.float32)
    norm = np.zeros(starts[-1] + SEG, np.float32)
    fade = np.hanning(SEG).astype(np.float32) + 1e-3  # +eps: the very first / last samples still count
    for i in range(0, len(wins), BATCH):
        mix = torch.from_numpy(wins[i:i + BATCH])[:, None].to(dev)  # (B, 1, SEG)
        with torch.autocast(dev, dtype=torch.bfloat16):
            est = model(mix).float()
        n = min(est.shape[-1], SEG)
        est = mixture_consistent(est[..., :n], mix[..., :n]).cpu().numpy()[:, :, 0]  # (B, 2, n)
        for e, s in zip(est, starts[i:i + BATCH]):
            out[:, s:s + n] += e * fade[:n]
            norm[s:s + n] += fade[:n]
    return (out / np.maximum(norm, 1e-6))[:, :len(x)]


def main(work: Path, variant: str, ckpt: Path) -> None:
    """Split 01_stems/<variant>/guitar.wav into guitar_1 / guitar_2 under 01_stems/<variant>_gmodel/."""
    dev = "cuda"
    audio, sr = sf.read(work / "01_stems" / variant / "guitar.wav", dtype="float32")
    audio = audio[:, None] if audio.ndim == 1 else audio
    model = load_model(ckpt, dev)
    parts = np.stack([split_channel(model, audio[:, c], dev) for c in range(audio.shape[1])], -1)  # (2, T, C)
    dst = work / "01_stems" / f"{variant}_gmodel"
    dst.mkdir(parents=True, exist_ok=True)
    for k in range(2):
        sf.write(dst / f"guitar_{k + 1}.wav", parts[k], sr, subtype="FLOAT")
    print(f"guitar split with {ckpt.name}: {audio.shape[0] / sr:.1f} s -> {dst}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]) if len(sys.argv) > 3 else MODEL)
