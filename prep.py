"""Stage 1: decode any input to a fixed-spec master.wav and write analysis.json.

Runs in envs/beat_this.  Usage: prep.py <input_audio> <work_dir>
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
PEAK_DBFS = -1.0
CLIP_LEVEL = 0.999


def decode(src: Path, dst: Path) -> None:
    """Decode any ffmpeg-readable file to 44.1 kHz stereo float32 WAV."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn",
         "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(dst)],
        check=True,
    )


def count_clipped(audio: np.ndarray) -> int:
    """Count samples at or near full scale, a sign the source was already clipped."""
    return int((np.abs(audio) >= CLIP_LEVEL).sum())


def peak_normalize(audio: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale so the peak sits at PEAK_DBFS; return the audio and the gain applied in dB."""
    peak = float(np.abs(audio).max())
    if peak == 0:
        return audio, 0.0
    gain = 10 ** (PEAK_DBFS / 20) / peak
    return audio * gain, 20 * np.log10(gain)


def lr_correlation(audio: np.ndarray, lo_hz: float | None = None, hi_hz: float | None = None) -> float:
    """Pearson correlation of L and R, optionally band-limited via an FFT mask. ~1 means mono."""
    left, right = audio[:, 0], audio[:, 1]
    if lo_hz is not None:
        freqs = np.fft.rfftfreq(len(left), 1 / SR)
        mask = (freqs >= lo_hz) & (freqs <= hi_hz)
        left = np.fft.irfft(np.fft.rfft(left) * mask, len(left))
        right = np.fft.irfft(np.fft.rfft(right) * mask, len(right))
    if left.std() == 0 or right.std() == 0:
        return 1.0
    return float(np.corrcoef(left, right)[0, 1])


def track_beats(wav: Path) -> tuple[list[float], list[float]]:
    """Return beat and downbeat times in seconds using beat_this (no DBN)."""
    from beat_this.inference import File2Beats
    beats, downbeats = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)(str(wav))
    return [float(b) for b in beats], [float(d) for d in downbeats]


def main(src: Path, work: Path) -> None:
    """Run all prep steps and write 00_prep/master.wav + analysis.json."""
    out = work / "00_prep"
    out.mkdir(parents=True, exist_ok=True)
    master = out / "master.wav"

    decode(src, master)
    audio, _ = sf.read(master, dtype="float32")
    clipped = count_clipped(audio)
    audio, gain_db = peak_normalize(audio)
    sf.write(master, audio, SR, subtype="FLOAT")

    beats, downbeats = track_beats(master)
    analysis = {
        "sample_rate": SR,
        "duration_s": len(audio) / SR,
        "gain_db": gain_db,
        "clipped_samples": clipped,
        "lr_corr_full": lr_correlation(audio),
        "lr_corr_200_5k": lr_correlation(audio, 200, 5000),
        "beats": beats,
        "downbeats": downbeats,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
