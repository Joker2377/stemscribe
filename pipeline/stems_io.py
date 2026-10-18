"""Shared stem writing for every separator script (needs only numpy + soundfile)."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def residual_report(master: np.ndarray, stems: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    """Return master minus sum of stems, and its energy relative to master in dB."""
    residual = master - sum(stems.values())
    ratio = (residual ** 2).sum() / max((master ** 2).sum(), 1e-12)
    return residual, float(10 * np.log10(max(ratio, 1e-12)))


def write_stems(out: Path, master: np.ndarray, stems: dict[str, np.ndarray], sr: int, meta: dict) -> None:
    """Write every stem, residual.wav and stems.json (meta + residual_db) into out/."""
    residual, residual_db = residual_report(master, stems)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in {**stems, "residual": residual}.items():
        sf.write(out / f"{name}.wav", data, sr, subtype="FLOAT")
    (out / "stems.json").write_text(json.dumps({**meta, "residual_db": residual_db}, indent=1))
