"""Download the full Slakh2100 (all stems, MIDI, mix) for training the two-guitar splitter.

Runs in envs/tools.  Usage: fetch_slakh_full.py [out_dir=dataset/slakh_flac] [split ...]
Source: WhiRik/slakh_flac (gated, accept terms on the website first), which keeps the original
per-file layout <split>/<track>/{metadata.yaml, mix.flac, stems/Sxx.flac, MIDI/Sxx.mid}.
Resumable: files already in out_dir are skipped. Track selection (>= 2 guitars) happens at training
time from metadata.yaml; non-guitar stems are kept for bleed simulation.
The "omitted" split is skipped on purpose: those tracks reuse MIDI from other splits (test leakage).
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "WhiRik/slakh_flac"
SPLITS = ("train", "validation", "test")
WORKERS = 8


def main(out: Path, splits: list[str]) -> None:
    """Mirror the given splits of the repo into out/."""
    snapshot_download(REPO, repo_type="dataset", local_dir=out, max_workers=WORKERS,
                      allow_patterns=[f"{s}/*" for s in splits])


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/slakh_flac"),
         sys.argv[2:] or list(SPLITS))
