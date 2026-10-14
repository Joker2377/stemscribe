"""Download the first N Slakh2100 test-split tracks that match our band.

Runs in envs/tools.  Usage: fetch_slakh.py <out_dir> [n_tracks=10]
The official 104 GB tarball is shuffled (one track's files are spread over the whole archive),
so we pull per-file from two Hugging Face mirrors instead:
  metadata.yaml + MIDI/Sxx.mid  <- aashishbishow/slakh2100_midi_only_redux
  mix.flac                      <- DreamyWanderer/Slakh2100-FLAC-Redux-Reduced
A track is kept when it has rendered Guitar, Piano, Bass and Drums stems with MIDI.
"""
import shutil
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

MIDI_REPO = "aashishbishow/slakh2100_midi_only_redux"
MIX_REPO = "DreamyWanderer/Slakh2100-FLAC-Redux-Reduced"
SPLIT = "test"
NEEDED = {"Guitar", "Piano", "Bass", "Drums"}


def fetch(repo: str, path: str, dest: Path) -> Path:
    """Download one dataset file to dest (skips if it is already there)."""
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(repo, path, repo_type="dataset"), dest)
    return dest


def wanted_stems(meta: dict) -> dict[str, dict]:
    """Stems that were rendered into the mix and have MIDI."""
    return {k: s for k, s in meta["stems"].items() if s.get("audio_rendered") and s.get("midi_saved")}


def fetch_track(name: str, out: Path) -> bool:
    """Download metadata; if the band matches, also MIDI files and mix. Returns kept or not."""
    track = out / name
    meta_file = fetch(MIDI_REPO, f"{SPLIT}/{name}/metadata.yaml", track / "metadata.yaml")
    stems = wanted_stems(yaml.safe_load(meta_file.read_text()))
    if not NEEDED <= {s["inst_class"] for s in stems.values()}:
        shutil.rmtree(track)
        return False
    for sid in stems:
        fetch(MIDI_REPO, f"{SPLIT}/{name}/MIDI/{sid}.mid", track / "MIDI" / f"{sid}.mid")
    fetch(MIX_REPO, f"data/{SPLIT}/{name}/mix.flac", track / "mix.flac")
    return True


def main(out: Path, n_tracks: int) -> None:
    """Walk the test split in order until n_tracks matching tracks are downloaded."""
    names = sorted(Path(f.path).name for f in
                   HfApi().list_repo_tree(MIDI_REPO, repo_type="dataset", path_in_repo=SPLIT))
    kept = 0
    for name in names:
        if fetch_track(name, out):
            kept += 1
            print(f"{name}: kept {kept}/{n_tracks}", flush=True)
            if kept >= n_tracks:
                break


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 10)
