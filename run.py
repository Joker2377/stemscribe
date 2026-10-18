"""Run the whole pipeline on one song, each stage in its own venv.

Usage: python run.py <input_audio> [--sep NAME] [--force stage,stage]
Output: work/<song_id>/ ; a stage is skipped when its output already exists.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
WORK = ROOT / "work"
PIPELINE = "pipeline"  # stage scripts, relative to ROOT

# separator name -> (venv, script, extra args); the name is also the stem/midi subfolder
SEPARATORS = {
    "htdemucs_6s": ("demucs", f"{PIPELINE}/separate.py", []),
    "bs_roformer_sw": ("separator", f"{PIPELINE}/separate_uvr.py", ["BS-Roformer-SW.ckpt", "bs_roformer_sw"]),
}


def final_variant(sep: str) -> str:
    """Folder name of the finished MIDI: separated -> dereverbed -> windowed transcription -> notes refined
    -> guitars split by the model."""
    return f"{sep}_dry_win_refined_modelassign"


def stages(src: Path, work: Path, sep: str) -> list[tuple[str, str, str, list[str], str]]:
    """(stage, venv, script, args, output that marks it done) for one separator variant."""
    venv, script, extra = SEPARATORS[sep]
    w, dry = str(work), f"{sep}_dry"
    midi, refined = f"{dry}_win", f"{dry}_win_refined"
    return [
        ("prep", "beat_this", f"{PIPELINE}/prep.py", [str(src), w], "00_prep/analysis.json"),
        ("separate", venv, script, [w, *extra], f"01_stems/{sep}/stems.json"),
        ("dereverb", "separator", f"{PIPELINE}/dereverb.py", [w, sep], f"01_stems/{dry}/stems.json"),
        ("split_guitars", "separator", f"{PIPELINE}/split_guitars_model.py", [w, dry], f"01_stems/{dry}_gmodel/guitar_2.wav"),
        ("transcribe", "muscriptor", f"{PIPELINE}/transcribe.py", [w, dry, "medium", "--window"], f"02_midi/{midi}.done.json"),
        ("refine", "tools", f"{PIPELINE}/refine_notes.py", [w, dry, midi], f"02_midi/{refined}/guitar.json"),
        ("assign_guitars", "tools", f"{PIPELINE}/assign_guitars.py", [w, dry, refined, "model"], f"02_midi/{final_variant(sep)}/guitar_1.json"),
        ("merge", "tools", f"{PIPELINE}/merge.py", [w, final_variant(sep)], f"02_midi/{final_variant(sep)}/song.mid"),
    ]


def song_id(path: Path) -> str:
    """First 12 hex chars of the file's SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def venv_python(name: str) -> str:
    """Path to the interpreter of envs/<name>."""
    return str(ROOT / "envs" / name / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))


def run_stage(venv: str, script: str, args: list[str]) -> float:
    """Run one script (path relative to ROOT) in its venv; return elapsed seconds. Raises on failure."""
    t0 = time.time()
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    subprocess.run([venv_python(venv), str(ROOT / script), *args], check=True, env=env)
    return time.time() - t0


def update_manifest(work: Path, stage: str, info: dict) -> None:
    """Record per-stage info (timing, args) in work/<id>/manifest.json."""
    path = work / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[stage] = info
    path.write_text(json.dumps(manifest, indent=1))


def main() -> None:
    """Parse args and run each stage whose output is missing (or forced)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--sep", default="bs_roformer_sw", choices=SEPARATORS)
    ap.add_argument("--force", default="", help="comma-separated stages to rerun")
    args = ap.parse_args()

    sid = song_id(args.input)
    work = WORK / sid
    work.mkdir(parents=True, exist_ok=True)
    forced = set(filter(None, args.force.split(",")))
    update_manifest(work, "input", {"path": str(args.input.resolve()), "song_id": sid})

    for stage, venv, script, stage_args, done in stages(args.input, work, args.sep):
        # ponytail: cache = "output exists"; hash params into manifest when experiments need it
        if (work / done).exists() and stage not in forced:
            print(f"[{stage}] cached", flush=True)
            continue
        print(f"[{stage}] running", flush=True)
        secs = run_stage(venv, script, stage_args)
        update_manifest(work, f"{stage}:{args.sep}", {"venv": venv, "script": script, "seconds": round(secs, 1)})
        print(f"[{stage}] done in {secs:.1f}s", flush=True)
    print(work / "02_midi" / final_variant(args.sep) / "song.mid")


if __name__ == "__main__":
    sys.exit(main())
