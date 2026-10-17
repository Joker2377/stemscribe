"""Transcribe every MoisesDB guitar track on its own (clean, isolated audio -> close to ground truth).

Runs in envs/muscriptor.  Usage: transcribe_moises.py [data=dataset/moisesdb_flac] [model_size=medium]
Writes <song>/guitar/<track_id>.json ({"notes": [...]}, raw MuScriptor times, no lag shift).
The notes stand in for the MIDI MoisesDB lacks: they give each guitar group's mean pitch (output order
for training) and the reference notes for scoring a split.  Resumable.
"""
import json
import sys
from pathlib import Path

from muscriptor.transcription_model import TranscriptionModel

from transcribe import is_silent, transcribe_stem


def guitar_tracks(song: Path) -> list[tuple[Path, str]]:
    """(flac, MoisesDB trackType) of every guitar track of a song; trackType names match MuScriptor's."""
    meta = json.loads((song / "data.json").read_text(encoding="utf-8"))
    return [(song / "guitar" / f"{t['id']}.flac", t["trackType"])
            for s in meta["stems"] if s["stemName"] == "guitar" for t in s["tracks"]]


def main(data: Path, size: str) -> None:
    """Transcribe every guitar track that has no json yet."""
    model = TranscriptionModel.load_model(size)
    todo = [(f, k) for song in sorted(data.iterdir()) if (song / "data.json").exists()
            for f, k in guitar_tracks(song) if not f.with_suffix(".json").exists()]
    for i, (f, kind) in enumerate(todo):
        notes = [] if is_silent(f) else transcribe_stem(model, f, [kind])
        f.with_suffix(".json").write_text(json.dumps({"model": f"muscriptor-{size}", "instrument": kind,
                                                      "notes": notes}))
        print(f"{i + 1}/{len(todo)} {f.parent.parent.name[:8]} {kind}: {len(notes)} notes", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/moisesdb_flac"),
         sys.argv[2] if len(sys.argv) > 2 else "medium")
