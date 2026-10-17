"""Unpack the MoisesDB songs with >= 2 guitar tracks from the official zip, as FLAC (half the disk of WAV).

Runs in envs/tools.  Usage: prep_moisesdb.py [zip=dataset/moisesdb.zip] [out=dataset/moisesdb_flac]
Layout: <out>/<song_id>/data.json and <out>/<song_id>/<stemName>/<track_id>.flac (same as the zip).
Non-guitar stems are kept for bleed simulation.  Resumable: finished songs (data.json written last) are skipped.
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import soundfile as sf

MIN_GUITARS = 2


def guitar_count(meta: dict) -> int:
    """Number of guitar track files in a song's data.json."""
    return sum(len(s["tracks"]) for s in meta["stems"] if s["stemName"] == "guitar")


def track_entries(meta: dict, song_dir: str) -> list[tuple[str, str]]:
    """(zip entry, relative output path) for every track file of a song; ids come from data.json, not the zip."""
    return [(f"{song_dir}/{s['stemName']}/{t['id']}.{t['extension']}", f"{s['stemName']}/{t['id']}.flac")
            for s in meta["stems"] for t in s["tracks"]]


def unpack_song(z: zipfile.ZipFile, meta_entry: str, out: Path) -> bool:
    """Convert one song's tracks to FLAC under out/<song_id>/; False if skipped (too few guitars)."""
    meta = json.loads(z.read(meta_entry))
    if guitar_count(meta) < MIN_GUITARS:
        return False
    song_dir = meta_entry.rsplit("/", 1)[0]
    dst = out / Path(song_dir).name
    for entry, rel in track_entries(meta, song_dir):
        path = (dst / rel).resolve()
        assert path.is_relative_to(out.resolve()), rel  # zip content is untrusted
        path.parent.mkdir(parents=True, exist_ok=True)
        data, sr = sf.read(io.BytesIO(z.read(entry)), dtype="float32")
        sf.write(path, data, sr, subtype="PCM_24")
    (dst / "data.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return True


def main(zip_path: Path, out: Path) -> None:
    """Unpack every qualifying song not already done."""
    z = zipfile.ZipFile(zip_path)
    metas = [n for n in z.namelist() if n.endswith("/data.json")]
    done = 0
    for i, m in enumerate(metas):
        if (out / Path(m).parent.name / "data.json").exists():
            done += 1
            continue
        if unpack_song(z, m, out):
            done += 1
            print(f"{i + 1}/{len(metas)} {Path(m).parent.name}", flush=True)
    print(f"{done} songs in {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/moisesdb.zip"),
         Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dataset/moisesdb_flac"))
