"""Self-check for the MoisesDB pair sampling on a fake 3-guitar song.  Run in envs/separator:
python test_train_guitar_split_moises.py"""
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from train_guitar_split import SEG, SR
from train_guitar_split_moises import sample_example, song_entry


def fake_song(root: Path) -> Path:
    """Song with guitars at pitch 40 (loud), 43 and 70, a bass track and the SW stem; 20 s of noise each."""
    song = root / "song"
    pitches = {"g_low": 40, "g_mid": 43, "g_high": 70}
    meta = {"stems": [{"stemName": "guitar", "tracks": [{"id": k, "extension": "flac"} for k in pitches]},
                      {"stemName": "bass", "tracks": [{"id": "b", "extension": "flac"}]}]}
    rng = np.random.default_rng(0)
    for stem, ids in (("guitar", list(pitches)), ("bass", ["b"])):
        (song / stem).mkdir(parents=True)
        for i in ids:
            amp = 0.3 if i == "g_low" else 0.05
            sf.write(song / stem / f"{i}.flac", rng.normal(0, amp, (20 * SR, 2)), SR)
            notes = [{"pitch": pitches.get(i, 30), "start": 0.0, "end": 20.0}]
            (song / stem / f"{i}.json").write_text(json.dumps({"notes": notes}))
    sf.write(song / "sw_guitar_dry.flac", rng.normal(0, 0.1, (20 * SR, 2)), SR)
    (song / "data.json").write_text(json.dumps(meta))
    return song


def demo() -> None:
    """Groups cut at the biggest pitch gap (40, 43 | 70); output 1 is the high group; shapes and levels sane."""
    with tempfile.TemporaryDirectory() as tmp:
        e = song_entry(fake_song(Path(tmp)))
        assert [[f.stem for f in g] for g in e["groups"]] == [["g_low", "g_mid"], ["g_high"]]
        assert [f.parent.name for f in e["others"]] == ["bass"]
        rng = random.Random(0)
        for real in (True, False):
            mix, y = sample_example(e, rng, need_both=True, real=real)
            assert mix.shape == (SEG,) and y.shape == (2, SEG) and np.isfinite(mix).all()
            assert np.std(y[0]) < np.std(y[1]), "output 1 must be the (quiet) high group, not the loud low one"
            assert -31 < 20 * np.log10(np.sqrt(np.mean(mix ** 2))) < -11
    print("ok")


if __name__ == "__main__":
    demo()
