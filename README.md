# music_decompose

Song → multitrack MIDI (vocals, bass, drums, piano, two guitars), with a beat-tracked tempo map that follows tempo changes.
Personal, non-commercial use (MuScriptor and BS-Roformer-SW weights are CC BY-NC).

## Pipeline (`run.py`)

| stage | script | venv | output in `work/<song_id>/` |
| --- | --- | --- | --- |
| prep | `prep.py` | beat_this | `00_prep/master.wav`, `analysis.json` (beats, downbeats, stereo stats) |
| separate | `separate_uvr.py` (BS-Roformer-SW; `separate.py` = htdemucs_6s baseline) | separator / demucs | `01_stems/<sep>/` |
| dereverb | `dereverb.py` | separator | `01_stems/<sep>_dry/` |
| transcribe | `transcribe.py` (MuScriptor medium) | muscriptor | `02_midi/<sep>_dry/<stem>.json` |
| assign_guitars | `assign_guitars.py` (per-note stereo position) | tools | `02_midi/<sep>_dry_noteassign/guitar_{1,2}.json` |
| merge | `merge.py` | tools | `02_midi/<sep>_dry_noteassign/song.mid` |

Each stage runs in its own venv as a subprocess and is skipped when its output exists (`--force stage,...` to redo).

```
python run.py song.mp3 [--sep bs_roformer_sw|htdemucs_6s] [--force transcribe,merge]
```

## Setup

Python 3.11 venvs under `envs/<name>/` (git-ignored), one per `requirements/<name>.txt`:

```
uv venv envs/tools --python 3.11
uv pip install --python envs/tools/Scripts/python.exe -r requirements/tools.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

Separation / dereverb weights download to `envs/models/` on first use. Set `PYTHONIOENCODING=utf-8` on Windows.

## Evaluation

```
envs/tools/Scripts/python.exe fetch_slakh.py dataset/slakh 10   # Slakh2100 test tracks with guitar+piano+bass+drums
python eval_batch.py                                              # -> work/eval/results.csv + mean table
```

`eval_midi.py` scores note F1 with mir_eval (onset, onset+offset; drums onset only), aligning Slakh's
octave-transposed patches. Self-checks: `test_merge.py` (tempo map), `test_eval_midi.py` (scoring).
