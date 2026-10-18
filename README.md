# music_decompose

Song → multitrack MIDI (vocals, bass, drums, piano, two guitars), with a beat-tracked tempo map that follows tempo changes.
Personal, non-commercial use (MuScriptor and BS-Roformer-SW weights are CC BY-NC).

## Pipeline (`run.py`)

| stage | script | venv | output in `work/<song_id>/` |
| --- | --- | --- | --- |
| prep | `prep.py` | beat_this | `00_prep/master.wav`, `analysis.json` (beats, downbeats, stereo stats) |
| separate | `separate_uvr.py` (BS-Roformer-SW; `separate.py` = htdemucs_6s baseline) | separator / demucs | `01_stems/<sep>/` |
| dereverb | `dereverb.py` (vocals: karaoke split into lead / backing first) | separator | `01_stems/<sep>_dry/` |
| transcribe | `transcribe.py` (MuScriptor medium) | muscriptor | `02_midi/<sep>_dry/<stem>.json` |
| split_guitars | `split_guitars_model.py` (trained high / low splitter, see below; outputs sum to the input) | separator | `01_stems/<sep>_dry_gmodel/guitar_{1,2}.wav` |
| refine | `refine_notes.py` (onset lag, note-tail trim, drop < 40 ms blips, cut notes at stem silence / in > 16-note pile-ups) | tools | `02_midi/<sep>_dry_refined/<stem>.json` |
| assign_guitars | `assign_guitars.py` (per-note energy in the model's two guitars; `smooth` mode also averages neighbours, fewer track hops) | tools | `02_midi/<sep>_dry_refined_modelassign/guitar_{1,2}.json` |
| merge | `merge.py` | tools | `02_midi/<sep>_dry_refined_modelassign/song.mid` |

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

## Web UI

```
python server.py [port]        # http://localhost:8000 (stdlib only, same Python as run.py)
```

Upload a song (queued, one pipeline run at a time); the left list shows each song's stage progress.
A finished song opens as a multitrack piano roll (one lane per track, bar/beat grid from the beat tracker).
MIDI plays in the browser with a small Web Audio synth (`web/synth.js`), scheduled on the same
`AudioContext` clock as the playhead and the original audio (`00_prep/master.wav`), so notes, sound and roll line up.
Overall volume, original-song / MIDI balance, per-track volume / mute / solo; `song.mid` download.

## Evaluation

```
envs/tools/Scripts/python.exe fetch_slakh.py dataset/slakh 10   # Slakh2100 test tracks with guitar+piano+bass+drums
python eval_batch.py                                              # -> work/eval/results.csv + mean table
python eval_batch.py dataset/slakh_flac/validation slakh_val_tracks.txt    # 12 validation tracks (full Slakh, below)
envs/tools/Scripts/python.exe oracle_stems.py <slakh_track>        # transcribe the true stems (no separation)
envs/tools/Scripts/python.exe error_breakdown.py <variant> <slakh_track>...   # what kind of errors
envs/tools/Scripts/python.exe note_continuity.py <variant> <slakh_track>...   # fragmented / glued notes, guitar track hopping
```

`eval_midi.py` scores note F1 with mir_eval (onset, onset+offset; drums onset only), aligning Slakh's
octave-transposed patches. `eval_guitar_split.py` scores the two-guitar split (balanced accuracy, 0.5 = no split).
Self-checks: `test_*.py` (each runs on its own: `python test_x.py` in the venv named in its docstring).

Current scores (onset F1, test 10 / validation 12; "oracle" = transcribing the true stems, validation):

| stem | test | val | oracle |
| --- | --- | --- | --- |
| guitar | 0.577 | 0.578 | 0.780 |
| piano | 0.710 | 0.780 | 0.902 |
| bass | 0.844 | 0.884 | 0.938 |
| drums | 0.860 | 0.820 | 0.865 |
| guitar split (balanced acc) | 0.756 | 0.676 | |

## Two-guitar splitter

A mono Mel-Roformer trained on Slakh2100 guitar stems to split one guitar signal into the higher and the lower part.

```
envs/tools/Scripts/python.exe fetch_slakh_full.py                  # full Slakh (flac) -> dataset/slakh_flac/
envs/separator/Scripts/python.exe train_guitar_split.py 10 [--resume | --init=<ckpt>]   # -> envs/models/guitar_split/
python score_checkpoints.py envs/models/guitar_split               # pick a checkpoint on the real task
```

The pipeline uses run 5 (`envs/models/guitar_split_run5_gap/best.pt`, set in `split_guitars_model.py`).
The current training script adds input augmentation (tone, bleed, limiting); that run (6) did not beat run 5.

Shelved code lives in `archive/` (git-ignored): MoisesDB real-multitrack training (`archive/moisesdb/`) and
fine-tuning MuScriptor to output the two guitars directly (`archive/plan_c_tx/`; did not beat the current flow,
see notes/roadmap.md).
