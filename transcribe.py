"""Stage 3a: transcribe each stem with MuScriptor into 02_midi/<variant>/<stem>.json note lists.

Runs in envs/muscriptor.  Usage: transcribe.py <work_dir> <stem_variant> [model_size] [--window]
Note format: {"pitch", "start", "end", "instrument"} with times in seconds.
--window: two half-offset passes without tie forcing, stitched from window centres (window_stitch.py),
written to 02_midi/<stem_variant>_win/. MuScriptor's tie forcing carries a wrong held note on chunk after
chunk (demo lead vocal: a stuck A4 through a 30 s silence, then 5 s of A4 where B4/C#5 is sung).
MIDI writing (tempo map) lives in merge.py so it is shared by every model.
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from muscriptor.events import NoteEndEvent
from muscriptor.transcription_model import TranscriptionModel

from window_stitch import WINDOW_S, combine

# Hard instrument constraint per stem; None = let the model decide.
STEM_INSTRUMENTS = {
    "vocals": ["voice"],
    "lead_vocals": ["voice"],
    "backing_vocals": ["voice"],
    "drums": ["drums"],
    "bass": ["electric_bass"],
    "piano": ["acoustic_piano", "electric_piano"],
    "guitar": ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"],
    "guitar_1": ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"],
    "guitar_2": ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"],
    "other": None,
}
SILENCE_DBFS = -50.0
WINDOW_BATCH = 8  # windows decoded together (only possible without tie forcing)


def is_silent(wav: Path) -> bool:
    """True if the stem's RMS is below SILENCE_DBFS (avoids hallucinated notes on empty stems)."""
    audio, _ = sf.read(wav, dtype="float32")
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20 * np.log10(max(rms, 1e-12)) < SILENCE_DBFS


def collect_notes(events, offset: float = 0.0) -> list[dict]:
    """Finished notes from MuScriptor's event stream, shifted by offset seconds."""
    notes = []
    for ev in events:
        if isinstance(ev, NoteEndEvent):
            s = ev.start_event
            notes.append({"pitch": s.pitch, "start": s.start_time + offset,
                          "end": ev.end_time + offset, "instrument": s.instrument})
    return sorted(notes, key=lambda n: (n["start"], n["pitch"]))


def transcribe_stem(model: TranscriptionModel, wav: Path, instruments: list[str] | None) -> list[dict]:
    """One pass over the stem with MuScriptor's default tie forcing."""
    # ponytail: raw stream has a constant lag up to ~25 ms; refine_notes.py takes it out per stem.
    return collect_notes(model.transcribe(str(wav), instruments=instruments))


def transcribe_pass(model: TranscriptionModel, audio: torch.Tensor, sr: int, offset: float,
                    instruments: list[str] | None) -> tuple[list[dict], set[float]]:
    """One untied pass over audio that starts `offset` s into the song: (notes, start times of runaway windows).
    A runaway window is one MuScriptor warns about for not emitting EOS within its token limit."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        notes = collect_notes(model.transcribe((audio, sr), instruments=instruments, prelude_forcing=False,
                                               batch_size=WINDOW_BATCH), offset)
    bad = {offset + float(m.group(1)) for w in caught if (m := re.search(r"seek=([\d.]+)s\) did not emit EOS", str(w.message)))}
    return notes, bad


def transcribe_windowed(model: TranscriptionModel, wav: Path, instruments: list[str] | None) -> list[dict]:
    """Two passes, the second shifted by half a window, no tie forcing; keep each window's centre."""
    audio, sr = sf.read(wav, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.T.copy())
    shift = int(WINDOW_S / 2 * sr)
    notes_a, bad_a = transcribe_pass(model, audio, sr, 0.0, instruments)
    notes_b, bad_b = transcribe_pass(model, audio[:, shift:], sr, shift / sr, instruments)
    if bad_a | bad_b:
        print(f"  runaway windows at {sorted(bad_a | bad_b)} s: centres taken from the other pass", flush=True)
    return combine(notes_a, notes_b, bad_a | bad_b)


def main(work: Path, variant: str, size: str, window: bool = False) -> None:
    """Transcribe every non-silent stem of 01_stems/<variant>/ into 02_midi/<variant>[_win]/."""
    midi_variant = f"{variant}_win" if window else variant
    stems, out = work / "01_stems" / variant, work / "02_midi" / midi_variant
    out.mkdir(parents=True, exist_ok=True)
    model = TranscriptionModel.load_model(size)
    summary = {"model": f"muscriptor-{size}" + ("-window" if window else ""), "notes": {}}
    for stem, instruments in STEM_INSTRUMENTS.items():
        wav = stems / f"{stem}.wav"
        if not wav.exists() or is_silent(wav):
            print(f"skip {stem}", flush=True)
            continue
        done = out / f"{stem}.json"
        if done.exists():  # per-stem cache; delete the json to redo one stem
            notes = json.loads(done.read_text())["notes"]
        else:
            notes = (transcribe_windowed if window else transcribe_stem)(model, wav, instruments)
            done.write_text(json.dumps(
                {"model": summary["model"], "instruments": instruments, "notes": notes}))
        summary["notes"][stem] = len(notes)
        print(f"{stem}: {len(notes)} notes", flush=True)
    (work / "02_midi" / f"{midi_variant}.done.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--window"]
    main(Path(args[0]), args[1], args[2] if len(args) > 2 else "medium", "--window" in sys.argv)
