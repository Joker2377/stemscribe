"""Stage 3a: transcribe each stem with MuScriptor into 02_midi/<variant>/<stem>.json note lists.

Runs in envs/muscriptor.  Usage: transcribe.py <work_dir> <stem_variant> [model_size]
Note format: {"pitch", "start", "end", "instrument"} with times in seconds.
MIDI writing (tempo map) lives in merge.py so it is shared by every model.
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from muscriptor.events import NoteEndEvent
from muscriptor.transcription_model import TranscriptionModel

# Hard instrument constraint per stem; None = let the model decide.
STEM_INSTRUMENTS = {
    "vocals": ["voice"],
    "drums": ["drums"],
    "bass": ["electric_bass"],
    "piano": ["acoustic_piano", "electric_piano"],
    "guitar": ["acoustic_guitar", "clean_electric_guitar", "distorted_electric_guitar"],
    "other": None,
}
SILENCE_DBFS = -50.0


def is_silent(wav: Path) -> bool:
    """True if the stem's RMS is below SILENCE_DBFS (avoids hallucinated notes on empty stems)."""
    audio, _ = sf.read(wav, dtype="float32")
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return 20 * np.log10(max(rms, 1e-12)) < SILENCE_DBFS


def transcribe_stem(model: TranscriptionModel, wav: Path, instruments: list[str] | None) -> list[dict]:
    """Collect finished notes from MuScriptor's event stream."""
    # ponytail: raw stream has a constant lag up to ~25 ms; transcribe_to_midi corrects it via
    # its own beat grid. Fix here if the alignment test song shows the offset.
    notes = []
    for ev in model.transcribe(str(wav), instruments=instruments):
        if isinstance(ev, NoteEndEvent):
            s = ev.start_event
            notes.append({"pitch": s.pitch, "start": s.start_time,
                          "end": ev.end_time, "instrument": s.instrument})
    return sorted(notes, key=lambda n: (n["start"], n["pitch"]))


def main(work: Path, variant: str, size: str) -> None:
    """Transcribe every non-silent stem of 01_stems/<variant>/ into 02_midi/<variant>/."""
    stems, out = work / "01_stems" / variant, work / "02_midi" / variant
    out.mkdir(parents=True, exist_ok=True)
    model = TranscriptionModel.load_model(size)
    summary = {"model": f"muscriptor-{size}", "notes": {}}
    for stem, instruments in STEM_INSTRUMENTS.items():
        wav = stems / f"{stem}.wav"
        if not wav.exists() or is_silent(wav):
            print(f"skip {stem}", flush=True)
            continue
        done = out / f"{stem}.json"
        if done.exists():  # per-stem cache; delete the json to redo one stem
            notes = json.loads(done.read_text())["notes"]
        else:
            notes = transcribe_stem(model, wav, instruments)
            done.write_text(json.dumps(
                {"model": summary["model"], "instruments": instruments, "notes": notes}))
        summary["notes"][stem] = len(notes)
        print(f"{stem}: {len(notes)} notes", flush=True)
    (work / "02_midi" / f"{variant}.done.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "medium")
