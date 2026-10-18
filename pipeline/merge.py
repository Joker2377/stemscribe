"""Stage 3b: turn 02_midi/<variant>/<stem>.json note lists into <stem>.mid and a multitrack song.mid.

Runs in envs/tools.  Usage: merge.py <work_dir> <stem_variant>
Tempo map: one set_tempo per tracked beat, so songs with tempo changes stay aligned.
Note times are mapped seconds -> ticks by piecewise-linear interpolation over the beats,
which is exactly what a player integrates back when it reads those tempo events.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import mido
import numpy as np

TPQ = 960
VELOCITY = 100
DRUM_CHANNEL = 9
DEFAULT_BPM = 120.0

# MuScriptor instrument name -> GM program (0-based).
GM_PROGRAM = {
    "acoustic_piano": 0, "electric_piano": 4, "chromatic_percussion": 11, "organ": 16,
    "acoustic_guitar": 25, "clean_electric_guitar": 27, "distorted_electric_guitar": 30,
    "acoustic_bass": 32, "electric_bass": 33, "violin": 40, "viola": 41, "cello": 42,
    "contrabass": 43, "orchestral_harp": 46, "timpani": 47, "string_ensemble": 48,
    "synth_strings": 50, "voice": 53, "orchestra_hit": 55, "trumpet": 56, "trombone": 57,
    "tuba": 58, "french_horn": 60, "brass_section": 61, "soprano_and_alto_sax": 65,
    "tenor_sax": 66, "baritone_sax": 67, "oboe": 68, "english_horn": 69, "bassoon": 70,
    "clarinet": 71, "flutes": 73, "synth_lead": 80, "synth_pad": 88,
}


def beats_per_bar(beats: list[float], downbeats: list[float]) -> int:
    """Median number of beats between consecutive downbeats (4 if unknown)."""
    if len(downbeats) < 2:
        return 4
    counts = np.diff(np.searchsorted(beats, downbeats))
    counts = counts[counts > 0]
    return int(np.median(counts)) if len(counts) else 4


def build_knots(beats: list[float], downbeats: list[float], bpb: int) -> tuple[np.ndarray, np.ndarray]:
    """Return matching (seconds, ticks) knots: t=0 -> tick 0, then beat i -> (k+i)*TPQ.

    k (beats of pickup before the first tracked beat) is chosen near the real gap at the
    first beat's tempo, then nudged so the first downbeat lands exactly on a bar line.
    """
    beats = [b for b in beats if b > 1e-3]
    if len(beats) < 2:
        return np.array([0.0, 60 / DEFAULT_BPM]), np.array([0, TPQ])
    k = max(1, round(beats[0] / (beats[1] - beats[0])))
    if downbeats:
        d = int(np.argmin(np.abs(np.array(beats) - downbeats[0])))
        k += (-(k + d)) % bpb
    times = np.array([0.0] + beats)
    ticks = np.array([0] + [(k + i) * TPQ for i in range(len(beats))])
    return times, ticks


def sec_to_tick(t: float, times: np.ndarray, ticks: np.ndarray) -> int:
    """Map seconds to ticks; past the last knot, continue at the last segment's tempo."""
    if t <= times[-1]:
        return int(round(np.interp(t, times, ticks)))
    slope = (ticks[-1] - ticks[-2]) / (times[-1] - times[-2])
    return int(round(ticks[-1] + (t - times[-1]) * slope))


def tempo_track(times: np.ndarray, ticks: np.ndarray, bpb: int) -> mido.MidiTrack:
    """Track 0: time signature plus one set_tempo at every knot."""
    events = [(0, mido.MetaMessage("time_signature", numerator=bpb, denominator=4))]
    for i in range(len(times) - 1):
        beats = (ticks[i + 1] - ticks[i]) / TPQ
        us = int(round((times[i + 1] - times[i]) / beats * 1e6))
        events.append((int(ticks[i]), mido.MetaMessage("set_tempo", tempo=us)))
    return to_track(events, "tempo")


def to_track(events: list[tuple[int, mido.Message]], name: str) -> mido.MidiTrack:
    """Convert (absolute_tick, message) pairs into a delta-timed MidiTrack."""
    track = mido.MidiTrack([mido.MetaMessage("track_name", name=name)])
    last = 0
    for tick, msg in events:
        track.append(msg.copy(time=tick - last))
        last = tick
    return track


def note_track(name: str, notes: list[dict], channel: int, program: int,
               times: np.ndarray, ticks: np.ndarray) -> mido.MidiTrack:
    """Build one instrument track; note-offs sort before note-ons on the same tick."""
    events = [(0, 0, mido.Message("program_change", channel=channel, program=program))]
    for n in notes:
        on = sec_to_tick(n["start"], times, ticks)
        off = max(on + 1, sec_to_tick(n["end"], times, ticks))
        events.append((on, 1, mido.Message("note_on", channel=channel, note=n["pitch"], velocity=VELOCITY)))
        events.append((off, 0, mido.Message("note_off", channel=channel, note=n["pitch"], velocity=0)))
    events.sort(key=lambda e: (e[0], e[1]))
    return to_track([(t, m) for t, _, m in events], name)


def drop_zero_length(notes: list[dict]) -> list[dict]:
    """Remove 0-length notes: MuScriptor sometimes repeats one pitch token many times on the same
    tick (demo piano: 2387 of 4284 notes), each repeat closing the previous note instantly."""
    return [n for n in notes if n["end"] - n["start"] > 1e-3]


def main_instrument(notes: list[dict]) -> str:
    """Most frequent instrument label among a stem's notes."""
    return Counter(n["instrument"] for n in notes).most_common(1)[0][0]


def group_tracks(stem_notes: dict[str, list[dict]]) -> list[tuple[str, list[dict], int, int]]:
    """One (name, notes, channel, program) per stem; program comes from its main instrument."""
    melodic = iter(c for c in range(16) if c != DRUM_CHANNEL)
    out = []
    for stem, notes in stem_notes.items():
        if not notes:
            continue
        inst = main_instrument(notes)
        if inst == "drums":
            out.append((stem, notes, DRUM_CHANNEL, 0))
        else:
            out.append((stem, notes, next(melodic), GM_PROGRAM.get(inst, 0)))
    return out


def write_midi(path: Path, stem_notes: dict[str, list[dict]],
               beats: list[float], downbeats: list[float]) -> None:
    """Write a type-1 MIDI with the beat-derived tempo map and one track per instrument."""
    bpb = beats_per_bar(beats, downbeats)
    times, ticks = build_knots(beats, downbeats, bpb)
    mid = mido.MidiFile(type=1, ticks_per_beat=TPQ)
    mid.tracks.append(tempo_track(times, ticks, bpb))
    for name, notes, channel, program in group_tracks(stem_notes):
        mid.tracks.append(note_track(name, notes, channel, program, times, ticks))
    mid.save(path)


def main(work: Path, variant: str) -> None:
    """Write <stem>.mid for each transcribed stem of 02_midi/<variant>/ and the combined song.mid."""
    analysis = json.loads((work / "00_prep" / "analysis.json").read_text())
    beats, downbeats = analysis["beats"], analysis["downbeats"]
    midi_dir = work / "02_midi" / variant
    stem_notes = {p.stem: drop_zero_length(json.loads(p.read_text())["notes"])
                  for p in sorted(midi_dir.glob("*.json"))}
    for stem, notes in stem_notes.items():
        write_midi(midi_dir / f"{stem}.mid", {stem: notes}, beats, downbeats)
    write_midi(midi_dir / "song.mid", stem_notes, beats, downbeats)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
