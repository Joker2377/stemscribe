from mcp.server.mcpserver.exceptions import ToolError
from mcp_server import bpm_from_beats, check_song_id, compact_status, filter_notes, note_name, track_summary

NOTES = [
    {"pitch": 60, "start": 1.0, "end": 2.0},
    {"pitch": 64, "start": 0.0, "end": 1.5},  # starts before the window but still sounds in it
    {"pitch": 67, "start": 3.0, "end": 4.0},
]


def test_note_name():
    assert note_name(60) == "C4"
    assert note_name(21) == "A0"
    assert note_name(61) == "C#4"


def test_filter_notes_window_sort_and_limit():
    rows, total = filter_notes(NOTES, 0.5, 2.5, 10)
    assert total == 2
    assert [r["note"] for r in rows] == ["E4", "C4"]  # sorted by start
    rows, total = filter_notes(NOTES, 0.0, None, 1)
    assert total == 3 and len(rows) == 1



def test_filter_notes_drum_names():
    hits = [{"pitch": 36, "start": 0.0, "end": 0.01}, {"pitch": 38, "start": 0.5, "end": 0.51}, {"pitch": 80, "start": 1.0, "end": 1.01}]
    rows, _ = filter_notes(hits, 0.0, None, 10, drums=True)
    assert [r["note"] for r in rows] == ["kick", "snare", "G#5"]  # unmapped pitch falls back to its note name


def test_track_summary():
    assert track_summary([]) == {"notes": 0}
    s = track_summary(NOTES)
    assert s["notes"] == 3 and s["range"] == "C4-G4" and s["last_s"] == 4.0


def test_bpm_from_beats():
    assert bpm_from_beats([0.0, 0.5, 1.0, 1.5, 3.0]) == 120.0  # one missed beat doesn't move the median
    assert bpm_from_beats([1.0]) is None


def test_check_song_id_rejects_paths():
    for bad in ["../../etc", "..", "000000000000"]:  # last one: right shape but no such song
        try:
            check_song_id(bad)
        except ToolError:
            continue
        raise AssertionError(bad)


def test_compact_status():
    info = {"id": "a", "name": "x.mp3", "status": "running", "stage": "transcribe", "position": None,
            "stages": ["prep", "separate", "transcribe"], "done": ["prep", "separate"], "log": ["l1"]}
    assert compact_status(info) == {"id": "a", "name": "x.mp3", "status": "running", "progress": "2/3 stages", "stage": "transcribe"}
    queued = compact_status({**info, "status": "queued", "position": 2, "done": []})
    assert queued["queue_position"] == 2 and "stage" not in queued
    assert compact_status({**info, "status": "failed"})["log_tail"] == ["l1"]
