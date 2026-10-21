"""MCP server: let an agent (e.g. Claude Code) drive the pipeline queue and read the transcribed notes as text.

Runs in the tools venv (needs `mcp`).  Registered in .mcp.json; stdio transport.
Queue tools (transcribe, list_songs, song_status, cancel) go through server.py's HTTP API, so the agent and the
web UI share one queue; server.py is started in the background if it isn't running.  summary / get_notes read the per-track JSON next to song.mid.
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlparse

import run
import server
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAX_NOTES = 500  # cap per get_notes call so one request can't flood the agent's context
# GM percussion names for the pitches the drum transcriber emits; others fall back to note_name
GM_DRUMS = {35: "kick", 36: "kick", 37: "side_stick", 38: "snare", 39: "clap", 40: "snare", 41: "tom_low", 42: "hihat_closed",
            43: "tom_low", 44: "hihat_pedal", 45: "tom_mid", 46: "hihat_open", 47: "tom_mid", 48: "tom_high", 49: "crash",
            50: "tom_high", 51: "ride", 52: "china", 53: "ride_bell", 55: "splash", 57: "crash", 59: "ride"}
SERVER_URL = os.environ.get("STEMSCRIBE_URL", "http://127.0.0.1:8000")

mcp = MCPServer("stemscribe")


def note_name(pitch: int) -> str:
    """MIDI pitch -> scientific name, e.g. 60 -> C4."""
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def filter_notes(notes: list[dict], start_s: float, end_s: float | None, limit: int, drums: bool = False) -> tuple[list[dict], int]:
    """Notes sounding inside [start_s, end_s) as compact rows, sorted by start; returns (first `limit` rows, total matches).
    With drums=True the name is the GM drum (kick, snare, ...) instead of a pitch name."""
    name = (lambda p: GM_DRUMS.get(p, note_name(p))) if drums else note_name
    hits = [n for n in notes if n["end"] > start_s and (end_s is None or n["start"] < end_s)]
    hits.sort(key=lambda n: (n["start"], n["pitch"]))
    rows = [{"note": name(n["pitch"]), "pitch": n["pitch"], "start": round(n["start"], 3),
             "dur": round(n["end"] - n["start"], 3)} for n in hits[:limit]]
    return rows, len(hits)


def track_summary(notes: list[dict]) -> dict:
    """Note count, pitch range and first/last onset of one track."""
    if not notes:
        return {"notes": 0}
    lo, hi = min(n["pitch"] for n in notes), max(n["pitch"] for n in notes)
    return {"notes": len(notes), "range": f"{note_name(lo)}-{note_name(hi)}",
            "first_s": round(min(n["start"] for n in notes), 2), "last_s": round(max(n["end"] for n in notes), 2)}


def bpm_from_beats(beats: list[float]) -> float | None:
    """Tempo from the median beat interval (robust to a few missed beats); None if too few beats."""
    if len(beats) < 2:
        return None
    return round(60 / statistics.median(b - a for a, b in zip(beats, beats[1:], strict=False)), 1)


def compact_status(info: dict) -> dict:
    """One song from server.py's /api/songs, trimmed for an agent: where it is in the queue and which stage it's on."""
    out = {"id": info["id"], "name": info["name"], "status": info["status"],
           "progress": f"{len(info['done'])}/{len(info['stages'])} stages"}
    if info["status"] == "running":
        out["stage"] = info["stage"]
    if info.get("position"):
        out["queue_position"] = info["position"]
    if info["status"] == "failed":
        out["log_tail"] = info["log"][-5:]
    return out


def check_song_id(sid: str) -> str:
    """Return sid if it looks like a song id (12 hex chars) and its work folder exists; else raise ToolError.
    Also keeps a caller-supplied id from walking out of work/ (it is joined into a path)."""
    if not re.fullmatch(r"[0-9a-f]{12}", sid) or not (run.WORK / sid / "manifest.json").exists():
        raise ToolError(f"unknown song id {sid!r}; call list_songs")
    return sid


def midi_dir(sid: str) -> Path:
    """Folder holding the finished song.mid and per-track note JSON of one song."""
    return run.WORK / sid / "02_midi" / run.final_variant(server.SEP)


def load_track(sid: str, track: str) -> list[dict]:
    """Note list of one track; raises ToolError naming the valid tracks when it doesn't exist."""
    tracks = {p.stem: p for p in midi_dir(sid).glob("*.json")}
    if track not in tracks:
        raise ToolError(f"no track {track!r} for song {sid}; tracks: {sorted(tracks)}")
    return json.loads(tracks[track].read_text())["notes"]


def request(method: str, path: str, data: bytes | None = None):
    """One HTTP call to server.py, returning its JSON (raises URLError when nothing listens)."""
    req = urllib.request.Request(SERVER_URL + path, data=data, method=method)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def start_server() -> None:
    """Launch server.py on SERVER_URL's port, detached so it outlives this MCP process (songs may still be running).
    Its output goes to work/server.log; it must never inherit our stdio, which is the MCP channel."""
    port = str(urlparse(SERVER_URL).port or 8000)
    log = run.WORK / "server.log"
    if os.name == "nt":
        # MCP clients may put us in a kill-on-close Job Object that forbids breakaway, so a plain detached Popen still
        # dies with us. A process created through WMI belongs to the WMI service, outside our job.
        cmdline = f'cmd /c ""{sys.executable}" "{run.ROOT / "server.py"}" {port} >> "{log}" 2>&1"'
        ps = "(Invoke-CimMethod Win32_Process Create -Arguments @{CommandLine=$env:SS_CMD; CurrentDirectory=$env:SS_CWD}).ReturnValue"
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], stdin=subprocess.DEVNULL, capture_output=True, check=True,
                       env={**os.environ, "SS_CMD": cmdline, "SS_CWD": str(run.ROOT)})
    else:
        with open(log, "a", encoding="utf-8") as f:  # the child keeps its own handle after we close ours
            subprocess.Popen([sys.executable, str(run.ROOT / "server.py"), port], cwd=run.ROOT, stdin=subprocess.DEVNULL,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True)


def api(method: str, path: str, data: bytes | None = None):
    """Call server.py and return its JSON; if it isn't running (and is local), start it and retry for up to 15 s."""
    try:
        return request(method, path, data)
    except urllib.error.URLError as e:
        if urlparse(SERVER_URL).hostname not in ("127.0.0.1", "localhost"):
            raise ToolError(f"server.py not reachable at {SERVER_URL} ({e})") from e
    start_server()
    for _ in range(30):
        time.sleep(0.5)
        try:
            return request(method, path, data)
        except urllib.error.URLError:
            continue
    raise ToolError(f"started server.py but {SERVER_URL} still doesn't answer; see work/server.log")


def all_songs() -> list[dict]:
    """Every song as server.py reports it (includes live queue state)."""
    return api("GET", "/api/songs")


@mcp.tool()
def transcribe(audio_path: str) -> dict:
    """Add a local audio file to the shared queue (same queue as the web UI). Returns its song id and queue state.
    A song takes minutes on GPU; poll song_status."""
    path = Path(audio_path).expanduser().resolve()
    if not path.is_file():
        raise ToolError(f"not a file: {path}")
    sid = api("PUT", f"/api/upload?name={quote(path.name)}", path.read_bytes())["id"]
    return song_status(sid)


@mcp.tool()
def list_songs() -> list[dict]:
    """All songs (newest first) with status queued/running/failed/done/incomplete, current stage and queue position."""
    return [compact_status(s) for s in all_songs()]


@mcp.tool()
def song_status(song_id: str) -> dict:
    """One song in detail: status, stage running now, finished stages, queue position, last log lines, tracks when done."""
    info = next((s for s in all_songs() if s["id"] == song_id), None)
    if info is None:
        raise ToolError(f"unknown song id {song_id!r}; call list_songs")
    return {**compact_status(info), "stages": info["stages"], "done": info["done"], "log": info["log"][-10:],
            "tracks": info["tracks"]}


@mcp.tool()
def cancel(song_id: str) -> dict:
    """Remove a waiting song from the queue, or kill it if it is running (finished stages stay cached)."""
    return api("DELETE", f"/api/jobs/{quote(song_id)}")


@mcp.tool()
def summary(song_id: str) -> dict:
    """Overview of a finished song: duration, tempo, song.mid path and per-track note count / range."""
    work = run.WORK / check_song_id(song_id)
    analysis = json.loads((work / "00_prep" / "analysis.json").read_text())
    tracks = {p.stem: track_summary(json.loads(p.read_text())["notes"]) for p in sorted(midi_dir(song_id).glob("*.json"))}
    return {"duration_s": round(analysis["duration_s"], 1), "bpm": bpm_from_beats(analysis.get("beats", [])),
            "midi": str(midi_dir(song_id) / "song.mid"), "tracks": tracks}


@mcp.tool()
def get_notes(song_id: str, track: str, start_s: float = 0.0, end_s: float | None = None, limit: int = 200) -> dict:
    """Notes of one track in a time window (seconds), as name/pitch/start/dur rows. Drum rows are named kick/snare/hihat_closed/...
    (GM drum pitches). Max 500 rows."""
    rows, total = filter_notes(load_track(check_song_id(song_id), track), start_s, end_s, min(limit, MAX_NOTES), drums=track == "drums")
    return {"total": total, "returned": len(rows), "notes": rows}


if __name__ == "__main__":
    mcp.run()
