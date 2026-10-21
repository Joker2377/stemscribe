"""Web UI backend: upload a song, run the pipeline in one background worker, serve results.

Runs with the same plain Python as run.py (stdlib only).  Usage: python server.py [port]
Open http://localhost:8000 .  Uploads go to work/uploads/; progress = which stage outputs exist.
The same queue is driven by mcp_server.py over this HTTP API.
"""
import contextlib
import json
import mimetypes
import os
import queue
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import run

ROOT = Path(__file__).parent
WEB = ROOT / "web"
UPLOADS = run.WORK / "uploads"
SEP = "bs_roformer_sw"
LOG_LINES = 30

# song_id -> {"status": queued|running|failed, "stage": str, "log": [str], "proc": Popen while running, "cancel": bool}
# insertion order = queue order (re-queued songs are popped and re-added at the end)
jobs: dict[str, dict] = {}
job_queue: queue.Queue = queue.Queue()


def worker() -> None:
    """Run queued songs through run.py one at a time (one GPU job at a time), tracking the current stage."""
    while True:
        sid, path = job_queue.get()
        job = jobs.get(sid)
        if job is None or job["status"] != "queued":  # cancelled while waiting, or a stale duplicate entry
            continue
        job["status"] = "running"
        proc = job["proc"] = subprocess.Popen([sys.executable, str(ROOT / "run.py"), str(path)], cwd=ROOT,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              text=True, encoding="utf-8", errors="replace", start_new_session=True,
                                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        for line in proc.stdout:
            job["log"] = (job["log"] + [line.rstrip()])[-LOG_LINES:]
            if line.startswith("[") and line.rstrip().endswith("running"):
                job["stage"] = line[1:line.index("]")]
        if proc.wait() == 0 or job.get("cancel"):
            del jobs[sid]  # finished (or cancelled) songs are described by their files alone
        else:
            job["status"] = "failed"


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill run.py and the stage script it started (that child is what holds the GPU)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        with contextlib.suppress(ProcessLookupError):  # it may have exited on its own just now
            os.killpg(proc.pid, signal.SIGTERM)  # worker starts run.py in its own session = process group


def cancel_job(sid: str) -> str:
    """Drop a queued song or kill a running one; returns what happened (not_queued / removed / killed)."""
    job = jobs.get(sid)
    if job is None or job["status"] not in ("queued", "running"):
        return "not_queued"
    if job["status"] == "queued":
        del jobs[sid]  # the worker skips its queue entry
        return "removed"
    job["cancel"] = True  # worker deletes the job once the process exits
    # ponytail: no lock; in the microseconds between "running" and Popen there is no proc yet and the kill is missed
    if proc := job.get("proc"):
        kill_tree(proc)
    return "killed"


def queue_position(sid: str) -> int | None:
    """1-based place among waiting songs (the running one not counted); None if not waiting."""
    waiting = [k for k, j in jobs.items() if j["status"] == "queued"]
    return waiting.index(sid) + 1 if sid in waiting else None


def song_info(work: Path) -> dict:
    """Name, status, finished stages and track list of one work/<id>/ folder."""
    sid = work.name
    manifest = json.loads((work / "manifest.json").read_text())
    src = Path(manifest["input"]["path"])
    done = [stage for stage, *_, marker in run.stages(src, work, SEP) if (work / marker).exists()]
    midi_dir = work / "02_midi" / run.final_variant(SEP)
    job = jobs.get(sid)
    finished = (midi_dir / "song.mid").exists()
    return {
        "id": sid,
        "name": f"{src.parent.name}/{src.name}" if src.stem == "mix" else src.name,
        "status": job["status"] if job else ("done" if finished else "incomplete"),
        "stage": job["stage"] if job else "",
        "position": queue_position(sid),
        "log": job["log"] if job else [],
        "stages": [s[0] for s in run.stages(src, work, SEP)],
        "done": done,
        "variant": run.final_variant(SEP),
        "tracks": sorted(p.stem for p in midi_dir.glob("*.json")) if finished else [],
    }


def is_eval_song(work: Path) -> bool:
    """Whether a work folder is a Slakh evaluation cache (input under dataset/), not a user's song."""
    src = Path(json.loads((work / "manifest.json").read_text())["input"]["path"])
    return src.is_relative_to(ROOT / "dataset")


def list_songs() -> list[dict]:
    """Every user song folder under work/ (with a manifest, not an eval cache), newest first."""
    dirs = [d for d in run.WORK.iterdir() if (d / "manifest.json").exists() and not is_eval_song(d)]
    dirs.sort(key=lambda d: (d / "manifest.json").stat().st_mtime, reverse=True)
    return [song_info(d) for d in dirs]


def enqueue_upload(name: str, data: bytes) -> str:
    """Save an uploaded file, queue it unless it is already queued/running; return its song id."""
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / Path(name).name  # basename only: no path tricks from the client
    path.write_bytes(data)
    sid = run.song_id(path)
    if jobs.get(sid, {}).get("status") not in ("queued", "running"):
        jobs.pop(sid, None)  # a failed song re-queues at the end
        jobs[sid] = {"status": "queued", "stage": "", "log": []}
        (run.WORK / sid).mkdir(parents=True, exist_ok=True)
        run.update_manifest(run.WORK / sid, "input", {"path": str(path.resolve()), "song_id": sid})
        job_queue.put((sid, path))
    return sid


def safe_path(base: Path, rel: str) -> Path | None:
    """base/rel if it stays inside base and is a file, else None."""
    p = (base / unquote(rel)).resolve()
    return p if p.is_relative_to(base.resolve()) and p.is_file() else None


class Handler(BaseHTTPRequestHandler):
    """Routes: / and /web/* static, /api/songs, PUT /api/upload?name=, DELETE /api/jobs/<id>, /files/<id>/<path>."""

    def do_GET(self) -> None:
        """Serve the UI, the song list, or a file from work/."""
        path = urlparse(self.path).path
        if path == "/":
            self.send_file(WEB / "index.html")
        elif path.startswith("/web/"):
            self.send_file(safe_path(WEB, path[5:]))
        elif path == "/api/songs":
            self.send_json(list_songs())
        elif path.startswith("/files/"):
            self.send_file(safe_path(run.WORK, path[7:]))
        else:
            self.send_error(404)

    def do_PUT(self) -> None:
        """Accept a raw file body as an upload (no multipart parsing needed)."""
        url = urlparse(self.path)
        name = parse_qs(url.query).get("name", [""])[0]
        if url.path != "/api/upload" or not name:
            return self.send_error(400)
        data = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_json({"id": enqueue_upload(name, data)})

    def do_DELETE(self) -> None:
        """Cancel a queued or running song (DELETE, not POST: a cross-site page can't send it without CORS)."""
        path = urlparse(self.path).path
        if not path.startswith("/api/jobs/"):
            return self.send_error(404)
        self.send_json({"result": cancel_job(path[len("/api/jobs/"):])})

    def send_json(self, obj) -> None:
        """Write obj as a JSON response."""
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path | None) -> None:
        """Stream a file (404 when missing or outside the allowed folder)."""
        if path is None or not path.is_file():
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                self.wfile.write(chunk)

    def log_message(self, *args) -> None:
        """Silence per-request logging (the UI polls every 2 s)."""


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    threading.Thread(target=worker, daemon=True).start()
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
