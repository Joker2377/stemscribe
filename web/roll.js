// Multitrack piano roll on one canvas: a ruler, then one lane per track, each with its own pitch range.
import { firstAtOrAfter } from "./player.js";

export const RULER_H = 24;
export const LANE_H = 96;
export const TIME_H = 20;  // mm:ss scale under the lanes
const TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300];  // seconds between time labels

export class Roll {
  // tracks: [{name, notes, color, dim}]; beats/downbeats in seconds; onSeek(time) on click.
  constructor(canvas, tracks, beats, downbeats, duration, onSeek) {
    this.canvas = canvas;
    this.tracks = tracks;
    this.beats = beats;
    this.downbeats = downbeats;
    this.duration = duration;
    this.start = 0;        // left edge of the view, seconds
    this.pxPerSec = 80;
    for (const t of tracks) prepareLane(t);
    canvas.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
    canvas.addEventListener("mousedown", (e) => onSeek(this.timeAt(e.offsetX)));
  }

  // Song time under canvas x.
  timeAt(x) {
    return Math.max(0, this.start + x / this.pxPerSec);
  }

  // Wheel scrolls in time; Ctrl+wheel zooms around the mouse.
  onWheel(e) {
    e.preventDefault();
    if (e.ctrlKey) {
      const t = this.timeAt(e.offsetX);
      this.pxPerSec = Math.min(1000, Math.max(5, this.pxPerSec * Math.exp(-e.deltaY / 300)));
      this.start = Math.max(0, t - e.offsetX / this.pxPerSec);
    } else {
      this.start = Math.max(0, this.start + (e.deltaX || e.deltaY) / this.pxPerSec);
    }
    this.userScrolled = performance.now();
  }

  // Keep the playhead on screen while playing (unless the user scrolled in the last 2 s).
  follow(time) {
    const w = this.canvas.clientWidth / this.pxPerSec;
    if (performance.now() - (this.userScrolled || 0) < 2000) return;
    if (time < this.start || time > this.start + w * 0.85) this.start = Math.max(0, time - w * 0.1);
  }

  // Redraw everything at playhead `time`.
  draw(time) {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const lanes = this.tracks;
    const w = c.clientWidth, h = RULER_H + lanes.length * LANE_H + TIME_H;
    if (c.width !== w * dpr || c.height !== h * dpr) {
      c.width = w * dpr;
      c.height = h * dpr;
      c.style.height = h + "px";
    }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.fillStyle = "#16181d";
    g.fillRect(0, 0, w, h);
    const end = this.start + w / this.pxPerSec;
    this.drawGrid(g, w, h, end);
    lanes.forEach((t, i) => this.drawLane(g, t, RULER_H + i * LANE_H, w, end, time));
    this.drawTimeScale(g, h - TIME_H, w, end);
    const x = (time - this.start) * this.pxPerSec;
    g.fillStyle = "#fff";
    g.fillRect(x - 0.5, 0, 1.5, h);
  }

  // Beat lines, bar lines with bar numbers in the ruler, lane separators.
  drawGrid(g, w, h, end) {
    const bars = new Set(this.downbeats);
    const showBeats = this.pxPerSec > 25;
    g.font = "11px sans-serif";
    for (const b of this.beats) {
      if (b < this.start || b > end) continue;
      const isBar = bars.has(b);
      if (!isBar && !showBeats) continue;
      const x = (b - this.start) * this.pxPerSec;
      g.fillStyle = isBar ? "#3a3f4b" : "#23262d";
      g.fillRect(x, RULER_H, 1, h - RULER_H);
      if (isBar) {
        g.fillStyle = "#8a90a0";
        g.fillText(String(this.downbeats.indexOf(b) + 1), x + 3, 15);
      }
    }
    g.fillStyle = "#2c3038";
    for (let y = RULER_H; y <= h - TIME_H; y += LANE_H) g.fillRect(0, y, w, 1);
  }

  // mm:ss labels along the bottom, spaced at least ~70 px apart for the current zoom.
  drawTimeScale(g, top, w, end) {
    const step = TIME_STEPS.find((s) => s * this.pxPerSec >= 70) || TIME_STEPS.at(-1);
    g.fillStyle = "#0f1115";
    g.fillRect(0, top, w, TIME_H);
    g.font = "11px sans-serif";
    for (let t = Math.ceil(this.start / step) * step; t <= end; t += step) {
      const x = (t - this.start) * this.pxPerSec;
      g.fillStyle = "#3a3f4b";
      g.fillRect(x, top, 1, 5);
      g.fillStyle = "#8a90a0";
      g.fillText(mmss(t), x + 3, top + 14);
    }
  }

  // Notes of one track inside its lane; notes sounding at `time` are drawn brighter.
  drawLane(g, t, top, w, end, time) {
    const rows = t.hi - t.lo + 1;
    const nh = Math.max(1.5, (LANE_H - 8) / rows);
    const first = firstAtOrAfter(t.notes, this.start - t.maxDur);
    for (let i = first; i < t.notes.length; i++) {
      const n = t.notes[i];
      if (n.start > end) break;
      if (n.end < this.start) continue;
      const x = (n.start - this.start) * this.pxPerSec;
      const nw = Math.max(2, (n.end - n.start) * this.pxPerSec);
      const y = top + 4 + (t.hi - n.pitch) * ((LANE_H - 8 - nh) / Math.max(1, rows - 1));
      g.fillStyle = n.start <= time && time < n.end ? "#fff" : t.color;
      g.globalAlpha = t.dim ? 0.3 : 1;
      g.fillRect(x, y, nw, nh);
    }
    g.globalAlpha = 1;
  }
}

// Whole seconds as m:ss.
function mmss(t) {
  return `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}`;
}

// Pitch range and longest note of a track (for lane scaling and the visible-note search).
function prepareLane(t) {
  const pitches = t.notes.map((n) => n.pitch);
  t.lo = pitches.length ? Math.min(...pitches) : 60;
  t.hi = pitches.length ? Math.max(...pitches) : 72;
  t.maxDur = t.notes.reduce((m, n) => Math.max(m, n.end - n.start), 0);
}
