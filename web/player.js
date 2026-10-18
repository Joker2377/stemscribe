// Transport: schedules MIDI notes and the original audio on one AudioContext clock,
// so what you hear and the playhead in the roll come from the same time source.
import { playNote } from "./synth.js";

// Schedule notes this far ahead. Long on purpose: a main-thread stall (redraw, a 40-note pile-up, a background
// tab's 1 s timer throttling) shorter than this cannot make notes late. Mix changes act on live gains and
// seek/pause drop the whole bus, so nothing waits on the lookahead.
const LOOKAHEAD_S = 1.5;
const TICK_MS = 25;        // scheduler interval

export class Player {
  // tracks: [{notes (sorted by start), kind, volume, mute, solo}]; audio: decoded original song or null.
  constructor(ctx, tracks, audio) {
    this.ctx = ctx;
    this.tracks = tracks;
    this.audio = audio;
    this.volume = 1;      // overall volume
    this.balance = 0.5;   // 0 = only the original song, 1 = only MIDI
    this.offset = 0;       // song time when playback (re)started
    this.started = 0;      // ctx time at that moment
    this.playing = false;
    this.bus = null;
  }

  // Current song time in seconds, corrected for output latency so it matches what is heard.
  time() {
    if (!this.playing) return this.offset;
    const lat = this.ctx.outputLatency || this.ctx.baseLatency || 0;
    return Math.max(0, this.offset + this.ctx.currentTime - this.started - lat);
  }

  // Start playing from song time `at`.
  async play(at = this.offset) {
    await this.ctx.resume();
    this.stopBus();
    this.offset = at;
    this.started = this.ctx.currentTime + 0.05;
    this.playing = true;
    this.makeBus();
    for (const t of this.tracks) t.next = firstAtOrAfter(t.notes, at);
    if (this.audio) {
      const src = this.ctx.createBufferSource();
      src.buffer = this.audio;
      src.connect(this.audioGain);
      src.start(this.started, at);
    }
    this.schedule();
    this.timer = setInterval(() => this.schedule(), TICK_MS);
  }

  // Pause, remembering the position.
  pause() {
    if (!this.playing) return;
    this.offset = this.time();
    this.playing = false;
    this.stopBus();
  }

  // Jump to song time `at`, keeping the play/pause state.
  seek(at) {
    if (this.playing) this.play(at);
    else this.offset = at;
  }

  // Fresh output graph for this playback run: dropping it later silences everything already scheduled.
  makeBus() {
    this.bus = this.ctx.createGain();
    this.bus.connect(this.ctx.destination);
    this.audioGain = this.ctx.createGain();
    this.audioGain.connect(this.bus);
    this.midiGain = this.ctx.createGain();
    this.midiGain.connect(this.bus);
    for (const t of this.tracks) {
      t.gain = this.ctx.createGain();
      t.gain.connect(this.midiGain);
    }
    this.applyMix();
  }

  // Disconnect the current output graph and stop the scheduler.
  stopBus() {
    clearInterval(this.timer);
    if (this.bus) this.bus.disconnect();
    this.bus = null;
  }

  // Push volume / balance / mute / solo settings into the live gains.
  applyMix() {
    if (!this.bus) return;
    const anySolo = this.tracks.some((t) => t.solo);
    for (const t of this.tracks) t.gain.gain.value = audible(t, anySolo) ? t.volume : 0;
    // balance: both at full in the middle, the other side fades out towards either end
    this.audioGain.gain.value = Math.min(1, 2 * (1 - this.balance));
    this.midiGain.gain.value = Math.min(1, 2 * this.balance);
    this.bus.gain.value = this.volume;
  }

  // Schedule every note that starts before now + lookahead.
  schedule() {
    const horizon = this.offset + this.ctx.currentTime - this.started + LOOKAHEAD_S;
    for (const t of this.tracks) {
      while (t.next < t.notes.length && t.notes[t.next].start < horizon) {
        const n = t.notes[t.next++];
        playNote(this.ctx, t.gain, t.kind, n, this.started + n.start - this.offset, n.end - n.start);
      }
    }
  }
}

// Whether a track should sound given the solo state.
export function audible(t, anySolo) {
  return anySolo ? t.solo : !t.mute;
}

// Index of the first note starting at or after `time` (binary search).
export function firstAtOrAfter(notes, time) {
  let lo = 0, hi = notes.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (notes[mid].start < time) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
