// Small Web Audio synth: one voice recipe per instrument family, GM drum map for drums.
// ponytail: oscillator voices, not samples — swap in a soundfont player if the timbre matters.

// Instrument label (MuScriptor name) -> voice recipe.
export function family(instrument) {
  if (instrument === "drums") return "drums";
  if (/bass|contrabass|tuba/.test(instrument)) return "bass";
  if (/guitar|harp/.test(instrument)) return "pluck";
  if (/piano|percussion|organ/.test(instrument)) return "keys";
  if (/voice|sax|flute|oboe|clarinet|trumpet|trombone|horn|lead/.test(instrument)) return "lead";
  return "pad";
}

const midiHz = (p) => 440 * 2 ** ((p - 69) / 12);

// Voice recipes: oscillator type, lowpass cutoff (Hz), attack / decay times (s), sustain level, release (s).
const VOICES = {
  bass:  { type: "square",   cutoff: 900,  a: 0.005, d: 0.25, s: 0.5,  r: 0.06 },
  pluck: { type: "sawtooth", cutoff: 2500, a: 0.003, d: 0.4,  s: 0.25, r: 0.08 },
  keys:  { type: "triangle", cutoff: 5000, a: 0.003, d: 0.8,  s: 0.2,  r: 0.15 },
  lead:  { type: "sawtooth", cutoff: 1800, a: 0.03,  d: 0.2,  s: 0.7,  r: 0.1  },
  pad:   { type: "sawtooth", cutoff: 1200, a: 0.08,  d: 0.3,  s: 0.7,  r: 0.2  },
};

// Schedule one pitched note on `out` at context time `when` lasting `dur` seconds.
function tone(ctx, out, kind, pitch, when, dur) {
  const v = VOICES[kind];
  const osc = ctx.createOscillator();
  const filt = ctx.createBiquadFilter();
  const amp = ctx.createGain();
  osc.type = v.type;
  osc.frequency.value = midiHz(pitch);
  filt.frequency.value = v.cutoff;
  const peak = 0.25, end = when + Math.max(dur, v.a);
  amp.gain.setValueAtTime(0, when);
  amp.gain.linearRampToValueAtTime(peak, when + v.a);
  amp.gain.setTargetAtTime(peak * v.s, when + v.a, v.d / 3);
  amp.gain.setTargetAtTime(0, end, v.r / 3);
  osc.connect(filt).connect(amp).connect(out);
  osc.start(when);
  osc.stop(end + v.r * 2);
}

let noiseBuf = null;
// One second of white noise, shared by all drum hits.
function noise(ctx) {
  if (!noiseBuf) {
    noiseBuf = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
    const d = noiseBuf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  }
  return noiseBuf;
}

// Filtered noise burst (snare, hats, cymbals).
function hiss(ctx, out, when, len, filterType, freq, gain) {
  const src = ctx.createBufferSource();
  const filt = ctx.createBiquadFilter();
  const amp = ctx.createGain();
  src.buffer = noise(ctx);
  filt.type = filterType;
  filt.frequency.value = freq;
  amp.gain.setValueAtTime(gain, when);
  amp.gain.exponentialRampToValueAtTime(0.001, when + len);
  src.connect(filt).connect(amp).connect(out);
  src.start(when);
  src.stop(when + len);
}

// Sine with a falling pitch (kick, toms).
function thump(ctx, out, when, fromHz, toHz, len, gain) {
  const osc = ctx.createOscillator();
  const amp = ctx.createGain();
  osc.frequency.setValueAtTime(fromHz, when);
  osc.frequency.exponentialRampToValueAtTime(toHz, when + len);
  amp.gain.setValueAtTime(gain, when);
  amp.gain.exponentialRampToValueAtTime(0.001, when + len);
  osc.connect(amp).connect(out);
  osc.start(when);
  osc.stop(when + len);
}

// One GM drum hit (pitch = GM percussion key).
function drum(ctx, out, pitch, when) {
  if (pitch === 35 || pitch === 36) return thump(ctx, out, when, 150, 45, 0.35, 0.9);
  if (pitch === 38 || pitch === 40 || pitch === 37 || pitch === 39) {
    thump(ctx, out, when, 220, 160, 0.1, 0.4);
    return hiss(ctx, out, when, 0.18, "highpass", 1500, 0.5);
  }
  if (pitch === 42 || pitch === 44) return hiss(ctx, out, when, 0.05, "highpass", 7000, 0.3);
  if (pitch === 46) return hiss(ctx, out, when, 0.3, "highpass", 7000, 0.3);
  if ([41, 43, 45, 47, 48, 50].includes(pitch)) return thump(ctx, out, when, midiHz(pitch + 12), midiHz(pitch), 0.3, 0.6);
  if (pitch === 51 || pitch === 59 || pitch === 53) return hiss(ctx, out, when, 0.4, "bandpass", 5000, 0.25);
  return hiss(ctx, out, when, 1.2, "highpass", 4000, 0.3); // crashes, splash, china, everything else
}

// Schedule one note of a track whose voice family is `kind`.
export function playNote(ctx, out, kind, note, when, dur) {
  if (kind === "drums") drum(ctx, out, note.pitch, when);
  else tone(ctx, out, kind, note.pitch, when, dur);
}
