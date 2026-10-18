// Page logic: song list + upload, progress view, result view (roll + MIDI playback).
import { family } from "./synth.js";
import { Player, audible } from "./player.js";
import { Roll } from "./roll.js";

const ORDER = ["lead_vocals", "backing_vocals", "guitar_1", "guitar_2", "piano", "bass", "drums", "other"];
const COLORS = ["#ff7a7a", "#ffb37a", "#ffe07a", "#b5f07a", "#7af0c8", "#7ac8ff", "#a88cff", "#ff8ce0"];
const STATUS = { done: "完成", running: "處理中", queued: "排隊中", failed: "失敗", incomplete: "未完成" };

const $ = (id) => document.getElementById(id);
let songs = [];
let current = null;   // {id, player, roll, tracks}
let selected = null;
let shown = null;     // song whose result view is shown or loading

// Fetch JSON or throw.
async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

// Refresh the song list (and the progress view of the selected song) from the server.
async function poll() {
  songs = await getJSON("/api/songs");
  $("songs").innerHTML = "";
  for (const s of songs) {
    const li = document.createElement("li");
    li.className = s.id === selected ? "sel" : "";
    li.innerHTML = `<span class="st">${STATUS[s.status]}</span>`;
    li.append(s.name);
    li.onclick = () => select(s.id);
    $("songs").append(li);
  }
  const s = songs.find((x) => x.id === selected);
  if (s && s.status !== "done") showProgress(s);
  if (s && s.status === "done" && shown !== s.id) showResult(s);
}

// Open a song: its result if finished, else its progress.
function select(id) {
  selected = id;
  const s = songs.find((x) => x.id === id);
  $("title").textContent = s.name;
  if (s.status === "done") showResult(s);
  else showProgress(s);
  poll();
}

// Stage checklist plus the tail of the log for a song that is not finished.
function showProgress(s) {
  closeResult();
  const items = s.stages.map((st) => {
    const mark = s.done.includes(st) ? "✔" : st === s.stage && s.status === "running" ? "⏳" : "·";
    return `<li>${mark} ${st}</li>`;
  }).join("");
  $("progress").innerHTML = `<b>${STATUS[s.status]}</b><ul>${items}</ul>`
    + (s.log.length ? `<pre>${s.log.join("\n").replace(/</g, "&lt;")}</pre>` : "");
  $("progress").hidden = false;
}

// Tear down the playing result view.
function closeResult() {
  if (current) {
    current.player.pause();
    current.player.ctx.close();
  }
  current = shown = null;
  $("roll").hidden = true;
  $("play").disabled = true;
  $("download").hidden = true;
}

// Most frequent instrument label among a track's notes.
function mainInstrument(notes) {
  const count = {};
  for (const n of notes) count[n.instrument] = (count[n.instrument] || 0) + 1;
  return Object.entries(count).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
}

// Load the finished MIDI tracks (same JSON merge.py turns into song.mid) in display order.
async function loadTracks(s) {
  const base = `/files/${s.id}/02_midi/${s.variant}/`;
  const names = [...s.tracks].sort((a, b) => rank(a) - rank(b));
  const tracks = await Promise.all(names.map(async (name, i) => {
    const notes = (await getJSON(base + name + ".json")).notes
      .filter((n) => n.end - n.start > 1e-3)  // same as merge.drop_zero_length
      .sort((a, b) => a.start - b.start);
    const instrument = mainInstrument(notes);
    return { name, notes, instrument, kind: family(instrument), color: COLORS[i % COLORS.length],
             volume: 1, mute: false, solo: false };
  }));
  return tracks.filter((t) => t.notes.length);
}

// Position of a track name in ORDER (unknown names last).
function rank(name) {
  const i = ORDER.indexOf(name);
  return i < 0 ? ORDER.length : i;
}

// Build the result view: lanes, transport, and the original audio (decoded in the background).
async function showResult(s) {
  closeResult();
  shown = s.id;
  $("progress").hidden = true;
  const analysis = await getJSON(`/files/${s.id}/00_prep/analysis.json`);
  const tracks = await loadTracks(s);
  if (shown !== s.id) return;
  const ctx = new AudioContext();
  const player = new Player(ctx, tracks, null);
  player.volume = Number($("volume").value);
  player.balance = Number($("balance").value);
  const roll = new Roll($("lanes"), tracks, analysis.beats, analysis.downbeats, analysis.duration_s,
                        (t) => player.seek(t));
  current = { id: s.id, player, roll, tracks, duration: analysis.duration_s };
  renderHeads(tracks, player);
  $("roll").hidden = false;
  $("play").disabled = false;
  $("download").href = `/files/${s.id}/02_midi/${s.variant}/song.mid`;
  $("download").download = s.name.replace(/\W+/g, "_") + ".mid";
  $("download").hidden = false;
  loadAudio(s, ctx, player);
}

// Decode 00_prep/master.wav (the file the notes were timed against) and hand it to the player.
async function loadAudio(s, ctx, player) {
  $("audio").textContent = "原曲（載入中）";
  const buf = await (await fetch(`/files/${s.id}/00_prep/master.wav`)).arrayBuffer();
  const audio = await ctx.decodeAudioData(buf);
  if (current?.player !== player) return;
  player.audio = audio;
  if (player.playing) player.play(player.time());  // pick up the audio mid-playback
  $("audio").textContent = "原曲";
}

// Track header column: name, instrument, mute / solo, volume.
function renderHeads(tracks, player) {
  const heads = $("heads");
  heads.innerHTML = `<div style="height:24px"></div>`;
  for (const t of tracks) {
    const div = document.createElement("div");
    div.className = "head";
    div.innerHTML = `<div class="name" style="color:${t.color}">${t.name}</div>`
      + `<div class="inst">${t.instrument} · ${t.notes.length} 音</div>`;
    const mute = button("M", () => { t.mute = !t.mute; mute.className = t.mute ? "on" : ""; updateMix(); });
    const solo = button("S", () => { t.solo = !t.solo; solo.className = t.solo ? "on" : ""; updateMix(); });
    const vol = slider(0, 1.5, t.volume, (v) => { t.volume = v; player.applyMix(); });
    div.append(mute, " ", solo, " ", vol);
    heads.append(div);
  }
}

// Small button with a click handler.
function button(label, onclick) {
  const b = document.createElement("button");
  b.textContent = label;
  b.onclick = onclick;
  return b;
}

// Range input calling onInput(value); double-click resets it to `initial`.
function slider(min, max, initial, onInput) {
  const s = document.createElement("input");
  Object.assign(s, { type: "range", min, max, step: 0.01, value: initial });
  s.oninput = () => onInput(Number(s.value));
  s.ondblclick = () => { s.value = initial; onInput(initial); };
  return s;
}

// Apply mute / solo to the audio and dim silent lanes.
function updateMix() {
  const anySolo = current.tracks.some((t) => t.solo);
  for (const t of current.tracks) t.dim = !audible(t, anySolo);
  current.player.applyMix();
}

// Toggle play / pause.
function togglePlay() {
  if (!current) return;
  const p = current.player;
  if (p.playing) p.pause();
  else p.play(p.time() >= current.duration ? 0 : p.time());
}

// m:ss.s
function fmt(t) {
  return `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
}

// Animation loop: redraw the roll at the player's clock.
function frame() {
  if (current) {
    const p = current.player, t = p.time();
    if (p.playing && t >= current.duration) p.pause();
    if (p.playing) current.roll.follow(t);
    current.roll.draw(t);
    $("clock").textContent = fmt(t);
    $("play").textContent = p.playing ? "⏸ 暫停" : "▶ 播放";
  }
  requestAnimationFrame(frame);
}

// Send the chosen file as the raw request body; the server queues it.
async function upload(file) {
  const r = await fetch(`/api/upload?name=${encodeURIComponent(file.name)}`, { method: "PUT", body: file });
  const { id } = await r.json();
  await poll();
  select(id);
}

$("play").onclick = togglePlay;
// Overall volume and original-vs-MIDI balance (kept across songs; applied to the current player).
for (const [id, key, initial] of [["volume", "volume", 1], ["balance", "balance", 0.5]]) {
  const s = $(id);
  const set = (v) => { s.value = v; if (current) { current.player[key] = v; current.player.applyMix(); } };
  s.oninput = () => set(Number(s.value));
  s.ondblclick = () => set(initial);
}
$("upload").onchange = (e) => e.target.files[0] && upload(e.target.files[0]);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target.tagName !== "INPUT") { e.preventDefault(); e.target.blur(); togglePlay(); }
});
poll();
setInterval(poll, 2000);
requestAnimationFrame(frame);
