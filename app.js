import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.wasm.min.mjs';
import { Coach, S } from './core.js';

const $ = id => document.getElementById(id);
const vid = $('vid'), ov = $('ov'), ctx = ov.getContext('2d');
const status = t => $('status').textContent = t;

// ---------- Modelle laden ----------
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
let seg, pose;
const VERSION = 'v1.3';
const ready = (async () => {
  const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  // Modelle liegen in Teilen (Upload-Grenze) und werden hier zusammengesetzt
  const load = async name => { const parts = []; for (let i = 0; ; i++) { const r = await fetch(`./models/${name}.part${i}`); if (!r.ok) break; parts.push(new Uint8Array(await r.arrayBuffer())); }
    const n = parts.reduce((s, p) => s + p.length, 0), all = new Uint8Array(n); let o = 0; for (const p of parts) { all.set(p, o); o += p.length; } return all; };
  status('Lade Modell 1/2 …'); seg = await ort.InferenceSession.create(await load('yolo11n-seg.onnx'), opt);
  status('Lade Modell 2/2 …'); pose = await ort.InferenceSession.create(await load('yolo11n-pose.onnx'), opt);
  status('Bereit. (' + VERSION + ')');
})().catch(e => status('Fehler beim Laden: ' + e.message));
async function run(name, data) {
  const s = name === 'seg' ? seg : pose;
  const o = await s.run({ images: new ort.Tensor('float32', data, [1, 3, S, S]) });
  return { out0: o.output0.data, out1: o.output1?.data };
}

// ---------- Ausschnitt fürs Modell ----------
const pc = new OffscreenCanvas(S, S), px = pc.getContext('2d', { willReadFrequently: true });
const buf = new Float32Array(3 * S * S);
function getPatch(box) {
  const W = vid.videoWidth, H = vid.videoHeight, b = box || [0, 0, W, H];
  const bw = b[2] - b[0], bh = b[3] - b[1], sc = S / Math.max(bw, bh), ox = (S - bw * sc) / 2, oy = (S - bh * sc) / 2;
  px.fillStyle = 'rgb(114,114,114)'; px.fillRect(0, 0, S, S);
  px.drawImage(vid, b[0], b[1], bw, bh, ox, oy, bw * sc, bh * sc);
  const d = px.getImageData(0, 0, S, S).data, n = S * S;
  for (let i = 0, j = 0; i < n; i++, j += 4) { buf[i] = d[j] / 255; buf[n + i] = d[j + 1] / 255; buf[2 * n + i] = d[j + 2] / 255; }
  return { data: buf, sc, ox, oy };
}

// ---------- Sprache ----------
// Ansagen als fertige Tondateien (laufen auf iOS über Kopfhörer wie Musik), Sprachsynthese nur als Notlösung
let voice = null, muted = false, busyUntil = 0, nSaid = 0, unlocked = false, sIdx = { 'Coach startet.': 's00' };
fetch('./sounds/index.json').then(r => r.json()).then(j => sIdx = j).catch(() => { });
const sfx = new Audio(); sfx.preload = 'auto'; sfx.setAttribute('playsinline', '');
function pickVoice() {
  const v = speechSynthesis.getVoices().filter(x => x.lang && x.lang.toLowerCase().startsWith('de'));
  const score = x => (/premium|erweitert|enhanced/i.test(x.name) ? 4 : 0) + (/anna|helena|petra|markus|yannick/i.test(x.name) ? 2 : 0) + (x.localService ? 1 : 0);
  voice = v.sort((a, b) => score(b) - score(a))[0] || null;
}
speechSynthesis.onvoiceschanged = pickVoice; pickVoice();
const snd = t => { const e = $('snd'); if (e) e.textContent = t; };
function tts(text) {
  try { if (speechSynthesis.speaking) speechSynthesis.cancel(); speechSynthesis.resume(); } catch { }
  const u = new SpeechSynthesisUtterance(text); u.lang = 'de-DE'; if (voice) u.voice = voice; u.volume = 1;
  u.onstart = () => snd('Ton ok (Synth) · ' + (++nSaid)); u.onerror = e => snd('Ton-Fehler: ' + e.error);
  speechSynthesis.speak(u);
}
function speak(text) {
  busyUntil = performance.now() + 900 + text.length * 70;
  const n = sIdx[text];
  if (!n) return tts(text);
  sfx.src = './sounds/' + n + '.mp3';
  sfx.play().then(() => snd('Ton ok · ' + (++nSaid) + ' Ansagen')).catch(e => { snd('Ton-Fehler: ' + e.name); tts(text); });
}
function unlock() { // muss direkt im Fingertipp laufen
  if (unlocked) return; unlocked = true; speak('Coach startet.');
  try { const u = new SpeechSynthesisUtterance(' '); u.volume = 0; speechSynthesis.speak(u); } catch { }
}
function say(text) {
  if (muted || performance.now() < busyUntil) return false;
  speak(text); return true;
}

// ---------- Ablauf ----------
const FREQ = { ruhig: { gapAny: 10, gapCat: 30, gapPraise: 20 }, normal: { gapAny: 6, gapCat: 20, gapPraise: 14 }, viel: { gapAny: 4, gapCat: 14, gapPraise: 10 } };
let coach = new Coach(FREQ.normal), running = false, t0 = 0, last = null, banner = null, log = [], stats, wake = null;
function resetStats() { stats = { frames: 0, horse: 0, ang: [], gait: {}, msgs: [], start: Date.now() }; }

async function startSession(src) {
  await ready; if (!seg) return;
  if (src === 'cam') {
    const st = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    vid.muted = true; vid.srcObject = st;
  } else { vid.srcObject = null; vid.muted = true; vid.src = URL.createObjectURL(src); vid.loop = false; }
  await vid.play();
  try { wake = await navigator.wakeLock?.request('screen'); } catch { }
  $('start').style.display = 'none'; $('panel').hidden = false; $('bar').hidden = false; $('report').style.display = 'none';
  coach = new Coach(FREQ[$('freq').value]); resetStats(); log = []; running = true; t0 = performance.now();
  vid.onended = () => stopSession();
  loop();
}
async function loop() {
  let n = 0, tf = performance.now();
  while (running) {
    if (vid.readyState < 2) { await new Promise(r => requestAnimationFrame(r)); continue; }
    const t = vid.srcObject ? (performance.now() - t0) / 1000 : vid.currentTime;
    const r = await coach.step(t, vid.videoWidth, vid.videoHeight, getPatch, run);
    last = r; record(r); draw(r); panel(r);
    if (r.msg) { banner = { msg: r.msg, t }; say(r.msg[0]); $('cnt').textContent = 'Ansagen: ' + (log.length + 1); log.push({ t: Math.round(t), text: r.msg[0], kind: r.msg[1] }); }
    n++; const now = performance.now(); if (now - tf > 2000) { $('fps').textContent = (n * 1000 / (now - tf)).toFixed(1) + ' Bilder/s'; n = 0; tf = now; }
    await new Promise(r => setTimeout(r, 0));
  }
}
function record(r) {
  stats.frames++; if (r.hb) stats.horse++;
  if (r.ang !== undefined) stats.ang.push(r.ang);
  if (r.gait) stats.gait[r.gait] = (stats.gait[r.gait] || 0) + 1;
}

// ---------- Anzeige ----------
function fit() { const R = ov.getBoundingClientRect(), d = devicePixelRatio || 1; ov.width = R.width * d; ov.height = R.height * d; }
addEventListener('resize', fit); fit();
function draw(r) {
  const W = vid.videoWidth, H = vid.videoHeight, cw = ov.width, ch = ov.height;
  const sc = Math.min(cw / W, ch / H), ox = (cw - W * sc) / 2, oy = (ch - H * sc) / 2, P = (x, y) => [ox + x * sc, oy + y * sc];
  ctx.clearRect(0, 0, cw, ch); const lw = Math.max(3, cw / 300);
  if (r.seg) {
    const [a, b] = r.seg, v = [b[0] - a[0], b[1] - a[1]], top = P(a[0] - 0.5 * v[0], a[1] - 0.5 * v[1]), ext = P(b[0] + 0.4 * v[0], b[1] + 0.4 * v[1]);
    ctx.lineWidth = lw * 2; ctx.strokeStyle = r.ang > -9 ? '#2fbf6a' : r.ang > -16 ? '#f2b33d' : '#e2453b';
    ctx.beginPath(); ctx.moveTo(...top); ctx.lineTo(...ext); ctx.stroke();
    ctx.lineWidth = lw; ctx.strokeStyle = '#fff'; ctx.beginPath(); ctx.moveTo(...top); ctx.lineTo(top[0], ext[1]); ctx.stroke();
  }
  if (r.rider) {
    const k = r.rider.kp, c = r.rider.kc, sd = c[5] + c[11] >= c[6] + c[12] ? 0 : 1;
    ctx.lineWidth = lw * 1.5; ctx.strokeStyle = '#3ce07a';
    for (let [a, b] of [[3, 5], [5, 7], [7, 9], [5, 11], [11, 13], [13, 15]]) { a += sd; b += sd; if (c[a] > .4 && c[b] > .4) { ctx.beginPath(); ctx.moveTo(...P(...k[a])); ctx.lineTo(...P(...k[b])); ctx.stroke(); } }
  }
  const bn = $('banner');
  if (banner && r.t - banner.t < 3.5) { bn.style.display = 'block'; bn.textContent = banner.msg[0]; bn.style.background = { c: '#b8322ae6', p: '#25874be6', info: '#24599be6' }[banner.msg[1]]; }
  else bn.style.display = 'none';
}
const FMT = { ang: [v => (v > 0 ? '+' : '') + v.toFixed(0) + '°', v => v < -16], gaze: [v => v.toFixed(0) + '°', v => v > 27], lean: [v => (v > 0 ? '+' : '') + v.toFixed(0) + '°', v => v > 6 || v < -8], hand: [v => v.toFixed(2), v => v > 0.16], elbow: [v => v.toFixed(0) + '°', v => v > 146], heel: [v => v.toFixed(2), v => v > 0.36] };
function panel(r) {
  $('gait').textContent = 'Gangart: ' + (r.gait || '–');
  for (const k in FMT) { const el = $('m-' + k), v = r.m?.[k]; if (v === null || v === undefined) { el.textContent = '–'; el.className = 'v'; } else { el.textContent = FMT[k][0](v); el.className = 'v ' + (FMT[k][1](v) ? 'bad' : 'ok'); } }
}

// ---------- Bericht ----------
function stopSession() {
  running = false; speechSynthesis.cancel(); sfx.pause(); unlocked = false; try { wake?.release(); } catch { }
  vid.pause(); vid.srcObject?.getTracks().forEach(t => t.stop());
  const a = stats.ang, mins = ((Date.now() - stats.start) / 60000).toFixed(1);
  const behind = a.length ? Math.round(100 * a.filter(v => v < -5).length / a.length) : null;
  const med = a.length ? [...a].sort((x, y) => x - y)[a.length >> 1].toFixed(0) : '–';
  const gsum = Object.values(stats.gait).reduce((s, v) => s + v, 0) || 1;
  const cnt = {}; log.filter(l => l.kind === 'c').forEach(l => cnt[l.text] = (cnt[l.text] || 0) + 1);
  const rows = [['Dauer', mins + ' min'], ['Pferd erkannt', Math.round(100 * stats.horse / Math.max(1, stats.frames)) + ' %'], ['Nase zur Senkrechten (Mittel)', med + '°'], ['Anteil hinter der Senkrechten', behind === null ? '–' : behind + ' %'],
  ...Object.entries(stats.gait).map(([g, v]) => ['Anteil ' + g, Math.round(100 * v / gsum) + ' %']), ['Ansagen gesamt', log.length], ['davon Korrekturen', log.filter(l => l.kind === 'c').length], ['davon Lob', log.filter(l => l.kind === 'p').length]];
  $('rep').innerHTML = '<table>' + rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('') + '</table>' +
    '<h3>Häufigste Korrekturen</h3><table>' + Object.entries(cnt).sort((x, y) => y[1] - x[1]).map(([k, v]) => `<tr><td>${k}</td><td>${v}×</td></tr>`).join('') + '</table>' +
    '<h3>Alle Ansagen</h3><table>' + log.map(l => `<tr><td>${l.text}</td><td>${l.t} s</td></tr>`).join('') + '</table>';
  $('report').style.display = 'block'; $('panel').hidden = true; $('bar').hidden = true;
}

// ---------- Bedienung ----------
$('go-cam').onclick = () => { unlock(); startSession('cam').catch(e => status('Kamera nicht verfügbar: ' + e.message)); };
$('file').onclick = () => unlock();
$('file').onchange = e => { const f = e.target.files[0]; if (f) startSession(f).catch(e => status('Video-Fehler: ' + e.message)); };
$('stop').onclick = stopSession;
$('mute').onclick = () => { muted = !muted; $('mute').textContent = muted ? 'Ton aus' : 'Ton an'; if (muted) speechSynthesis.cancel(); };
$('freq').onchange = () => Object.assign(coach.opt, FREQ[$('freq').value]);
$('test').onclick = () => { unlock(); speak('Hand vor, Nase raus.'); };
$('copy').onclick = () => navigator.clipboard?.writeText($('rep').innerText).catch(() => { });
$('again').onclick = () => { $('report').style.display = 'none'; $('start').style.display = 'grid'; status('Bereit.'); };
window.__state = () => ({ log, frames: stats?.frames, horse: stats?.horse, fps: $('fps').textContent, gait: last?.gait, ang: last?.m?.ang });
