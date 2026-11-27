import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.wasm.min.mjs';
import { Coach, S, RULES } from './core.js';

const $ = id => document.getElementById(id);
const vid = $('vid'), ov = $('ov'), ctx = ov.getContext('2d');
const status = t => $('status').textContent = t;

// ---------- Modelle laden ----------
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
let seg, pose;
const VERSION = 'v1.9.2';
let backend = 'CPU';
const ready = (async () => {
  const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  // Modelle liegen in Teilen (Upload-Grenze) und werden hier zusammengesetzt
  const load = async name => { const parts = []; for (let i = 0; ; i++) { const r = await fetch(`./models/${name}.part${i}`); if (!r.ok) break; parts.push(new Uint8Array(await r.arrayBuffer())); }
    const n = parts.reduce((s, p) => s + p.length, 0), all = new Uint8Array(n); let o = 0; for (const p of parts) { all.set(p, o); o += p.length; } return all; };
  status('Lade Modell 1/2 …'); const bSeg = await load('yolo11n-seg.onnx');
  status('Lade Modell 2/2 …'); const bPose = await load('yolo11n-pose.onnx');
  seg = await ort.InferenceSession.create(bSeg.slice(), opt); pose = await ort.InferenceSession.create(bPose.slice(), opt);
  // Grafikchip (WebGPU) versuchen; nur nutzen, wenn Ergebnisse gleich und schneller sind
  if (false) {
    const tmo = (p, ms) => Promise.race([p, new Promise((_, r) => setTimeout(() => r(new Error('Zeitüberschreitung')), ms))]);
    try {
      status('Prüfe Grafikchip …');
      const g = { executionProviders: ['webgpu'], graphOptimizationLevel: 'all' };
      const sg = await tmo(ort.InferenceSession.create(bSeg.slice(), g), 20000), pg = await tmo(ort.InferenceSession.create(bPose.slice(), g), 20000);
      const x = new Float32Array(3 * S * S); for (let i = 0; i < x.length; i++) x[i] = ((i * 7919) % 997) / 997;
      const T = d => new ort.Tensor('float32', d, [1, 3, S, S]);
      const same = async (a, b) => { const p = (await a.run({ images: T(x) })).output0.data, q = (await b.run({ images: T(x) })).output0.data; let m = 0, d = 0; for (let i = 0; i < p.length; i++) { m = Math.max(m, Math.abs(p[i])); d = Math.max(d, Math.abs(p[i] - q[i])); } return d <= 0.02 * m; };
      const speed = async s => { await s.run({ images: T(x) }); const t = performance.now(); for (let i = 0; i < 3; i++) await s.run({ images: T(x) }); return (performance.now() - t) / 3; };
      if (await same(seg, sg) && await same(pose, pg)) {
        const tc = await speed(seg) + await speed(pose), tg = await speed(sg) + await speed(pg);
        if (tg < tc * 0.9) { seg = sg; pose = pg; backend = 'GPU'; } else backend = 'CPU (GPU nicht schneller)';
      } else backend = 'CPU (GPU ungenau)';
    } catch (e) { backend = 'CPU (kein GPU: ' + String(e.message || e).slice(0, 40) + ')'; }
  }
  status('Bereit. (' + VERSION + ' · ' + backend + ')');
})().catch(e => status('Fehler beim Laden: ' + e.message));
let inT = null;
async function run(name, data) {
  const s = name === 'seg' ? seg : pose;
  if (!inT || inT.data !== data) inT = new ort.Tensor('float32', data, [1, 3, S, S]);
  const o = await s.run({ images: inT });
  return { out0: o.output0.data, out1: o.output1?.data };
}

// ---------- Ausschnitt fürs Modell ----------
const pc = Object.assign(document.createElement('canvas'), { width: S, height: S }), px = pc.getContext('2d', { willReadFrequently: true });
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
// Web Audio: einmal per Fingertipp gestartet, danach dürfen Ansagen jederzeit kommen (iOS)
try { if (navigator.audioSession) navigator.audioSession.type = 'playback'; } catch { } // auch bei Stumm-Schalter
let actx = null; const bufs = {};
async function loadSounds() {
  try { const j = await (await fetch('./sounds/index.json')).json(); sIdx = j;
    await Promise.all(Object.values(j).map(async n => { const ab = await (await fetch('./sounds/' + n + '.mp3')).arrayBuffer(); bufs[n] = await actx.decodeAudioData(ab); }));
    snd('Ton bereit (' + Object.keys(bufs).length + ' Ansagen geladen)');
  } catch (e) { snd('Ton-Fehler beim Laden: ' + e.message); }
}
function keepAlive() { // stiller Dauerton hält Audio und Bluetooth wach
  const b = actx.createBuffer(1, actx.sampleRate, actx.sampleRate), s = actx.createBufferSource(); s.buffer = b; s.loop = true; s.connect(actx.destination); s.start();
}
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
  const n = sIdx[text], b = n && bufs[n];
  if (!actx || !b) return tts(text);
  if (actx.state !== 'running') actx.resume();
  const s = actx.createBufferSource(); s.buffer = b; s.connect(actx.destination); s.start();
  snd('Ton ok · ' + (++nSaid) + ' Ansagen · ' + actx.state);
}
function unlock() { // muss direkt im Fingertipp laufen
  if (!actx) { actx = new (window.AudioContext || window.webkitAudioContext)(); keepAlive(); loadSounds().then(() => speak('Coach startet.')); }
  actx.resume();
  if (!unlocked) { unlocked = true; try { const u = new SpeechSynthesisUtterance(' '); u.volume = 0; speechSynthesis.speak(u); } catch { } }
}
function say(text) {
  if (muted || performance.now() < busyUntil) return false;
  speak(text); return true;
}

// ---------- Ablauf ----------
const FREQ = { ruhig: { gapAny: 12, gapCat: 30, gapPraise: 24, maxPerMin: 3, gapCatSlow: 60 }, normal: { gapAny: 8, gapCat: 20, gapPraise: 16, maxPerMin: 5, gapCatSlow: 40 }, viel: { gapAny: 6, gapCat: 14, gapPraise: 10, maxPerMin: 8, gapCatSlow: 30 } };
let coach = new Coach(FREQ.normal), running = false, t0 = 0, last = null, banner = null, log = [], stats, wake = null;
function resetStats() { learn = { samples: [], marks: [] }; lastSampleT = -9; stats = { frames: 0, horse: 0, ang: [], gait: {}, msgs: [], start: Date.now(), fault: {}, seen: {}, work: 0, maxWork: 0, lastT: null, walks: 0, prevG: null }; }

async function startSession(src) {
  try { const d = JSON.parse(localStorage.getItem('rc_last') || 'null'); if (d && !d.sent && confirm('Der letzte Bericht (' + (d.summary?.d || '') + ') wurde noch nicht geteilt. OK = jetzt teilen, Abbrechen = trotzdem starten.')) { await sendData(d, false); return; } } catch { }
  await ready; if (!seg) return;
  if (src === 'cam') {
    const st = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    vid.muted = true; vid.srcObject = st;
  } else { vid.srcObject = null; vid.muted = true; vid.src = URL.createObjectURL(src); vid.loop = false; }
  await vid.play();
  if (!vid.srcObject) { vid.pause(); vid.currentTime = 0; } // Datei: Bild fuer Bild (VIDEO_STEP), nicht in Echtzeit
  try { wake = await navigator.wakeLock?.request('screen'); } catch { }
  $('start').style.display = 'none'; $('panel').hidden = false; $('bar').hidden = false; $('report').style.display = 'none'; $('lmbar').hidden = !learnOn;
  coach = new Coach(FREQ[$('freq').value]); resetStats(); log = []; running = true; t0 = performance.now();
  vid.onended = () => stopSession();
  loop();
}
const VIDEO_STEP = 0.25; // Sekunden Videozeit je Auswertung (= 4 Bilder/s, immer gleich, egal wie schnell der Rechner ist)
const seek = x => new Promise(res => { let to; const done = () => { vid.removeEventListener('seeked', done); clearTimeout(to); res(); }; to = setTimeout(done, 3000); vid.addEventListener('seeked', done); vid.currentTime = x; });
async function loop() {
  let n = 0, tf = performance.now();
  while (running) {
    if (vid.readyState < 2) { await new Promise(r => requestAnimationFrame(r)); continue; }
    const t = vid.srcObject ? (performance.now() - t0) / 1000 : vid.currentTime;
    const r = await coach.step(t, vid.videoWidth, vid.videoHeight, getPatch, run);
    last = r; record(r); sample(r); draw(r); panel(r);
    if (r.msg) { banner = { msg: r.msg, t }; say(r.msg[0]); $('cnt').textContent = 'Ansagen: ' + (log.length + 1); log.push({ t: Math.round(t), text: r.msg[0], kind: r.msg[1] }); }
    n++; const now = performance.now(); if (now - tf > 2000) { $('fps').textContent = (n * 1000 / (now - tf)).toFixed(1) + ' Bilder/s'; n = 0; tf = now; }
    if (!vid.srcObject) { // Datei: nächstes Bild holen, am Ende Bericht
      const nt = t + VIDEO_STEP;
      if (!isFinite(vid.duration) || nt >= vid.duration - 0.05) { stopSession(); break; }
      await seek(nt); $('cnt').textContent = 'Ansagen: ' + log.length + ' · Video ' + Math.round(nt) + ' / ' + Math.round(vid.duration) + ' s';
    }
    await new Promise(r => setTimeout(r, 0));
  }
}
// ---------- Lernmodus: Messwerte je Sekunde + Markierungen ----------
let learnOn = false, learn = { samples: [], marks: [] }, lastSampleT = -9;
const snap = r => { const m = r.m || {}, o = { t: +r.t.toFixed(1), gait: r.gait || null }; for (const k of ['ang', 'hy', 'ny', 'gaze', 'lean', 'hand', 'heel', 'elbow', 'wsd', 'wjit', 'angsd', 'flick']) o[k] = m[k] === null || m[k] === undefined ? null : +(+m[k]).toFixed(3); return o; };
function sample(r) { if (!learnOn || !r.m || r.t - lastSampleT < 1) return; lastSampleT = r.t; learn.samples.push(snap(r)); }
function mark(label) { if (!last || !last.m) return; learn.marks.push({ label, ...snap(last) }); const b = $('lm-info'); if (b) b.textContent = 'Markiert: ' + label + ' (' + learn.marks.length + ')'; }
function record(r) {
  stats.frames++; if (r.hb) stats.horse++;
  if (r.ang !== undefined) stats.ang.push(r.ang);
  if (r.gait) stats.gait[r.gait] = (stats.gait[r.gait] || 0) + 1;
  for (const [k, , , cat] of RULES) { const v = r.m?.[k]; if (v === null || v === undefined) continue; stats.seen[cat] = (stats.seen[cat] || 0) + 1; }
  for (const [k, test, , cat] of RULES) { const v = r.m?.[k]; if (v !== null && v !== undefined && test(v)) stats.fault[cat] = (stats.fault[cat] || 0) + 1; }
  // Arbeitsphasen ohne Schrittpause
  const dt = stats.lastT === null ? 0 : Math.min(2, r.t - stats.lastT); stats.lastT = r.t;
  if (r.gait === 'Trab' || r.gait === 'Galopp') { stats.work += dt; stats.maxWork = Math.max(stats.maxWork, stats.work); }
  else if (r.gait === 'Schritt' || r.gait === 'Halt') { if (stats.work > 30) stats.walks++; stats.work = 0; }
}
const GOOD = { kopf: 'Kopfhaltung', blick: 'Blick nach vorn', oberkoerper: 'Oberkörper aufrecht', hand: 'Handhöhe', handruhe: 'ruhige Hände', zuegel: 'gleichmäßige Anlehnung', ellbogen: 'Ellbogen weich', bein: 'Beinlage' };
const CAT = { kopf: ['Kopf/Hals (Nase hinter der Senkrechten)', 'Zügel länger lassen, Hand vor, Hals dehnen lassen.'], blick: ['Blick nach unten', 'Bewusst vorausschauen: nächster Buchstabe, nächste Linie.'], oberkoerper: ['Oberkörper', 'Groß machen, Schulterblätter zurück, ruhig sitzen.'], hand: ['Hände zu hoch', 'Hände tief und ruhig über dem Widerrist, Linie Ellbogen–Hand–Maul.'], zuegel: ['Zügel flackert (Hand zupft, Kopf unruhig)', 'Hand stehen lassen, gleichmäßig annehmen, nicht zupfen.'], handruhe: ['Hände unruhig', 'Hand stehen lassen, aus dem Sitz mitgehen, Bein macht die Arbeit.'], ellbogen: ['Arme gestreckt', 'Ellbogen locker an der Hüfte, Arm weich.'], bein: ['Bein/Absatz', 'Bein lang, Absatz unter die Hüfte.'] };

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
  if (!running) return; // Video-Ende und Stopp-Knopf dürfen nicht beide den Bericht bauen
  running = false; speechSynthesis.cancel(); unlocked = false; try { wake?.release(); } catch { }
  vid.pause(); vid.srcObject?.getTracks().forEach(t => t.stop());
  const a = stats.ang, mins = (!vid.srcObject && isFinite(vid.currentTime) ? vid.currentTime / 60 : (Date.now() - stats.start) / 60000).toFixed(1);
  const behind = a.length ? Math.round(100 * a.filter(v => v < -5).length / a.length) : null;
  const med = a.length ? [...a].sort((x, y) => x - y)[a.length >> 1].toFixed(0) : '–';
  const gsum = Object.values(stats.gait).reduce((s, v) => s + v, 0) || 1;
  const cnt = {}; log.filter(l => l.kind === 'c').forEach(l => cnt[l.text] = (cnt[l.text] || 0) + 1);
  const rows = [['Dauer', mins + ' min'], ['Pferd erkannt', Math.round(100 * stats.horse / Math.max(1, stats.frames)) + ' %'], ['Nase zur Senkrechten (Mittel)', med + '°'], ['Anteil hinter der Senkrechten', behind === null ? '–' : behind + ' %'],
  ...Object.entries(stats.gait).map(([g, v]) => ['Anteil ' + g, Math.round(100 * v / gsum) + ' %']), ['Ansagen gesamt', log.length], ['davon Korrekturen', log.filter(l => l.kind === 'c').length], ['davon Lob', log.filter(l => l.kind === 'p').length]];
  // Fehleranteile je Bereich, Punktzahl, Schwerpunkte
  const share = {}; for (const c in CAT) if ((stats.seen[c] || 0) > 10) share[c] = Math.round(100 * (stats.fault[c] || 0) / stats.seen[c]);
  const vals = Object.values(share), score = vals.length ? Math.max(0, Math.round(100 - vals.reduce((s, v) => s + v, 0) / vals.length)) : null;
  const worst = Object.entries(share).filter(([, v]) => v >= 15).sort((x, y) => y[1] - x[1]).slice(0, 3), good = Object.entries(share).filter(([, v]) => v < 10).map(([c]) => GOOD[c]);
  const src = vid.srcObject ? 'Live' : 'Video';
  const sum = { d: new Date().toISOString().slice(0, 16).replace('T', ' '), src, mins: +mins, score, med: med === '–' ? null : +med, behind, share };
  let hist = []; try { hist = JSON.parse(localStorage.getItem('rc_hist') || '[]'); } catch { }
  const prev = hist.filter(h => h.src === src).slice(-1)[0];
  if (+mins >= 0.5) { hist.push(sum); hist = hist.slice(-30); try { localStorage.setItem('rc_hist', JSON.stringify(hist)); } catch { } }
  const trend = (n, o, better) => o === null || o === undefined || n === null ? '' : n === o ? ' (=)' : ((better === 'up' ? n > o : n < o) ? ' (↑ besser als letztes Mal: ' + o + ')' : ' (↓ letztes Mal: ' + o + ')');
  rows.unshift(['Gesamtpunkte', score === null ? '–' : score + ' / 100' + trend(score, prev?.score, 'up')]);
  rows.push(['Schrittpausen nach Arbeit', stats.walks], ['Längste Arbeitsphase ohne Pause', Math.round(stats.maxWork / 60 * 10) / 10 + ' min'], ['Quelle', src], ['Technik', VERSION + ' · ' + backend + ' · ' + $('fps').textContent]);
  const behindRow = rows.find(r => r[0] === 'Anteil hinter der Senkrechten'); if (behindRow && behind !== null) behindRow[1] += trend(behind, prev?.behind, 'down');
  $('rep').innerHTML = '<table>' + rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('') + '</table>' +
    '<h3>Schwerpunkte fürs nächste Mal</h3>' + (worst.length ? '<table>' + worst.map(([c, v]) => `<tr><td><b>${CAT[c][0]}</b> (${v} % der Zeit)<br>${CAT[c][1]}</td></tr>`).join('') + '</table>' : '<p>Keine deutlichen Schwächen erkannt.</p>') +
    (good.length ? '<h3>Das lief gut</h3><p>' + good.join(', ') + '</p>' : '') +
    '<h3>Anteil der Zeit mit Fehler je Bereich</h3><table>' + Object.entries(share).map(([c, v]) => `<tr><td>${CAT[c][0]}</td><td>${v} %${trend(v, prev?.share?.[c], 'down')}</td></tr>`).join('') + '</table>' +
    '<h3>Häufigste Korrekturen</h3><table>' + Object.entries(cnt).sort((x, y) => y[1] - x[1]).map(([k, v]) => `<tr><td>${k}</td><td>${v}×</td></tr>`).join('') + '</table>' +
    '<h3>Verlauf (letzte Einheiten)</h3><table><tr><td>Datum</td><td>Punkte · hinter Senkr. · Min</td></tr>' + hist.slice(-8).reverse().map(h => `<tr><td>${h.d} (${h.src || 'Live'})${h.rating ? ' ★' + h.rating : ''}</td><td>${h.score ?? '–'} · ${h.behind ?? '–'} % · ${h.mins}</td></tr>`).join('') + '</table>' +
    '<h3>Alle Ansagen</h3><table>' + log.map(l => `<tr><td>${l.text}</td><td>${l.t} s</td></tr>`).join('') + '</table>';
  $('rep').insertAdjacentHTML('beforeend', '<h3>War es hilfreich für die Reiterin?</h3><div id="rate" style="display:flex;gap:8px;flex-wrap:wrap">' + [1, 2, 3, 4, 5].map(n => `<button data-n="${n}">${n}</button>`).join('') + '</div><div id="rate-info" style="color:#9a958b;margin-top:6px">1 = gar nicht, 5 = sehr</div>');
  lastSum = sum; lastHist = hist;
  $('rate').onclick = e => { const n = +e.target.dataset?.n; if (!n) return; sum.rating = n; try { localStorage.setItem('rc_hist', JSON.stringify(hist)); } catch { } $('rate-info').textContent = 'Gespeichert: ' + n + ' von 5'; };
  try { localStorage.setItem('rc_last', JSON.stringify({ version: VERSION, summary: sum, bericht: $('rep').innerText, ansagen: log, lernmodus: learn, sent: false })); } catch { }
  setTimeout(() => { try { sendData(JSON.parse(localStorage.getItem('rc_last')), true); } catch { } }, 300);
  $('report').style.display = 'block'; $('panel').hidden = true; $('bar').hidden = true; $('lmbar').hidden = true;
}
let lastSum = null, lastHist = null;
async function sendData(data, auto) {
  const txt = data.bericht || '';
  const name = 'reitcoach_' + (data.summary?.d || 'bericht').replace(/[: ]/g, '-') + '.json';
  const file = new File([JSON.stringify(data, null, 1)], name, { type: 'application/json' });
  if (!auto) { try { if (navigator.canShare?.({ files: [file] })) { await navigator.share({ files: [file], title: 'Reitcoach-Bericht', text: txt.slice(0, 1500) }); markSent(); return; } } catch (e) { if (e.name === 'AbortError') return; } }
  const u = URL.createObjectURL(file), l = document.createElement('a'); l.href = u; l.download = name; document.body.appendChild(l); l.click(); l.remove(); setTimeout(() => URL.revokeObjectURL(u), 5000);
  if (!auto) markSent();
}
function markSent() { try { const d = JSON.parse(localStorage.getItem('rc_last') || 'null'); if (d) { d.sent = true; localStorage.setItem('rc_last', JSON.stringify(d)); } } catch { } }
async function shareReport() {
  let d; try { d = JSON.parse(localStorage.getItem('rc_last') || 'null'); } catch { }
  if (!d) d = { version: VERSION, summary: lastSum, bericht: $('rep').innerText, ansagen: log, lernmodus: learn };
  else { d.summary = lastSum || d.summary; d.bericht = $('rep').innerText || d.bericht; }
  await sendData(d, false);
}

// ---------- Bedienung ----------
$('go-cam').onclick = () => { unlock(); startSession('cam').catch(e => status('Kamera nicht verfügbar: ' + e.message)); };
$('file').onclick = () => unlock(); document.querySelector('label[for=file]').addEventListener('click', unlock);
$('file').onchange = e => { const f = e.target.files[0]; if (f) startSession(f).catch(e => status('Video-Fehler: ' + e.message)); };
$('stop').onclick = stopSession;
$('lm').onclick = () => { learnOn = !learnOn; $('lm').textContent = learnOn ? 'Lernmodus an' : 'Lernmodus'; $('lmbar').hidden = !learnOn; };
for (const [id, label] of [['lm-ok', 'korrekt'], ['lm-push', 'drückt raus'], ['lm-behind', 'dahinter'], ['lm-flick', 'flackert']]) $(id).onclick = () => mark(label);
$('share').onclick = shareReport;
$('mute').onclick = () => { muted = !muted; $('mute').textContent = muted ? 'Ton aus' : 'Ton an'; if (muted) speechSynthesis.cancel(); };
$('freq').onchange = () => Object.assign(coach.opt, FREQ[$('freq').value]);
$('test').onclick = () => { unlock(); speak('Hand vor, Nase raus.'); };
document.addEventListener('visibilitychange', () => { if (!document.hidden && actx) actx.resume(); });
$('copy').onclick = () => navigator.clipboard?.writeText($('rep').innerText).catch(() => { });
$('again').onclick = () => { $('report').style.display = 'none'; $('start').style.display = 'grid'; status('Bereit.'); };
window.__state = () => ({ log, frames: stats?.frames, horse: stats?.horse, fps: $('fps').textContent, gait: last?.gait, ang: last?.m?.ang });
