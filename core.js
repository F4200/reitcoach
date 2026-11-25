// KI-Reitcoach – Kernlogik (läuft im Browser und in Node zum Testen)
export const S = 320;               // Modell-Eingangsgröße
const HORSE = 17;

// ---------- Hilfen ----------
const sigmoid = x => 1 / (1 + Math.exp(-x));
const median = a => { if (!a.length) return null; const b = [...a].sort((x, y) => x - y); const m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
const std = a => { if (a.length < 2) return null; const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
const deg = r => r * 180 / Math.PI;

// ---------- Ausgaben der Modelle lesen ----------
// seg: out0 [1,116,N], out1 [1,32,80,80]; patch-Koordinaten (0..S)
export function decodeHorse(out0, out1, N = 2100, minConf = 0.3) {
  let best = -1, bs = minConf;
  for (let i = 0; i < N; i++) { const s = out0[(4 + HORSE) * N + i]; if (s > bs) { bs = s; best = i; } }
  if (best < 0) return null;
  const cx = out0[best], cy = out0[N + best], w = out0[2 * N + best], h = out0[3 * N + best];
  const box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
  const coef = new Float32Array(32); for (let k = 0; k < 32; k++) coef[k] = out0[(84 + k) * N + best];
  // Maske 80x80 -> bilinear auf SxS, nur innerhalb der Box
  const P = 80, low = new Float32Array(P * P);
  for (let k = 0; k < 32; k++) { const c = coef[k], off = k * P * P; for (let j = 0; j < P * P; j++) low[j] += c * out1[off + j]; }
  const mask = new Uint8Array(S * S), r = P / S;
  const x0 = Math.max(0, Math.floor(box[0])), x1 = Math.min(S - 1, Math.ceil(box[2])), y0 = Math.max(0, Math.floor(box[1])), y1 = Math.min(S - 1, Math.ceil(box[3]));
  for (let y = y0; y <= y1; y++) {
    const fy = Math.min(P - 1.001, Math.max(0, (y + 0.5) * r - 0.5)), iy = fy | 0, dy = fy - iy;
    for (let x = x0; x <= x1; x++) {
      const fx = Math.min(P - 1.001, Math.max(0, (x + 0.5) * r - 0.5)), ix = fx | 0, dx = fx - ix;
      const v = low[iy * P + ix] * (1 - dx) * (1 - dy) + low[iy * P + ix + 1] * dx * (1 - dy) + low[(iy + 1) * P + ix] * (1 - dx) * dy + low[(iy + 1) * P + ix + 1] * dx * dy;
      if (v > 0) mask[y * S + x] = 1;   // sigmoid(v)>0.5  <=> v>0
    }
  }
  return { box, conf: bs, mask };
}
// pose: out0 [1,56,N]
export function decodePersons(out0, N = 2100, minConf = 0.35) {
  const c = [];
  for (let i = 0; i < N; i++) { const s = out0[4 * N + i]; if (s > minConf) c.push([s, i]); }
  c.sort((a, b) => b[0] - a[0]);
  const res = [];
  for (const [s, i] of c) {
    const cx = out0[i], cy = out0[N + i], w = out0[2 * N + i], h = out0[3 * N + i];
    const box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
    if (res.some(p => iou(p.box, box) > 0.5)) continue;
    const kp = [], kc = [];
    for (let k = 0; k < 17; k++) { kp.push([out0[(5 + 3 * k) * N + i], out0[(6 + 3 * k) * N + i]]); kc.push(out0[(7 + 3 * k) * N + i]); }
    res.push({ box, conf: s, kp, kc }); if (res.length >= 5) break;
  }
  return res;
}
function iou(a, b) { const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])); const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy; return u > 0 ? ix * iy / u : 0; }

export function pickRider(persons, hb) {
  let best = null, ba = 0;
  for (const p of persons) {
    const b = p.box, cx = (b[0] + b[2]) / 2;
    if (hb[0] < cx && cx < hb[2] && b[3] < hb[3] && b[1] < hb[1] + 0.25 * (hb[3] - hb[1])) { const a = (b[2] - b[0]) * (b[3] - b[1]); if (a > ba) { ba = a; best = p; } }
  }
  return best;
}
export function facingLeft(p) {
  const d = []; for (const [a, b] of [[13, 11], [14, 12]]) if (p.kc[a] > 0.4 && p.kc[b] > 0.4) d.push(p.kp[a][0] - p.kp[b][0]);
  return d.length ? (d.reduce((s, v) => s + v, 0) / d.length) < 0 : null;
}

// ---------- Kopfhaltung des Pferdes aus dem Umriss ----------
export function headGeom(mask, fl) {
  const get = fl ? (x, y) => mask[y * S + x] : (x, y) => mask[y * S + (S - 1 - x)];
  let x0 = S, x1 = -1, y0 = S, y1 = -1, n = 0;
  const xf = new Int32Array(S).fill(1e6);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) if (get(x, y)) { n++; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; if (xf[y] === 1e6) xf[y] = x; }
  if (n < 1500) return null;
  const W = x1 - x0, H = y1 - y0; if (H <= 0 || W / H < 1.15) return null;
  let ear = -1; const xe = x0 + Math.floor(0.3 * W);
  for (let y = 0; y < S && ear < 0; y++) for (let x = x0; x < xe; x++) if (get(x, y)) { ear = y; break; }
  if (ear < 0) return null;
  let run = 1e6, nose = null;
  for (let y = ear; y < y1; y++) { if (xf[y] >= 1e6) continue; if (xf[y] - run > 0.07 * W && y - ear > 0.12 * H) { nose = y - 1; break; } run = Math.min(run, xf[y]); }
  if (nose === null) return null;
  const L = nose - ear; if (L < 0.15 * H || L > 0.75 * H) return null;
  const ya = Math.floor(ear + 0.30 * L), yb = Math.floor(ear + 0.85 * L);
  let sy = 0, sx = 0, syy = 0, sxy = 0, m = 0;
  for (let y = ya; y < yb; y++) { if (xf[y] >= 1e6) continue; sy += y; sx += xf[y]; syy += y * y; sxy += y * xf[y]; m++; }
  if (m < 4) return null;
  const k = (m * sxy - sy * sx) / (m * syy - sy * sy);
  const ang = -deg(Math.atan(k));
  let pa = [xf[ya], ya], pb = [xf[yb - 1], yb - 1];
  if (!fl) { pa = [S - 1 - pa[0], pa[1]]; pb = [S - 1 - pb[0], pb[1]]; }
  return { ang, pa, pb, hy: (y1 - ear) / H, ny: (y1 - nose) / H };
}

// ---------- Sitz der Reiterin ----------
export function riderMetrics(p, fl) {
  const k = p.kp, c = p.kc, s = fl ? -1 : 1;
  const side = (c[5] + c[11] + c[13]) >= (c[6] + c[12] + c[14]) ? 0 : 1;
  const [SH, EL, WR, HP, AN, EAR] = [5, 7, 9, 11, 15, 3].map(i => i + side);
  const o = {};
  if (c[SH] > .5 && c[HP] > .5) {
    const T = Math.hypot(k[SH][0] - k[HP][0], k[SH][1] - k[HP][1]);
    o.lean = deg(Math.atan2(s * (k[SH][0] - k[HP][0]), k[HP][1] - k[SH][1]));
    if (c[WR] > .4) o.hand = (k[HP][1] - k[WR][1]) / T;
    if (c[AN] > .4) o.heel = s * (k[AN][0] - k[HP][0]) / T;
    if (c[0] > .5 && c[EAR] > .5) o.gaze = deg(Math.atan2(k[0][1] - k[EAR][1], Math.max(1e-3, s * (k[0][0] - k[EAR][0]))));
    if (Math.min(c[SH], c[EL], c[WR]) > .5) {
      const a = [k[SH][0] - k[EL][0], k[SH][1] - k[EL][1]], b = [k[WR][0] - k[EL][0], k[WR][1] - k[EL][1]];
      o.elbow = deg(Math.acos(Math.max(-1, Math.min(1, (a[0] * b[0] + a[1] * b[1]) / Math.hypot(...a) / Math.hypot(...b)))));
    }
  }
  return o;
}

// ---------- Gangart aus dem Bewegungsrhythmus ----------
function spectrum(t, x, fs = 15) {
  const t0 = t[0], n = Math.floor((t[t.length - 1] - t0) * fs); if (n < 20) return null;
  const u = new Float64Array(n); let j = 0;
  for (let i = 0; i < n; i++) { const tt = t0 + i / fs; while (j < t.length - 2 && t[j + 1] < tt) j++; const a = t[j], b = t[j + 1]; u[i] = b > a ? x[j] + (x[j + 1] - x[j]) * (tt - a) / (b - a) : x[j]; }
  // Hochpass: gleitenden Mittelwert (1,2 s) abziehen -> Kamerazoom/Schwenk raus
  const hw = Math.round(0.6 * fs), v = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = 0, c = 0; for (let k = Math.max(0, i - hw); k <= Math.min(n - 1, i + hw); k++) { s += u[k]; c++; } v[i] = u[i] - s / c; }
  let sd = 0; for (let i = 0; i < n; i++) { sd += v[i] * v[i]; u[i] = v[i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1))); }
  sd = Math.sqrt(sd / n);
  let bf = 0, bp = -1;
  for (let f = 0.9; f < 3.6; f += 0.02) { let re = 0, im = 0; for (let i = 0; i < n; i++) { const w = 2 * Math.PI * f * i / fs; re += u[i] * Math.cos(w); im -= u[i] * Math.sin(w); } const p = re * re + im * im; if (p > bp) { bp = p; bf = f; } }
  return { f: bf, sd };
}
function solve3(A, b) { const M = A.map((r, i) => [...r, b[i]]); for (let i = 0; i < 3; i++) { let p = i; for (let r = i + 1; r < 3; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;[M[i], M[p]] = [M[p], M[i]]; for (let r = 0; r < 3; r++) if (r !== i) { const f = M[r][i] / M[i][i]; for (let c = i; c < 4; c++) M[r][c] -= f * M[i][c]; } } return M.map((r, i) => r[3] / r[i]); }
export function classifyGait(w) {
  // kalibriert an eigenen Videos (Zirkel + lange Seite), Okt/Nov 2026
  const { head_f: hf = 0, asp_f: af, hip_f: pf, hip_sd: hs, asp_sd: asd } = w;
  const near = (x, y, d) => Math.abs(x - y) < d;
  if (asd < 0.02 && hs < 0.012) return 'Halt';
  if (af >= 2.2 && af <= 3.0 && (near(hf, af / 2, 0.25) || near(pf, af / 2, 0.25) || near(pf, af, 0.3))) return 'Trab';
  if (pf >= 1.1 && pf <= 1.45 && hf >= 1.1 && hf <= 1.45) return 'Trab';           // Leichttraben: Reiter 1x pro Tritt
  const f = [af, pf, hf].filter(x => x >= 1.35 && x <= 2.2);
  if (f.length >= 2) { const fm = f.reduce((s, x) => s + x, 0) / f.length; return (asd > 0.085 || (asd > 0.076 && fm <= 1.62)) ? 'Galopp' : 'Schritt'; }
  return null;
}

// ---------- Ansagen ----------
export const RULES = [
  ['ang', v => v < -16, ['Hand vor, Nase raus.', 'Gib nach, lass sie vorne raus.', 'Zügel etwas länger, Hals lang machen.', 'Hals lang, lass sie sich dehnen.'], 'kopf'],
  ['gaze', v => v > 20, ['Blick hoch!', 'Schau nach vorne, zwischen den Ohren durch.', 'Schau voraus, plan deine Linie.', 'Vorausschauen, nicht aufs Pferd gucken.'], 'blick'],
  ['lean', v => v > 6, ['Oberkörper aufrichten.', 'Mach dich groß, Schulterblätter zurück.', 'Ruhig sitzen bleiben, nicht vorbeugen.'], 'oberkoerper'],
  ['lean', v => v < -8, ['Nicht nach hinten lehnen, senkrecht sitzen.'], 'oberkoerper'],
  ['hand', v => v > 0.16, ['Hände tiefer.', 'Hände ruhig über dem Widerrist tragen.', 'Hand, Ellbogen, Maul: eine Linie.'], 'hand'],
  ['wjit', v => v > 0.045, ['Hände ruhig halten.', 'Hände stehen lassen, nicht mitwackeln.', 'Hand ruhig, das Bein macht die Arbeit.'], 'handruhe'],
  ['flick', v => v > 0.5, ['Zügel gleichmäßig dran lassen, nicht zupfen.'], 'zuegel'],
  ['elbow', v => v > 146, ['Ellbogen anwinkeln, locker an den Körper.', 'Arme nicht strecken, Ellbogen weich.', 'Ellbogen an die Hüfte, ganz locker.'], 'ellbogen'],
  ['heel', v => v > 0.36, ['Absatz unter die Hüfte, Bein zurück.', 'Bein lang, Absatz tief.', 'Bein ruhig dran lassen.'], 'bein'],
];
export const PRAISE = ['Gut so!', 'Prima, so bleiben.', 'Sehr schön, Nase vorne.', 'Genau so, weiter.', 'Klasse, weiter so.', 'Schön, bleib so ruhig.'];
const GAIT_SAY = {
  Schritt: ['Schritt. Zügel lang, locker lassen.', 'Schritt. Lob sie, Zügel hingeben.'],
  Trab: ['Trab. Schön gleichmäßig.', 'Trab. Rhythmus finden, Hand ruhig.'],
  Galopp: ['Galopp. Ruhig mitsitzen.', 'Galopp. Rhythmus halten, vorausschauen.'],
  Halt: ['Halt. Lob sie.'],
};
const DOWN_SAY = ['Gut durchpariert. Hand nachgeben.', 'Schöner Übergang, weich bleiben.'];
const RANK = { Halt: 0, Schritt: 1, Trab: 2, Galopp: 3 };
const TEMPO = {
  Trab: { fast: ['Tempo ruhiger, Rhythmus halten.'], slow: ['Mehr vorwärts, treib sie fleißiger.', 'Mehr Schwung, Bein dran.'] },
  Galopp: { fast: ['Galopp ruhiger, gleichmäßiger Rhythmus.'], slow: ['Mehr Galopp, frischer nach vorne.'] },
};
const pick = a => a[Math.floor(Math.random() * a.length)];

export class Coach {
  constructor(opt = {}) { this.opt = Object.assign({ gapAny: 8, gapCat: 20, gapPraise: 16, maxPerMin: 5, gapCatSlow: 40 }, opt); this.reset(); }
  reset() { this.hist = []; this.sig = []; this.last = {}; this.lastAny = -99; this.lastPraise = -99; this.gaitHist = []; this.prevGait = null; this.baseF = []; this.track = null; this.miss = 0; this.gait = null; this.gw = null; this.lastGaitCalc = -99; this.baseG = {}; this.walkSince = undefined; this.walkSaid = false; this.lastWalk = undefined; this._lastG = null; this._kopfBad = false; this._fc = {}; this.said = []; }

  // getPatch(box) -> {data:Float32Array(3*S*S), sc, ox, oy} ; box in Bildkoordinaten oder null = ganzes Bild
  // run(name, data) -> Promise<{out0,out1}>
  async step(t, frameW, frameH, getPatch, run) {
    const r = { t };
    if (!this.track || this.miss > 3) {
      // Suche: ganzes Bild, sonst überlappende Ausschnitte (kleines/fernes Pferd)
      const s = Math.min(frameW, frameH), L = Math.max(frameW, frameH), tiles = [null];
      for (let o = 0; o + s <= L + 1; o += s / 2) tiles.push(frameW >= frameH ? [o, 0, o + s, s] : [0, o, s, o + s]);
      this._ti = ((this._ti || 0) + 1) % tiles.length;
      let found = null;
      for (const tb of [tiles[0], tiles[this._ti || 1]]) {
        const P0 = getPatch(tb); const o = await run('seg', P0.data); const h = decodeHorse(o.out0, o.out1);
        if (h) { const off = tb || [0, 0]; found = h.box.map((v, i) => off[i % 2] + (v - (i % 2 ? P0.oy : P0.ox)) / P0.sc); break; }
      }
      if (!found) { this.track = null; this.hist.push(r); return this.after(r); }
      this.track = found; this.miss = 0;
    }
    const tb = this.track, bw = tb[2] - tb[0], bh = tb[3] - tb[1];
    const crop = [Math.max(0, tb[0] - 0.15 * bw), Math.max(0, tb[1] - 0.8 * bh), Math.min(frameW, tb[2] + 0.15 * bw), Math.min(frameH, tb[3] + 0.08 * bh)];
    const P = getPatch(crop);
    const os = await run('seg', P.data), op = await run('pose', P.data);
    const h = decodeHorse(os.out0, os.out1);
    if (!h) { this.miss++; this.hist.push(r); return this.after(r); }
    const toF = (x, y) => [crop[0] + (x - P.ox) / P.sc, crop[1] + (y - P.oy) / P.sc];
    const hb = [...toF(h.box[0], h.box[1]), ...toF(h.box[2], h.box[3])];
    this.track = hb; this.miss = 0; r.hb = hb;
    const persons = decodePersons(op.out0).map(p => ({ ...p, box: [...toF(p.box[0], p.box[1]), ...toF(p.box[2], p.box[3])], kp: p.kp.map(q => toF(q[0], q[1])) }));
    const rd = pickRider(persons, hb); r.rider = rd;
    if (rd) {
      const fl = facingLeft(rd);
      if (fl !== null) {
        Object.assign(r, riderMetrics(rd, fl));
        const g = headGeom(h.mask, fl);
        if (g) { r.ang = g.ang; r.hy = g.hy; r.ny = g.ny; r.seg = [toF(...g.pa), toF(...g.pb)]; }
      }
      // Rhythmus-Signale
      const c = rd.kc, k = rd.kp, H = hb[3] - hb[1], W = hb[2] - hb[0];
      if (c[11] > .4 || c[12] > .4) {
        const hip = c[11] >= c[12] ? k[11] : k[12];
        this.sig.push({ t, hip: (hb[3] - hip[1]) / H, head: c[0] > .4 ? (hb[3] - k[0][1]) / H : null, asp: W / H });
      }
    }
    this.hist.push(r);
    return this.after(r);
  }

  after(r) {
    const t = r.t, o = this.opt;
    this.hist = this.hist.filter(x => x.t > t - 3); this.sig = this.sig.filter(x => x.t > t - 4.5);
    const win = this.hist.filter(x => x.t > t - 3);
    const med = k => { const v = win.filter(x => x[k] !== undefined).map(x => x[k]); return v.length >= 3 ? median(v) : null; };
    const m = {}; for (const k of ['ang', 'gaze', 'lean', 'hand', 'heel', 'elbow', 'hy', 'ny']) m[k] = med(k);
    const hw = win.filter(x => x.hand !== undefined), hv = hw.map(x => x.hand); m.wsd = hv.length >= 5 ? std(hv) : null;
    // schnelles Zupfen: Hand minus gleitender Mittelwert (ca. 1 s) -> langsames Führen zählt nicht
    m.wjit = null;
    if (hw.length >= 6) { const d = hw.map((x, i) => { const nb = hw.filter(y => Math.abs(y.t - x.t) <= 0.6).map(y => y.hand); return x.hand - nb.reduce((s, v) => s + v, 0) / nb.length; }); m.wjit = std(d); }
    const av = win.filter(x => x.ang !== undefined).map(x => x.ang); m.angsd = av.length >= 6 ? std(av) : null;
    m.flick = m.wjit !== null && m.angsd !== null ? (m.wjit > 0.045 && m.angsd > 4 ? 1 : 0) : null;
    r.m = m;
    // Gangart (jede Sekunde neu)
    if (t - this.lastGaitCalc >= 1) {
      this.lastGaitCalc = t; const s = this.sig;
      if (s.length >= 10 && s[s.length - 1].t - s[0].t > 3.2 && s.length / (s[s.length - 1].t - s[0].t) >= 2.5) {
        const tt = s.map(x => x.t), w = {};
        const a = spectrum(tt, s.map(x => x.asp)), b = spectrum(tt, s.map(x => x.hip));
        const hs = s.filter(x => x.head !== null), c = hs.length > 12 ? spectrum(hs.map(x => x.t), hs.map(x => x.head)) : null;
        if (a && b) { w.asp_f = a.f; w.asp_sd = a.sd; w.hip_f = b.f; w.hip_sd = b.sd; if (c) w.head_f = c.f; this.gw = w; const g = classifyGait(w); this.gaitHist.push(g); this.gaitHist = this.gaitHist.slice(-3); const v = this.gaitHist.filter(Boolean); this.gait = v.length ? v.sort((x, y) => v.filter(z => z === y).length - v.filter(z => z === x).length)[0] : null; }
      } else { this.gait = null; this.gw = null; }
    }
    r.gait = this.gait;
    // Ansage wählen
    let msg = null;
    if (this.t0 === undefined) this.t0 = t;
    const warm = t - this.t0 < 5;
    if (t - (this._stT ?? -99) >= 1) { this._stT = t; this._stable = (this._stable || []).concat([this.gait]).slice(-4); }
    const stable = this._stable.length === 4 && this._stable[0] && this._stable.every(g => g === this._stable[0]) ? this._stable[0] : null;
    if (stable && stable !== this.prevGait && t - this.lastAny > 3 && t - (this.last.gait ?? -99) > 10) { this.last.gait = t; const first = this.prevGait === null; this.prevGait = stable; if (!first) { const prev = this._lastG; msg = [prev && RANK[stable] < RANK[prev] && stable !== 'Halt' ? pick(DOWN_SAY) : pick(GAIT_SAY[stable]), 'info']; } this._lastG = stable; if (stable === 'Schritt' || stable === 'Halt') { this.walkSince = t; this.walkSaid = false; this.lastWalk = t; } }
    if (!msg && this.gw && TEMPO[this.gait] && t - this.lastGaitCalc < 0.01) {
      const g = this.gait, f = this.gw.asp_f; this.baseG = this.baseG || {}; const B = (this.baseG[g] = (this.baseG[g] || []).concat([f]).slice(-40)); const bf = median(B);
      if (B.length > 15 && t - (this.last.tempo ?? -99) > o.gapCat && t - this.lastAny > o.gapAny) {
        if (f > bf * 1.08) { msg = [pick(TEMPO[g].fast), 'c']; this.last.tempo = t; }
        else if (f < bf * 0.92) { msg = [pick(TEMPO[g].slow), 'c']; this.last.tempo = t; }
      }
    }
    // Jungpferd: Pausen im Schritt nutzen, nach langer Arbeit an Pause erinnern
    if (this.lastWalk === undefined) this.lastWalk = t;
    if (!msg && this.gait === 'Schritt' && this.walkSince !== undefined && !this.walkSaid && t - this.walkSince > 25 && t - this.lastAny > o.gapAny) { msg = ['Pause gut nutzen, Zügel ganz hingeben.', 'info']; this.walkSaid = true; }
    if (!msg && (this.gait === 'Trab' || this.gait === 'Galopp') && t - this.lastWalk > 300 && t - (this.last.pause ?? -99) > 120 && t - this.lastAny > o.gapAny) { msg = ['Denk an eine Schrittpause für sie.', 'info']; this.last.pause = t; }
    let faults = RULES.filter(([k, test]) => m[k] !== null && m[k] !== undefined && test(m[k]));
    // Kopf: Hysterese (Korrektur unter -16 Grad, Entwarnung erst über -13 Grad)
    if (m.ang !== null && m.ang !== undefined) { if (m.ang < -16) this._kopfBad = true; else if (m.ang > -13) this._kopfBad = false; }
    faults = faults.filter(([k]) => k !== 'ang' || this._kopfBad);
    if (this._kopfBad && m.ang !== null && m.ang >= -16 && m.ang <= -13) { const r0 = RULES[0]; if (!faults.includes(r0)) faults.unshift(r0); }
    // Sitz-Themen nur, wenn sie anhalten (mind. 3 Prüfungen = ca. 3 s) -> kein Dauerfeuer
    this._fc = this._fc || {}; const seenCat = new Set(faults.map(f => f[3]));
    for (const cat of ['bein', 'oberkoerper']) this._fc[cat] = seenCat.has(cat) ? (this._fc[cat] || 0) + 1 : 0;
    faults = faults.filter(f => !['bein', 'oberkoerper'].includes(f[3]) || this._fc[f[3]] >= 12);
    this.said = (this.said || []).filter(x => x > t - 60);
    const capped = this.said.length >= (o.maxPerMin ?? 5);
    if (!msg && t - this.lastAny > o.gapAny && t - this.lastPraise > o.gapPraise && m.ang !== null && m.ang > -9 && m.ang < 12 && faults.length <= 1 && !faults.some(f => ['kopf', 'zuegel', 'handruhe'].includes(f[3]))) { msg = [pick(PRAISE), 'p']; this.lastPraise = t; }
    if (!msg && !capped && t - this.lastAny > o.gapAny) {
      const slow = c2 => ['bein', 'oberkoerper'].includes(c2) ? (o.gapCatSlow ?? o.gapCat * 2) : (c2 === 'blick' ? Math.min(o.gapCat, 12) : o.gapCat);
      const due = faults.filter(f => t - (this.last[f[3]] ?? -99) > slow(f[3])).sort((a, b) => (this.last[a[3]] ?? -99) - (this.last[b[3]] ?? -99));
      if (due.length) { const [, , ph, cat] = due[0]; msg = [pick(ph), 'c']; this.last[cat] = t; }
    }
    if (warm) msg = null;
    if (msg) { this.lastAny = t; if (msg[1] === 'c' || msg[1] === 'p') this.said.push(t); }
    r.msg = msg;
    return r;
  }
}
