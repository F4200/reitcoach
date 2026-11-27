# KI-Reitcoach

Web-App, die beim Reiten (Live-Kamera oder Video-Datei) Pferd und Reiterin erkennt und kurze
Ansagen gibt (Korrektur, Lob, Gangart-Info). Erkennung mit YOLO11 (Segmentierung + Pose) über
onnxruntime-web 1.31.1 im Browser. Läuft auf GitHub Pages, kein Build-Schritt.

Antworten auf Deutsch, knapp. Vor größeren Umbauten fragen. Immer kurz sagen, was geändert wird.

## Dateien
- `index.html` – Oberfläche
- `app.js` – Kamera/Video, Modelle laden, Sprachausgabe, Bericht (Download als JSON,
  `localStorage`: `rc_last` letzter Bericht mit `sent`-Flag, `rc_hist` Verlauf), Lernmodus.
  `VERSION` hier pflegen. Ansage-Stufen in `FREQ` (ruhig / normal / viel).
- `core.js` – Auswertung ohne DOM: Messwerte, Regeln (`RULES`), Gangart, Ansage-Auswahl
  (`Coach.step`). Wird auch vom Offline-Test benutzt.
- `sounds/` – vertonte Ansagen. **Nicht ändern.**
- `models/` – ONNX-Modelle, in `.part0–3` geteilt. **Nicht ändern.**
- `LIESMICH-v1.9.md` – Änderungsnotizen v1.9

## Regeln
- **Push nur nach ausdrücklicher Bestätigung.** Commits lokal sind ok.
- Nichts in `sounds/` und `models/` ändern.
- Ansage-Obergrenze `maxPerMin` gilt für alle Ansagen außer Gangartwechsel.

## Testverfahren (offline)
Werkzeug: `C:\Users\frank\Documents\Reit-Projekt\tools\testlauf\analyze.mjs`
(gleiche `core.js`, gleiche Modelle, Bild für Bild mit 0,25 s Schritt).

- Clips: `C:\Users\frank\Documents\Reit-Projekt\medien\videos\gopro\20261124\clips\` (A–E, je 180 s)
- Ergebnisse: `C:\Users\frank\Documents\Reit-Projekt\medien\berichte\testlauf-offline\<version>\`
- Stufe: `normal`
- Voraussetzungen: Node.js (installiert unter `C:\Program Files\nodejs`), ffmpeg/ffprobe im PATH,
  `onnxruntime-web@1.31.1` per npm in einem Arbeitsordner.
- Das Skript enthält feste Linux-Pfade. Für den Lauf eine Kopie im Arbeitsordner anpassen
  (Original nicht ändern):
  - Import von `core.js` → `file:///C:/Users/frank/Documents/reitcoach/core.js`
  - Modellpfad `M` → Ordner mit zusammengesetzten Modellen
    (`cat yolo11n-seg.onnx.part0 … part3 > yolo11n-seg.onnx`, ebenso `-pose`)
  - Versionslabel `offline-v…` anpassen
- Aufruf je Clip (Läufe können parallel laufen; 5 parallel ca. 33 min):
  `node analyze.mjs <clip.mp4> <ausgabe.json> normal`
- Auswerten: `share.zuegel`, `share.handruhe`, `ansagenProMin`; zusätzlich die größte Zahl
  an Ansagen in einem 60-s-Fenster aus `log` prüfen.

## Zielwerte (Stufe normal)
- Höchstens 5 Ansagen pro Minute
- Zügel- und Handruhe-Fehleranteil deutlich unter 75 %

Stand v1.9.2 (5 Clips): Zügel 1–2 %, Handruhe 10–17 %, 5,0–5,7 Ansagen/min
(ohne Gangartwechsel 3,7–4,0; der Rest über 5 kommt nur von Gangartwechseln).
