# Reitcoach v1.9 (Entwurf, noch nicht getestet im Browser)

Hochladen auf GitHub: nur app.js, core.js, index.html (sounds und models bleiben wie sie sind).

## Neu gegenüber v1.8
- Blick-Korrektur ab 20° (vorher 27°), mind. alle 12 s möglich
- Hand: langsames Führen zählt nicht mehr, nur schnelles Zupfen (Messwert wjit, Schwelle 0,045 vorläufig)
- Neue Ansage "Zügel gleichmäßig dran lassen, nicht zupfen." (Hand zupft + Kopf unruhig). Noch ohne Tondatei, nutzt Browser-Sprachausgabe
- Lob nur bei Nase −9° bis +12° und ohne Kopf-, Zügel- oder Handfehler
- Kopf-Korrektur mit Hysterese (an unter −16°, aus über −13°)
- Gangerkennung ab 2,5 statt 4 Bilder/s (vorläufig, an Video prüfen)
- Ansagen begrenzt: je Stufe max. 3 / 5 / 8 pro Minute, seltenstes Thema zuerst, Bein und Oberkörper erst nach ca. 12 s Dauer und nur alle 40 s
- Verlauf speichert nicht mehr doppelt
- Bericht: Quelle (Live/Video), Bewertung 1 bis 5, Knopf "Bericht teilen" (JSON mit Messdaten)
- Lernmodus: Knöpfe Korrekt / Drückt raus / Dahinter / Flackert, Messwerte je Sekunde inkl. Kopfhöhe (hy Genick, ny Nase)

## Offen
- Tondateien: "gut" neu vertonen, Zügel-Ansage vertonen
- Schwellen für Bein/Oberkörper bleiben wie v1.8, bis Lernmodus-Daten da sind
