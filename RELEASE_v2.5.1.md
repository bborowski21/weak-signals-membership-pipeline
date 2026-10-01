## v2.5.1

Stand des Manuskripts. Nur Benennung und Dokumentation; keine Rechnung und keine Zahl ändert sich.

### Geändert

- **TF-IDF statt c-TF-IDF für die führenden Begriffe aus Schritt 1.** Ihre IDF wird über die einzelnen Publikationen
  gerechnet, nicht über die Topics. Klassenbasiertes c-TF-IDF rechnet nur die Topic-Güte (Schritt 3c). Angepasst sind
  die README der Datendateien (`build_supplement.py`), die Beschriftung von Abbildung S1 (`paper_figures_rp.py`),
  `README.md` und `CITATION.cff`.
- Bezeichner im Code bleiben, damit Code und Laufausgaben zum berichteten Lauf passen.

### Kompatibilität

`v2.2` (Masterarbeit), `v2.3.1`, `v2.4`, `v2.4.1` und `v2.5` bleiben unverändert bestehen.
