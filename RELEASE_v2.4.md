## v2.4

Stand des Manuskripts. Der dort berichtete Lauf ist mit v2.4 gerechnet, vollständig neu ab der
Datenaufbereitung (29./30.09.2026). Seine Zahlen weichen von v2.3.1 ab, auch auf der Topic-Ebene.

### Behoben

- **DS3 (Review-Absenz) zählte keine Reviews.** Der Vergleich des Dokumenttyps hing an der
  Schreibweise: In den aufbereiteten Daten stand `review`, verglichen wurde mit `"Review"`. DS3 war
  damit in allen Topics eine reine Funktion der Topicgröße. Der Vergleich ist jetzt unabhängig von
  der Schreibweise und erkennt mehrteilige Angaben.
- In der Datenaufbereitung ging bei `early access article|review` der Typ Review verloren;
  „early access article“ gilt jetzt als Status, nicht als Typ.
- Schritt 3 (Faktorenanalyse) läuft mit scikit-learn ab 1.8. Muster, Phi und Kommunalitäten sind
  bitgleich mit scikit-learn 1.5.

### Geändert

- **Glättung von DS3 zum Review-Anteil der Phase**, neuer Standard
  `REVIEW_ABSENCE_PRIOR = "phase_share"`: DS3 = 1 − (r + 2α·p₀)/(n + 2α) mit p₀ = Σr/Σn über die
  Topics der Phase, bei unveränderter Stärke 2α = 10. Der symmetrische Prior (Mittelwert 0,5) hielt
  DS3 bei einem Review-Anteil von 2 bis 3 Prozent an die Topicgröße gebunden. Er bleibt als Option
  `"symmetric"` erhalten.
- Die Datendateien des Supplements tragen keine S-Nummern mehr im Namen; das Abbildungsskript liest
  die Topiczahlen aus den Daten.

### Hinzugefügt

- Revision des Sprachmodells festgeschrieben (`SBERT_MODEL_REVISION` in `config.py`), vollständige
  Lock-Datei der Laufumgebung (64 Distributionen, Python 3.12.7), `CITATION.cff`.
- Schritt 6 (Übergänge der dominanten Konfiguration zwischen den Phasen), Stabilität der
  Phasengrenze mit Abbildung S16, Prüfskript zur Reproduzierbarkeit, Robustheitsexperimente
  (Perturbation, zwei Nullmodelle, Standardisierungsvarianten), Abbildungssatz und Datendateien für
  die Einreichung.

### Reproduzierbarkeit

In derselben Umgebung ist der Lauf wiederholbar: HDBSCAN und UMAP ergeben aus den gespeicherten
Eingängen in beiden Phasen exakt die gespeicherten Ausgaben, und die SBERT-Vektoren waren bei einer
Wiederholung am Folgetag bitgleich (geprüft an Phase 1). Gegen einen Lauf vom Mai 2026 mit denselben
Fassungen von torch, transformers und sentence-transformers weichen die Vektoren um höchstens
1,7e−7 ab; die Topics ändern sich dann etwa so stark wie bei einem anderen Seed.

### Kompatibilität

`v2.2` (Masterarbeit) und `v2.3.1` bleiben unverändert bestehen. Einzelheiten im `CHANGELOG.md`.
