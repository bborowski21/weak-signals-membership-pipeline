## v2.3.1

Patch auf v2.3, entstanden aus einer erneuten kritischen Prüfung: v2.3 dokumentierte die
Cosine-Quellen-Abweichung, stellte sie aber an zwei Stellen selbst neu auf.

### Behoben

- **Der Orchestrator-Lauf ist jetzt durchgehend methodenkonform.** `run_all_phases.py` rief
  `step01b` weiterhin ohne `--with-sbert` auf; ein voller Lauf erzeugte damit c-TF-IDF-Matches
  neben der SBERT-basierten Sensitivitätsdatei, also exakt die Mischlage, die v2.3 als Ursache
  beschreibt.
- **Provenienz steht in den Artefakten selbst.** Neue Spalte `cosine_source`
  (`sbert_centroid` oder `ctfidf`) in den `topic_matches`-CSVs und in
  `sensitivity_hybrid_alpha.csv`. Ausgaben der beiden Betriebsmodi (Thesis-Reproduktion ohne
  Flag, methodenkonform mit Flag) bleiben auch nach dem Lauf unterscheidbar.
- **Kein Konfidenz-Vokabular mehr in der Diagnostik:** „Score-Schwelle (deskriptiv)" statt
  „Konfidenzschwelle", „Review-Kandidaten" statt „Unsichere Matches". Begründung im
  v2.3-CHANGELOG: Der Hybrid-Score misst Ähnlichkeit, nicht Korrektheit.
- README: Die gelistete Konstante `HYBRID_ALPHA` existierte in `config.py` nicht; korrigiert.
  Neuer Abschnitt „Zwei Betriebsmodi des Cross-Phase-Matchings".
- Drei als *Präzisierung v2.3.1* markierte Nachträge im v2.3-Eintrag: die Masterarbeit ist
  zwischen Kapitel 3 (SBERT-Zentroide) und Anhang D (c-TF-IDF, dokumentiert korrekt den
  tatsächlichen Lauf) intern uneinheitlich; die Margin-Aussage ist als fehlender Beleg zu lesen,
  nicht als Gegenbeweis; Herkunft und Grenzen der Richtig/Falsch-Beurteilungen sind offengelegt.

### Verifikation

Beide Modi wurden nach den Änderungen gegen die Referenzläufe geprüft: ohne Flag identische
99 Mutual-Paare (Thesis-Stand v2.2), mit Flag identische 101 Paare, `cosine_source` jeweils
korrekt gestempelt.

### Kompatibilität

`v2.2` bleibt unverändert der reproduzierbare Stand der Masterarbeit (Aufruf ohne
`--with-sbert`). Für die Hintergründe der Cosine-Quellen-Abweichung siehe den v2.3-Eintrag im
`CHANGELOG.md`.
