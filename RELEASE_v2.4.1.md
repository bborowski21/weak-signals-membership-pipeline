## v2.4.1

Korrekturen aus dem Systemcheck vom 30.09.2026. Keine Zahl im Manuskript ändert sich.

### Behoben

- **Faktorkorrelationen der EFA in falscher Reihenfolge.** factor_analyzer 0.5.1 sortiert die Spalten der
  Mustermatrix nach der Rotation um, die Faktorkorrelationen nicht. Φ steht jetzt in der Reihenfolge der
  Musterspalten; berichtigt sind die Φ-Dateien von Schritt 3 und Panel A der Abbildung der
  Faktorkorrelationen. Mustermatrizen und Faktorzahlen bleiben gleich.
- **Kommunalitäten bei obliquer Rotation** jetzt als diag(P Φ P') statt als Summe der quadrierten
  Musterladungen; ein Heywood-Fall wird gemeldet.
- **Sensitivitätsanalyse:** Die Ablation der Emerging-Concept-Subindikatoren wirkt jetzt auch auf m_ec,
  der Endjahrtest transformiert die Wachstumsrate wie Schritt 2. Beides wirkt erst bei einem neuen Lauf
  von Schritt 5; die im Supplement berichteten Größen bleiben gleich.
- Drei Stellen ohne Wirkung auf den berichteten Lauf: Streuungsfilter auf den Rohwerten, leere
  Keyword-Einträge, DS3 unter pandas 3 bei fehlendem Dokumenttyp.

### Hinzugefügt

- Die Skripte, mit denen die Abbildungen (`paper_figures_rp.py`, `rp_style.py`), Tabelle S3
  (`make_table_s3.py`) und die Klassenanteile unter den Nullmodellen (`nullmodel_class_shares.py`) des
  Manuskripts entstanden sind, und in `berichteter_lauf/` der Treiber des berichteten Laufs.

### Geändert

- README (Ablauf des berichteten Laufs, Zitations-Kohärenz über `step02b_run_with_kati.py`, Smoke-Test,
  Repository-Struktur), `requirements.txt` auf die Major-Versionen des berichteten Laufs, Supplement-
  Datendateien mit den Konfigurationsnamen des Manuskripts.

### Berichtigung zu v2.4

Der berichtete Lauf ist mit v2.4 gerechnet bis auf fünf Dateien, die danach geändert wurden (in
`config.py` der Standard der DS3-Glättung, den im Lauf der Treiber je Variante setzte, und die Revision
des Sprachmodells; in `step01_topic_modeling.py` das Laden in dieser Revision; der Aufsatz für
scikit-learn in `step03_efa_pca.py`; Dateinamen in `build_supplement.py`; Optionen in
`pruefe_reproduzierbarkeit.py`). Schritt 3 ist mit v2.4.1 neu gerechnet. Einzelheiten und Prüfsummen im
`CHANGELOG.md`.

### Kompatibilität

`v2.2` (Masterarbeit), `v2.3.1` und `v2.4` bleiben unverändert bestehen.
