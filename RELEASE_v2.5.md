## v2.5

Stand des Manuskripts. Der im Manuskript berichtete Lauf ist mit diesem Stand gerechnet.

### Geändert

- **EO1 und DS2 als gewichtete Jaccard-Distanz.** EO1 (`keyword_volatility`) vergleicht die relativen
  Häufigkeiten aller Author Keywords zwischen der frühen und der späten Hälfte eines Topics, DS2
  (`terminological_instability`) zwischen aufeinanderfolgenden Jahren, gemittelt. Bis v2.4.1 waren es die 20
  beziehungsweise 15 häufigsten Keywords als Mengen, bei Gleichständen abhängig von der Reihenfolge der
  Datensätze.
- **Nicht bestimmbare Werte** von EO1, DS2 und IP2 (`citation_momentum`) erhalten den Median der Phase. Bis
  v2.4.1 zählte eine Seite ohne Keywords als vollständiger Wechsel (1,0), die übrigen Fälle erhielten 0,5.
- **Tabelle S3** (`make_table_s3.py`): Beschriftung wie im Supplement.
- Beleg der Phasengrenze (`output_phase_boundary/phase_boundary_stability.json`) aus dem neuen Lauf; README und
  `berichteter_lauf/README.md` beschreiben, wie der Lauf entstand.

### Wirkung auf den berichteten Lauf

Die übrigen 13 Indikatoren, die Topics (146 und 256) und die 105 gegenseitig besten Paare bleiben gleich. Die
dominante Konfiguration wechselt bei 14 Topics in Phase 1 und 32 in Phase 2; Einzelheiten in `CHANGELOG.md`.

### Kompatibilität

`v2.2` (Masterarbeit), `v2.3.1`, `v2.4` und `v2.4.1` bleiben unverändert bestehen.
