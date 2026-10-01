# Der im Manuskript berichtete Lauf

`neulauf_voll.py` ist der Treiber, mit dem der im Manuskript berichtete Lauf gerechnet ist, in zwei Varianten
von DS3. Er liegt hier unverändert (SHA-256 beginnt mit `8db27e5c`, wie in den Laufprotokollen `herkunft.json`)
und belegt, welche Skripte in welcher Reihenfolge und mit welchen Einstellungen liefen. Die Pipeline-Skripte ruft
er unverändert auf, jeden Schritt in einem eigenen Python-Prozess; er setzt dort nur die Pfade auf den Laufordner
und die Variante von DS3 in `config.REVIEW_ABSENCE_PRIOR`.

Seit v2.5 berichtet das Manuskript den Lauf vom 30.09.2026 (Laufordner `output_neulauf_v2.5`, siehe „Der Lauf
v2.5“ unten). Er rechnet ab Schritt 2 neu und übernimmt die Einheiten des vollständigen Neulaufs vom
29./30.09.2026 (`output_neulauf_voll_2026-09-29`, ab der Datenaufbereitung gerechnet, berichtet mit v2.4 und
v2.4.1).

Hinweis zur Benennung: Im Treiber heißt „berichteter Lauf“ noch der ältere Lauf vom Mai 2026 (Ausgaben in
`output_phase1/` und `output_phase2/` des Pipeline-Ordners), gegen den er die neuen Einheiten prüft.

## Stufen

| Stufe | Inhalt | Skripte |
|---|---|---|
| `daten` | KATI-Lieferung in WoS-Spalten, Textbereinigung, Abgleich | `prepare_kati_data.py`, `clean_pipeline_data.py` |
| `einheiten` | Schritt 1 je Phase, Matching 1b mit SBERT-Zentroiden, Referenzkohärenz 2b aus den KATI-Referenzlisten, Topic-Güte 3c, Reproduzierbarkeitsprüfung | `run_phase.py`, `step01b_cross_phase_matching.py --with-sbert`, `step02b_run_with_kati.py`, `step03c_topic_quality.py`, `pruefe_reproduzierbarkeit.py` |
| `saat` | Übernahme der Einheiten in jede Variante, per Prüfsumme bestätigt | Treiber |
| `pipeline` | Schritte 2, 3, 3b, 4, 5b und 5 je Phase, dann 5c | `run_phase_indicators.py`, `run_phase_efa.py`, `run_phase_validation.py`, `run_phase_viz.py`, `step05b_artifacts.py`, `run_phase_sensitivity.py`, `step05c_cross_phase_sensitivity.py` |
| `profil` | Citation-Topic-Profil 2c (deskriptiv) | `run_step02c_phases.py` |
| `robustheit` | Übergänge zwischen den Phasen (6), Perturbation, zwei Nullmodelle, Klassenanteile unter den Nullmodellen, Standardisierungsvarianten | `step06_cross_phase_transitions.py`, `perturbation_experiment.py`, `nullmodel_experiment.py`, `nullmodel_class_shares.py`, `standardisation_variants.py` |
| `grenze` | Stabilität gegenüber der Phasengrenze | `run_phase_boundary_stability.py` |
| `abbildungen` | Abbildung S16, Abbildungen des Manuskripts | `make_figS16.py`, `paper_figures_rp.py` |
| `supplement` | Datendateien des Supplements | `build_supplement.py` |
| `pruefung` | Einheiten gegen den älteren Lauf, Reproduzierbarkeit in derselben Umgebung, DS3 unabhängig nachgerechnet, Kontrolle, dass außerhalb des Laufordners nichts geändert wurde | Treiber |

Tabelle S3 des Supplements lässt sich mit `make_table_s3.py` aus dem Laufordner der Variante `prior`
erzeugen; das Ergebnis ist bytegleich mit der Tabelle im Supplement.

## Varianten

Je Variante ein Unterordner: `prior` (Glättung von DS3 zum Review-Anteil der Phase, `"phase_share"`, im
Manuskript berichtet) und `alpha5` (symmetrischer Prior, `"symmetric"`, zum Vergleich).

## Der Lauf v2.5

Der Lauf vom 30.09.2026 übernimmt aus dem vollständigen Neulauf vom 29./30.09.2026 die Phasendateien und den
Ordner `einheiten/` (Stufen `daten` und `einheiten`) unverändert, 32 Dateien samt Treiber, per SHA-256
bestätigt (Liste in `uebernahme_aus_lauf_2026-09-29.json` im Laufordner), und rechnet die Stufen `saat` bis
`pruefung` für beide Varianten neu. Gerechnet hat derselbe Treiber auf einer Kopie des Codes von v2.4.1, in der
nur `step02_indicators.py` geändert war, in der Fassung von v2.5 (EO1 und DS2 als gewichtete Jaccard-Distanz,
Median der Phase für nicht bestimmbare Werte; Einzelheiten in `CHANGELOG.md`). Von den 53 Skripten, die das
Laufprotokoll aus diesem Code-Ordner erfasst, haben 52 den Stand von v2.5; `make_table_s3.py` trägt in v2.5 die
Beschriftung von Tabelle S3 aus dem Supplement und wird vom Treiber nicht aufgerufen. Die Abbildungen baute in
Schritt A.2 die damals außerhalb des Repos gepflegte Fassung von `paper_figures_rp.py`, die sich von der des
Repos nur im Docstring und in FigA9 unterscheidet; FigA9 kam in Schritt A.3 aus der Fassung des Repos. Der
Treiber meldet nach beiden Varianten keine Änderung außerhalb des Laufordners.

Mit diesem Stand lässt sich der Lauf wiederholen, sofern der Lauf vom 29./30.09.2026 vorliegt:

```bash
mkdir output_neulauf_v2.5 && cp berichteter_lauf/neulauf_voll.py output_neulauf_v2.5/
cp -Rp output_neulauf_voll_2026-09-29/einheiten output_neulauf_voll_2026-09-29/wos_qc_phase*.csv output_neulauf_v2.5/
python output_neulauf_v2.5/neulauf_voll.py --nur saat,pipeline,profil,robustheit,grenze,abbildungen,supplement,pruefung
```

## Aufruf

Der Treiber erwartet, in einem Laufordner innerhalb des Pipeline-Ordners zu liegen: Standard für `--repo`
ist der übergeordnete Ordner, geschrieben wird nur in den eigenen Ordner. Ordner `output_*` sind in
`.gitignore` erfasst.

```bash
mkdir output_neulauf && cp berichteter_lauf/neulauf_voll.py output_neulauf/
python output_neulauf/neulauf_voll.py --trocken      # nur den Plan zeigen
python output_neulauf/neulauf_voll.py --kati "<KATI-Ordner>" --publikation "<Ordner>"
```

Einzelne Stufen lassen sich mit `--nur`, eine Variante mit `--variante prior` und ein Wiedereinstieg in
die Pipeline mit `--ab-schritt` (etwa `3.1`) und `--bis-schritt` wählen.

## Annahmen über Ordner außerhalb des Repos

- `--kati`: die KATI-Lieferung beider Phasen samt den Referenzlisten `QC_2000-2015 References.csv` und
  `QC_2016-2025 References.csv`. Lizenzierte Daten, nicht Teil des Repos.
- `--publikation`: Beide Läufe nahmen dort `Analysen/Abbildungen_RP/paper_figures_rp.py` und `rp_style.py`
  (Stufe `abbildungen`, Schritt A.2) sowie `Analysen/Nullmodell/nullmodel_class_shares.py` (Stufe
  `robustheit`, Schritt R.3); Abbildung A9 kam jeweils aus der Fassung im Pipeline-Ordner (Schritt A.3). Seit
  v2.4.1 liegen diese Skripte im Repo, und `paper_figures_rp.py` des Repos baut alle Abbildungen einschließlich
  A9. Für einen Lauf außerhalb dieser Ordnerstruktur die drei Dateien aus dem Repo in einen Ordner mit derselben
  Unterstruktur legen und diesen mit `--publikation` übergeben.
- Die Stufen `einheiten` und `pruefung` vergleichen mit dem älteren Lauf in `output_phase1/` und
  `output_phase2/` des Pipeline-Ordners und setzen dessen Ausgaben voraus.
- `--sicherung`: Sicherung des älteren Laufs; wird nur angelegt, wenn sie fehlt.

## Nachträge

- Lauf vom 29./30.09.2026: Nach dem Systemcheck vom 30.09.2026 ist Schritt 3 (EFA) der Variante `prior` mit
  dem Stand v2.4.1 neu gerechnet (`--nur pipeline --variante prior --ab-schritt 3.1 --bis-schritt 3.2`), wegen
  der Korrektur der Faktorkorrelationen und Kommunalitäten; Einzelheiten in `CHANGELOG.md`. Das Laufprotokoll
  vor diesem Schritt liegt im Laufordner als `herkunft_vor_systemcheck.json`.
