# F3-Pipeline: Membership-Scoring zur Detektion von Weak Signals

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20283613.svg)](https://doi.org/10.5281/zenodo.20283613)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue.svg)](https://www.python.org/downloads/)

Reproduktionscode zur Masterarbeit *Weak Signals in Foresight: Ein Operationalisierungsframework für Frühindikatoren
potenzieller Entwicklungen* (Ben-Nicholas Borowski,
Data Science & Analytics, 2026). Die Pipeline operationalisiert das in F2
entwickelte fünfdimensionale Framework über 16 bibliometrische Indikatoren
und überführt sie in vier kontinuierliche Memberships (`m_ws`, `m_trend`,
`m_ec`, `m_latent`).

## Zitation

Zwei Arbeiten zitieren das Repositorium, jede mit eigenem Stand. Die
Masterarbeit referenziert den Stand `v2.2`, das daraus hervorgegangene Manuskript
den Stand `v2.5.1` (davor `v2.5`, `v2.4.1`, `v2.4` und `v2.3.1`). Alle Stände liegen unter
derselben Concept-DOI
[10.5281/zenodo.20283613](https://doi.org/10.5281/zenodo.20283613), die stets
auf die jeweils neueste Version auflöst; das Badge oben zeigt diese
Concept-DOI. Die versionsgenauen DOIs stehen auf der Zenodo-Seite unter
*Versions*.

Für den Stand `v2.5.1` (Manuskript) bitte zitieren als:

> Borowski, Ben-Nicholas (2026). *Weak-signal membership pipeline: topic
> model, sixteen indicators and four configurational memberships*
> (Version v2.5.1) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.23078208

BibTeX. Der Eintragstyp ist `misc`, so exportiert Zenodo selbst und so
verlangt es klassisches BibTeX; unter biblatex kann er auf `software`
geändert werden.

```bibtex
@misc{borowski_weak_2026,
  author    = {Borowski, Ben-Nicholas},
  title     = {Weak-signal membership pipeline: topic model, sixteen indicators
               and four configurational memberships},
  version   = {v2.5.1},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23078208},
  url       = {https://doi.org/10.5281/zenodo.23078208}
}
```

Für den Stand `v2.2` (Masterarbeit) gilt dieselbe Form mit
`version = {v2.2}`, dem deutschen Titel des damaligen Deposits und der
zugehörigen Versions-DOI; für `v2.3.1` mit der Versions-DOI
[10.5281/zenodo.22128929](https://doi.org/10.5281/zenodo.22128929), für `v2.4` mit
[10.5281/zenodo.23058951](https://doi.org/10.5281/zenodo.23058951), für `v2.4.1`
mit [10.5281/zenodo.23070432](https://doi.org/10.5281/zenodo.23070432), für `v2.5` mit
[10.5281/zenodo.23077271](https://doi.org/10.5281/zenodo.23077271).

## Architektur

Die Pipeline ist als sequentielle Schritte organisiert, die in der
Masterarbeit (Kapitel 3) methodisch verankert sind. Jeder Schritt
entspricht einem Modul; gemeinsame Konfiguration in `config.py`.

| Schritt | Beschreibung | Modul |
|---------|--------------|-------|
| 0       | KATI-Lieferung in WoS-Spalten (Deduplizierung nach UID, Dokumenttyp ohne „early access“, Länder in Großbuchstaben) | `prepare_kati_data.py` |
| 0a      | Textbereinigung für Schritt 1 (LaTeX, HTML, Copyright-Hinweise, Unicode NFKC, Leerraum); schreibt die `_clean`-Dateien | `text_preprocessing.py`, `clean_pipeline_data.py` |
| 1       | Topic Modeling (SBERT + UMAP + HDBSCAN) | `step01_topic_modeling.py` |
| 1b      | Phasenübergreifendes Topic-Matching (Hybrid-Score) | `step01b_cross_phase_matching.py` |
| 1c      | TEM-Robustheitsdiagnostik (nicht im berichteten Lauf) | `step01c_tem_robustness.py` |
| 2       | 16 Indikatoren über 5 Dimensionen | `step02_indicators.py` |
| 2       | Membership-Scoring (kontinuierlich, Sigmoid) | `step02_memberships.py` |
| 2b      | Zitations-Kohärenz ($\rho_t$) aus den KATI-Referenzlisten | `step02b_run_with_kati.py` (Funktionen aus `step02b_reference_overlap.py`) |
| 2c      | Citation-Topic-Profil (Macro/Meso/Micro, deskriptiv) | `step02c_citation_topic_profile.py` |
| 3       | EFA (minres, Oblimin): interne Strukturkohärenz; PCA nur als etikettierter Robustheitscheck | `step03_efa_pca.py` |
| 3b      | Externe Konstruktvalidierung (RTW/CTW) | `step03b_external_validation.py` |
| 3c      | Topic-Modell-Güte ($C_v$/$C_{\text{NPMI}}$/$C_{\text{UMass}}$, Diversität) | `step03c_topic_quality.py` |
| 4       | Phaseninterne Visualisierungen | `step04_visualizations.py` |
| 4b      | Cross-Phase-Visualisierungen (nicht im berichteten Lauf) | `step04b_cross_phase_viz.py` |
| 5       | OAT-Sensitivitätsanalyse ($k \times \lambda$-Grid) | `step05_sensitivity.py` |
| 5b      | Sensitivitäts-Artefakte (Vorberechnung) | `step05b_artifacts.py` |
| 5c      | Cross-Phase-Sensitivität (Hybrid-$\alpha_H$) | `step05c_cross_phase_sensitivity.py` |
| 6       | Übergänge der dominanten Konfiguration zwischen den Phasen | `step06_cross_phase_transitions.py` |

Die Wrapper `run_*.py` orchestrieren die Schritte phasen- und
übergreifend. Die wesentlichen Einstiegspunkte:

```bash
python run_all_phases.py                  # Schritte 1 bis 5c beider Phasen (ohne 2b, 2c, 3c, 6 und Robustheit)
python run_phase.py 1                     # Schritt 1 (Topic Modeling), Phase 1 (2000–2015)
python run_phase.py 2                     # Schritt 1 (Topic Modeling), Phase 2 (2016–2025)
python run_phase_viz.py 1                 # Visualisierungen Phase 1
python run_phase_viz.py 2                 # Visualisierungen Phase 2
python run_phase_sensitivity.py 1         # OAT-Sensitivität Phase 1
```

### Der im Manuskript berichtete Lauf

Der im Manuskript berichtete Lauf (30.09.2026, Stand v2.5) ist mit dem Treiber
`berichteter_lauf/neulauf_voll.py` gerechnet, ab Schritt 2 auf den Einheiten des
vollständigen Neulaufs vom 29./30.09.2026; dort beschreibt eine README Stufen,
Aufruf, Ordnerannahmen und die Entstehung des Laufs. Der Treiber rechnet die
Schritte von `run_all_phases.py`
(1 und 1b mit `--with-sbert`, dann 2 bis 5c) und dazu, in dieser Reihenfolge,
`prepare_kati_data.py` und `clean_pipeline_data.py` vor Schritt 1,
`step02b_run_with_kati.py`, `step03c_topic_quality.py` und
`pruefe_reproduzierbarkeit.py` nach Schritt 1b, `run_step02c_phases.py`,
`step06_cross_phase_transitions.py`, `perturbation_experiment.py`,
`nullmodel_experiment.py`, `nullmodel_class_shares.py`, `standardisation_variants.py`,
`run_phase_boundary_stability.py`, `make_figS16.py`, `paper_figures_rp.py` und
`build_supplement.py` nach Schritt 5c. Tabelle S3 des Supplements erzeugt
`make_table_s3.py` aus dem Laufordner.

### Zwei Betriebsmodi des Cross-Phase-Matchings

`step01b_cross_phase_matching.py` kennt zwei Cosine-Quellen, und seit v2.3.1
sind beide an den Artefakten selbst ablesbar (Spalte `cosine_source` in den
`topic_matches`-CSVs sowie in `sensitivity_hybrid_alpha.csv`):

| Modus | Aufruf | Cosine-Quelle | Zweck |
|---|---|---|---|
| methodenkonform | `--with-sbert` (Standard über `run_all_phases.py`) | SBERT-Topic-Zentroide (Gl. 3.2 der Arbeit) | Weiterentwicklung, Paper |
| Thesis-Reproduktion | ohne Flag | Vektoren der führenden Topic-Begriffe aus Schritt 1 (TF-IDF; `cosine_source` = `ctfidf`) | exakte Reproduktion der in der Masterarbeit berichteten Zahlen (Tag `v2.2`) |

Hintergrund und Zahlenvergleich beider Modi: `CHANGELOG.md`, Einträge v2.3
und v2.3.1.

### Visualisierungs-Codierungen (Schritte 4 und 4c)

Die phaseninternen und phasenübergreifenden Visualisierungen tragen die
in `step02_memberships.py` operativ verankerten Margin-Schwellen
($\Delta = m_{(1)} - m_{(2)}$) als zusätzliche Codierungsebene:

- **`dimension_heatmap.png`**: Topic-Labels enthalten `(Δ=...)` analog
  zur Membership-Heatmap. Die Eindeutigkeit der Argmax-Zuordnung pro
  Zeile bleibt damit direkt ablesbar.
- **`extended_tem.png`**: Bubble-Alpha und Outline-Stil codieren drei
  Margin-Stufen: $\Delta \geq 0{,}10$ opak/weiße Outline (klar);
  $0{,}05 \leq \Delta < 0{,}10$ transparent gestrichelt (Übergang);
  $\Delta < 0{,}05$ durchscheinend gestrichelt (mehrdeutig). Die
  Farbe codiert weiterhin den Signaltyp, die Bubble-Größe die
  Epistemische Offenheit. Quadrantenlabels sitzen in den Plot-Ecken;
  die Legende ist außerhalb des Datenbereichs platziert.
- **`migration_sankey.png`** (cross-phase): Bänder pro
  Argmax-Migration sind in einen klaren Anteil
  ($\Delta \geq 0{,}10$ in beiden Phasen, vollflächig) und einen
  knappen Anteil ($\Delta < 0{,}10$ in P1 oder P2, gehatched `//`)
  aufgeteilt. Die argmax-reduzierte Sankey-Sicht wird damit gegenüber
  knappen Klassenwechseln transparent.

Diese Codierung implementiert die in Kapitel 5 der Masterarbeit
(Abschnitt *Dreistufige Margin-Lesart*) entwickelte Interpretationsskala
auch visuell und vermeidet, dass die argmax-Reduktion die in V2
zurückgewiesene kategoriale Reifizierung visuell reproduziert. Die
Abbildungen des Manuskripts baut `paper_figures_rp.py` (Stil in `rp_style.py`).

## Installation

Python ≥ 3.11. Empfohlen: Virtual Environment.

```bash
git clone https://github.com/bborowski21/weak-signals-membership-pipeline.git
cd weak-signals-membership-pipeline
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Die exakte Umgebung des im Manuskript berichteten Laufs (Python 3.12.7,
macOS arm64, 64 Distributionen) steht in `requirements.lock.txt`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
```

## Daten

Die F3-Pipeline operiert auf einem Web-of-Science-Korpus, der über
das KATI-System (FKIE) zusammengestellt wurde. Aus lizenzbedingten
Gründen ist der Korpus nicht Teil des Repositories.
Die Pipeline ist jedoch reproduzierbar auf jeder WoS-Lieferung, die
die 17 in der Methoden-Sektion 3.1.1 dokumentierten Felder umfasst.

### Indikator-Datenstatus (finale Mai-2026-Lieferung)

Die finale KATI-Tranche (Mai 2026) liefert die konstitutiven WoS-Felder
einschließlich *Author Full Names*; damit sind **alle 16 Indikatoren aktiv**
(inkl. `DI1`, Autoren-Konzentration). Die Referenzen liegen als eigene
Listen je Phase vor (`QC_2000-2015 References.csv`, `QC_2016-2025 References.csv`);
aus ihnen rechnet `step02b_run_with_kati.py` die Zitations-Kohärenz $\rho_t$.
`step02b_reference_overlap.py` liest stattdessen das WoS-Feld *Cited References*,
das `prepare_kati_data.py` leer lässt, und liefert auf KATI-Daten deshalb keine
Werte; es bleibt für WoS-Exporte mit diesem Feld erhalten. Der dokumentierte
NaN-Fallback bleibt als defensive Vorrichtung für reduzierte Datenlieferungen
erhalten.

Synthetische Artefakte der Schritte 1 und 2 für einen Smoke-Test erzeugt
`generate_synthetic_artifacts.py` (nach `output_smoke/`):

```bash
python generate_synthetic_artifacts.py --output-dir output_smoke --n-docs 500
```

Der danach angezeigte Folgeschritt (`run_sensitivity_hparam.py`, ebenso
`--run-sensitivity`) ist in v2.5.1 nicht lauffähig.

## Konfiguration

Zentrale Hyperparameter in `config.py`:

```python
SBERT_MODEL              = "all-MiniLM-L6-v2"
SBERT_MODEL_REVISION     = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # Hub-Commit (ab v2.4)
HDBSCAN_MIN_CLUSTER_SIZE = 25
MEMBERSHIP_SIGMOID_K     = 1.0    # Trennschärfe-Parameter (V2)
MEMBERSHIP_LAMBDA_WP     = 0.5    # WP-Gewichtung in m_trend (V2)
REVIEW_ABSENCE_ALPHA     = 5      # Stärke der Glättung von DS3: 2 * alpha
REVIEW_ABSENCE_PRIOR     = "phase_share"  # Mittelwert: Review-Anteil der Phase (ab v2.4)
SENSITIVITY_HYBRID_ALPHA_GRID = [0.4, 0.6, 0.8]  # alpha-Grid der Cross-Phase-Sensitivitaet
```

Das Cosine-Gewicht des Hybrid-Scores selbst ist kein `config.py`-Parameter,
sondern liegt als `DEFAULT_ALPHA = 0.6` in `step01b_cross_phase_matching.py`
und ist dort über `--alpha` einstellbar.

Die Sensitivitätsanalyse (`step05`) variiert diese Parameter über
ein zweidimensionales $k \times \lambda$-Grid; Design und Ergebnisse
sind im Methoden- bzw. Ergebniskapitel der Masterarbeit dokumentiert.

## Repository-Struktur

| Bereich | Dateien |
|---|---|
| Dokumentation | `README.md`, `CHANGELOG.md`, `RELEASE_v2.3.md`, `RELEASE_v2.3.1.md`, `RELEASE_v2.4.md`, `RELEASE_v2.4.1.md`, `RELEASE_v2.5.md`, `RELEASE_v2.5.1.md`, `CITATION.cff`, `LICENSE` |
| Umgebung | `requirements.txt`, `requirements.lock.txt`, `config.py`, `.gitignore` |
| Daten | `prepare_kati_data.py`, `text_preprocessing.py`, `clean_pipeline_data.py` |
| Pipeline | `step01_topic_modeling.py` bis `step06_cross_phase_transitions.py` (Tabelle oben), Wrapper `run_all_phases.py`, `run_phase.py`, `run_phase_indicators.py`, `run_phase_efa.py`, `run_phase_validation.py`, `run_phase_viz.py`, `run_phase_sensitivity.py`, `run_step02c_phases.py` |
| Robustheit | `perturbation_experiment.py`, `nullmodel_experiment.py`, `nullmodel_class_shares.py`, `standardisation_variants.py`, `run_phase_boundary_stability.py` |
| Manuskript | `paper_figures_rp.py`, `rp_style.py`, `make_figS16.py`, `build_supplement.py`, `make_table_s3.py` |
| Belege des berichteten Laufs | `pruefe_reproduzierbarkeit.py`, `reproduzierbarkeit_phase1.json`, `reproduzierbarkeit_phase2.json`, `topic_quality_results.json`, `output_phase_boundary/phase_boundary_stability.json`, `berichteter_lauf/` (Treiber und README) |
| Nicht im berichteten Lauf | `run_all.py` (veraltet), `run_phase_clean.py`, `run_sensitivity_hparam.py`, `run_wp1_rightedge.py`, `run_cross_phase_viz.py`, `step01c_tem_robustness.py`, `step04b_cross_phase_viz.py`, `paper_figures.py`, `plot_style.py`, `rerender_pub.py`, `rerender_loading_matrices.py`, `render_efa_pub.py`, `generate_ws_topic_tables.py`, `analyse_konfidenz.py`, `generate_synthetic_artifacts.py` |

## Versionierung

Die in der finalen Fassung der Masterarbeit referenzierte Version ist
über das Git-Tag `v2.2` fixiert und besitzt eine eigene Zenodo-DOI.
Das daraus hervorgegangene Manuskript referenziert `v2.5.1` (davor `v2.5`,
`v2.4.1`, `v2.4` und `v2.3.1`); Versionen mit GitHub-Release archiviert Zenodo
jeweils mit eigener DOI. Alle Tags bleiben unverändert bestehen. v2.5.1 ändert
keine Rechnung: Die führenden Begriffe je Topic aus Schritt 1 heißen in der
Dokumentation, in der README der Datendateien und in Abbildung S1 wie im
Manuskript TF-IDF-Begriffe; klassenbasiertes c-TF-IDF rechnet nur Schritt 3c
für die Topic-Güte. v2.5 rechnet EO1
(`keyword_volatility`) und DS2 (`terminological_instability`) als gewichtete
Jaccard-Distanz der relativen Keyword-Häufigkeiten und setzt für nicht
bestimmbare Werte von EO1, DS2 und IP2 den Median der Phase ein; der
berichtete Lauf ist mit diesem Stand gerechnet, seine Zahlen weichen ab
Schritt 2 von v2.4.1 ab. v2.4.1 berichtigt die Faktorkorrelationen und
Kommunalitäten der EFA, zwei Rechenwege der Sensitivitätsanalyse und drei
Stellen ohne Wirkung auf den berichteten Lauf, nimmt die Skripte für
Abbildungen, Tabelle S3 und die Klassenanteile unter den Nullmodellen sowie
den Treiber des berichteten Laufs auf und berichtigt die Beschreibung;
keine Zahl des Manuskripts ändert sich. v2.4 behebt die Zählung der Reviews
in DS3, stellt die Glättung von DS3 auf den Review-Anteil der Phase um und
hält Laufumgebung und Revision des Sprachmodells fest; Einzelheiten in
`CHANGELOG.md`.
v2.2 vereinheitlicht die Step-Benennung auf ein durchgängig
sequenzielles Schema und ergänzt das Diagnostikmodul
`step02c_citation_topic_profile.py` (deskriptive Citation-Topic-
Charakterisierung je Topic, ohne Indikatorwirkung) samt Zwei-Phasen-
Wrapper `run_step02c_phases.py`. v2.1 stellt Schritt 3 von einer
PCA-Realisierung auf eine gemeinsame Faktorenanalyse um (minres-
Extraktion, Oblimin-Rotation; Pattern- und Faktorkorrelationsmatrizen,
Horn-Parallelanalyse auf der unreduzierten Korrelationsmatrix mit
dokumentierter SMC-Diagnostik); die PCA bleibt als etikettierter
Robustheitscheck erhalten. v1.2 erweiterte v1.1 um
das Topic-Modell-Güte-Modul (`step03c_topic_quality.py`). Spätere
Weiterentwicklungen erscheinen unter weiteren Tags, ohne den
zitierten Stand zu modifizieren.

## Lizenz

MIT-Lizenz (siehe `LICENSE`). Der Code darf in Forschung und Lehre
frei verwendet werden; eine Zitation der Masterarbeit oder dieses
Repositories ist erbeten.

## Autor

Ben-Nicholas Borowski, Master Data Science & Analytics, 2026
