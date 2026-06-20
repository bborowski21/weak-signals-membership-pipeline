# F3-Pipeline: Membership-Scoring zur Detektion von Weak Signals

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20283613.svg)](https://doi.org/10.5281/zenodo.20283613)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python ≥3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/downloads/)

Reproduktionscode zur Masterarbeit *Weak Signals in Foresight: Ein Operationalisierungsframework für Frühindikatoren
potenzieller Entwicklungen* (Ben-Nicholas Borowski,
Data Science & Analytics, 2026). Die Pipeline operationalisiert das in F2
entwickelte fünfdimensionale Framework über 16 bibliometrische Indikatoren
und überführt sie in vier kontinuierliche Memberships (`m_ws`, `m_trend`,
`m_ec`, `m_latent`).

## Zitation

Wenn dieser Code in akademischen Arbeiten verwendet wird, bitte zitieren als:

> Borowski, Ben-Nicholas (2026). *F3-Pipeline: Membership-Scoring zur Detektion
> von Weak Signals* (Version v2.2) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.20283613

BibTeX:

```bibtex
@software{borowski_pipeline_2026,
  author    = {Borowski, Ben-Nicholas},
  title     = {F3-Pipeline: Membership-Scoring zur Detektion von Weak Signals},
  version   = {v2.2},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20283613},
  url       = {https://doi.org/10.5281/zenodo.20283613}
}
```

## Architektur

Die Pipeline ist als sequentielle Schritte organisiert, die in der
Masterarbeit (Kapitel 3) methodisch verankert sind. Jeder Schritt
entspricht einem Modul; gemeinsame Konfiguration in `config.py`.

| Schritt | Beschreibung | Modul |
|---------|--------------|-------|
| 0       | KATI-WoS-Datenaufbereitung; ISO-3-Normalisierung | `prepare_kati_data.py` |
| 0a      | Textbereinigung (Lemmatisierung, Token-Filter) | `text_preprocessing.py` |
| 1       | Topic Modeling (SBERT + UMAP + HDBSCAN) | `step01_topic_modeling.py` |
| 1b      | Phasenübergreifendes Topic-Matching (Hybrid-Score) | `step01b_cross_phase_matching.py` |
| 1c      | TEM-Robustheitsdiagnostik | `step01c_tem_robustness.py` |
| 2       | 16 Indikatoren über 5 Dimensionen | `step02_indicators.py` |
| 2       | Membership-Scoring (kontinuierlich, Sigmoid) | `step02_memberships.py` |
| 2b      | Zitations-Kohärenz ($\rho_t$) | `step02b_reference_overlap.py` |
| 2c      | Citation-Topic-Profil (Macro/Meso/Micro, deskriptiv) | `step02c_citation_topic_profile.py` |
| 3       | EFA (minres, Oblimin) — interne Strukturkohärenz; PCA nur als etikettierter Robustheitscheck | `step03_efa_pca.py` |
| 3b      | Externe Konstruktvalidierung (RTW/CTW) | `step03b_external_validation.py` |
| 3c      | Topic-Modell-Güte ($C_v$/$C_{\text{NPMI}}$/$C_{\text{UMass}}$, Diversität) | `step03c_topic_quality.py` |
| 4       | Phaseninterne Visualisierungen | `step04_visualizations.py` |
| 4b      | Cross-Phase-Visualisierungen | `step04b_cross_phase_viz.py` |
| 5       | OAT-Sensitivitätsanalyse ($k \times \lambda$-Grid) | `step05_sensitivity.py` |
| 5b      | Sensitivitäts-Artefakte (Vorberechnung) | `step05b_artifacts.py` |
| 5c      | Cross-Phase-Sensitivität (Hybrid-$\alpha_H$) | `step05c_cross_phase_sensitivity.py` |

Die Wrapper `run_*.py` orchestrieren die Schritte phasen- und
übergreifend. Die wesentlichen Einstiegspunkte:

```bash
python run_all_phases.py                  # Vollständige Pipeline (P1 + P2 + Cross)
python run_phase.py 1                     # Phase 1 (2000–2015), Schritte 1–3
python run_phase.py 2                     # Phase 2 (2016–2025), Schritte 1–3
python run_phase_viz.py 1                 # Visualisierungen Phase 1
python run_phase_viz.py 2                 # Visualisierungen Phase 2
python run_cross_phase_viz.py             # Cross-Phase: Sankey, Shift-Heatmap, Strukturradar
python run_phase_sensitivity.py 1         # OAT-Sensitivität Phase 1
```

### Visualisierungs-Codierungen (Schritte 4 und 4c)

Die phaseninternen und phasenübergreifenden Visualisierungen tragen die
in `step02_memberships.py` operativ verankerten Margin-Schwellen
($\Delta = m_{(1)} - m_{(2)}$) als zusätzliche Codierungsebene:

- **`dimension_heatmap.png`** — Topic-Labels enthalten `(Δ=...)` analog
  zur Membership-Heatmap. Die Eindeutigkeit der Argmax-Zuordnung pro
  Zeile bleibt damit direkt ablesbar.
- **`extended_tem.png`** — Bubble-Alpha und Outline-Stil codieren drei
  Margin-Stufen: $\Delta \geq 0{,}10$ opak/weiße Outline (klar);
  $0{,}05 \leq \Delta < 0{,}10$ transparent gestrichelt (Übergang);
  $\Delta < 0{,}05$ durchscheinend gestrichelt (mehrdeutig). Die
  Farbe codiert weiterhin den Signaltyp, die Bubble-Größe die
  Epistemische Offenheit. Quadrantenlabels sitzen in den Plot-Ecken;
  die Legende ist außerhalb des Datenbereichs platziert.
- **`migration_sankey.png`** (cross-phase) — Bänder pro
  Argmax-Migration sind in einen klaren Anteil
  ($\Delta \geq 0{,}10$ in beiden Phasen, vollflächig) und einen
  knappen Anteil ($\Delta < 0{,}10$ in P1 oder P2, gehatched `//`)
  aufgeteilt. Die argmax-reduzierte Sankey-Sicht wird damit gegenüber
  knappen Klassenwechseln transparent.

Diese Codierung implementiert die in Kapitel 5 der Masterarbeit
(Abschnitt *Dreistufige Margin-Lesart*) entwickelte Interpretationsskala
auch visuell und vermeidet, dass die argmax-Reduktion die in V2
zurückgewiesene kategoriale Reifizierung visuell reproduziert.

## Installation

Python ≥ 3.10. Empfohlen: Virtual Environment.

```bash
git clone https://github.com/bborowski21/weak-signals-membership-pipeline.git
cd weak-signals-membership-pipeline
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Daten

Die F3-Pipeline operiert auf einem Web-of-Science-Korpus, der über
das KATI-System (FKIE) zusammengestellt wurde. Aus lizenzbedingten
Gründen ist der Korpus nicht Teil des Repositories.
Die Pipeline ist jedoch reproduzierbar auf jeder WoS-Lieferung, die
die 17 in der Methoden-Sektion 3.1.1 dokumentierten Felder umfasst.

### Indikator-Datenstatus (finale Mai-2026-Lieferung)

Die finale KATI-Tranche (Mai 2026) liefert alle 17 konstitutiven
WoS-Felder, einschließlich *Author Full Names* und *Cited References*.
Damit sind **alle 16 Indikatoren aktiv** (inkl. `DI1`,
Autoren-Konzentration) und die Zitations-Kohärenzprüfung
(`step02b_reference_overlap.py`, $\rho_t$) ist auf beiden Phasen
durchgeführt (Cited-References-Abdeckung: P1 >99,9 %, P2 99,5 %).
Der dokumentierte NaN-Fallback bleibt als defensive Vorrichtung für
reduzierte Datenlieferungen erhalten.


Ein synthetischer Mini-Korpus zum Smoke-Test der Pipeline kann über
`generate_synthetic_artifacts.py` erzeugt werden:

```bash
python generate_synthetic_artifacts.py --output data/synthetic_demo.csv --n 500
```

## Konfiguration

Zentrale Hyperparameter in `config.py`:

```python
SBERT_MODEL              = "all-MiniLM-L6-v2"
HDBSCAN_MIN_CLUSTER_SIZE = 25
MEMBERSHIP_SIGMOID_K     = 1.0    # Trennschärfe-Parameter (V2)
MEMBERSHIP_LAMBDA_WP     = 0.5    # WP-Gewichtung in m_trend (V2)
HYBRID_ALPHA             = 0.6    # Cosine-Anteil im Hybrid-Score
```

Die Sensitivitätsanalyse (`step05`) variiert diese Parameter über
ein zweidimensionales $k \times \lambda$-Grid; Design und Ergebnisse
sind im Methoden- bzw. Ergebniskapitel der Masterarbeit dokumentiert.

## Repository-Struktur

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── config.py
├── prepare_kati_data.py
├── text_preprocessing.py
├── step01_topic_modeling.py
├── step01b_cross_phase_matching.py
├── step01c_tem_robustness.py
├── step02_indicators.py
├── step02_memberships.py
├── step02b_reference_overlap.py
├── step02c_citation_topic_profile.py
├── step03_efa_pca.py
├── step03b_external_validation.py
├── step04_visualizations.py
├── step04b_cross_phase_viz.py
├── step05_sensitivity.py
├── run_all.py
├── run_all_phases.py
├── run_phase.py
├── run_phase_clean.py
├── run_phase_indicators.py
├── run_phase_efa.py
├── run_phase_validation.py
├── run_phase_viz.py
├── run_phase_sensitivity.py
├── run_step02c_phases.py
├── run_cross_phase_viz.py
├── step05c_cross_phase_sensitivity.py
├── run_sensitivity_hparam.py
├── step05b_artifacts.py
├── clean_pipeline_data.py
├── generate_synthetic_artifacts.py
├── rerender_loading_matrices.py
└── render_efa_pub.py
```

## Versionierung

Die in der finalen Fassung der Masterarbeit referenzierte Version ist
über das Git-Tag `v2.2` fixiert und besitzt eine eigene Zenodo-DOI.
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

Ben-Nicholas Borowski — Master Data Science & Analytics — 2026
