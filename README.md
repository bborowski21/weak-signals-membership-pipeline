# F3-Pipeline: Membership-Scoring zur Detektion von Weak Signals

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20283613.svg)](https://doi.org/10.5281/zenodo.20283613)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python ≥3.10](https://img.shields.io/badge/python-%E2%89%A53.10-blue.svg)](https://www.python.org/downloads/)

Reproduktionscode zur Masterarbeit *Weak Signals in Foresight: Eine
mehrdimensionale, membership-basierte Operationalisierung* (Ben Borowski,
Data Science & Analytics, 2026). Die Pipeline operationalisiert das in F2
entwickelte fünfdimensionale Framework über 16 bibliometrische Indikatoren
und überführt sie in vier kontinuierliche Memberships (`m_ws`, `m_trend`,
`m_ec`, `m_latent`).

## Zitation

Wenn dieser Code in akademischen Arbeiten verwendet wird, bitte zitieren als:

> Borowski, B. (2026). *F3-Pipeline: Membership-Scoring zur Detektion
> von Weak Signals* (Version v1.0-thesis) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.20283613

BibTeX:

```bibtex
@software{borowski_pipeline_2026,
  author    = {Borowski, Ben},
  title     = {F3-Pipeline: Membership-Scoring zur Detektion von Weak Signals},
  version   = {v1.0-thesis},
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
| 1c      | Phasenübergreifendes Topic-Matching (Hybrid-Score) | `step01c_cross_phase_matching.py` |
| 1d      | TEM-Robustheitsdiagnostik | `step01d_tem_robustness.py` |
| 2       | 16 Indikatoren über 5 Dimensionen | `step02_indicators.py` |
| 2b      | Zitations-Kohärenz ($\rho_t$) | `step02b_reference_overlap.py` |
| 2b'     | Membership-Scoring (kontinuierlich, Sigmoid) | `step02b_memberships.py` |
| 3       | EFA/PCA — interne Strukturkohärenz | `step03_efa_pca.py` |
| 3b      | Externe Konstruktvalidierung (RTW/CTW) | `step03b_external_validation.py` |
| 4       | Phaseninterne Visualisierungen | `step04_visualizations.py` |
| 4c      | Cross-Phase-Visualisierungen | `step04c_cross_phase_viz.py` |
| 5       | OAT-Sensitivitätsanalyse ($k \times \lambda$-Grid) | `step05_sensitivity.py` |

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
das KATI-System (FKIE) zusammengestellt wurde. Aus lizenz- und
embargobedingten Gründen ist der Korpus nicht Teil des Repositories.
Die Pipeline ist jedoch reproduzierbar auf jeder WoS-Lieferung, die
die 17 in der Methoden-Sektion 3.1.2 dokumentierten Felder umfasst.

Ein synthetischer Mini-Korpus zum Smoke-Test der Pipeline kann über
`generate_synthetic_artifacts.py` erzeugt werden:

```bash
python generate_synthetic_artifacts.py --output data/synthetic_demo.csv --n 500
```

## Konfiguration

Zentrale Hyperparameter in `config.py`:

```python
SBERT_MODEL              = "all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE         = 15
MEMBERSHIP_SIGMOID_K     = 1.0    # Trennschärfe-Parameter (V2)
MEMBERSHIP_LAMBDA_WP     = 0.5    # WP-Gewichtung in m_trend (V2)
HYBRID_ALPHA             = 0.6    # Cosine-Anteil im Hybrid-Score
```

Die Sensitivitätsanalyse (`step05`) variiert diese Parameter über
ein zweidimensionales $k \times \lambda$-Grid, dokumentiert in
Tabelle 3.6 der Masterarbeit.

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
├── step01c_cross_phase_matching.py
├── step01d_tem_robustness.py
├── step02_indicators.py
├── step02b_memberships.py
├── step02b_reference_overlap.py
├── step03_efa_pca.py
├── step03b_external_validation.py
├── step04_visualizations.py
├── step04c_cross_phase_viz.py
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
├── run_cross_phase_viz.py
├── run_cross_phase_sensitivity.py
├── run_sensitivity_hparam.py
├── build_step5_artifacts.py
├── clean_pipeline_data.py
├── generate_synthetic_artifacts.py
└── rerender_loading_matrices.py
```

## Versionierung

Die in der Masterarbeit referenzierte Version ist über das Git-Tag
`v1.0-thesis` fixiert und besitzt eine eigene Zenodo-DOI. Spätere
Weiterentwicklungen erscheinen unter weiteren Tags (`v1.1`, `v2.0`),
ohne den Thesis-Stand zu modifizieren.

## Lizenz

MIT-Lizenz (siehe `LICENSE`). Der Code darf in Forschung und Lehre
frei verwendet werden; eine Zitation der Masterarbeit oder dieses
Repositories ist erbeten.

## Autor

Ben Borowski — Master Data Science & Analytics — 2026
