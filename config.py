"""
Zentrale Konfiguration für die gesamte F3-Pipeline.
Alle Parameter an einer Stelle — hier ändern, überall wirksam.
"""

from pathlib import Path

# =============================================================================
# PFADE
# =============================================================================

# Basis-Verzeichnis: Dort wo dieses Script liegt
BASE_DIR = Path(__file__).parent

# Input: WoS CSV (Quantum Computing, zwei Phasen)
# Phase 1: 2000-2015, Phase 2: 2016-2025
# Die Pipeline wird pro Phase separat ausgeführt; Konfiguration der Phase
# erfolgt per Umschalten dieses Pfads UND der Phasen-Jahresgrenzen
# (PHASE_YEAR_MIN / PHASE_YEAR_MAX).
DATA_PATH = BASE_DIR.parent / "wos_qc_phase2_2016_2025.csv"

# Phasen-Jahresgrenzen (inklusiv). Diese Werte werden in
# step01_topic_modeling.load_and_clean() und step02_indicators.run() als
# Year-Filter auf den Roh-Korpus angewandt, BEVOR Topics gebildet bzw.
# Indikatoren berechnet werden. Default-Werte korrespondieren zu DATA_PATH
# (Phase 2). Die Wrapper run_phase.py / run_phase_clean.py patchen diese
# Werte zur Laufzeit phasen-spezifisch (Phase 1: 2000-2015, Phase 2: 2016-2025).
#
# Funktionale Wirkung: Records mit Year < PHASE_YEAR_MIN oder
# Year > PHASE_YEAR_MAX werden vor allen analytischen Schritten verworfen.
# Damit ist der Phasenvergleich strikt auf das jeweilige Fenster beschränkt
# und WoS-Indexierungslatenz (Early-Access-Records mit PY > PHASE_YEAR_MAX)
# wirkt sich nicht auf Topic-Bildung oder Indikatoren aus.
PHASE_YEAR_MIN = 2016
PHASE_YEAR_MAX = 2025

# WoS-spezifische Spaltennamen (canonical mapping; werden im Ingest benutzt)
WOS_COLUMNS = {
    "year":         "Publication Year",
    "authors":      "Author Full Names",
    "authors_short":"Authors",
    "orcids":       "ORCIDs",
    "affiliations": "Affiliations",
    "addresses":    "Addresses",
    "source":       "Source Title",
    "doctype":      "Document Type",
    "wos_cat":      "WoS Categories",
    "author_kw":    "Author Keywords",
    "keywords_plus":"Keywords Plus",
    "times_cited":  "Times Cited, WoS Core",
    "funding_orgs": "Funding Orgs",
    "cited_refs":   "Cited References",
    # Clarivate-vorberechnete Felder für die externe Validierung (Schritt 3b):
    "rtw":          "RTW",   # Reference Topic Width
    "ctw":          "CTW",   # Citation Topic Width
}

# Output: Alle Ergebnisse landen hier
OUTPUT_DIR = BASE_DIR / "output"

# =============================================================================
# SCHRITT 1: TOPIC MODELING
# =============================================================================

# SBERT Modell — all-MiniLM-L6-v2 ist der Standard bei Ebadi et al. (2026)
# Alternativen: "all-mpnet-base-v2" (besser, aber langsamer)
#               "allenai/specter2" (speziell für wissenschaftliche Papers)
SBERT_MODEL = "all-MiniLM-L6-v2"

# UMAP Parameter
UMAP_N_COMPONENTS = 15      # Zieldimensionen (15 = BERTopic-Standard)
UMAP_N_NEIGHBORS = 15       # Lokale Nachbarschaft
UMAP_MIN_DIST = 0.0         # Kompakte Cluster für HDBSCAN
UMAP_METRIC = "cosine"      # Cosine für Textembeddings

# HDBSCAN Parameter
HDBSCAN_MIN_CLUSTER_SIZE = 25   # Min. Dokumente pro Topic
HDBSCAN_MIN_SAMPLES = 8        # Kernsample-Dichte
HDBSCAN_CLUSTER_METHOD = "eom"  # "eom" (Excess of Mass) oder "leaf"

# c-TF-IDF
CTFIDF_TOP_N_WORDS = 15     # Keywords pro Topic

# =============================================================================
# SCHRITT 2: INDIKATOREN
# =============================================================================

CURRENT_YEAR = 2026

# Globales Beobachtungsfenster (phasenidentisch)
# Wird für Temporale Neuheit (EP1) und Altersnormierung (DI4, WP2) genutzt.
Y_MIN = 2000
Y_MAX = 2025
Y_CUTOFF = 2025              # Referenzjahr der Altersnormierung

# Bayesianische Glättung der Review-Absenz (EP3)
REVIEW_ABSENCE_ALPHA = 5

# Indikator-Dimensionszuordnung
INDICATOR_DIMENSIONS = {
    "Epistemische Offenheit": [
        "keyword_volatility",
        "disciplinary_entropy",
        "semantic_incoherence",
    ],
    "Wahrnehmbarkeit": [
        "relative_proportion_inv",
        "noise_ratio",
        "journal_specificity",
    ],
    "Entwicklungsphase": [
        "temporal_novelty",
        "terminological_instability",
        "review_absence",
    ],
    "Diffusion": [
        "author_concentration",
        "institutional_concentration",
        "geographic_concentration",
        "citation_concentration",
    ],
    "Wirkungspotenzial": [
        "growth_rate",
        "citation_momentum",
        "field_breadth",
    ],
}

# Alle Dimensionsnamen in fester Reihenfolge
DIM_NAMES = list(INDICATOR_DIMENSIONS.keys())

# Kurzcodes der Dimensionen — fuer Achsenbeschriftungen, Annotationen und
# Plot-Legenden. Interne DataFrame-Spaltennamen bleiben die langen Bezeichner.
DIM_SHORT_CODES = {
    "Epistemische Offenheit": "EO",
    "Wahrnehmbarkeit":        "WN",
    "Entwicklungsphase":      "EP",
    "Diffusion":              "DI",
    "Wirkungspotenzial":      "WP",
}
DIM_SHORT_LIST = [DIM_SHORT_CODES[d] for d in DIM_NAMES]

# =============================================================================
# SCHRITT 4: EFA/PCA
# =============================================================================

EFA_MIN_VARIANCE = 0.01     # Indikatoren mit std < dies werden ausgeschlossen
EFA_PARALLEL_N_ITER = 200   # Iterationen für Parallelanalyse

# =============================================================================
# SCHRITT 2b: MEMBERSHIP-SCORES (Pipeline V2 — Membership-Paradigma)
# =============================================================================
# Kontinuierliche Klassen-Memberships pro Topic statt harter Klassifikation.
# Theoretischer Anker:
#   - Hiltunen 2008: WS = hohe Unsicherheit über alle Kerndimensionen (EO, WN,
#                    EP, DI); WP konstitutiv NICHT Teil des WS-Begriffs.
#   - Rotolo et al. 2015: EC = Novelty (EP1) + Growth (WP1) + Impact (WP2) +
#                         Field-Breadth (WP3).
# Mathematik (vgl. step02b_memberships.py):
#   z_d(t) = (ind_d(t) − Median_d) / max(IQR_d/2, eps)        (robust z-Score)
#   sigma(x) = 1 / (1 + exp(−k · x))                          (Sigmoid)
#   m_WS    = sigma( mean(z_EO, z_WN, z_EP, z_DI) )
#   m_Trend = sigma( −mean(z_EO, z_WN, z_EP, z_DI) + λ · z_WP )
#   m_EC    = sigma( mean(z_EP1, z_WP1, z_WP2, z_WP3) )
#   m_Lat   = sigma( −mean(z_EO, z_WN, z_EP, z_DI, z_WP) )

MEMBERSHIP_SIGMOID_K = 1.0     # Sigmoid-Skalenparameter (1.0 = sanfte Trennung)
MEMBERSHIP_LAMBDA_WP = 0.5     # WP-Gewicht im Trend-Score

# =============================================================================
# VISUALISIERUNG
# =============================================================================

SIGNAL_COLORS = {
    "Weak Signal": "#E74C3C",
    "Emerging Concept": "#F39C12",
    "Trend": "#3498DB",
    "Latent/Mixed": "#95A5A6",
}

DIM_COLORS = {
    "Epistemische Offenheit": "#E74C3C",
    "Wahrnehmbarkeit": "#3498DB",
    "Entwicklungsphase": "#2ECC71",
    "Diffusion": "#F39C12",
    "Wirkungspotenzial": "#9B59B6",
}

FIG_DPI = 150

# =============================================================================
# SCHRITT 5b: SENSITIVITÄTSANALYSE
# =============================================================================
# Die Grids sind auf drei Ebenen organisiert (vgl. 03_methods.tex,
# Abschnitt "Schritt 5: Sensitivitätsanalyse"):
#   (i)   Einheitenbildung  — BERTopic-Hyperparameter
#   (ii)  Indikatorkonstruktion — Bayesianische Glättung α (EP3)
#   (iii) Klassifikationsregel — 2D-Grid k* × q*

# (i) Einheitenbildung — BERTopic-Hyperparameter-Grids
#     Stabilitätsmaß: Spearman-ρ der Indikator-Perzentile vs. Referenzlauf.
#     ρ > 0.9 → robust; ρ < 0.7 → explizite Reflexion in der Interpretation.
SENSITIVITY_MIN_CLUSTER_GRID     = [15, 20, 25, 30, 40]  # HDBSCAN min_cluster_size
SENSITIVITY_MIN_SAMPLES_GRID     = [5, 8, 12]            # HDBSCAN min_samples
SENSITIVITY_MIN_TOPIC_SIZE_GRID  = [5, 10, 15, 20]       # BERTopic min_topic_size
SENSITIVITY_N_NEIGHBORS_GRID     = [10, 15, 30]          # UMAP n_neighbors

# (ii) Indikatorkonstruktion — Bayes-Prior für EP3 (Review-Absenz)
#      α = 5 ist Referenz; {2, 10} decken schwache bzw. starke Regularisierung ab.
SENSITIVITY_ALPHA_GRID = [2, 5, 10]

# (ii-b) Cross-Phase-Hybrid-Gewicht α_H für Topic-Matching (step01c).
#       Hybrid-Score H = α_H · cosine + (1 − α_H) · jaccard.
#       α_H = 0.6 Referenz (Repräsentations-Cosine dominiert leicht);
#       {0.4, 0.8} decken jaccard-lastige bzw. cosine-lastige Varianten ab.
#       Stabilitätsmaß: Jaccard-Überlappung der Mutual-Best-Paarmenge vs. Referenz
#       sowie Spearman-Rangkorrelation der Hybrid-Scores.
SENSITIVITY_HYBRID_ALPHA_GRID = [0.4, 0.6, 0.8]

# (iii) Membership-Klassifikation — 2D-Grid Sigmoid-k × WP-Gewicht λ
#       k = Sigmoid-Skalenparameter (steuert Schärfe der Membership-Übergänge);
#       λ = Gewicht von WP im Trend-Membership-Score.
#       Stabilitätsmaß: Spearman-Rangkorrelation der Membership-Vektoren
#       (m_ws, m_trend, m_ec, m_latent) vs. Referenzlauf (k=1.0, λ=0.5).
#       ρ > 0.9 → robust; ρ < 0.7 → explizite Reflexion in der Interpretation.
SENSITIVITY_SIGMOID_K_GRID = [0.5, 1.0, 2.0]
SENSITIVITY_LAMBDA_GRID    = [0.3, 0.5, 0.7]

# Ergänzende Analysen
# UMAP-Seed-Stabilität (stochastische Einheitenbildung)
SENSITIVITY_SEEDS = [0, 1, 7, 42, 2026]

# Leave-One-Out-Indikator-Ablation
SENSITIVITY_ABLATION_SET = "ALL"    # "ALL" = alle 16 Indikatoren einzeln

# Phasen-Alternative: Splitjahr-Variation ±1 Jahr sowie Doubling-Time-
# informierte Referenz (Scheidsteger et al. 2021, Quantum Reports,
# DOI: 10.3390/quantum3030036; td = 5–7 Jahre → Split ∈ {2017,…,2020}).
SENSITIVITY_PHASE_SPLITS = [2014, 2015, 2016, 2017, 2018, 2019, 2020]

# Indexierungs-/Rezeptionslatenz: rezeptionsbasierte Indikatoren (DI4, WP2)
# werden in einer Variante um die letzten N Monate des Beobachtungsfensters
# bereinigt (WoS-Indexierungs- und Zitationsakkumulationslatenz).
SENSITIVITY_INDEXING_CUTOFF_MONTHS = 12

# =============================================================================
# SCHRITT 3b: EXTERNE VALIDIERUNG (Clarivate-Topic-Breitenmaße)
# =============================================================================
# RTW (Reference Topic Width) und CTW (Citation Topic Width) sind paper-level
# von Clarivate vorberechnete Streumaße über die WoS-„Citation Topics".
# Operativ als externe Konstruktindikatoren genutzt:
#   - RTW (intellectual diversity inputs) → Konvergenz mit EO2 disciplinary_entropy
#   - CTW (intellectual diversity outputs) → Konvergenz mit WP3 field_breadth
#   - Diskriminanz: schwach mit DI3 geographic_concentration, DI1 author_concentration
# Die Aggregation erfolgt topic-level via Median (robust gegenüber Outliern in
# stark rechtsschiefen RTW/CTW-Verteilungen).
EXTERNAL_VALIDATION_AGGREGATION = "median"   # "median" oder "mean"

# Korrelations-Schwellwerte für die Beurteilung (s. Methodik §Schritt 3b):
EXTVAL_CONVERGENT_MIN = 0.50    # ρ ≥ 0.50 → konvergent
EXTVAL_DISCRIMINANT_MAX = 0.30  # ρ ≤ 0.30 → diskriminant
