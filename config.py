
from pathlib import Path


BASE_DIR = Path(__file__).parent

DATA_PATH = BASE_DIR.parent / "wos_qc_phase2_2016_2025.csv"

PHASE_YEAR_MIN = 2016
PHASE_YEAR_MAX = 2025

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
    "rtw":          "RTW",
    "ctw":          "CTW",
}

OUTPUT_DIR = BASE_DIR / "output"


SBERT_MODEL = "all-MiniLM-L6-v2"

UMAP_N_COMPONENTS = 15
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"

HDBSCAN_MIN_CLUSTER_SIZE = 25
HDBSCAN_MIN_SAMPLES = 8
HDBSCAN_CLUSTER_METHOD = "eom"

CTFIDF_TOP_N_WORDS = 15


CURRENT_YEAR = 2026

Y_MIN = 2000
Y_MAX = 2025
Y_CUTOFF = 2025

REVIEW_ABSENCE_ALPHA = 5

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

DIM_NAMES = list(INDICATOR_DIMENSIONS.keys())

DIM_SHORT_CODES = {
    "Epistemische Offenheit": "EO",
    "Wahrnehmbarkeit":        "WN",
    "Entwicklungsphase":      "EP",
    "Diffusion":              "DI",
    "Wirkungspotenzial":      "WP",
}
DIM_SHORT_LIST = [DIM_SHORT_CODES[d] for d in DIM_NAMES]


EFA_MIN_VARIANCE = 0.01
EFA_PARALLEL_N_ITER = 200


MEMBERSHIP_SIGMOID_K = 1.0
MEMBERSHIP_LAMBDA_WP = 0.5


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

FIG_DPI = 300


SENSITIVITY_MIN_CLUSTER_GRID     = [15, 20, 25, 30, 40]
SENSITIVITY_MIN_SAMPLES_GRID     = [5, 8, 12]
SENSITIVITY_MIN_TOPIC_SIZE_GRID  = [5, 10, 15, 20]
SENSITIVITY_N_NEIGHBORS_GRID     = [10, 15, 30]

SENSITIVITY_ALPHA_GRID = [2, 5, 10]

SENSITIVITY_HYBRID_ALPHA_GRID = [0.4, 0.6, 0.8]

SENSITIVITY_SIGMOID_K_GRID = [0.5, 1.0, 2.0]
SENSITIVITY_LAMBDA_GRID    = [0.3, 0.5, 0.7]

SENSITIVITY_SEEDS = [0, 1, 7, 42, 2026]

SENSITIVITY_ABLATION_SET = "ALL"

SENSITIVITY_PHASE_SPLITS = [2014, 2015, 2016, 2017, 2018, 2019, 2020]

SENSITIVITY_INDEXING_CUTOFF_MONTHS = 12

EXTERNAL_VALIDATION_AGGREGATION = "median"

EXTVAL_CONVERGENT_MIN = 0.50
EXTVAL_DISCRIMINANT_MAX = 0.30
