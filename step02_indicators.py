"""
SCHRITT 2: Indikatorberechnung — 16 Indikatoren auf 5 Dimensionen
===================================================================

Berechnet alle 16 Indikatoren des F2-Frameworks pro Topic.
Datengrundlage: Web of Science (WoS) CSV-Export.

Revidierte Spezifikation (Stand: finaler Korpus):
  - EO2 disciplinary_entropy: Shannon-H über `WoS Categories` (nicht Journals)
  - WN1 relative_proportion_inv: rangbasierte phasen-interne Normalisierung
  - EP1 temporal_novelty: Normierung auf globales Fenster Y_MIN–Y_MAX
  - EP3 review_absence: additive Bayes-Glättung mit α = REVIEW_ABSENCE_ALPHA
  - DI1 author_concentration: Disambiguierung via `Author Full Names` + ORCID
  - DI2 institutional_concentration: kuratiertes Feld `Affiliations`
  - DI3 geographic_concentration: bevorzugt `Country List` (KATI, ISO-3,
    bereits dedupliziert); Fallback: ISO-3166-Heuristik auf `Addresses`
  - DI4 citation_concentration: Gini auf altersnormierter Rate
  - WP2 citation_momentum: altersnormierte Rate (Y_CUTOFF = 2025)
  - WP3 field_breadth: normalisierte Shannon-H über `Keywords Plus`
  - DI1 ist blockiert, solange Author Full Names / ORCIDs fehlen; die
    Pipeline schreibt in diesem Fall NaN (kein Abbruch). DI2/DI3 sind nach
    der KATI-Mai-2026-Lieferung operativ.

Dimensionen und Indikatoren:
  1. Epistemische Offenheit (3):
     - keyword_volatility: Jaccard-Distanz der Top-Keywords (early vs. late)
     - disciplinary_entropy: Normalisierte Shannon-H über WoS Categories
     - semantic_incoherence: 1 − mittlere Cosine-Similarity innerhalb des Topics

  2. Wahrnehmbarkeit (3):
     - relative_proportion_inv: 1 − rank(prop_t) / T_phase  (rangbasiert)
     - noise_ratio: Anteil Noise-Dokumente im UMAP-Umfeld
     - journal_specificity: Anteil Nischen-Journals im Topic

  3. Entwicklungsphase (3):
     - temporal_novelty: 1 − (mean_year − Y_MIN) / (Y_MAX − Y_MIN)
     - terminological_instability: Jahr-für-Jahr Keyword-Volatilität
     - review_absence: 1 − (n_rev + α) / (n_t + 2α)  (Bayes-geglättet)

  4. Diffusion (4):
     - author_concentration: Gini über Author Full Names + ORCID
     - institutional_concentration: Logistische Institutions-Konzentration
     - geographic_concentration: HHI der Länder-Verteilung (ISO-3166)
     - citation_concentration: Gini auf altersnormierter Rate

  5. Wirkungspotenzial (3):
     - growth_rate: Annualisierte Wachstumsrate (aus TEM)
     - citation_momentum: Verhältnis altersnormierter Raten (spät/früh)
     - field_breadth: Shannon-H über Keywords Plus

Autor: Ben Borowski
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import entropy
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import (
    OUTPUT_DIR, DATA_PATH, CURRENT_YEAR,
    PHASE_YEAR_MIN, PHASE_YEAR_MAX,
    Y_MIN, Y_MAX, Y_CUTOFF, REVIEW_ABSENCE_ALPHA,
    INDICATOR_DIMENSIONS, DIM_NAMES,
)


# =============================================================================
# WoS-SPALTENMAPPING
# =============================================================================
# Der WoS-Export nutzt andere Spaltennamen als Scopus. Die Pipeline arbeitet
# intern mit kanonischen Keys; die Rename-Logik wird beim Laden (run())
# angewandt, sodass alle Indikator-Funktionen unabhängig vom Quellformat sind.
CANON_COLUMNS = {
    "Publication Year":     "Year",
    "Author Full Names":    "Authors",
    "Affiliations":         "Affiliations",
    "Addresses":            "Addresses",
    "Source Title":         "Source title",
    "Document Type":        "Document Type",
    "WoS Categories":       "WoS Categories",
    "Author Keywords":      "Author Keywords",
    "Keywords Plus":        "Keywords Plus",
    "Times Cited, WoS Core":"Times Cited",
    "ORCIDs":               "ORCIDs",
    "Funding Orgs":         "Funding Orgs",
    "RTW":                  "RTW",
    "CTW":                  "CTW",
    # KATI-Erweiterung Mai 2026: vorbereitete Länderliste (ISO-3, Semikolon)
    "Country List":         "Country List",
}


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def safe_entropy(counts: list, base: int = 2) -> float:
    """Shannon-Entropie mit Absicherung für leere Verteilungen."""
    if not counts or sum(counts) == 0:
        return 0.0
    probs = np.array(counts) / sum(counts)
    return entropy(probs, base=base)


def jaccard_distance(set_a: set, set_b: set) -> float:
    """Jaccard-Distanz zwischen zwei Mengen."""
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return 1 - len(set_a & set_b) / len(union)


def hhi(counts: list) -> float:
    """Herfindahl-Hirschman Index."""
    if not counts or sum(counts) == 0:
        return 1.0
    total = sum(counts)
    shares = [c / total for c in counts]
    return sum(s ** 2 for s in shares)


def gini(values: np.ndarray) -> float:
    """Gini-Koeffizient."""
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    if n == 0 or sorted_vals.sum() == 0:
        return 0.5
    cumulative = np.cumsum(sorted_vals)
    total = cumulative[-1]
    return (2 * np.sum((np.arange(1, n + 1) * sorted_vals))) / (n * total) - (n + 1) / n


def extract_keywords(series: pd.Series, top_n: int = 20) -> set:
    """Top-N Keywords aus einer Author-Keywords-Spalte extrahieren."""
    kws = []
    for kw_str in series.dropna():
        kws.extend([k.strip().lower() for k in kw_str.split(";")])
    return set([k for k, _ in Counter(kws).most_common(top_n)])


# =============================================================================
# DIMENSION 1: EPISTEMISCHE OFFENHEIT
# =============================================================================

def compute_keyword_volatility(df: pd.DataFrame, topic_ids: list,
                               labels: np.ndarray, top_n: int = 20) -> dict:
    """
    Keyword-Volatilität: Jaccard-Distanz der Top-Keywords zwischen
    Early- und Late-Periode (Median-Split nach Jahr).
    Hohe Volatilität → instabile Terminologie → hohe epistemische Offenheit.
    """
    df_t = df.copy()
    df_t["topic"] = labels
    results = {}

    for tid in topic_ids:
        topic_df = df_t[df_t["topic"] == tid]
        years = topic_df["Year"].sort_values()

        if len(years) < 4:
            results[tid] = 0.5
            continue

        mid_year = years.median()
        kw_early = extract_keywords(topic_df[topic_df["Year"] <= mid_year]["Author Keywords"], top_n)
        kw_late = extract_keywords(topic_df[topic_df["Year"] > mid_year]["Author Keywords"], top_n)

        results[tid] = jaccard_distance(kw_early, kw_late) if (kw_early or kw_late) else 0.5

    return results


def compute_disciplinary_entropy(df: pd.DataFrame, topic_ids: list,
                                  labels: np.ndarray) -> dict:
    """
    Disziplinäre Entropie: Normalisierte Shannon-H über die Verteilung der
    `WoS Categories` (WC) innerhalb des Topics.

    WC ist eine kuratierte, disziplinär konsolidierte Klassifikation
    (ca. 250 Kategorien). Die Messung auf WC statt auf Source-Title-Ebene
    entkoppelt das Konstrukt von journalspezifischen Publikationsstrategien
    und liefert eine weniger verrauschte Entropiebasis.

    Hohe Entropie → breite disziplinäre Streuung → hohe epistemische Offenheit.
    """
    df_t = df.copy()
    df_t["topic"] = labels
    results = {}

    # Fallback-Spalte, falls der Export keine WoS Categories enthält
    wc_col = "WoS Categories" if "WoS Categories" in df.columns else "Source title"

    for tid in topic_ids:
        raw = df_t.loc[df_t["topic"] == tid, wc_col].dropna()
        if len(raw) < 2:
            results[tid] = 0.0
            continue

        # WC ist ein Semikolon-separiertes Multi-Label-Feld
        cats = []
        for s in raw:
            cats.extend([c.strip() for c in str(s).split(";") if c.strip()])

        if len(cats) < 2:
            results[tid] = 0.0
            continue

        counts = list(Counter(cats).values())
        max_ent = np.log2(len(counts)) if len(counts) > 1 else 1.0
        results[tid] = safe_entropy(counts) / max_ent if max_ent > 0 else 0.0

    return results


def compute_semantic_incoherence(df: pd.DataFrame, topic_ids: list,
                                  labels: np.ndarray,
                                  embeddings_sbert: np.ndarray) -> dict:
    """
    Semantische Inkohärenz: 1 − mittlere paarweise Cosine-Similarity
    der SBERT-Embeddings innerhalb des Topics.
    Hohe Inkohärenz → diverse Dokumente → hohe epistemische Offenheit.

    Nutzt die Original-384D-SBERT-Embeddings (nicht die UMAP-reduzierten),
    weil SBERT bereits L2-normalisiert ist → Dot-Product = Cosine-Similarity.
    """
    results = {}

    for tid in topic_ids:
        indices = np.where(labels == tid)[0]

        if len(indices) < 3:
            results[tid] = 0.5
            continue

        # Sampling bei großen Topics (Effizienz)
        if len(indices) > 200:
            sample_idx = np.random.RandomState(42).choice(indices, 200, replace=False)
        else:
            sample_idx = indices

        topic_vecs = embeddings_sbert[sample_idx]
        sim_matrix = cosine_similarity(topic_vecs)

        # Mittlere paarweise Similarity (oberes Dreieck)
        n = sim_matrix.shape[0]
        upper_tri = sim_matrix[np.triu_indices(n, k=1)]
        mean_sim = upper_tri.mean()

        # Invertieren: hohe Inkohärenz = hohe EO
        results[tid] = 1.0 - mean_sim

    return results


# =============================================================================
# DIMENSION 2: WAHRNEHMBARKEIT
# =============================================================================

def compute_relative_proportion(proportions: pd.DataFrame,
                                topic_ids: list) -> dict:
    """
    Relative Topic-Proportion (invertiert, rangbasiert).

    Revidiert (20.04.2026): Statt Skalennormierung auf das größte Topic
    wird eine rangbasierte Inversion innerhalb der Phase verwendet.
    Formel: P_obs = 1 - rank_phase(prop_t) / T_phase.

    Die rang- statt skalenbasierte Normierung eliminiert die Abhängigkeit
    vom phasenspezifischen Gesamtvolumen und macht den Indikator
    phasenvergleichstauglich: Ein Topic mit niedrigem Rang gilt in beiden
    Phasen als gleichermaßen wenig sichtbar, unabhängig vom absoluten
    Volumen der Phase.

    Niedrige Proportion → geringe Sichtbarkeit → Weak-Signal-Eigenschaft.
    """
    results = {}
    avg_props = proportions.mean(axis=0)
    # Topics nach mittlerer Proportion sortieren; rank 1 = kleinste
    ranked = avg_props.rank(method="average", ascending=True)
    n_topics = len(ranked)

    for tid in topic_ids:
        if tid in ranked.index and n_topics > 0:
            # rangbasiert: hoher Rang = große Proportion; invertiert
            results[tid] = 1.0 - (ranked[tid] / n_topics)
        else:
            results[tid] = 1.0

    return results


def compute_noise_ratio(labels: np.ndarray, embeddings_reduced: np.ndarray,
                        topic_ids: list) -> dict:
    """
    Noise-Ratio: Anteil der Noise-Dokumente im UMAP-Umfeld eines Topics.
    Hoher Noise-Anteil → schwer abgrenzbar → geringe Wahrnehmbarkeit.

    Nutzt die UMAP-reduzierten Embeddings (15D), da HDBSCAN dort clustert.
    """
    results = {}

    for tid in topic_ids:
        mask = labels == tid
        if mask.sum() < 3:
            results[tid] = 0.5
            continue

        centroid = embeddings_reduced[mask].mean(axis=0).reshape(1, -1)
        k = min(mask.sum() * 2, len(labels))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(embeddings_reduced)
        _, indices = nn.kneighbors(centroid)

        neighbor_labels = labels[indices.flatten()]
        noise_count = (neighbor_labels == -1).sum()
        results[tid] = noise_count / len(neighbor_labels)

    return results


def compute_journal_specificity(df: pd.DataFrame, topic_ids: list,
                                 labels: np.ndarray) -> dict:
    """
    Journal-Spezifität: Anteil Publikationen in Nischen-Journals
    (unterhalb des Median der globalen Journal-Größe).
    Hohe Spezifität → Nische → geringe Wahrnehmbarkeit.
    """
    df_t = df.copy()
    df_t["topic"] = labels
    journal_sizes = df["Source title"].value_counts()
    median_size = journal_sizes.median()

    results = {}
    for tid in topic_ids:
        journals = df_t.loc[df_t["topic"] == tid, "Source title"].dropna()
        if len(journals) < 2:
            results[tid] = 0.5
            continue

        niche_count = sum(1 for j in journals if journal_sizes.get(j, 0) <= median_size)
        results[tid] = niche_count / len(journals)

    return results


# =============================================================================
# DIMENSION 3: ENTWICKLUNGSPHASE
# =============================================================================

def compute_temporal_novelty(df: pd.DataFrame, topic_ids: list,
                              labels: np.ndarray) -> dict:
    """
    Temporale Neuheit: (mean_year - Y_MIN) / (Y_MAX - Y_MIN).

    Direktional als *Recency* operationalisiert: Ein höherer Wert
    bedeutet, dass der Publikationsschwerpunkt eines Topics näher am
    rechten Rand des globalen Beobachtungsfensters liegt — also
    "jünger" im Sinne der wissenschaftlichen Aktualität ist. Diese
    Direktionalität ist zwingend, weil EP1 in der Membership-Logik als
    Novelty-Anchor für m_ec dient (Rotolo 2015: "radical novelty"); ein
    invertiertes Vorzeichen würde Alter statt Neuheit messen.

    Revidiert (20.04.2026): Normierung auf das globale Beobachtungsfenster
    Y_MIN=2000, Y_MAX=2025 -- identisch für beide Phasen. Die Normierung
    auf das Gesamtfenster statt auf die jeweilige Phase sichert die
    semantische Invarianz: Ein Topic, dessen Publikationsschwerpunkt
    2013-2015 liegt, trägt in Phase 1 dann nicht artifizielle
    Maximalneuheit, sondern seine reale Reifeposition innerhalb der
    Gesamtzeitachse.

    Junge Topics (hohe Recency) → hoher EP1-Wert → frühe Entwicklungsphase.
    """
    df_t = df.copy()
    df_t["topic"] = labels
    span = Y_MAX - Y_MIN
    results = {}

    for tid in topic_ids:
        years = df_t.loc[df_t["topic"] == tid, "Year"].dropna()
        if len(years) == 0 or span <= 0:
            results[tid] = 0.5
            continue
        mean_year = float(years.mean())
        # Clipping, falls mean_year außerhalb des Fensters liegt
        nov = (mean_year - Y_MIN) / span
        results[tid] = float(np.clip(nov, 0.0, 1.0))

    return results


def compute_terminological_instability(df: pd.DataFrame, topic_ids: list,
                                        labels: np.ndarray) -> dict:
    """
    Terminologische Instabilität: Mittlere Jahr-für-Jahr
    Jaccard-Distanz der Top-Keywords.
    Hohe Instabilität → Terminologie noch nicht konsolidiert → frühe Phase.
    """
    df_t = df.copy()
    df_t["topic"] = labels
    results = {}

    for tid in topic_ids:
        topic_df = df_t[df_t["topic"] == tid]
        years = sorted(topic_df["Year"].unique())

        if len(years) < 3:
            results[tid] = 0.5
            continue

        volatilities = []
        for i in range(1, len(years)):
            kw1 = extract_keywords(
                topic_df[topic_df["Year"] == years[i - 1]]["Author Keywords"], 15
            )
            kw2 = extract_keywords(
                topic_df[topic_df["Year"] == years[i]]["Author Keywords"], 15
            )
            if kw1 or kw2:
                volatilities.append(jaccard_distance(kw1, kw2))

        results[tid] = np.mean(volatilities) if volatilities else 0.5

    return results


def compute_review_absence(df: pd.DataFrame, topic_ids: list,
                            labels: np.ndarray,
                            alpha: float = None) -> dict:
    """
    Review-Absenz, additiv (Bayes) geglättet:
        R_abs = 1 − (n_rev + α) / (n_t + 2α)  mit α = REVIEW_ABSENCE_ALPHA

    Revidiert (20.04.2026): Die Bayesianische Glättung korrigiert den
    Small-Topic-Bias. Ohne Glättung kollabiert der Indikator bei kleinen
    Topics (n_t < 20) strukturell auf Werte nahe 1 und verliert
    Differenzierungskraft. Bei n_t → ∞ konvergiert die geglättete Formel
    gegen die unverzerrte Rohdefinition.

    Parameter
    ---------
    alpha : optional
        Überschreibt den Referenz-α aus config.REVIEW_ABSENCE_ALPHA (= 5).
        Wird von Schritt 5 (Sensitivitätsanalyse) mit α ∈ {2, 5, 10} genutzt,
        um die Robustheit der EP3-Rangfolge gegen die Prior-Wahl zu prüfen.

    WoS-Dokumenttyp: "Review" (im Feld `Document Type`).
    Wenige Reviews → nicht konsolidiert → frühe Phase.
    """
    df_t = df.copy()
    df_t["topic"] = labels
    alpha = REVIEW_ABSENCE_ALPHA if alpha is None else float(alpha)
    results = {}

    for tid in topic_ids:
        types = df_t.loc[df_t["topic"] == tid, "Document Type"].dropna()
        n_t = len(types)
        if n_t == 0:
            results[tid] = 0.5
            continue
        # WoS führt "Review" sowohl standalone als auch in Kombination
        # (z.B. "Article; Early Access") -- substring-Match ist hier
        # absichtlich strenger gehalten: exakter Match auf "Review".
        review_count = int((types.astype(str).str.strip() == "Review").sum())
        results[tid] = 1.0 - (review_count + alpha) / (n_t + 2 * alpha)

    return results


# =============================================================================
# DIMENSION 4: DIFFUSION (invertiert: hoher Wert = geringe Diffusion)
# =============================================================================

def compute_author_concentration(df: pd.DataFrame, topic_ids: list,
                                  labels: np.ndarray) -> dict:
    """
    Autoren-Konzentration: Gini-Koeffizient der Publikationsverteilung
    über Autoren.

    Revidiert (20.04.2026): Disambiguierung primär über
    `Author Full Names` (im kanonischen Spaltennamen `Authors`), mit
    `ORCIDs` als Tie-Breaker bei Namensgleichheit (z. B. "Zhang, Y.").
    Ein ORCID-Treffer überschreibt den Namens-Key.

    Hoher Gini → wenige dominante Autoren → geringe Diffusion.

    Datenverfügbarkeit: Wenn das Feld `Authors` (Author Full Names) im
    Korpus fehlt oder durchgängig leer ist, liefert der Indikator NaN
    für alle Topics — die nachgelagerte Dimensionsaggregation klammert
    NaN-Indikatoren konsistent aus.
    """
    df_t = df.copy()
    df_t["topic"] = labels
    has_orcid = "ORCIDs" in df_t.columns

    # Datenverfügbarkeits-Guard: DI1 ist blockiert, wenn `Authors` fehlt
    # oder im gesamten Korpus leer ist.
    if "Authors" not in df_t.columns or \
       df_t["Authors"].fillna("").astype(str).str.strip().eq("").all():
        return {tid: float("nan") for tid in topic_ids}

    results = {}

    for tid in topic_ids:
        topic_df = df_t[df_t["topic"] == tid]
        authors_raw = topic_df["Authors"].fillna("")
        orcids_raw = topic_df["ORCIDs"].fillna("") if has_orcid else None

        all_authors = []
        for i, a_str in enumerate(authors_raw):
            names = [a.strip() for a in str(a_str).split(";") if a.strip()]
            orcids = []
            if orcids_raw is not None:
                orcids = [o.strip() for o in str(orcids_raw.iloc[i]).split(";")]
            # Tie-Breaker: prefer ORCID-Key, sonst Name
            for j, name in enumerate(names):
                key = name
                if j < len(orcids) and orcids[j] and "/" in orcids[j]:
                    # ORCIDs werden im WoS-Export oft als "Name/0000-0000"
                    # geliefert; rechter Teil ist die ID
                    parts = orcids[j].split("/")
                    if len(parts) >= 2 and parts[-1].strip():
                        key = parts[-1].strip()
                all_authors.append(key)

        if len(all_authors) < 2:
            results[tid] = 1.0
            continue

        counts = np.array(list(Counter(all_authors).values()), dtype=float)
        results[tid] = max(0, min(1, gini(counts)))

    return results


def compute_institutional_concentration(df: pd.DataFrame, topic_ids: list,
                                         labels: np.ndarray) -> dict:
    """
    Institutionelle Konzentration: Logistischer Index.
    Formel: 1 / (1 + log(1 + n_unique_institutions / n_docs))

    Datenquelle: kuratiertes WoS-Feld `Affiliations` (nicht `Addresses`).
    `Affiliations` ist eine bereits auflösungsbereinigte Institutionenliste
    (verschachtelte Hierarchien sind in WoS auf eine kanonische Form
    reduziert). Die Normalisierung sorgt für konsistentere Ergebnisse als
    der Freitext in `Addresses`.

    Niedrige Diversität → hoher Wert → geringe Diffusion.

    Datenverfügbarkeit: Wenn das Feld `Affiliations` fehlt oder im
    gesamten Korpus leer ist, liefert der Indikator NaN für alle Topics.
    """
    df_t = df.copy()
    df_t["topic"] = labels

    if "Affiliations" not in df_t.columns or \
       df_t["Affiliations"].fillna("").astype(str).str.strip().eq("").all():
        return {tid: float("nan") for tid in topic_ids}

    results = {}

    for tid in topic_ids:
        mask = df_t["topic"] == tid
        n_docs = int(mask.sum())
        affils = df_t.loc[mask, "Affiliations"].dropna()

        all_insts = []
        for a_str in affils:
            # WoS-Affiliations sind Semikolon-separiert
            all_insts.extend([i.strip() for i in str(a_str).split(";") if i.strip()])

        if len(all_insts) < 2 or n_docs == 0:
            results[tid] = 1.0
            continue

        n_unique = len(set(all_insts))
        results[tid] = 1 / (1 + np.log(1 + n_unique / n_docs))

    return results


# ISO-3166-Normalisierung: Kompakte Abbildung häufiger WoS-Ländervarianten
# auf einen kanonischen Schlüssel. Bewusst nicht erschöpfend, sondern auf
# die im Quantum-Computing-Korpus erwartbaren Varianten fokussiert.
_COUNTRY_ALIASES = {
    "USA": "USA", "U S A": "USA", "U.S.A.": "USA", "UNITED STATES": "USA",
    "UNITED STATES OF AMERICA": "USA",
    "UK": "UK", "U.K.": "UK", "UNITED KINGDOM": "UK",
    "ENGLAND": "UK", "SCOTLAND": "UK", "WALES": "UK", "NORTHERN IRELAND": "UK",
    "RUSSIA": "RUS", "RUSSIAN FEDERATION": "RUS",
    "SOUTH KOREA": "KOR", "KOREA": "KOR", "REPUBLIC OF KOREA": "KOR",
    "CZECH REPUBLIC": "CZE", "CZECHIA": "CZE",
    "PEOPLES R CHINA": "CHN", "PEOPLE'S REPUBLIC OF CHINA": "CHN",
    "CHINA": "CHN",
    "TAIWAN": "TWN", "REPUBLIC OF CHINA": "TWN",
}


def _normalize_country(raw: str) -> str | None:
    """ISO-3166-artige Normalisierung eines Ländertexts."""
    if not raw:
        return None
    key = raw.strip().upper().rstrip(".")
    if 1 < len(key) < 50:
        return _COUNTRY_ALIASES.get(key, key)
    return None


def compute_geographic_concentration(df: pd.DataFrame, topic_ids: list,
                                      labels: np.ndarray) -> dict:
    """
    Geografische Konzentration: HHI der Länder-Verteilung.

    Revidiert (20.04.2026): Extraktion aus `Addresses` (Freitext, Land am
    Zeilenende jeder Adresse). Jedes Land wird pro Record genau einmal
    gezählt (set-Aggregation), um Mehrfachzählung bei mehreren Co-Autor:innen
    aus demselben Land zu vermeiden. Normalisierung über _normalize_country().

    Erweiterung (Mai 2026, KATI-Lieferung Dr. John): Sofern die Spalte
    `Country List` vorliegt (Semikolon-getrennte ISO-3-Codes, bereits pro UID
    algorithmisch dedupliziert), wird diese bevorzugt — sie eliminiert die
    Heuristik der Freitext-Extraktion und ersetzt die Alias-Normalisierung
    durch eine konsistente ISO-3-Kodierung.

    Datenverfügbarkeit: Wenn weder `Country List` noch `Addresses` noch
    `Affiliations` befüllt sind, liefert der Indikator NaN für alle Topics.
    """
    df_t = df.copy()
    df_t["topic"] = labels

    # Quellen-Priorität: Country List > Addresses > Affiliations
    has_country_list = (
        "Country List" in df_t.columns
        and not df_t["Country List"].fillna("").astype(str)
                                   .str.strip().eq("").all()
    )

    if has_country_list:
        results = {}
        for tid in topic_ids:
            cls = df_t.loc[df_t["topic"] == tid, "Country List"].fillna("")
            countries = []
            for s in cls:
                # KATI-Format: "CHE; USA" — pro Record set für Eindeutigkeit
                record = {p.strip().upper() for p in str(s).split(";")
                          if p.strip()}
                countries.extend(record)

            if len(countries) < 2:
                results[tid] = 1.0
                continue

            results[tid] = hhi(list(Counter(countries).values()))
        return results

    # Fallback: Freitext-Heuristik aus Addresses bzw. Affiliations
    src = "Addresses" if "Addresses" in df_t.columns else \
          ("Affiliations" if "Affiliations" in df_t.columns else None)

    if src is None or \
       df_t[src].fillna("").astype(str).str.strip().eq("").all():
        return {tid: float("nan") for tid in topic_ids}

    results = {}

    for tid in topic_ids:
        addrs = df_t.loc[df_t["topic"] == tid, src].dropna()
        countries = []
        for a_str in addrs:
            record_countries = set()
            for inst in str(a_str).split(";"):
                parts = inst.strip().split(",")
                if parts:
                    c = _normalize_country(parts[-1])
                    if c:
                        record_countries.add(c)
            countries.extend(record_countries)

        if len(countries) < 2:
            results[tid] = 1.0
            continue

        results[tid] = hhi(list(Counter(countries).values()))

    return results


def _age_normalized_rates(topic_df: pd.DataFrame) -> np.ndarray:
    """
    Altersnormierte Zitationsrate pro Dokument:
        C_rate(d) = TC(d) / (Y_CUTOFF - y_d + 1)

    Neutralisiert das unterschiedliche Akkumulationsfenster zwischen den
    Phasen (Phase 1: 10-25 Jahre, Phase 2: 0-10 Jahre). Y_CUTOFF = 2025.
    """
    tc = topic_df["Times Cited"].fillna(0).astype(float).values
    years = topic_df["Year"].astype(float).values
    age = np.clip(Y_CUTOFF - years + 1, 1.0, None)
    return tc / age


def compute_citation_concentration(df: pd.DataFrame, topic_ids: list,
                                    labels: np.ndarray) -> dict:
    """
    Zitations-Konzentration: Gini auf der altersnormierten Rate
    C_rate(d) = TC(d) / (Y_CUTOFF - y_d + 1).

    Revidiert (20.04.2026): Die Altersnormierung macht die
    Konzentrationsmessung phasen-kompatibel, ohne die konzeptionelle
    Identität des Indikators zu verändern. Der Gini-Koeffizient ist
    skaleninvariant und reagiert somit sauber auf die strukturelle
    Verteilung, nicht auf das unterschiedliche Akkumulationsfenster.

    Hoher Gini → wenige Papers dominieren die Rezeption.
    """
    df_t = df.copy()
    df_t["topic"] = labels
    results = {}

    for tid in topic_ids:
        topic_df = df_t[df_t["topic"] == tid]

        if len(topic_df) < 3:
            results[tid] = 0.5
            continue

        rates = _age_normalized_rates(topic_df)
        if rates.sum() == 0:
            results[tid] = 0.5
            continue

        results[tid] = max(0, min(1, gini(np.sort(rates))))

    return results


# =============================================================================
# DIMENSION 5: WIRKUNGSPOTENZIAL
# =============================================================================

def compute_growth_rate(tem_metrics: pd.DataFrame, topic_ids: list) -> dict:
    """
    Wachstumsrate: Annualisierte Wachstumsrate aus TEM (nach Ebadi et al.).
    Normalisiert auf [0,1] durch Clipping.
    """
    results = {}
    for tid in topic_ids:
        row = tem_metrics[tem_metrics["topic"] == tid]
        if len(row) > 0:
            gr = row.iloc[0]["growth_rate"]
            results[tid] = np.clip((gr + 0.5) / 1.5, 0, 1)
        else:
            results[tid] = 0.5
    return results


def compute_citation_momentum(df: pd.DataFrame, topic_ids: list,
                               labels: np.ndarray) -> dict:
    """
    Zitations-Momentum: Verhältnis altersnormierter Raten zwischen später
    und früher Hälfte der Topic-Lebensdauer.

    Revidiert (20.04.2026): Statt Rohzitationen werden altersnormierte
    Raten C_rate(d) = TC(d) / (Y_CUTOFF - y_d + 1) verwendet. Der
    Median-Split erfolgt phasen-intern (pro Topic) über das Publikations-
    jahr. Die Normierung entkoppelt das Momentum vom absoluten
    Akkumulationsfenster und macht es phasenvergleichstauglich.

    Steigendes normiertes Momentum → wachsendes Wirkungspotenzial.
    """
    df_t = df.copy()
    df_t["topic"] = labels
    results = {}

    for tid in topic_ids:
        topic_df = df_t[df_t["topic"] == tid]
        if len(topic_df) < 4:
            results[tid] = 0.5
            continue

        mid_year = topic_df["Year"].median()
        rates_all = _age_normalized_rates(topic_df)
        years = topic_df["Year"].astype(float).values

        early_rates = rates_all[years <= mid_year]
        late_rates = rates_all[years > mid_year]

        if len(early_rates) == 0 or len(late_rates) == 0:
            results[tid] = 0.5
            continue

        early_mean = float(early_rates.mean())
        late_mean = float(late_rates.mean())

        if early_mean > 0:
            ratio = late_mean / early_mean
            results[tid] = float(np.clip(ratio, 0, 2) / 2)
        else:
            results[tid] = 0.5 if late_mean > 0 else 0.0

    return results


def compute_field_breadth(df: pd.DataFrame, topic_ids: list,
                           labels: np.ndarray) -> dict:
    """
    Feld-Breite: Normalisierte Shannon-Entropie über die Verteilung der
    `Keywords Plus` innerhalb des Topics.

    Revidiert (20.04.2026): Ersetzt die frühere Type-Token-Ratio durch
    eine entropiebasierte Messung. TTR reagiert stark auf die Topic-Größe
    und ist bei großen Topics strukturell gedrückt. Die normalisierte
    Shannon-H (H / log|K|) ist hingegen konzentrationssensitiv und liefert
    ein größenrobustes Maß für die Streuung thematischer Anschlusspunkte.

    `Keywords Plus` ist WoS-algorithmisch generiert (aus Referenztiteln)
    und bildet damit die *rezipierte* thematische Breite des Topics ab --
    semantisch komplementär zu `Author Keywords` (emittierte Perspektive).

    Breite, gleichverteilte Keyword-Plus-Streuung → hohes
    Anschlussfähigkeits- und Wirkungspotenzial.
    """
    df_t = df.copy()
    df_t["topic"] = labels
    results = {}

    for tid in topic_ids:
        kw_series = df_t.loc[df_t["topic"] == tid, "Keywords Plus"].dropna()
        all_kw = []
        for kw_str in kw_series:
            all_kw.extend([k.strip().lower() for k in str(kw_str).split(";") if k.strip()])

        if len(all_kw) < 5:
            results[tid] = 0.5
            continue

        counts = list(Counter(all_kw).values())
        if len(counts) < 2:
            results[tid] = 0.0
            continue

        max_ent = np.log2(len(counts))
        results[tid] = safe_entropy(counts) / max_ent if max_ent > 0 else 0.0

    return results


# =============================================================================
# AGGREGATION: ALLE 16 INDIKATOREN BERECHNEN
# =============================================================================

def compute_all_indicators(
    df: pd.DataFrame,
    labels: np.ndarray,
    embeddings_sbert: np.ndarray,
    embeddings_reduced: np.ndarray,
    proportions: pd.DataFrame,
    tem_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Berechnet alle 16 Indikatoren für alle Topics.

    Args:
        df: WoS DataFrame, bereits mit kanonischen Spaltennamen (Year,
            Authors, Affiliations, Addresses, WoS Categories, Keywords Plus,
            Author Keywords, Document Type, Source title, Times Cited, ORCIDs).
            Das Mapping erfolgt in run() via CANON_COLUMNS.
        labels: HDBSCAN Topic-Labels
        embeddings_sbert: Original 384D SBERT-Embeddings (für Cosine-Sim)
        embeddings_reduced: UMAP 15D Embeddings (für Noise-Ratio)
        proportions: Jährliche Topic-Proportionen (aus Schritt 1)
        tem_metrics: TEM-Metriken mit avg_proportion und growth_rate

    Returns:
        DataFrame: rows = topics, columns = 16 Indikatoren
    """
    topic_ids = sorted(set(labels[labels >= 0]))
    print(f"\nBerechne 16 Indikatoren für {len(topic_ids)} Topics...")

    # --- Dimension 1: Epistemische Offenheit ---
    print("  [1/5] Epistemische Offenheit...")
    kw_vol = compute_keyword_volatility(df, topic_ids, labels)
    disc_ent = compute_disciplinary_entropy(df, topic_ids, labels)
    sem_inc = compute_semantic_incoherence(df, topic_ids, labels, embeddings_sbert)

    # --- Dimension 2: Wahrnehmbarkeit ---
    print("  [2/5] Wahrnehmbarkeit...")
    rel_prop = compute_relative_proportion(proportions, topic_ids)
    noise_r = compute_noise_ratio(labels, embeddings_reduced, topic_ids)
    jour_spec = compute_journal_specificity(df, topic_ids, labels)

    # --- Dimension 3: Entwicklungsphase ---
    print("  [3/5] Entwicklungsphase...")
    temp_nov = compute_temporal_novelty(df, topic_ids, labels)
    term_inst = compute_terminological_instability(df, topic_ids, labels)
    rev_abs = compute_review_absence(df, topic_ids, labels)

    # --- Dimension 4: Diffusion ---
    print("  [4/5] Diffusion (invertiert: hoher Wert = geringe Diffusion)...")
    auth_conc = compute_author_concentration(df, topic_ids, labels)
    inst_conc = compute_institutional_concentration(df, topic_ids, labels)
    geo_conc = compute_geographic_concentration(df, topic_ids, labels)
    cit_conc = compute_citation_concentration(df, topic_ids, labels)

    # --- Dimension 5: Wirkungspotenzial ---
    print("  [5/5] Wirkungspotenzial...")
    growth = compute_growth_rate(tem_metrics, topic_ids)
    cit_mom = compute_citation_momentum(df, topic_ids, labels)
    field_br = compute_field_breadth(df, topic_ids, labels)

    # Zusammenführen
    records = []
    for tid in topic_ids:
        records.append({
            "topic": tid,
            "keyword_volatility": kw_vol.get(tid, 0.5),
            "disciplinary_entropy": disc_ent.get(tid, 0.5),
            "semantic_incoherence": sem_inc.get(tid, 0.5),
            "relative_proportion_inv": rel_prop.get(tid, 0.5),
            "noise_ratio": noise_r.get(tid, 0.5),
            "journal_specificity": jour_spec.get(tid, 0.5),
            "temporal_novelty": temp_nov.get(tid, 0.5),
            "terminological_instability": term_inst.get(tid, 0.5),
            "review_absence": rev_abs.get(tid, 0.5),
            "author_concentration": auth_conc.get(tid, 0.5),
            "institutional_concentration": inst_conc.get(tid, 0.5),
            "geographic_concentration": geo_conc.get(tid, 0.5),
            "citation_concentration": cit_conc.get(tid, 0.5),
            "growth_rate": growth.get(tid, 0.5),
            "citation_momentum": cit_mom.get(tid, 0.5),
            "field_breadth": field_br.get(tid, 0.5),
        })

    indicator_df = pd.DataFrame(records).set_index("topic")
    print(f"  Indikator-Matrix: {indicator_df.shape}")
    return indicator_df


# =============================================================================
# DIMENSIONSAGGREGATION (z-Standardisierung → Mittelwert)
# =============================================================================

def compute_dimension_scores(indicator_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert Indikatoren zu Dimensionsscores:
    1. z-Standardisierung aller 16 Indikatoren
    2. Mittelwert der z-Werte pro Dimension
    """
    scaler = StandardScaler()
    z_scores = pd.DataFrame(
        scaler.fit_transform(indicator_df),
        index=indicator_df.index,
        columns=indicator_df.columns,
    )

    dim_scores = pd.DataFrame(index=indicator_df.index)
    for dim_name, indicators in INDICATOR_DIMENSIONS.items():
        # Nur Indikatoren mit Varianz > 0 einbeziehen
        valid = [ind for ind in indicators if ind in z_scores.columns
                 and z_scores[ind].std() > 0.01]
        if valid:
            dim_scores[dim_name] = z_scores[valid].mean(axis=1)
        else:
            dim_scores[dim_name] = 0.0

    return dim_scores


# =============================================================================
# KLASSIFIKATION ENTFERNT (Pipeline V2 — Membership-Paradigma)
# =============================================================================
# Die V1-Funktion classify_signals() ist in V2 durch das eigenständige Modul
# step02b_memberships.py ersetzt. Step 2 produziert hier nur noch die 16
# Indikatoren und die Dimensionsscores; die kontinuierlichen Klassen-Member-
# ships werden im Folgeschritt aus diesen Artefakten berechnet.


# =============================================================================
# MAIN
# =============================================================================

def run():
    import re

    print("=" * 70)
    print("SCHRITT 2: INDIKATORBERECHNUNG — 16 Indikatoren × 5 Dimensionen")
    print("=" * 70)

    # --- Daten laden (identische Filterung wie Schritt 1) ---
    # Reihenfolge: Spaltenmapping → Year-Filter [PHASE_YEAR_MIN, PHASE_YEAR_MAX]
    # → Längen-Filter (≥ 10 Wörter Title+Abstract). Diese Sequenz muss
    # bit-identisch zu step01_topic_modeling.load_and_clean() sein, damit
    # die Indizierung auf model_results.pkl (labels/embeddings) konsistent
    # bleibt — sonst greift das assert() weiter unten.
    print("\nLade Daten und Topic-Model-Ergebnisse...")
    df = pd.read_csv(DATA_PATH)

    # WoS-Export → kanonische Spaltennamen. Alle Indikator-Funktionen
    # arbeiten mit den internen Keys (Year, Authors, Times Cited, ...),
    # unabhängig vom Quellformat. Nur tatsächlich vorhandene Spalten
    # werden umbenannt (fehlende WoS-Felder werden ignoriert).
    rename_map = {src: dst for src, dst in CANON_COLUMNS.items() if src in df.columns}
    df = df.rename(columns=rename_map)

    df["text"] = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")
    df["text_clean"] = df["text"].apply(
        lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", t.lower())).strip()
        if isinstance(t, str) else ""
    )

    # (i) Phasen-Year-Filter (siehe step01_topic_modeling.load_and_clean)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[(df["Year"] >= PHASE_YEAR_MIN) & (df["Year"] <= PHASE_YEAR_MAX)].copy()

    # (ii) Längen-Heuristik
    df = df[df["text_clean"].str.split().str.len() >= 10].reset_index(drop=True)
    print(f"  {len(df)} Dokumente geladen (Year-Filter "
          f"[{PHASE_YEAR_MIN}, {PHASE_YEAR_MAX}], Längen-Filter ≥ 10 Wörter; "
          f"WoS-Spaltenmapping: {len(rename_map)} Spalten)")

    # --- Model-Ergebnisse aus Schritt 1 ---
    with open(OUTPUT_DIR / "model_results.pkl", "rb") as f:
        model_data = pickle.load(f)

    labels = model_data["labels"]
    embeddings_sbert = model_data["embeddings_sbert"]
    embeddings_reduced = model_data["embeddings_reduced"]

    assert len(labels) == len(df), (
        f"Label-Länge ({len(labels)}) ≠ DataFrame-Länge ({len(df)}). "
        "Bitte Schritt 1 erneut ausführen."
    )

    # TEM-Metriken & Proportionen
    tem_metrics = pd.read_csv(OUTPUT_DIR / "tem_metrics.csv")
    proportions = pd.read_csv(OUTPUT_DIR / "topic_proportions_yearly.csv", index_col=0)
    proportions.columns = proportions.columns.astype(int)

    # --- Indikatoren berechnen ---
    indicator_df = compute_all_indicators(
        df=df,
        labels=labels,
        embeddings_sbert=embeddings_sbert,
        embeddings_reduced=embeddings_reduced,
        proportions=proportions,
        tem_metrics=tem_metrics,
    )

    # --- Dimensionsscores ---
    print("\nBerechne Dimensionsscores (z-standardisiert)...")
    dim_scores = compute_dimension_scores(indicator_df)

    # ===== SPEICHERN =====
    # Step 2 in V2 schreibt nur Indikatoren und Dimensionsscores. Die
    # Klassen-Memberships entstehen im Folgeschritt (step02b_memberships).
    print("\nSpeichere Ergebnisse...")
    indicator_df.to_csv(OUTPUT_DIR / "indicators_16.csv")
    dim_scores.to_csv(OUTPUT_DIR / "dimension_scores.csv")

    # ===== ZUSAMMENFASSUNG =====
    print(f"\n{'=' * 70}")
    print("INDIKATORERGEBNIS (V2 — Membership-Pipeline)")
    print(f"{'=' * 70}")
    print(f"\nIndikator-Matrix: {indicator_df.shape[0]} Topics × "
          f"{indicator_df.shape[1]} Indikatoren")
    print(f"Dimensionsscores: {dim_scores.shape[0]} Topics × "
          f"{dim_scores.shape[1]} Dimensionen")
    print(f"\nDimensions-Statistiken (z-standardisierte Mittelwerte):")
    for dim in DIM_NAMES:
        vals = dim_scores[dim]
        print(f"  {dim:25s}: mean={vals.mean():+.3f}, std={vals.std():.3f}, "
              f"min={vals.min():+.2f}, max={vals.max():+.2f}")

    print(f"\nAlles gespeichert in {OUTPUT_DIR}/")
    print(f"Nächster Schritt: python step02b_memberships.py")
    return indicator_df, dim_scores


if __name__ == "__main__":
    run()
