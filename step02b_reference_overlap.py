"""
SCHRITT 2b: Zitations-informierte Robustheitsprüfung der Topic-Bildung
======================================================================

Zweck
-----
Die semantische Topic-Bildung (Schritt 1) erzeugt Einheiten auf Grundlage
von Titel-, Abstract- und Keyword-Texten. Xie & Waltman (2025,
Scientometrics, DOI: 10.1007/s11192-025-05324-z) zeigen, dass die
Überlappung zwischen Topic-Modeling-Clustern und citation-basierten
Clustern typischerweise gering ist: Beide Sichten bilden unterschiedliche
Aspekte der Feldstruktur ab. Topic Modeling erfasst semantische und
gesellschaftsbezogene Problemlagen; citation-based Clustering erfasst
intellektuelle Mikro-Communities.

Dieses Modul liefert eine pragmatische Zweitsicht, ohne die Pipeline zu
zerbrechen: Für jedes semantisch gebildete Topic wird die durchschnittliche
paarweise Überlappung der Cited-References-Mengen seiner Dokumente
berechnet. Hohe Überlappung = starke intellektuelle Kohärenz der
semantischen Einheit; niedrige Überlappung = semantische Bündelung ohne
gemeinsamen Referenzkern (möglicher Topic-Artefakt oder tatsächlich
interdisziplinäres Bündel).

Metriken
--------
  RO_topic   Mean paarweiser Jaccard der Cited-References-Mengen innerhalb
             des Topics (N_pairs = min(200, C(n, 2)) via Sampling).
  RO_global  Gleiche Metrik auf ein globales Referenz-Sample (Sanity-Check).
  n_refs     Mittlere Referenz-Anzahl pro Dokument im Topic.

Interpretation
--------------
Die Metrik hat keinen normativen Schwellenwert. Sie ist als
Robustheitsindikator in der Diskussion (Kap. 5) zu verwenden:
Topics mit RO_topic deutlich über RO_global (z. B. > 2×) sind
intellektuell kohärent; Topics mit RO_topic ≈ RO_global signalisieren
ein rein semantisch gebildetes, referenz-heterogenes Bündel.

Input
-----
  df          WoS-DataFrame mit kanonischen Spaltennamen und einer Spalte
              `Cited References` (Semikolon-separiert, WoS-Export-Format).
  labels      HDBSCAN-Topiclabels (gleiche Länge wie df).
  topic_ids   Liste der zu analysierenden Topic-IDs.

Output
------
  reference_overlap.csv   Pro Topic: RO_topic, n_refs, ratio vs. RO_global.

Literatur
---------
  Xie, Q., & Waltman, L. (2025). A comparison of citation-based clustering
    and topic modeling for science mapping. Scientometrics, 130(5),
    2497–2522.
  Kessler, M. M. (1963). Bibliographic coupling between scientific papers.
    American Documentation, 14, 10–25.  [Jaccard-basierte Kopplung]
  Small, H. (1973). Co-citation in the scientific literature. Journal of
    the American Society for Information Science, 24(4), 265–269.

Autor: Ben Borowski
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

def parse_cited_references(ref_string) -> set:
    """
    Zerlegt das WoS-Feld `Cited References` in eine Menge normalisierter
    Referenz-Identifikatoren.

    WoS-Format pro Referenz (Semikolon-separiert):
      "AUTHOR, YEAR, JOURNAL, VOLUME, PAGE, DOI DOI ..."

    Für die Jaccard-Rechnung wird ein kompakter Key gebildet:
      erste drei Komma-Felder (Autor, Jahr, Journal) — in lowercase,
      DOI bevorzugt, wenn vorhanden.
    """
    if pd.isna(ref_string) or not isinstance(ref_string, str):
        return set()

    refs = set()
    for raw in ref_string.split(";"):
        r = raw.strip()
        if not r:
            continue
        # Falls eine DOI am Ende steht, als eindeutigen Key nutzen
        doi_idx = r.lower().find("doi ")
        if doi_idx >= 0:
            doi = r[doi_idx + 4:].strip().lower()
            doi = doi.split()[0] if doi else ""
            if doi:
                refs.add("doi:" + doi)
                continue
        # Fallback: erste drei Felder normalisieren
        parts = [p.strip().lower() for p in r.split(",")[:3] if p.strip()]
        if parts:
            refs.add("|".join(parts))
    return refs


def jaccard(a: set, b: set) -> float:
    """Jaccard-Koeffizient zweier Mengen; 0.0 bei leerer Vereinigung."""
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def mean_pairwise_jaccard(ref_sets: list, max_pairs: int = 200,
                           seed: int = 42) -> float:
    """
    Mittlerer paarweiser Jaccard über eine Liste von Referenz-Mengen.
    Bei >max_pairs Paaren wird per RandomState gesampelt, um die
    Laufzeit O(n²) bei großen Topics zu beschränken.
    """
    valid = [s for s in ref_sets if s]
    if len(valid) < 2:
        return 0.0

    rng = np.random.RandomState(seed)
    all_pairs = list(combinations(range(len(valid)), 2))
    if len(all_pairs) > max_pairs:
        idx = rng.choice(len(all_pairs), size=max_pairs, replace=False)
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs

    scores = [jaccard(valid[i], valid[j]) for i, j in pairs]
    return float(np.mean(scores)) if scores else 0.0


# =============================================================================
# KERNFUNKTION
# =============================================================================

def compute_reference_overlap(df: pd.DataFrame, labels: np.ndarray,
                              topic_ids: list,
                              max_pairs_per_topic: int = 200,
                              global_sample_size: int = 500,
                              seed: int = 42) -> pd.DataFrame:
    """
    Berechnet RO_topic für jedes Topic sowie ein globales Vergleichs-Sample
    (RO_global) zur Kontextualisierung.

    Die WoS-Spalte `Cited References` wird vorausgesetzt. Fehlt sie,
    gibt die Funktion einen DataFrame mit NaN-Werten zurück und eine
    Warnung auf stdout.
    """
    ref_col = None
    for candidate in ["Cited References", "References", "CitedReferences"]:
        if candidate in df.columns:
            ref_col = candidate
            break

    if ref_col is None:
        print("  [step02b] WARNUNG: Keine Cited-References-Spalte gefunden.")
        return pd.DataFrame({"topic": topic_ids,
                             "RO_topic": np.nan,
                             "RO_global": np.nan,
                             "n_refs": np.nan,
                             "ratio_vs_global": np.nan}).set_index("topic")

    print(f"  [step02b] Referenz-Overlap pro Topic (Spalte: '{ref_col}')")

    # Referenz-Mengen pro Dokument (einmal parsen, dann wiederverwenden)
    ref_sets_all = df[ref_col].apply(parse_cited_references).tolist()

    # Globaler Baseline-Wert: Zufalls-Sample aus dem Gesamtkorpus
    rng = np.random.RandomState(seed)
    n_total = len(ref_sets_all)
    sample_size = min(global_sample_size, n_total)
    global_idx = rng.choice(n_total, size=sample_size, replace=False)
    global_sample = [ref_sets_all[i] for i in global_idx]
    ro_global = mean_pairwise_jaccard(global_sample,
                                       max_pairs=max_pairs_per_topic,
                                       seed=seed)
    print(f"  [step02b] RO_global (baseline, n={sample_size}) = {ro_global:.4f}")

    records = []
    for tid in topic_ids:
        mask = labels == tid
        topic_refs = [ref_sets_all[i] for i in np.where(mask)[0]]
        non_empty = [r for r in topic_refs if r]

        ro_topic = mean_pairwise_jaccard(topic_refs,
                                          max_pairs=max_pairs_per_topic,
                                          seed=seed)
        n_refs = np.mean([len(r) for r in non_empty]) if non_empty else 0.0
        ratio = (ro_topic / ro_global) if ro_global > 0 else np.nan

        records.append({
            "topic": tid,
            "RO_topic": ro_topic,
            "RO_global": ro_global,
            "n_refs": n_refs,
            "ratio_vs_global": ratio,
        })

    result = pd.DataFrame(records).set_index("topic")
    return result


# =============================================================================
# MAIN
# =============================================================================

def run(df: pd.DataFrame = None, labels: np.ndarray = None,
        topic_ids: list = None, output_dir: Path = None):
    """
    Eigenständig oder als Teil der Pipeline ausführbar.

    Wird ohne Argumente aufgerufen, werden die Inputs aus dem OUTPUT_DIR
    (Schritt 1 Artefakte) geladen. In der Pipeline-Orchestrierung werden
    die Objekte direkt durchgereicht.
    """
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df is None or labels is None or topic_ids is None:
        # Standalone-Modus: Artefakte aus Schritt 1 laden
        import pickle
        with open(output_dir / "step1_artifacts.pkl", "rb") as f:
            art = pickle.load(f)
        df = art["df"]
        labels = art["labels"]
        topic_ids = sorted(set(labels[labels >= 0]))

    result = compute_reference_overlap(df, labels, topic_ids)
    out_path = output_dir / "reference_overlap.csv"
    result.to_csv(out_path)
    print(f"  [step02b] Gespeichert: {out_path}")
    return result


if __name__ == "__main__":
    run()
