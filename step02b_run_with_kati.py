"""
SCHRITT 2b (KATI-Runner): Reference-Overlap auf Basis einer KATI-Edge-Liste
==========================================================================

Hintergrund
-----------
Das Originalmodul ``step02b_reference_overlap.py`` erwartet die
WoS-Spalte ``Cited References`` (Semikolon-separierter String pro Dokument)
und parst diese inline. Im vorliegenden Projekt liegen die Zitationsdaten
jedoch als separate KATI-Edge-Liste im Long-Format vor:

    UID, UID_ref

Diese Form ist robuster (kein Excel-Zeilenlimit auf Inline-Strings; keine
String-Parsing-Heuristiken), erzwingt aber einen anderen Datenpfad.
Außerdem überschreitet die Phase-2-Datei das Excel-Zeilenlimit
(1.048.576), weshalb der Umweg über das Spreadsheet ausscheidet und direkt
mit pandas eingelesen werden muss.

Dieser Runner kapselt diesen alternativen Datenpfad, ohne das
inhaltlich-methodische Modul (``step02b_reference_overlap.py``) zu
verändern. Die methodische Logik (Mean Pairwise Jaccard, globales
Baseline-Sample, Sampling-Strategie) wird unverändert aus dem Original-
modul importiert. Damit bleibt die methodische Quelle der Wahrheit
ein einziges, reviewfähiges Modul.

Datenquellen
------------
  KATI_REFS  CSV im Long-Format mit Spalten ``UID``, ``UID_ref``.
             Jede Zeile = eine Zitation (Quell-UID zitiert Ziel-UID).
  TOPICS     CSV mit Topic-Zuweisungen pro Dokument (mind. ``UID``,
             ``topic``).

Methodik
--------
  RO_topic        Mean paarweiser Jaccard der Ref-Mengen innerhalb des
                  Topics (max_pairs=200, seed=42).
  RO_global       Gleiche Metrik auf ein Zufalls-Sample (n=500) aus dem
                  Korpus aller Dokumente mit ≥ 1 Referenz.
  ratio_vs_global RO_topic / RO_global; > 1 = überzufällig kohärent,
                  ≈ 1 = referenz-heterogenes Bündel.

Output pro Phase
----------------
  reference_overlap_<phase>.csv mit Spalten:
    topic, n_docs, n_docs_with_refs, coverage, RO_topic, RO_global,
    n_refs_mean, ratio_vs_global

Reproduzierbarkeit
------------------
  Sampling-Seed: 42 (in Original- und Runner-Modul identisch).
  Globales Sample: 500 zufällig gezogene Dokumente mit Refs.
  max_pairs: 200 (Original-Default beibehalten).

Aufruf
------
  python step02b_run_with_kati.py --phase 1
  python step02b_run_with_kati.py --phase 2

Literatur
---------
  Xie, Q., & Waltman, L. (2025). A comparison of citation-based clustering
    and topic modeling for science mapping. Scientometrics, 130(5),
    2497-2522.

Autor: Ben Borowski
"""

import argparse
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

# Methodische Kernfunktionen aus dem Originalmodul — unveraendert
from step02b_reference_overlap import jaccard, mean_pairwise_jaccard  # noqa: F401


# =============================================================================
# KONFIGURATION
# =============================================================================

# Skript-Verzeichnis als Anker; macht die Defaults unabhaengig vom CWD.
_HERE = Path(__file__).resolve().parent

DEFAULT_PATHS: Dict[int, Dict[str, Path]] = {
    1: {
        "kati_refs": _HERE / ".." / ".." / "Data Kati"
                            / "Phase 1 2000-2015"
                            / "QC_2000-2015 References.csv",
        "topics":    _HERE / "output_phase1" / "topic_assignments.csv",
        "out":       _HERE / "output_phase1" / "reference_overlap_p1.csv",
    },
    2: {
        "kati_refs": _HERE / ".." / ".." / "Data Kati"
                            / "Phase 2 2016-2025"
                            / "QC_2016-2025 References.csv",
        "topics":    _HERE / "output_phase2" / "topic_assignments.csv",
        "out":       _HERE / "output_phase2" / "reference_overlap_p2.csv",
    },
}

GLOBAL_SAMPLE_SIZE = 500
MAX_PAIRS = 200
SEED = 42


# =============================================================================
# DATEN-LOADER
# =============================================================================

def _normalize_uid(s):
    """Trimmt Whitespace und doppelte Quotes aus exportierten UIDs."""
    if isinstance(s, str):
        return s.strip().strip('"').strip()
    return s


def load_kati_refs(path: Path) -> Dict[str, Set[str]]:
    """
    Laedt die KATI-Edge-Liste und aggregiert sie zu einem Dict
    UID -> Set[UID_ref]. Robust gegen umschliessende Quotes und
    Whitespace in Header und Werten.
    """
    df = pd.read_csv(path, skipinitialspace=True, dtype=str)
    df.columns = [c.strip().strip('"') for c in df.columns]
    if "UID" not in df.columns or "UID_ref" not in df.columns:
        raise ValueError(
            f"KATI-Datei muss Spalten 'UID' und 'UID_ref' enthalten. "
            f"Gefunden: {list(df.columns)}"
        )
    df["UID"] = df["UID"].apply(_normalize_uid)
    df["UID_ref"] = df["UID_ref"].apply(_normalize_uid)
    return df.groupby("UID")["UID_ref"].apply(set).to_dict()


def load_topics(path: Path) -> pd.DataFrame:
    """
    Laedt die Topic-Zuweisungen pro Dokument. Erwartete Spalten:
    UID, topic. Andere Spalten werden ignoriert.
    """
    df = pd.read_csv(path, low_memory=False)
    if "UID" not in df.columns or "topic" not in df.columns:
        raise ValueError(
            f"Topic-Datei muss Spalten 'UID' und 'topic' enthalten. "
            f"Gefunden: {list(df.columns)}"
        )
    df["UID"] = df["UID"].apply(_normalize_uid)
    return df[["UID", "topic"]].copy()


# =============================================================================
# KERN-RUNNER
# =============================================================================

def run_phase(phase: int,
              kati_path: Path,
              topics_path: Path,
              out_path: Path,
              max_pairs: int = MAX_PAIRS,
              global_sample_size: int = GLOBAL_SAMPLE_SIZE,
              seed: int = SEED) -> Tuple[pd.DataFrame, float]:
    """
    Berechnet RO_topic pro Topic und RO_global fuer eine Phase.
    Nutzt mean_pairwise_jaccard aus dem Originalmodul, damit die
    methodische Logik nicht dupliziert wird.
    """
    print(f"[step2b-kati] Phase {phase}: lade KATI-Refs ...")
    refs_by_uid = load_kati_refs(kati_path)
    print(f"             {len(refs_by_uid):,} UIDs mit Ref-Sets")

    print(f"[step2b-kati] Phase {phase}: lade Topic-Zuweisungen ...")
    topics = load_topics(topics_path)
    print(f"             {len(topics):,} Dokumente, "
          f"{topics['topic'].nunique()} Topic-IDs")

    # Ref-Set pro Dokument anhaengen
    topics["ref_set"] = (
        topics["UID"]
        .map(refs_by_uid)
        .apply(lambda x: x if isinstance(x, set) else set())
    )
    topics["has_refs"] = topics["ref_set"].apply(lambda s: len(s) > 0)
    coverage_total = topics["has_refs"].mean()
    print(f"             Coverage (Docs mit Refs): {coverage_total:.1%}")

    # Globales Baseline-Sample (n=500 Zufallsdokumente mit Refs)
    rng = np.random.RandomState(seed)
    valid_idx = topics.index[topics["has_refs"]].tolist()
    sample_size = min(global_sample_size, len(valid_idx))
    g_idx = rng.choice(valid_idx, size=sample_size, replace=False)
    global_sample = topics.loc[g_idx, "ref_set"].tolist()
    ro_global = mean_pairwise_jaccard(global_sample,
                                      max_pairs=max_pairs, seed=seed)
    print(f"[step2b-kati] RO_global (n={sample_size}) = {ro_global:.5f}")

    # Pro Topic: RO_topic + Diagnosen
    rows = []
    topic_ids = sorted([t for t in topics["topic"].unique() if t >= 0])
    for tid in topic_ids:
        mask = topics["topic"] == tid
        topic_refs = topics.loc[mask, "ref_set"].tolist()
        non_empty = [r for r in topic_refs if r]

        ro_topic = mean_pairwise_jaccard(topic_refs,
                                         max_pairs=max_pairs, seed=seed)
        n_refs_mean = (float(np.mean([len(r) for r in non_empty]))
                       if non_empty else 0.0)
        coverage = (len(non_empty) / len(topic_refs)) if topic_refs else 0.0
        ratio = (ro_topic / ro_global) if ro_global > 0 else np.nan

        rows.append({
            "topic":           tid,
            "n_docs":          int(mask.sum()),
            "n_docs_with_refs": len(non_empty),
            "coverage":        round(coverage, 3),
            "RO_topic":        round(ro_topic, 5),
            "RO_global":       round(ro_global, 5),
            "n_refs_mean":     round(n_refs_mean, 1),
            "ratio_vs_global": (round(ratio, 2)
                                if not np.isnan(ratio) else np.nan),
        })

    result = pd.DataFrame(rows).set_index("topic")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path)
    print(f"[step2b-kati] Gespeichert: {out_path}")
    print(f"             n_topics={len(result)}, "
          f"ratio>2: {(result['ratio_vs_global']>2).sum()}, "
          f"ratio<1: {(result['ratio_vs_global']<1).sum()}")
    return result, ro_global


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 2b Runner mit KATI-Edge-List Eingang."
    )
    parser.add_argument("--phase", type=int, choices=[1, 2], required=True,
                        help="Phase 1 (2000-2015) oder Phase 2 (2016-2025).")
    parser.add_argument("--kati-refs", type=Path, default=None,
                        help="Override: Pfad zur KATI-CSV (UID, UID_ref).")
    parser.add_argument("--topics", type=Path, default=None,
                        help="Override: Pfad zur Topic-CSV (UID, topic).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Override: Ausgabe-CSV.")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    parser.add_argument("--global-sample-size", type=int,
                        default=GLOBAL_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    defaults = DEFAULT_PATHS[args.phase]
    kati_path = args.kati_refs or defaults["kati_refs"]
    topics_path = args.topics or defaults["topics"]
    out_path = args.out or defaults["out"]

    run_phase(
        phase=args.phase,
        kati_path=kati_path,
        topics_path=topics_path,
        out_path=out_path,
        max_pairs=args.max_pairs,
        global_sample_size=args.global_sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
