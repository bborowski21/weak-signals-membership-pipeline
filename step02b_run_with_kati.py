
import argparse
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

from step02b_reference_overlap import jaccard, mean_pairwise_jaccard



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



def _normalize_uid(s):
    if isinstance(s, str):
        return s.strip().strip('"').strip()
    return s


def load_kati_refs(path: Path) -> Dict[str, Set[str]]:
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
    df = pd.read_csv(path, low_memory=False)
    if "UID" not in df.columns or "topic" not in df.columns:
        raise ValueError(
            f"Topic-Datei muss Spalten 'UID' und 'topic' enthalten. "
            f"Gefunden: {list(df.columns)}"
        )
    df["UID"] = df["UID"].apply(_normalize_uid)
    return df[["UID", "topic"]].copy()



def run_phase(phase: int,
              kati_path: Path,
              topics_path: Path,
              out_path: Path,
              max_pairs: int = MAX_PAIRS,
              global_sample_size: int = GLOBAL_SAMPLE_SIZE,
              seed: int = SEED) -> Tuple[pd.DataFrame, float]:
    print(f"[step2b-kati] Phase {phase}: lade KATI-Refs ...")
    refs_by_uid = load_kati_refs(kati_path)
    print(f"             {len(refs_by_uid):,} UIDs mit Ref-Sets")

    print(f"[step2b-kati] Phase {phase}: lade Topic-Zuweisungen ...")
    topics = load_topics(topics_path)
    print(f"             {len(topics):,} Dokumente, "
          f"{topics['topic'].nunique()} Topic-IDs")

    topics["ref_set"] = (
        topics["UID"]
        .map(refs_by_uid)
        .apply(lambda x: x if isinstance(x, set) else set())
    )
    topics["has_refs"] = topics["ref_set"].apply(lambda s: len(s) > 0)
    coverage_total = topics["has_refs"].mean()
    print(f"             Coverage (Docs mit Refs): {coverage_total:.1%}")

    rng = np.random.RandomState(seed)
    valid_idx = topics.index[topics["has_refs"]].tolist()
    sample_size = min(global_sample_size, len(valid_idx))
    g_idx = rng.choice(valid_idx, size=sample_size, replace=False)
    global_sample = topics.loc[g_idx, "ref_set"].tolist()
    ro_global = mean_pairwise_jaccard(global_sample,
                                      max_pairs=max_pairs, seed=seed)
    print(f"[step2b-kati] RO_global (n={sample_size}) = {ro_global:.5f}")

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
