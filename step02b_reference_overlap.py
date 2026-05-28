
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations

try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = Path(__file__).parent / "output"



def parse_cited_references(ref_string) -> set:
    if pd.isna(ref_string) or not isinstance(ref_string, str):
        return set()

    refs = set()
    for raw in ref_string.split(";"):
        r = raw.strip()
        if not r:
            continue
        doi_idx = r.lower().find("doi ")
        if doi_idx >= 0:
            doi = r[doi_idx + 4:].strip().lower()
            doi = doi.split()[0] if doi else ""
            if doi:
                refs.add("doi:" + doi)
                continue
        parts = [p.strip().lower() for p in r.split(",")[:3] if p.strip()]
        if parts:
            refs.add("|".join(parts))
    return refs


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def mean_pairwise_jaccard(ref_sets: list, max_pairs: int = 200,
                           seed: int = 42) -> float:
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



def compute_reference_overlap(df: pd.DataFrame, labels: np.ndarray,
                              topic_ids: list,
                              max_pairs_per_topic: int = 200,
                              global_sample_size: int = 500,
                              seed: int = 42) -> pd.DataFrame:
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

    ref_sets_all = df[ref_col].apply(parse_cited_references).tolist()

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



def run(df: pd.DataFrame = None, labels: np.ndarray = None,
        topic_ids: list = None, output_dir: Path = None):
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df is None or labels is None or topic_ids is None:
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
