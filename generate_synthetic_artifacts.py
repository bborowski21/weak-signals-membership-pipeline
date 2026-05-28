
import argparse
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

if "sentence_transformers" not in sys.modules:
    _st_stub = types.ModuleType("sentence_transformers")
    _st_stub.SentenceTransformer = lambda *a, **k: None
    sys.modules["sentence_transformers"] = _st_stub


_WORDPOOL = [
    "quantum", "neural", "graph", "energy", "polymer", "catalysis",
    "ontology", "swarm", "metabolic", "blockchain", "synthesis",
    "fault", "stochastic", "spectroscopy", "transition", "hybrid",
    "diffusion", "adaptation", "kernel", "cascade", "emerging",
    "framework", "regime", "feedback", "anomaly", "precision",
]

_JOURNALS = [
    f"Journal of Synthetic Field {i}" for i in range(30)
]
_INSTITUTIONS = [f"Institute {chr(65 + i)}" for i in range(20)]
_COUNTRIES = ["DEU", "USA", "GBR", "CHN", "JPN", "FRA", "NLD", "CHE",
              "ITA", "ESP", "CAN", "AUS", "KOR", "SWE", "BRA"]
_DOCTYPES = (["Article"] * 80
             + ["Review"] * 12
             + ["Conference Paper"] * 6
             + ["Editorial"] * 2)
_WOS_CATS = [f"Synth Cat {i:02d}" for i in range(40)]


def _make_text(rng: np.random.Generator, cluster_id: int) -> str:
    base = rng.choice(_WORDPOOL, size=4, replace=False)
    extra = rng.choice(_WORDPOOL, size=8)
    seed_words = [f"cluster{cluster_id}_word{i}" for i in range(3)]
    words = list(base) + list(extra) + seed_words
    rng.shuffle(words)
    return " ".join(words) + ". " + " ".join(rng.choice(_WORDPOOL, size=15))


def _make_keywords(rng: np.random.Generator, cluster_id: int) -> str:
    n = rng.integers(3, 7)
    base = list(rng.choice(_WORDPOOL, size=n, replace=False))
    base.append(f"topic_{cluster_id}_kw")
    return "; ".join(base)


def _make_authors(rng: np.random.Generator) -> str:
    n = rng.integers(2, 6)
    return "; ".join(f"Author {rng.integers(0, 500)}" for _ in range(n))


def _make_affiliations(rng: np.random.Generator) -> str:
    n = rng.integers(1, 4)
    return "; ".join(rng.choice(_INSTITUTIONS, size=n, replace=False))


def _make_addresses(rng: np.random.Generator) -> str:
    n = rng.integers(1, 4)
    return "; ".join(f"Inst, City, {c}" for c in
                     rng.choice(_COUNTRIES, size=n, replace=True))


def _make_orcids(rng: np.random.Generator) -> str:
    if rng.random() < 0.3:
        return ""
    n = rng.integers(1, 3)
    return "; ".join(
        f"{rng.integers(1000, 9999):04d}-{rng.integers(1000, 9999):04d}-"
        f"{rng.integers(1000, 9999):04d}-{rng.integers(1000, 9999):04d}"
        for _ in range(n))


def build_df(n_docs: int, n_clusters: int, year_min: int,
             year_max: int, rng: np.random.Generator) -> pd.DataFrame:
    docs_per_cluster = n_docs // n_clusters
    cluster_assign = np.repeat(np.arange(n_clusters), docs_per_cluster)
    if len(cluster_assign) < n_docs:
        rest = n_docs - len(cluster_assign)
        cluster_assign = np.concatenate(
            [cluster_assign, rng.integers(0, n_clusters, rest)])
    rng.shuffle(cluster_assign)

    years = rng.choice(
        np.arange(year_min, year_max + 1),
        size=n_docs,
        p=_year_weights(year_min, year_max),
    )

    rows = []
    for i in range(n_docs):
        cid = int(cluster_assign[i])
        rows.append({
            "Title":            f"Synthetic title {i} (cluster {cid})",
            "Abstract":         _make_text(rng, cid),
            "Year":             int(years[i]),
            "Author Keywords":  _make_keywords(rng, cid),
            "Keywords Plus":    _make_keywords(rng, cid),
            "Index Keywords":   _make_keywords(rng, cid),
            "Document Type":    rng.choice(_DOCTYPES),
            "Source title":     rng.choice(_JOURNALS),
            "Authors":          _make_authors(rng),
            "Affiliations":     _make_affiliations(rng),
            "Addresses":        _make_addresses(rng),
            "ORCIDs":           _make_orcids(rng),
            "WoS Categories":   "; ".join(rng.choice(_WOS_CATS, size=2,
                                                       replace=False)),
            "Times Cited":      int(max(0, rng.normal(15, 18))),
            "Cited by":         int(max(0, rng.normal(15, 18))),
            "Usage Count 180":  int(max(0, rng.normal(10, 12))),
            "Funding Orgs":     "Synthetic Funder",
        })

    df = pd.DataFrame(rows)

    df["text"] = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")
    df["text_clean"] = df["text"].str.lower().str.replace(
        r"[^a-z0-9\s\-]", " ", regex=True).str.replace(r"\s+", " ", regex=True
    ).str.strip()

    df["_synthetic_cluster"] = cluster_assign
    return df


def _year_weights(year_min: int, year_max: int) -> np.ndarray:
    n = year_max - year_min + 1
    w = np.linspace(1.0, 2.0, n)
    return w / w.sum()



def build_embeddings(df: pd.DataFrame, dim: int,
                       rng: np.random.Generator,
                       noise_scale: float = 0.08) -> np.ndarray:
    n_clusters = int(df["_synthetic_cluster"].max() + 1)

    centroids = np.zeros((n_clusters, dim), dtype=np.float32)
    for c in range(n_clusters):
        centroids[c, c] = 1.0
        centroids[c] += rng.normal(0, 0.02, size=dim)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9

    noise = rng.normal(0, noise_scale, size=(len(df), dim))
    emb = centroids[df["_synthetic_cluster"].to_numpy()] + noise
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    return emb.astype(np.float32)



def build_baseline(df: pd.DataFrame, emb: np.ndarray) -> tuple:
    import umap
    import hdbscan
    from config import (UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS,
                          UMAP_MIN_DIST, UMAP_METRIC,
                          HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES,
                          HDBSCAN_CLUSTER_METHOD)

    print(f"  UMAP: {emb.shape[1]}D -> {UMAP_N_COMPONENTS}D ...", flush=True)
    reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS,
                          n_neighbors=UMAP_N_NEIGHBORS,
                          min_dist=UMAP_MIN_DIST,
                          metric=UMAP_METRIC, random_state=42)
    reduced = reducer.fit_transform(emb)

    print(f"  HDBSCAN: min_cluster={HDBSCAN_MIN_CLUSTER_SIZE} ...",
          flush=True)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=HDBSCAN_CLUSTER_METHOD)
    labels = clusterer.fit_predict(reduced)
    n_topics = len(set(labels[labels >= 0]))
    print(f"  Topics: {n_topics} | Noise: {(labels == -1).sum()}/"
          f"{len(labels)}", flush=True)
    return labels, reduced


def build_indicators(df: pd.DataFrame, labels: np.ndarray,
                       emb: np.ndarray, reduced: np.ndarray) -> tuple:
    from step01_topic_modeling import compute_tem_metrics
    from step02_indicators import compute_all_indicators
    print("  TEM-Metriken (proportions + tem) ...", flush=True)
    tem_df, proportions = compute_tem_metrics(df, labels)
    print("  compute_all_indicators (17 Indikatoren) ...", flush=True)
    ind_df = compute_all_indicators(
        df=df, labels=labels,
        embeddings_sbert=emb, embeddings_reduced=reduced,
        proportions=proportions, tem_metrics=tem_df)
    print(f"  Indikator-Tabelle: {ind_df.shape}", flush=True)
    return ind_df, proportions, tem_df



def save_artifacts(output_dir: Path, df: pd.DataFrame, labels: np.ndarray,
                     emb: np.ndarray, reduced: np.ndarray,
                     ind_df: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "step1_artifacts.pkl", "wb") as f:
        pickle.dump({
            "df":                df.drop(columns=["_synthetic_cluster"]),
            "labels":            labels,
            "embeddings_sbert":  emb,
            "embeddings_reduced": reduced,
        }, f)
    with open(output_dir / "step2_artifacts.pkl", "wb") as f:
        pickle.dump({"indicator_df": ind_df}, f)
    print(f"\n  Artefakte gespeichert: {output_dir}/", flush=True)



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic step1/step2-Artefakte fuer Smoke-Test "
                    "des BERTopic-Hyperparameter-Runners.")
    parser.add_argument("--n-docs",       type=int, default=800)
    parser.add_argument("--n-clusters",   type=int, default=12)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--year-min",     type=int, default=2014)
    parser.add_argument("--year-max",     type=int, default=2023)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--noise-scale",  type=float, default=0.08,
                          help="Streuung pro Cluster (klein = saubere "
                               "Cluster; gross = unscharfe Grenzen, "
                               "variable Spearman-rho-Werte).")
    parser.add_argument("--output-dir",   default="output_smoke")
    parser.add_argument("--run-sensitivity", action="store_true",
                          help="Im Anschluss den BERTopic-Hyperparameter-"
                               "Sensitivitaets-Batch ausfuehren.")
    args = parser.parse_args()

    print("=" * 70)
    print("SYNTHETIC ARTIFACT GENERATOR (Smoke-Test)")
    print("=" * 70)
    print(f"n_docs={args.n_docs} | n_clusters={args.n_clusters} | "
          f"dim={args.embedding_dim} | seed={args.seed}")
    print(f"Years: {args.year_min}-{args.year_max} | "
          f"output_dir={args.output_dir}")

    rng = np.random.default_rng(args.seed)

    print("\n[1/4] Baue synthetisches DataFrame ...")
    df = build_df(args.n_docs, args.n_clusters,
                    args.year_min, args.year_max, rng)
    print(f"      df.shape = {df.shape}, "
          f"Spalten = {len(df.columns)}")

    print(f"\n[2/4] Baue geclusterte Embeddings (Cosine-normiert, "
          f"noise_scale={args.noise_scale}) ...")
    emb = build_embeddings(df, args.embedding_dim, rng,
                             noise_scale=args.noise_scale)
    print(f"      embeddings.shape = {emb.shape}")

    print("\n[3/4] Baseline UMAP + HDBSCAN + 17 Indikatoren ...")
    labels, reduced = build_baseline(df, emb)
    ind_df, _props, _tem = build_indicators(df, labels, emb, reduced)

    print("\n[4/4] Persistenz ...")
    save_artifacts(Path(args.output_dir), df, labels, emb, reduced, ind_df)

    print("\n" + "=" * 70)
    if args.run_sensitivity:
        print("Starte BERTopic-Hyperparameter-Sensitivitaet (Smoke-Test) ...")
        print("=" * 70)
        from step05_sensitivity import bertopic_hyperparameter_sensitivity
        out = bertopic_hyperparameter_sensitivity(
            df=df.drop(columns=["_synthetic_cluster"]),
            baseline_labels=labels,
            baseline_indicators=ind_df,
            embeddings_sbert=emb,
        )
        out_path = Path(args.output_dir) / "sensitivity_parameter_hparam.csv"
        out.to_csv(out_path, index=False)
        print(f"\n[smoke] Ergebnisse gespeichert: {out_path}")
        print("\n--- Spearman-rho (mean ueber 17 Indikatoren) ---")
        print(out[["param", "value", "n_topics",
                    "n_matched_pairs", "spearman_rho_mean"]]
              .to_string(index=False))
    else:
        print("FERTIG. Naechster Schritt:")
        print(f"  python run_sensitivity_hparam.py "
              f"--output-dir {args.output_dir}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
