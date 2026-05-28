
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from collections import Counter

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import hdbscan

from config import (
    DATA_PATH, OUTPUT_DIR,
    PHASE_YEAR_MIN, PHASE_YEAR_MAX,
    SBERT_MODEL, UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS,
    UMAP_MIN_DIST, UMAP_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE, HDBSCAN_MIN_SAMPLES, HDBSCAN_CLUSTER_METHOD,
    CTFIDF_TOP_N_WORDS,
)



def load_and_clean(path: Path) -> pd.DataFrame:
    print(f"Lade Daten aus {path.name}...")
    df = pd.read_csv(path)

    df["text"] = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")

    df["text_clean"] = df["text"].apply(
        lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", t.lower())).strip()
        if isinstance(t, str) else ""
    )

    n0 = len(df)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[(df["Year"] >= PHASE_YEAR_MIN) & (df["Year"] <= PHASE_YEAR_MAX)].copy()
    n_year = len(df)
    print(f"  {n0} → {n_year} Dokumente (Year-Filter "
          f"[{PHASE_YEAR_MIN}, {PHASE_YEAR_MAX}])")

    df = df[df["text_clean"].str.split().str.len() >= 10].reset_index(drop=True)
    print(f"  {n_year} → {len(df)} Dokumente (Längen-Filter ≥ 10 Wörter)")
    print(f"  Zeitraum: {int(df['Year'].min())} – {int(df['Year'].max())}")
    return df



def compute_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    print(f"\nLade SBERT-Modell: {model_name}...")
    model = SentenceTransformer(model_name)

    device = model.device
    print(f"  Device: {device}")

    print(f"  Erzeuge Embeddings für {len(texts)} Dokumente...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,
        normalize_embeddings=True,
    )

    print(f"  Embedding-Matrix: {embeddings.shape}")
    return embeddings



def reduce_dimensions(embeddings: np.ndarray) -> np.ndarray:
    print(f"\nUMAP: {embeddings.shape[1]}D → {UMAP_N_COMPONENTS}D...")
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=42,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)
    print(f"  Ergebnis: {reduced.shape}")
    return reduced



def cluster_topics(embeddings_reduced: np.ndarray) -> np.ndarray:
    print(f"\nHDBSCAN (min_cluster={HDBSCAN_MIN_CLUSTER_SIZE}, "
          f"min_samples={HDBSCAN_MIN_SAMPLES})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=HDBSCAN_CLUSTER_METHOD,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings_reduced)

    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    noise = (labels == -1).sum()
    print(f"  Topics: {n_topics}")
    print(f"  Noise: {noise} ({noise / len(labels) * 100:.1f}%)")
    return labels



def compute_topic_keywords(
    df: pd.DataFrame, labels: np.ndarray, top_n: int = CTFIDF_TOP_N_WORDS
) -> dict:
    print(f"\nc-TF-IDF Topic-Keywords (top {top_n} pro Topic)...")

    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    vectorizer.fit(df["text_clean"].tolist())
    vocab = vectorizer.get_feature_names_out()

    topics = sorted(set(labels[labels >= 0]))
    topic_texts = []
    for tid in topics:
        mask = labels == tid
        topic_texts.append(" ".join(df.loc[mask, "text_clean"].tolist()))

    topic_tfidf = vectorizer.transform(topic_texts)

    topic_keywords = {}
    for i, tid in enumerate(topics):
        row = topic_tfidf[i].toarray().flatten()
        top_idx = row.argsort()[-top_n:][::-1]
        topic_keywords[tid] = [(vocab[j], float(row[j])) for j in top_idx if row[j] > 0]

    return topic_keywords, vectorizer



def compute_tem_metrics(df: pd.DataFrame, labels: np.ndarray) -> tuple:
    print("\nTemporale Metriken (TEM)...")
    df_t = df.copy()
    df_t["topic"] = labels
    df_t = df_t[df_t["topic"] >= 0]

    counts = df_t.groupby(["Year", "topic"]).size().unstack(fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)

    metrics = []
    for tid in proportions.columns:
        series = proportions[tid].sort_index()
        avg_prop = series.mean()

        first_nz = series[series > 0]
        if len(first_nz) >= 2:
            p_s, p_e = first_nz.iloc[0], first_nz.iloc[-1]
            n_y = first_nz.index[-1] - first_nz.index[0]
            growth = (p_e / p_s) ** (1 / n_y) - 1 if n_y > 0 and p_s > 0 else 0.0
        else:
            growth = 0.0

        metrics.append({
            "topic": tid,
            "avg_proportion": avg_prop,
            "growth_rate": growth,
        })

    return pd.DataFrame(metrics), proportions



def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SCHRITT 1: EINHEITENBILDUNG — SBERT + UMAP + HDBSCAN")
    print("=" * 70)

    df = load_and_clean(DATA_PATH)

    embeddings = compute_embeddings(df["text"].tolist(), SBERT_MODEL)

    embeddings_reduced = reduce_dimensions(embeddings)

    labels = cluster_topics(embeddings_reduced)

    topic_keywords, vectorizer = compute_topic_keywords(df, labels)

    tem_df, proportions = compute_tem_metrics(df, labels)

    print("\nSpeichere Ergebnisse...")

    df["topic"] = labels
    desired_cols = [
        "Title", "Year", "topic",
        "UID", "DOI",
        "Author Keywords", "Keywords Plus", "WoS Categories",
        "Times Cited, WoS Core",
        "Document Type", "Source Title",
        "Author Full Names", "ORCIDs", "Affiliations", "Addresses",
        "RTW", "CTW",
        "Index Keywords", "Cited by", "Source title", "Authors",
        "Authors with affiliations",
    ]
    out_cols = [c for c in desired_cols if c in df.columns]
    df[out_cols].to_csv(
        OUTPUT_DIR / "topic_assignments.csv", index=False
    )

    kw_records = []
    for tid, kws in topic_keywords.items():
        for kw, score in kws:
            kw_records.append({"topic": tid, "keyword": kw, "score": score})
    pd.DataFrame(kw_records).to_csv(OUTPUT_DIR / "topic_keywords.csv", index=False)

    tem_df.to_csv(OUTPUT_DIR / "tem_metrics.csv", index=False)
    proportions.to_csv(OUTPUT_DIR / "topic_proportions_yearly.csv")

    with open(OUTPUT_DIR / "model_results.pkl", "wb") as f:
        pickle.dump({
            "labels": labels,
            "embeddings_sbert": embeddings,
            "embeddings_reduced": embeddings_reduced,
        }, f)

    with open(OUTPUT_DIR / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    n_topics = len(topic_keywords)
    topic_sizes = pd.Series(labels[labels >= 0]).value_counts()

    print(f"\n{'=' * 70}")
    print(f"ERGEBNIS: {n_topics} Topics aus {len(df)} Dokumenten")
    print(f"{'=' * 70}")

    print(f"\nTop 15 Topics nach Größe:")
    for tid, size in topic_sizes.head(15).items():
        kw = topic_keywords.get(tid, [])
        kw_str = ", ".join([w for w, _ in kw[:5]])
        print(f"  T{tid:3d}: {size:5d} docs | {kw_str}")

    print(f"\nTop 10 nach Wachstumsrate:")
    for _, row in tem_df.sort_values("growth_rate", ascending=False).head(10).iterrows():
        tid = int(row["topic"])
        kw = topic_keywords.get(tid, [])
        kw_str = ", ".join([w for w, _ in kw[:3]])
        print(f"  T{tid:3d}: growth={row['growth_rate']:+.3f}, "
              f"prop={row['avg_proportion']:.4f} | {kw_str}")

    print(f"\nAlles gespeichert in {OUTPUT_DIR}/")
    return df, labels, embeddings, embeddings_reduced, topic_keywords


if __name__ == "__main__":
    run()
