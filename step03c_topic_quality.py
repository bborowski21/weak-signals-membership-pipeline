"""
Topic-Modell-Güte: Kohärenz (C_v, C_NPMI, C_UMass) und Diversität pro Phase.

Berechnet ergänzend zur Konstruktvalidierung (EFA, MTMM) die in der
Topic-Modeling-Literatur etablierten Intrinsic-Metriken auf den
bereits trainierten BERTopic-Lösungen.

- Kohärenz: Röder, Both, Hinneburg (2015) für C_v und C_NPMI;
  Mimno et al. (2011) für C_UMass. Top-N Wörter pro Topic: 10.
- Diversität: Dieng, Ruiz, Blei (2020). Angewandte Top-N: 100
  (anschlussfähig an gängige BERTopic-Reports; weicht vom Dieng-
  Original-Default top-25 ab, was im Methodenabschnitt benannt wird).
- Outlier-Topic (HDBSCAN-Label = -1) wird konsequent ausgeschlossen.
- Tokenisierung identisch zu step01_topic_modeling.py.

Ausgabe je Phase:
  output_phase{1,2}/topic_quality_summary.json
  output_phase{1,2}/topic_quality_per_topic.csv
Aggregat:
  topic_quality_results.json
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parent

PHASE_CONFIG = {
    "phase1": {
        "input_csv": ROOT.parent / "wos_qc_phase1_2000_2015_clean.csv",
        "model_pkl": ROOT / "output_phase1" / "model_results.pkl",
        "output_dir": ROOT / "output_phase1",
        "year_min": 2000,
        "year_max": 2015,
    },
    "phase2": {
        "input_csv": ROOT.parent / "wos_qc_phase2_2016_2025_clean.csv",
        "model_pkl": ROOT / "output_phase2" / "model_results.pkl",
        "output_dir": ROOT / "output_phase2",
        "year_min": 2016,
        "year_max": 2025,
    },
}

TOP_N_COHERENCE = 10
TOP_N_DIVERSITY = 100
COHERENCE_MEASURES = ("c_v", "c_npmi", "u_mass")
MIN_WORDS = 10


def clean_text_step01(t):
    """Tokenisierung identisch zu step01_topic_modeling.load_and_clean."""
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", t.lower())).strip()


def load_phase_corpus(cfg):
    df = pd.read_csv(cfg["input_csv"])
    df["text"] = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")
    df["text_clean"] = df["text"].apply(clean_text_step01)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[(df["Year"] >= cfg["year_min"]) & (df["Year"] <= cfg["year_max"])]
    df = df[df["text_clean"].str.split().str.len() >= MIN_WORDS].reset_index(drop=True)
    return df


def load_labels(model_pkl):
    with open(model_pkl, "rb") as f:
        d = pickle.load(f)
    return np.asarray(d["labels"])


def extract_topic_words_ctfidf(documents, labels, top_n):
    """c-TF-IDF Top-N Wörter pro Topic (BERTopic-Logik, manuell reimplementiert).

    c-TF: Klassen-relativ-Frequenz (Term-Anteil innerhalb des Topics)
    c-IDF: log(1 + A / f_t), mit A = mittl. Klassen-Frequenz, f_t = Term-Klassenfreq.
    """
    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b[a-z][a-z0-9\-]+\b",
        min_df=2,
    )
    X = vectorizer.fit_transform(documents)
    vocab = vectorizer.get_feature_names_out()

    label_array = np.asarray(labels)
    unique_labels = sorted({int(l) for l in label_array if l != -1})
    n_topics = len(unique_labels)
    n_terms = len(vocab)

    cf = np.zeros((n_topics, n_terms), dtype=np.float64)
    for idx, t in enumerate(unique_labels):
        mask = label_array == t
        cf[idx] = X[mask].sum(axis=0).A1

    row_sums = cf.sum(axis=1, keepdims=True)
    tf_norm = cf / np.maximum(row_sums, 1.0)

    avg_freq = cf.sum() / max(n_topics, 1)
    term_class_freq = cf.sum(axis=0)
    idf = np.log(1.0 + avg_freq / np.maximum(term_class_freq, 1.0))

    ctfidf = tf_norm * idf

    topic_terms = {}
    for idx, t in enumerate(unique_labels):
        top_indices = np.argsort(-ctfidf[idx])[:top_n]
        topic_terms[t] = [vocab[i] for i in top_indices]
    return topic_terms


def compute_coherence_per_topic(topic_terms, tokenized_docs, dictionary, measure):
    """Gensim CoherenceModel mit explizit gesetztem topn (Default wäre 20)."""
    topics_list = list(topic_terms.values())
    cm = CoherenceModel(
        topics=topics_list,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence=measure,
        topn=TOP_N_COHERENCE,
    )
    return cm.get_coherence_per_topic()


def compute_diversity(topic_terms, top_n):
    """Dieng et al. 2020: |unique words across top-N| / (N * top_n)."""
    n_topics = len(topic_terms)
    if n_topics == 0 or top_n == 0:
        return float("nan")
    all_words = set()
    for words in topic_terms.values():
        all_words.update(words[:top_n])
    return len(all_words) / (n_topics * top_n)


def summarize_array(arr):
    arr = np.asarray(arr, dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "n_valid": 0, "n_nan": int(len(arr))}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid, ddof=0)),
        "median": float(np.median(valid)),
        "n_valid": int(len(valid)),
        "n_nan": int(len(arr) - len(valid)),
    }


def run_phase(name, cfg):
    print(f"\n=== {name.upper()} ===")
    df = load_phase_corpus(cfg)
    n_docs = len(df)
    print(f"  Korpus: {n_docs} Dokumente nach Säuberung (Year+Längenfilter)")

    labels = load_labels(cfg["model_pkl"])
    print(f"  Labels: {len(labels)}")
    if len(labels) != n_docs:
        raise RuntimeError(f"Längendifferenz Labels ({len(labels)}) vs. Korpus ({n_docs})")

    documents = df["text_clean"].tolist()
    tokenized = [doc.split() for doc in documents]

    print(f"  Extrahiere Top-{TOP_N_DIVERSITY} via c-TF-IDF...")
    topic_words_top100 = extract_topic_words_ctfidf(documents, labels, TOP_N_DIVERSITY)
    n_topics = len(topic_words_top100)
    print(f"  Topics (ohne Outlier): {n_topics}")

    topic_words_top10 = {t: w[:TOP_N_COHERENCE] for t, w in topic_words_top100.items()}

    print(f"  Gensim-Dictionary...")
    dictionary = Dictionary(tokenized)

    coherence = {}
    for measure in COHERENCE_MEASURES:
        print(f"  Kohärenz {measure} (topn={TOP_N_COHERENCE})...")
        per_topic = compute_coherence_per_topic(topic_words_top10, tokenized, dictionary, measure)
        summary = summarize_array(per_topic)
        summary["per_topic"] = [float(x) for x in per_topic]
        coherence[measure] = summary
        print(f"    mean={summary['mean']:.4f}, std={summary['std']:.4f}, "
              f"median={summary['median']:.4f}, NaN={summary['n_nan']}")

    diversity = compute_diversity(topic_words_top100, TOP_N_DIVERSITY)
    print(f"  Diversität (top-{TOP_N_DIVERSITY}): {diversity:.4f}")

    label_array = np.asarray(labels)
    n_outlier = int((label_array == -1).sum())
    outlier_share = n_outlier / max(len(label_array), 1)

    topic_sizes = pd.Series(label_array[label_array != -1]).value_counts()

    summary = {
        "phase": name,
        "n_documents": int(len(label_array)),
        "n_topics": int(n_topics),
        "n_outlier_docs": int(n_outlier),
        "outlier_share": float(outlier_share),
        "topic_size_min": int(topic_sizes.min()),
        "topic_size_max": int(topic_sizes.max()),
        "topic_size_median": float(topic_sizes.median()),
        "topic_diversity_top100": float(diversity),
        "top_n_coherence": TOP_N_COHERENCE,
        "top_n_diversity": TOP_N_DIVERSITY,
        "coherence": coherence,
    }

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    summary_short = {k: v for k, v in summary.items() if k != "coherence"}
    summary_short["coherence"] = {
        m: {k: v for k, v in d.items() if k != "per_topic"}
        for m, d in coherence.items()
    }
    with open(cfg["output_dir"] / "topic_quality_summary.json", "w") as f:
        json.dump(summary_short, f, indent=2, ensure_ascii=False)

    per_topic_df = pd.DataFrame({
        "topic": list(topic_words_top10.keys()),
        "c_v": coherence["c_v"]["per_topic"],
        "c_npmi": coherence["c_npmi"]["per_topic"],
        "u_mass": coherence["u_mass"]["per_topic"],
    })
    per_topic_df.to_csv(cfg["output_dir"] / "topic_quality_per_topic.csv", index=False)

    return summary


def main():
    results = {}
    for name, cfg in PHASE_CONFIG.items():
        results[name] = run_phase(name, cfg)

    print("\n\n=== AGGREGAT ===")
    for name, s in results.items():
        c = s["coherence"]
        print(f"\n{name} ({s['n_topics']} Topics, {s['outlier_share']:.1%} Outlier-Anteil)")
        for m in ("c_v", "c_npmi", "u_mass"):
            print(f"  {m:7s}: mean={c[m]['mean']:.3f} std={c[m]['std']:.3f} median={c[m]['median']:.3f}")
        print(f"  diversity (top-{TOP_N_DIVERSITY}): {s['topic_diversity_top100']:.3f}")

    agg = {}
    for name, s in results.items():
        d = {k: v for k, v in s.items() if k != "coherence"}
        d["coherence"] = {m: {k: v for k, v in cd.items() if k != "per_topic"}
                          for m, cd in s["coherence"].items()}
        agg[name] = d
    with open(ROOT / "topic_quality_results.json", "w") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    print(f"\nAggregat-JSON: {ROOT / 'topic_quality_results.json'}")


if __name__ == "__main__":
    main()
