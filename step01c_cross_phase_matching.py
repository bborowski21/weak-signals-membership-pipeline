"""
Cross-Phase Topic Matching (Schritt 1c)
========================================

Verbindet die unabhängig pro Phase trainierten BERTopic-Modelle über ein
Hybrid-Maß aus zwei komplementären Signalquellen:

  (1) Topic-Repräsentations-Cosine
        Aus topic_keywords.csv: jedes Topic ist ein gewichteter Bag-of-Words.
        Dieser Vektor wird als TF-IDF-ähnliche Repräsentation auf das gemeinsame
        Vokabular projiziert; Cosine-Distanz zwischen Phasen-Topics.

  (2) Top-K Keyword-Jaccard
        Mengenbasierter Overlap der Top-K Keywords beider Topics.

  Optional (3) SBERT-Centroid-Cosine
        Wenn model_results.pkl mit BERTopic + topic_embeddings_ verfügbar:
        Cosine zwischen den nativen Topic-Embeddings.
        Nicht erforderlich für Lauffähigkeit; aktiviert mit ``--with-sbert``.

Hybrid-Score:
    score = alpha * representation_cosine + (1 - alpha) * keyword_jaccard
    Default alpha = 0.6 (Repräsentations-Cosine dominiert leicht, Jaccard als
    robuste lexikalische Verankerung).

Methodische Begründung:
  Die beiden Signale haben unterschiedliche Versagensmodi:
   - Cosine ist sensitiv für gemeinsame Vokabularverteilung, kann aber durch
     Topic-Größenunterschiede verzerrt werden.
   - Jaccard ist robust gegen Größeneffekte, ignoriert aber Gewichtungen.
  Hybrid mit konfigurierbarem alpha erlaubt Sensitivitätsanalysen.

Output:
    output_cross_phase/topic_matches.csv
        phase1_topic, phase2_topic, cosine, jaccard, hybrid,
        rank_p1_to_p2, rank_p2_to_p1, mutual_best
    output_cross_phase/match_diagnostics.txt
        Verteilungs-Statistiken, unsichere Matches.

Aufruf:
    python step01c_cross_phase_matching.py
    python step01c_cross_phase_matching.py --alpha 0.5 --topk 15
    python step01c_cross_phase_matching.py --threshold 0.3   # Konfidenz-Schwelle

Autor: Ben Borowski
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# KONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_PHASE1_DIR = BASE_DIR / "output_phase1"
DEFAULT_PHASE2_DIR = BASE_DIR / "output_phase2"
DEFAULT_OUT_DIR = BASE_DIR / "output_cross_phase"

DEFAULT_ALPHA = 0.6
DEFAULT_TOPK = 15
DEFAULT_THRESHOLD = 0.25  # Hybrid-Score unter dieser Schwelle = unsicher


# =============================================================================
# REPRÄSENTATIONS-AUFBAU
# =============================================================================

def load_topic_keywords(out_dir: Path, topk: int) -> pd.DataFrame:
    """Liest topic_keywords.csv und beschränkt auf Top-K pro Topic.

    Erwartet Spalten: topic, keyword, score
    """
    path = out_dir / "topic_keywords.csv"
    if not path.exists():
        raise FileNotFoundError(f"topic_keywords.csv fehlt in {out_dir}")
    df = pd.read_csv(path)
    if not {"topic", "keyword", "score"}.issubset(df.columns):
        raise ValueError(
            f"topic_keywords.csv muss Spalten topic, keyword, score haben. "
            f"Gefunden: {list(df.columns)}"
        )
    df = df[df["topic"] != -1].copy()  # Outlier-Cluster ausklammern
    # NaN-Keywords entfernen (entstehen durch leere c-TF-IDF-Tokens)
    df = df.dropna(subset=["keyword"]).copy()
    df["keyword"] = df["keyword"].astype(str)
    df = df.sort_values(["topic", "score"], ascending=[True, False])
    df = df.groupby("topic").head(topk).copy()
    return df


def build_vocab(p1_kw: pd.DataFrame, p2_kw: pd.DataFrame) -> Dict[str, int]:
    """Vereinigtes Vokabular aus beiden Phasen → Wort-Index-Mapping."""
    vocab = sorted(set(p1_kw["keyword"]).union(p2_kw["keyword"]))
    return {w: i for i, w in enumerate(vocab)}


def build_topic_matrix(
    kw_df: pd.DataFrame, vocab: Dict[str, int]
) -> Tuple[csr_matrix, List[int]]:
    """Sparse-Matrix [n_topics × n_vocab] mit Keyword-Scores als Gewichte.

    Zeilen-Index ist Topic-ID-Reihenfolge (sortiert).
    """
    topic_ids = sorted(kw_df["topic"].unique())
    topic_to_row = {t: i for i, t in enumerate(topic_ids)}

    rows, cols, data = [], [], []
    for _, r in kw_df.iterrows():
        kw = r["keyword"]
        if kw not in vocab:
            continue
        rows.append(topic_to_row[r["topic"]])
        cols.append(vocab[kw])
        data.append(float(r["score"]))

    mat = csr_matrix(
        (data, (rows, cols)),
        shape=(len(topic_ids), len(vocab)),
        dtype=np.float64,
    )
    return mat, topic_ids


def jaccard_topk(set_a: set, set_b: set) -> float:
    """Klassischer Jaccard-Index zweier Mengen."""
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


# =============================================================================
# SBERT-CENTROIDE (aus 384D-Embeddings + Cluster-Labels berechnet)
# =============================================================================
#
# step01_topic_modeling.py persistiert in model_results.pkl ein Dict mit:
#   - 'labels'             : ndarray [n_docs]       Cluster-Label pro Dokument
#   - 'embeddings_sbert'   : ndarray [n_docs, 384]  Original-SBERT-Embeddings
#   - 'embeddings_reduced' : ndarray [n_docs, 15]   UMAP-reduzierte Embeddings
#
# Methodisch wichtig: Für phasen-übergreifende Vergleiche verwenden wir
# zwingend die Original-SBERT-Embeddings (384D), NICHT die UMAP-reduzierten.
# UMAP ist unbeaufsichtigt pro Phase gefittet — die 15D-Räume sind nicht
# kommensurabel zwischen Phasen. Das 384D-SBERT-Space ist hingegen über das
# gemeinsame Modell (all-MiniLM-L6-v2) global stabil.

def try_load_topic_centroids(
    out_dir: Path, topic_ids: list[int]
) -> np.ndarray | None:
    """Berechnet Topic-Centroide aus den persistierten 384D-SBERT-Embeddings.

    Args:
        out_dir   : Pfad zu output_phaseN/, enthält model_results.pkl
        topic_ids : geordnete Liste der Topic-IDs, deren Centroide zurück-
                    gegeben werden sollen (in dieser Reihenfolge)

    Returns:
        ndarray der Form (len(topic_ids), 384) mit gemittelten SBERT-
        Embeddings pro Topic. None bei fehlendem Pickle oder unerwartetem
        Format. Topics ohne Dokumente erhalten Null-Vektoren.
    """
    pkl = out_dir / "model_results.pkl"
    if not pkl.exists():
        return None
    try:
        import pickle  # lokal — vermeidet Modul-Top-Level-Cost
        with pkl.open("rb") as f:
            obj = pickle.load(f)
    except Exception as e:  # nosec — alle Fehlertypen abfangen
        print(f"  [info] Pickle nicht ladbar ({type(e).__name__}): "
              f"{str(e)[:80]}")
        return None

    if not isinstance(obj, dict):
        print(f"  [info] model_results.pkl: unerwarteter Typ "
              f"{type(obj).__name__}, erwartet dict")
        return None
    if "labels" not in obj or "embeddings_sbert" not in obj:
        print(f"  [info] model_results.pkl: fehlende Keys "
              f"(gefunden: {list(obj.keys())})")
        return None

    labels = np.asarray(obj["labels"])
    embs = np.asarray(obj["embeddings_sbert"])
    if labels.shape[0] != embs.shape[0]:
        print(f"  [info] Labels/Embeddings Mismatch: "
              f"{labels.shape[0]} vs {embs.shape[0]}")
        return None

    dim = embs.shape[1]
    centroids = np.zeros((len(topic_ids), dim), dtype=np.float64)
    for i, t in enumerate(topic_ids):
        mask = labels == t
        n = int(mask.sum())
        if n == 0:
            continue  # Null-Vektor bleibt; Cosine wird 0 sein
        centroids[i] = embs[mask].mean(axis=0)
    return centroids


# =============================================================================
# MATCHING
# =============================================================================

def compute_pairwise_scores(
    p1_kw: pd.DataFrame,
    p2_kw: pd.DataFrame,
    alpha: float,
    use_sbert: bool,
    p1_dir: Path,
    p2_dir: Path,
) -> pd.DataFrame:
    """Liefert long-form DataFrame mit allen Topic-Paar-Scores.

    Spalten: phase1_topic, phase2_topic, cosine, jaccard, hybrid
    """
    vocab = build_vocab(p1_kw, p2_kw)
    print(f"Gemeinsames Vokabular: {len(vocab):,} Keywords")

    M1, topics1 = build_topic_matrix(p1_kw, vocab)
    M2, topics2 = build_topic_matrix(p2_kw, vocab)
    print(f"Phase 1 Topic-Matrix: {M1.shape}")
    print(f"Phase 2 Topic-Matrix: {M2.shape}")

    cos_kw = cosine_similarity(M1, M2)  # [n1, n2]
    print(f"Repräsentations-Cosine berechnet: shape={cos_kw.shape}")

    # Optional SBERT-Centroid-Cosine, ersetzt cos_kw, falls geladen.
    # Centroide werden direkt zu topics1/topics2 ausgerichtet — kein
    # nachträgliches Slicing/Alignment mehr nötig.
    cos_used = cos_kw
    sbert_active = False
    if use_sbert:
        cen1 = try_load_topic_centroids(p1_dir, topics1)
        cen2 = try_load_topic_centroids(p2_dir, topics2)
        if cen1 is not None and cen2 is not None:
            # Null-Centroid-Detektion: Topics ohne Dokumente liefern Null-Vektoren,
            # die in der Cosine-Berechnung NaN ergeben würden. Sklearn handhabt
            # das via Norm-Division — Null-Vektoren werden zu Cosine 0.
            cos_sbert = cosine_similarity(cen1, cen2)
            cos_used = cos_sbert
            sbert_active = True
            n_zero1 = int((np.linalg.norm(cen1, axis=1) == 0).sum())
            n_zero2 = int((np.linalg.norm(cen2, axis=1) == 0).sum())
            print(f"SBERT-Centroid-Cosine aktiv (384D-Original-Embeddings)")
            print(f"  Centroid-Shape Phase 1: {cen1.shape}  "
                  f"(Null-Centroide: {n_zero1})")
            print(f"  Centroid-Shape Phase 2: {cen2.shape}  "
                  f"(Null-Centroide: {n_zero2})")
        else:
            print("[info] SBERT-Centroide nicht verfügbar, Fallback c-TF-IDF.")

    # Jaccard auf Top-K Keyword-Sets (kw_df ist bereits topk-beschränkt)
    p1_sets = {t: set(g["keyword"])
               for t, g in p1_kw.groupby("topic")}
    p2_sets = {t: set(g["keyword"])
               for t, g in p2_kw.groupby("topic")}

    rows = []
    for i, t1 in enumerate(topics1):
        s1 = p1_sets[t1]
        for j, t2 in enumerate(topics2):
            jac = jaccard_topk(s1, p2_sets[t2])
            cos = float(cos_used[i, j])
            hyb = alpha * cos + (1.0 - alpha) * jac
            rows.append((t1, t2, cos, jac, hyb))

    out = pd.DataFrame(rows, columns=[
        "phase1_topic", "phase2_topic", "cosine", "jaccard", "hybrid",
    ])
    out.attrs["sbert_active"] = sbert_active
    return out


def best_matches(scores: pd.DataFrame) -> pd.DataFrame:
    """Pro phase1_topic das beste phase2-Match (und vice versa).

    Liefert eine annotierte Tabelle mit:
      - rank_p1_to_p2: Rang dieses p2-Topics für p1 (1 = best)
      - rank_p2_to_p1: Rang dieses p1-Topics für p2 (1 = best)
      - mutual_best: True, wenn beide Ränge = 1.
    """
    df = scores.copy()
    df["rank_p1_to_p2"] = (
        df.sort_values(["phase1_topic", "hybrid"], ascending=[True, False])
          .groupby("phase1_topic").cumcount() + 1
    )
    df["rank_p2_to_p1"] = (
        df.sort_values(["phase2_topic", "hybrid"], ascending=[True, False])
          .groupby("phase2_topic").cumcount() + 1
    )
    df["mutual_best"] = (df["rank_p1_to_p2"] == 1) & (df["rank_p2_to_p1"] == 1)
    return df


# =============================================================================
# REPORT
# =============================================================================

def write_report(
    matches: pd.DataFrame,
    p1_kw: pd.DataFrame,
    p2_kw: pd.DataFrame,
    threshold: float,
    out_dir: Path,
    sbert_active: bool,
    alpha: float,
    topk: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top-3-Keywords pro Topic für lesbare Ausgabe
    p1_top = (p1_kw.sort_values(["topic", "score"], ascending=[True, False])
              .groupby("topic").head(3)
              .groupby("topic")["keyword"].apply(lambda x: ", ".join(x)))
    p2_top = (p2_kw.sort_values(["topic", "score"], ascending=[True, False])
              .groupby("topic").head(3)
              .groupby("topic")["keyword"].apply(lambda x: ", ".join(x)))

    matches["phase1_keywords"] = matches["phase1_topic"].map(p1_top)
    matches["phase2_keywords"] = matches["phase2_topic"].map(p2_top)

    # Vollständige Pair-Score-Tabelle
    full_path = out_dir / "topic_matches_full.csv"
    matches.to_csv(full_path, index=False)
    print(f"\nAlle Topic-Paar-Scores: {full_path.name}  ({len(matches):,} Zeilen)")

    # Best-Match pro Phase-1-Topic (rang 1)
    best_p1 = matches[matches["rank_p1_to_p2"] == 1].copy()
    best_p1 = best_p1.sort_values("hybrid", ascending=False)
    best_path = out_dir / "topic_matches_best_p1_to_p2.csv"
    best_p1.to_csv(best_path, index=False)
    print(f"Beste P2-Matches je P1-Topic: {best_path.name}")

    # Reziproke Matches
    mutual = matches[matches["mutual_best"]].copy()
    mutual_path = out_dir / "topic_matches_mutual.csv"
    mutual.sort_values("hybrid", ascending=False).to_csv(mutual_path, index=False)
    print(f"Mutual-Best Matches: {mutual_path.name}  ({len(mutual)} Paare)")

    # Diagnostik-Bericht
    diag = []
    diag.append("Cross-Phase Topic Matching — Diagnostik")
    diag.append("=" * 60)
    diag.append(f"alpha (Cosine-Gewicht): {alpha}")
    diag.append(f"top_k Keywords:         {topk}")
    diag.append(f"Konfidenzschwelle:      hybrid >= {threshold}")
    diag.append(f"Cosine-Quelle:          "
                f"{'SBERT-Centroid' if sbert_active else 'c-TF-IDF (Fallback)'}")
    diag.append("")
    n_p1 = matches["phase1_topic"].nunique()
    n_p2 = matches["phase2_topic"].nunique()
    diag.append(f"Topics: Phase 1 = {n_p1}   Phase 2 = {n_p2}")
    diag.append(f"Mutual-Best Paare:      {len(mutual)}  "
                f"({len(mutual) / max(n_p1, n_p2) * 100:.1f}% des größeren Sets)")
    diag.append("")

    # Verteilung der Best-Match-Hybrid-Scores
    hyb = best_p1["hybrid"].describe()
    diag.append("Best-P1→P2 Hybrid-Score-Verteilung:")
    for k, v in hyb.items():
        diag.append(f"  {k:>6s}: {v:.3f}")
    diag.append("")

    n_low = (best_p1["hybrid"] < threshold).sum()
    diag.append(f"Unsichere Matches (hybrid < {threshold}): {n_low} / {len(best_p1)}")
    diag.append("")
    diag.append("Top-10 sicherste Mutual-Best Paare:")
    for _, r in mutual.nlargest(10, "hybrid").iterrows():
        diag.append(
            f"  P1#{int(r['phase1_topic']):>3d} [{r['phase1_keywords']}]  "
            f"⇄  P2#{int(r['phase2_topic']):>3d} [{r['phase2_keywords']}]  "
            f"(hyb={r['hybrid']:.3f}, cos={r['cosine']:.3f}, "
            f"jac={r['jaccard']:.3f})"
        )
    diag.append("")
    diag.append("Top-10 unsicherste Best-P1→P2 (Review-Kandidaten):")
    for _, r in best_p1.nsmallest(10, "hybrid").iterrows():
        diag.append(
            f"  P1#{int(r['phase1_topic']):>3d} [{r['phase1_keywords']}]  "
            f"→  P2#{int(r['phase2_topic']):>3d} [{r['phase2_keywords']}]  "
            f"(hyb={r['hybrid']:.3f})"
        )

    diag_path = out_dir / "match_diagnostics.txt"
    diag_path.write_text("\n".join(diag), encoding="utf-8")
    print(f"Diagnostik-Bericht: {diag_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1_DIR)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2_DIR)
    parser.add_argument("--out-dir",    type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Gewicht der Cosine-Komponente im Hybrid (default 0.6)")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                        help="Top-K Keywords pro Topic (default 15)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="Hybrid-Score-Schwelle für 'sichere' Matches")
    parser.add_argument("--with-sbert", action="store_true",
                        help="SBERT-Centroid-Cosine aus model_results.pkl verwenden")
    args = parser.parse_args()

    print("Cross-Phase Topic Matching")
    print("=" * 60)
    print(f"Phase 1: {args.phase1_dir}")
    print(f"Phase 2: {args.phase2_dir}")
    print(f"Output : {args.out_dir}")
    print(f"alpha={args.alpha}  topk={args.topk}  threshold={args.threshold}")
    print(f"SBERT-Centroid: {'an' if args.with_sbert else 'aus'}")
    print()

    p1_kw = load_topic_keywords(args.phase1_dir, args.topk)
    p2_kw = load_topic_keywords(args.phase2_dir, args.topk)
    print(f"Phase 1 Topics (ohne -1): {p1_kw['topic'].nunique()}")
    print(f"Phase 2 Topics (ohne -1): {p2_kw['topic'].nunique()}")

    scores = compute_pairwise_scores(
        p1_kw, p2_kw,
        alpha=args.alpha,
        use_sbert=args.with_sbert,
        p1_dir=args.phase1_dir,
        p2_dir=args.phase2_dir,
    )
    sbert_active = scores.attrs.get("sbert_active", False)

    matches = best_matches(scores)
    write_report(
        matches, p1_kw, p2_kw,
        threshold=args.threshold,
        out_dir=args.out_dir,
        sbert_active=sbert_active,
        alpha=args.alpha,
        topk=args.topk,
    )

    print()
    print("Fertig.")


if __name__ == "__main__":
    main()
