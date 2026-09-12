#!/usr/bin/env python3
"""
Phasengrenzen-Stabilitaet.

Frage: Ueberstehen die vier Memberships eine Verschiebung der Phasengrenze um
plus/minus ein Jahr?

Die vorhandene Ausgabe sensitivity_phase.csv beantwortet das nicht. Sie zaehlt
nur, wie viele Dokumente bei einem alternativen Splitjahr in welche Phase
fielen. Ausserdem wird sie je Phase geschrieben, obwohl die erzeugende Funktion
den Gesamtkorpus erwartet; jede der beiden Dateien enthaelt deshalb nur eine
Haelfte der Tabelle.

Verfahren je Zelle (alternatives Splitjahr x Phase):
  1. Dokumentmenge aus den beiden bereinigten Phasendateien neu schneiden.
  2. SBERT-Vektoren aus dem Cache holen (model_results.pkl beider Phasen).
     Es wird nicht neu eingebettet: Das Modell ist deterministisch und fuer
     beide Phasen dasselbe, die Vereinigung beider Caches deckt den Korpus.
  3. UMAP und HDBSCAN mit den Referenzparametern aus config.py neu rechnen.
  4. Die sechzehn Indikatoren, die fuenf Dimensionen und die vier Memberships
     neu rechnen.
  5. Topics ueber Dokumentueberlappung gegen den Referenzlauf matchen und je
     Membership die Spearman-Rangkorrelation bilden.

Berichtet wird rho_min, dieselbe Kennzahl wie in der Sensitivitaetstabelle des
Supplements.

KONTROLLZELLE: Splitjahr 2015 ist die Referenzwahl. Ihre Zellen muessen
rho_min = 1.0 und dieselbe Topicanzahl wie der Referenzlauf ergeben. Tun sie
das nicht, stimmt etwas an der Wiederverwendung der Vektoren nicht und die
uebrigen Zellen sind nicht zu gebrauchen.

Aufruf:
    python run_phase_boundary_stability.py
    python run_phase_boundary_stability.py --split-years 2014,2016
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

PHASES = {
    1: {"csv": "wos_qc_phase1_2000_2015_clean.csv",
        "out": "output_phase1", "ymin": 2000, "ymax": 2015},
    2: {"csv": "wos_qc_phase2_2016_2025_clean.csv",
        "out": "output_phase2", "ymin": 2016, "ymax": 2025},
}

REFERENCE_SPLIT = 2015



# ---------------------------------------------------------------------------
# Wortgleiche Kopien aus step01_topic_modeling. Sie stehen hier, damit dieses
# Skript ohne sentence_transformers auskommt: Es bettet nichts neu ein, sondern
# rechnet auf den gespeicherten Vektoren. Beide Funktionen sind reines pandas
# und numpy. Bei Aenderungen an step01_topic_modeling hier nachziehen.
# ---------------------------------------------------------------------------

def load_and_clean(path: Path, year_min: int, year_max: int) -> pd.DataFrame:
    from step02_indicators import CANON_COLUMNS

    print(f"Lade Daten aus {path.name}...", flush=True)
    df = pd.read_csv(path)

    # Dieselbe Umbenennung wie in step02_indicators.run(). Ohne sie fehlt den
    # Indikatoren unter anderem die Spalte "Source title". Importiert statt
    # kopiert, damit es eine einzige Quelle bleibt.
    rename_map = {src: dst for src, dst in CANON_COLUMNS.items()
                  if src in df.columns}
    df = df.rename(columns=rename_map)
    print(f"  Spaltenmapping: {len(rename_map)} Spalten umbenannt")

    df["text"] = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")

    df["text_clean"] = df["text"].apply(
        lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", t.lower())).strip()
        if isinstance(t, str) else ""
    )

    n0 = len(df)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[(df["Year"] >= year_min) & (df["Year"] <= year_max)].copy()
    n_year = len(df)
    print(f"  {n0} -> {n_year} Dokumente (Year-Filter [{year_min}, {year_max}])")

    df = df[df["text_clean"].str.split().str.len() >= 10].reset_index(drop=True)
    print(f"  {n_year} -> {len(df)} Dokumente (Laengen-Filter >= 10 Woerter)")
    return df


def compute_tem_metrics(df: pd.DataFrame, labels: np.ndarray) -> tuple:
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


def check_dependencies() -> None:
    fehlend = []
    for mod in ("umap", "hdbscan", "sklearn", "scipy"):
        try:
            __import__(mod)
        except ImportError:
            fehlend.append(mod)
    if fehlend:
        raise SystemExit(
            "Fehlende Pakete: " + ", ".join(fehlend) + ".\n"
            "Dieses Skript braucht numpy, pandas, scipy, scikit-learn, "
            "umap-learn und hdbscan.\n"
            "sentence-transformers wird NICHT gebraucht: Es wird nicht neu "
            "eingebettet.\n"
            "Bitte die Umgebung aktivieren, in der die Pipeline gelaufen ist.")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_reference_phase(phase: int):
    """df, Referenzlabels und gecachte SBERT-Vektoren einer Phase."""
    cfg = PHASES[phase]
    df = load_and_clean(PARENT_DIR / cfg["csv"], cfg["ymin"], cfg["ymax"])

    import pickle
    with open(BASE_DIR / cfg["out"] / "model_results.pkl", "rb") as f:
        model = pickle.load(f)
    labels = np.asarray(model["labels"])
    emb = np.asarray(model["embeddings_sbert"])
    del model

    if not (len(df) == len(labels) == len(emb)):
        raise SystemExit(
            f"Laengen-Mismatch Phase {phase}: df={len(df)}, "
            f"labels={len(labels)}, embeddings={len(emb)}. "
            f"Die bereinigte CSV passt nicht zum gespeicherten Lauf.")
    log(f"  Phase {phase}: {len(df)} Dokumente, "
        f"{len(set(labels[labels >= 0]))} Referenztopics, "
        f"Vektoren {emb.shape} {emb.dtype}")
    return df, labels, emb


def cluster(embeddings: np.ndarray, seed: int = 42):
    import umap
    import hdbscan
    from config import (UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
                        UMAP_METRIC, HDBSCAN_MIN_CLUSTER_SIZE,
                        HDBSCAN_MIN_SAMPLES, HDBSCAN_CLUSTER_METHOD)
    # Parametrierung zeichengleich mit step01_topic_modeling.reduce_dimensions
    # und cluster_topics. low_memory und prediction_data gehoeren dazu: Sie
    # aendern den Naechste-Nachbarn-Pfad von UMAP und damit das Ergebnis.
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=seed,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=HDBSCAN_CLUSTER_METHOD,
        prediction_data=True,
    )
    return clusterer.fit_predict(reduced), reduced


def evaluate_cell(df_all, emb_all, idx_ref, ref_labels, ref_memberships,
                  idx_alt, phase, split_year):
    import step02_indicators as s2
    import step05_sensitivity as s5

    n_all = len(df_all)
    df_alt = df_all.iloc[idx_alt].copy().reset_index(drop=True)
    emb_alt = np.ascontiguousarray(emb_all[idx_alt])

    log(f"  UMAP und HDBSCAN auf {len(df_alt)} Dokumenten ...")
    t0 = time.time()
    labels_alt, reduced_alt = cluster(emb_alt)
    n_topics = len(set(labels_alt[labels_alt >= 0]))
    log(f"  {n_topics} Topics, {int((labels_alt == -1).sum())} Rauschen "
        f"({time.time() - t0:.0f} s)")

    log("  Indikatoren, Dimensionen, Memberships ...")
    tem_metrics, proportions = compute_tem_metrics(df_alt, labels_alt)
    ind_alt = s2.compute_all_indicators(
        df=df_alt, labels=labels_alt,
        embeddings_sbert=emb_alt, embeddings_reduced=reduced_alt,
        proportions=proportions, tem_metrics=tem_metrics,
    )
    dims_alt = s5._aggregate_dimensions(ind_alt)
    memb_alt = s5._compute_memberships_from_dims(
        dim_scores=dims_alt, indicator_df=ind_alt)

    # Labelvektoren ueber den gesamten Korpus, damit die Ueberlappung die
    # Dokumentidentitaet vergleicht und nicht Positionen innerhalb einer Phase.
    base_full = np.full(n_all, -1, dtype=int)
    base_full[idx_ref] = ref_labels
    new_full = np.full(n_all, -1, dtype=int)
    new_full[idx_alt] = labels_alt

    log("  Matching gegen den Referenzlauf ...")
    mapping = s5._match_topics_by_overlap(base_full, new_full)
    rhos = s5._spearman_membership_aligned(ref_memberships, memb_alt, mapping)

    return {
        "alt_split_year": split_year,
        "phase": phase,
        "is_reference": split_year == REFERENCE_SPLIT,
        "n_docs": int(len(df_alt)),
        "n_docs_reference": int(len(idx_ref)),
        "n_topics": int(n_topics),
        "n_topics_reference": int(len(set(ref_labels[ref_labels >= 0]))),
        "n_noise": int((labels_alt == -1).sum()),
        **rhos,
    }, memb_alt, labels_alt



def evaluate_against_control(out_dir, years, phases, split_years,
                             reference_split=REFERENCE_SPLIT):
    """Zweite Auswertung: jede Alternative gegen die HEUTE gerechnete
    Referenzwahl statt gegen den gespeicherten Lauf.

    Weil die Kontrollzelle den gespeicherten Lauf nicht exakt reproduziert
    (Bibliotheksversionen sind gewandert), mischt der Vergleich gegen die
    Ablage zwei Effekte: Umgebungsdrift und Phasengrenze. Hier stammen beide
    Seiten aus demselben Lauf, die Drift faellt heraus.
    """
    import step05_sensitivity as s5

    n_all = len(years)

    def idx_for(sy, phase):
        return np.where((years <= sy) if phase == 1 else (years > sy))[0]

    rows = []
    for phase in phases:
        lab_c_path = out_dir / f"labels_split{reference_split}_phase{phase}.npy"
        memb_c_path = out_dir / f"memberships_split{reference_split}_phase{phase}.csv"
        if not (lab_c_path.exists() and memb_c_path.exists()):
            log(f"  Kontrollzelle Phase {phase} fehlt, uebersprungen")
            continue
        idx_c = idx_for(reference_split, phase)
        lab_c = np.load(lab_c_path)
        memb_c = pd.read_csv(memb_c_path, index_col=0)
        if len(lab_c) != len(idx_c):
            log(f"  Kontrollzelle Phase {phase}: Laengen passen nicht, uebersprungen")
            continue

        for sy in split_years:
            if sy == reference_split:
                continue
            lab_a_path = out_dir / f"labels_split{sy}_phase{phase}.npy"
            memb_a_path = out_dir / f"memberships_split{sy}_phase{phase}.csv"
            if not (lab_a_path.exists() and memb_a_path.exists()):
                continue
            idx_a = idx_for(sy, phase)
            lab_a = np.load(lab_a_path)
            memb_a = pd.read_csv(memb_a_path, index_col=0)
            if len(lab_a) != len(idx_a):
                continue

            base = np.full(n_all, -1, dtype=int)
            base[idx_c] = lab_c
            new = np.full(n_all, -1, dtype=int)
            new[idx_a] = lab_a
            mapping = s5._match_topics_by_overlap(base, new)
            rhos = s5._spearman_membership_aligned(memb_c, memb_a, mapping)
            rows.append({
                "alt_split_year": sy,
                "phase": phase,
                "n_control": int(len(idx_c)),
                "n_alt": int(len(idx_a)),
                "n_shared_docs": int(len(np.intersect1d(idx_c, idx_a))),
                "n_topics_control": int(len(set(lab_c[lab_c >= 0]))),
                "n_topics_alt": int(len(set(lab_a[lab_a >= 0]))),
                **rhos,
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stabilitaet der Memberships gegen die Phasengrenze")
    parser.add_argument("--split-years", default="2015,2014,2016,2017,2018,2019,2020",
                        help="Kommaliste; 2015 ist die Kontrollzelle")
    parser.add_argument("--output-dir", default="output_phase_boundary")
    parser.add_argument("--phases", default="1,2")
    args = parser.parse_args()

    check_dependencies()

    split_years = [int(x) for x in args.split_years.split(",") if x.strip()]
    phases = [int(x) for x in args.phases.split(",") if x.strip()]
    out_dir = BASE_DIR / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 66)
    log("PHASENGRENZE: STABILITAET DER MEMBERSHIPS")
    log("=" * 66)
    log(f"Splitjahre: {split_years}   Phasen: {phases}")
    log(f"Ausgabe   : {out_dir}")

    log("Referenzlaeufe laden ...")
    df1, lab1, emb1 = load_reference_phase(1)
    df2, lab2, emb2 = load_reference_phase(2)

    df_all = pd.concat([df1, df2], ignore_index=True)
    emb_all = np.vstack([emb1, emb2])
    n1 = len(df1)
    idx_ref = {1: np.arange(0, n1), 2: np.arange(n1, n1 + len(df2))}
    ref_labels = {1: lab1, 2: lab2}
    del emb1, emb2, df1, df2

    ref_memb = {}
    for p in (1, 2):
        path = BASE_DIR / PHASES[p]["out"] / "signal_memberships.csv"
        ref_memb[p] = pd.read_csv(path, index_col=0)
    log(f"Gesamtkorpus: {len(df_all)} Dokumente, Vektoren {emb_all.shape}")

    years = pd.to_numeric(df_all["Year"], errors="coerce").to_numpy()

    cells, records = [], []
    for sy in split_years:
        for p in phases:
            mask = (years <= sy) if p == 1 else (years > sy)
            cells.append((sy, p, np.where(mask)[0]))
    cells.sort(key=lambda c: (0 if c[0] == REFERENCE_SPLIT else 1, len(c[2])))

    csv_path = out_dir / "phase_boundary_stability.csv"
    for k, (sy, p, idx_alt) in enumerate(cells, 1):
        log("-" * 66)
        log(f"Zelle {k}/{len(cells)}: Splitjahr {sy}, Phase {p}, "
            f"{len(idx_alt)} Dokumente"
            + ("   [KONTROLLE]" if sy == REFERENCE_SPLIT else ""))
        t0 = time.time()
        try:
            rec, memb_alt, labels_alt = evaluate_cell(
                df_all, emb_all, idx_ref[p], ref_labels[p], ref_memb[p],
                idx_alt, p, sy)
        except Exception as exc:  # eine gescheiterte Zelle beendet nicht den Lauf
            log(f"  FEHLER: {type(exc).__name__}: {exc}")
            records.append({"alt_split_year": sy, "phase": p,
                            "n_docs": int(len(idx_alt)),
                            "error": f"{type(exc).__name__}: {exc}"})
            pd.DataFrame(records).to_csv(csv_path, index=False)
            continue

        rec["seconds"] = round(time.time() - t0, 1)
        records.append(rec)
        tag = f"split{sy}_phase{p}"
        memb_alt.to_csv(out_dir / f"memberships_{tag}.csv")
        np.save(out_dir / f"labels_{tag}.npy", labels_alt)
        pd.DataFrame(records).to_csv(csv_path, index=False)

        log(f"  rho_min={rec.get('rho_min')}  rho_mean={rec.get('rho_mean')}  "
            f"Paare={rec.get('n_matched_pairs')}  ({rec['seconds']} s)")
        if rec["is_reference"]:
            ok = (rec.get("rho_min") is not None
                  and not pd.isna(rec.get("rho_min"))
                  and rec["rho_min"] > 0.999
                  and rec["n_topics"] == rec["n_topics_reference"])
            log(f"  KONTROLLE {'bestanden' if ok else 'NICHT bestanden'}: "
                f"Topics {rec['n_topics']} gegen {rec['n_topics_reference']}, "
                f"rho_min {rec.get('rho_min')}")

    log("-" * 66)
    log("Zweite Auswertung: gegen die heute gerechnete Referenzwahl "
        f"{REFERENCE_SPLIT} (driftfrei) ...")
    ctrl_df = evaluate_against_control(out_dir, years, phases, split_years)
    ctrl_path = out_dir / "phase_boundary_vs_control.csv"
    if len(ctrl_df):
        ctrl_df.to_csv(ctrl_path, index=False)
        log(f"  {len(ctrl_df)} Zellen -> {ctrl_path.name}")
    else:
        log("  keine vergleichbaren Zellen gefunden")

    summary = {
        "split_years": split_years,
        "phases": phases,
        "n_corpus": int(len(df_all)),
        "cells": records,
        "cells_vs_control": ctrl_df.to_dict("records") if len(ctrl_df) else [],
    }
    with open(out_dir / "phase_boundary_stability.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log("=" * 66)
    log(f"FERTIG. {csv_path}")
    log("=" * 66)
    print("\n--- gegen den gespeicherten Referenzlauf ---", flush=True)
    print(pd.DataFrame(records).to_string(index=False), flush=True)
    if len(ctrl_df):
        cols = [c for c in ["alt_split_year", "phase", "n_alt", "n_topics_alt",
                            "n_topics_control", "rho_m_ws", "rho_m_trend",
                            "rho_m_ec", "rho_m_latent", "rho_mean", "rho_min",
                            "n_matched_pairs"] if c in ctrl_df.columns]
        print("\n--- gegen die heute gerechnete Referenzwahl (driftfrei) ---",
              flush=True)
        print(ctrl_df[cols].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
