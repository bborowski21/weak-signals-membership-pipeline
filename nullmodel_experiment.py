# -*- coding: utf-8 -*-
"""Nullmodell (Validierungsoption D): Bilden die vier Konfigurationen Struktur ab oder die Konstruktion?

Frage: Die Memberships entstehen aus Sigmoiden von Dimensionsmitteln. Waere die beobachtete
Verteilung der Profile (insbesondere der Anteil eindeutiger Profile mit Margin >= 0,10) auch dann
zu erwarten, wenn zwischen den Indikatoren keinerlei Kovarianz bestuende? Das Nullmodell zerstoert
gezielt Struktur und laesst alles andere unveraendert.

Zwei Nullmodelle, beide erhalten die Randverteilung jedes Indikators exakt:
  N1 "independent"  Jede der 16 Indikatorspalten wird unabhaengig ueber die Topics permutiert.
                    Zerstoert jede Kovarianz, auch innerhalb einer Dimension.
                    Test: Gibt es ueberhaupt Struktur, die das Framework aufgreift?
  N2 "block"        Alle Indikatoren einer Dimension werden mit derselben Permutation verschoben.
                    Erhaelt die Kovarianz innerhalb jeder Dimension, zerstoert die Ausrichtung
                    zwischen den Dimensionen. Test: Traegt das Konfigurationsmuster ueber die
                    einzelnen Dimensionen hinaus Information?

Rechnung: Dimensionsscores und Memberships mit dem unveraenderten Pipeline-Code
(step02_indicators.compute_dimension_scores, step02_memberships.compute_memberships).
Kennzahlen je Replikat: Anteil Margin >= 0,10 (eindeutige Profile), Anteil < 0,05, Median und
Mittel der Margin, Mittel des hoechsten Memberships, Anteil je Konfiguration, mittlere
Interkorrelation der Kerndimensionen. Der p-Wert ist der Anteil der Replikate, deren Statistik
den realen Wert erreicht oder uebertrifft (einseitig, exakt in der Richtung der Hypothese).

Aufruf (im Pipeline-Ordner, nach run_all_phases):
    python3 nullmodel_experiment.py [--root PFAD] [--code PFAD] [--reps 1000] [--seed 20260904]
Liest  : output_phase{1,2}/indicators_16.csv, signal_memberships.csv
Schreibt: output_phase{1,2}/nullmodel_summary.csv, nullmodel_replicates.csv;
          nullmodel_summary_both_phases.csv; figures_paper/nullmodel_margin.png/.pdf
Keine Rohdaten (WoS/KATI) beteiligt; nur abgeleitete Indikatorwerte.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MEMB = ["m_ws", "m_trend", "m_ec", "m_latent"]
LABEL = {"m_ws": "Weak signal", "m_trend": "Trend", "m_ec": "Emerging concept", "m_latent": "Latent/mixed"}
MODELS = ["independent", "block"]


def coherence(ind: pd.DataFrame, groups: list) -> tuple:
    """Mittlere Interkorrelation innerhalb der Dimensionen und zwischen ihnen (Strukturmass)."""
    c = np.corrcoef(ind.values, rowvar=False)
    within, between = [], []
    gid = np.full(c.shape[0], -1)
    for g, cols in enumerate(groups):
        for j in cols:
            gid[j] = g
    for i in range(c.shape[0]):
        for j in range(i + 1, c.shape[1]):
            (within if gid[i] == gid[j] else between).append(abs(c[i, j]))
    return float(np.mean(within)), float(np.mean(between))


def stats(memb: pd.DataFrame, ind: pd.DataFrame, groups: list) -> dict:
    m = memb["margin"].values
    arg = memb[MEMB].values.argmax(axis=1)
    top = memb[MEMB].values.max(axis=1)
    w, b = coherence(ind, groups)
    mm = np.corrcoef(memb[MEMB].values, rowvar=False)
    out = {"share_clear": float((m >= 0.10).mean()),
           "share_ambiguous": float((m < 0.05).mean()),
           "median_margin": float(np.median(m)),
           "mean_margin": float(m.mean()),
           "mean_top_membership": float(top.mean()),
           "mean_within_dim_r": w,                       # Kohaerenz der Dimensionen
           "mean_between_dim_r": b,
           "r_ws_ec": float(mm[0, 2]),                   # Verschraenkung der beiden Kernkonfigurationen
           "mean_abs_r_memberships": float(np.mean([abs(mm[i, j]) for i in range(4) for j in range(i + 1, 4)]))}
    for i, k in enumerate(MEMB):
        out[f"share_{k}"] = float((arg == i).mean())
    return out


def permute(ind: pd.DataFrame, model: str, groups: list, rng) -> pd.DataFrame:
    v = ind.values.copy()
    n = v.shape[0]
    if model == "independent":
        for j in range(v.shape[1]):
            v[:, j] = v[rng.permutation(n), j]
    else:                                   # block: eine Permutation je Dimension
        for cols in groups:
            p = rng.permutation(n)
            v[:, cols] = v[p][:, cols]
    return pd.DataFrame(v, index=ind.index, columns=ind.columns)


def run_phase(root: Path, phase: int, reps: int, seed: int):
    from step02_indicators import compute_dimension_scores
    from step02_memberships import compute_memberships
    from config import INDICATOR_DIMENSIONS

    out_dir = root / f"output_phase{phase}"
    ind = pd.read_csv(out_dir / "indicators_16.csv", index_col=0)
    ref = pd.read_csv(out_dir / "signal_memberships.csv", index_col=0)
    base = compute_memberships(ind, compute_dimension_scores(ind))
    assert np.abs(base[MEMB].values - ref.loc[base.index, MEMB].values).max() < 1e-6, "Baseline weicht ab"
    cols = list(ind.columns)
    groups = [[cols.index(c) for c in inds if c in cols] for inds in INDICATOR_DIMENSIONS.values()]
    real = stats(base, ind, groups)

    rows, summary = [], []
    for model in MODELS:
        rng = np.random.default_rng(seed + phase * 10 + MODELS.index(model))
        rec = []
        for r in range(reps):
            perm = permute(ind, model, groups, rng)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                memb = compute_memberships(perm, compute_dimension_scores(perm))
            s = stats(memb, perm, groups)
            s.update({"phase": phase, "model": model, "rep": r})
            rec.append(s)
        rec = pd.DataFrame(rec)
        rows.append(rec)
        row = {"phase": phase, "model": model, "reps": reps, "n_topics": len(ind)}
        for k in ("share_clear", "share_ambiguous", "median_margin", "mean_margin", "mean_top_membership",
                  "mean_within_dim_r", "mean_between_dim_r", "r_ws_ec", "mean_abs_r_memberships"):
            row[f"real_{k}"] = real[k]
            row[f"null_{k}_mean"] = rec[k].mean()
            row[f"null_{k}_p2.5"] = np.percentile(rec[k], 2.5)
            row[f"null_{k}_p97.5"] = np.percentile(rec[k], 97.5)
        # einseitige p-Werte in Richtung der Hypothese
        row["p_share_clear"] = float((rec["share_clear"].values >= real["share_clear"]).mean())
        row["p_median_margin"] = float((rec["median_margin"].values >= real["median_margin"]).mean())
        row["p_share_ambiguous"] = float((rec["share_ambiguous"].values <= real["share_ambiguous"]).mean())
        row["p_within_dim_r"] = float((rec["mean_within_dim_r"].values >= real["mean_within_dim_r"]).mean())
        row["p_r_ws_ec"] = float((rec["r_ws_ec"].values >= real["r_ws_ec"]).mean())
        for k in MEMB:
            row[f"real_share_{k}"] = real[f"share_{k}"]
            row[f"null_share_{k}_mean"] = rec[f"share_{k}"].mean()
        summary.append(row)
        print(f"P{phase} {model:12s}: eindeutig real {100 * real['share_clear']:.1f} % vs. null "
              f"{100 * rec['share_clear'].mean():.1f} % [{100 * np.percentile(rec['share_clear'], 2.5):.1f}, "
              f"{100 * np.percentile(rec['share_clear'], 97.5):.1f}] | p = {row['p_share_clear']:.3f} | "
              f"Median Margin real {real['median_margin']:.3f} vs. null {rec['median_margin'].mean():.3f} | "
              f"r(dim intern) real {real['mean_within_dim_r']:.3f} vs. null {rec['mean_within_dim_r'].mean():.3f} (p = {row['p_within_dim_r']:.3f}) | "
              f"r(m_ws, m_ec) real {real['r_ws_ec']:.3f} vs. null {rec['r_ws_ec'].mean():.3f}")
    summ = pd.DataFrame(summary)
    summ.to_csv(out_dir / "nullmodel_summary.csv", index=False)
    pd.concat(rows).to_csv(out_dir / "nullmodel_replicates.csv", index=False)
    return summ, pd.concat(rows), real


def make_figure(root: Path, results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = root / "figures_paper"
    fig_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.6), sharey=True)
    colors = {"independent": "#c6d7ea", "block": "#5b8ec4"}
    for ax, (phase, (summ, rec, real)) in zip(axes, sorted(results.items())):
        for model in MODELS:
            v = 100 * rec[rec.model == model]["share_clear"].values
            ax.hist(v, bins=30, color=colors[model], edgecolor="white", linewidth=0.3,
                    label=f"null model: {model}", alpha=0.9)
        ax.axvline(100 * real["share_clear"], color="#bb4717", lw=1.4,
                   label=f"observed ({100 * real['share_clear']:.0f} %)")
        ax.set_xlabel("Share of topics with margin $\\geq$ 0.10 (%)")
        ax.set_title(f"Phase {phase} (n = {summ.iloc[0]['n_topics']:.0f} topics)", fontsize=8, loc="left")
        ax.grid(axis="y", color="#eef1f4", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Replicates")
    axes[0].legend(frameon=False, fontsize=7, loc="upper center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"nullmodel_margin.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    # Zweite Abbildung: die beiden Strukturmasse, beobachtet gegen Nullverteilung
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 1.9))
    specs = [("mean_within_dim_r", "Mean correlation within dimensions", "independent"),
             ("r_ws_ec", "Correlation between $m_{ws}$ and $m_{ec}$", "independent")]
    for ax, (key, label, model) in zip(axes, specs):
        ys, labels = [], []
        for k, (phase, (summ, rec, real)) in enumerate(sorted(results.items())):
            v = rec[rec.model == model][key].values
            lo, hi = np.percentile(v, [2.5, 97.5])
            ax.plot([lo, hi], [k, k], color="#5b8ec4", lw=5, solid_capstyle="butt",
                    label="null model, 95 % of replicates" if k == 0 else None)
            ax.plot(v.mean(), k, "o", color="#07519d", ms=4,
                    label="null model, mean" if k == 0 else None)
            ax.plot(real[key], k, "D", color="#bb4717", ms=6,
                    label="observed" if k == 0 else None)
            ys.append(k); labels.append(f"Phase {phase}")
        ax.set_yticks(ys); ax.set_yticklabels(labels); ax.set_ylim(-0.5, len(ys) - 0.5)
        ax.set_xlim(0, max(0.6, ax.get_xlim()[1]))
        ax.set_xlabel(label)
        ax.grid(axis="x", color="#eef1f4", lw=0.5); ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"nullmodel_structure.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--code", default=None)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    sys.path.insert(0, str(Path(a.code).resolve() if a.code else root))
    if a.figures_only:
        results = {}
        for ph in (1, 2):
            d = root / f"output_phase{ph}"
            summ = pd.read_csv(d / "nullmodel_summary.csv")
            rec = pd.read_csv(d / "nullmodel_replicates.csv")
            real = {c[len("real_"):]: summ.iloc[0][c] for c in summ.columns if c.startswith("real_")}
            results[ph] = (summ, rec, real)
    else:
        results = {ph: run_phase(root, ph, a.reps, a.seed) for ph in (1, 2)}
    make_figure(root, results)
    if not a.figures_only:
        pd.concat([r[0] for r in results.values()]).to_csv(root / "nullmodel_summary_both_phases.csv", index=False)
    print("fertig:", root / "nullmodel_summary_both_phases.csv")


if __name__ == "__main__":
    main()
