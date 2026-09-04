# -*- coding: utf-8 -*-
"""Perturbationsexperiment (Option A): Stabilität der Memberships gegenüber Rauschen auf den 16 Indikatoren.

Frage: Wie oft kippt die kategoriale Zuordnung (Argmax der vier Memberships), wenn die Indikatoren
mit kleinem Rauschen versehen werden, und wie stark verschieben sich dabei die Membership-Werte selbst?

Design
  Rauschen  : je Indikator j additiv normalverteilt, Standardabweichung s * SD_j (SD über die Topics der Phase),
              Stufen s in NOISE_LEVELS; Rauschen unabhängig je Topic, Indikator und Replikat.
  Replikate : R = 1000 je Stufe und Phase, Seed 20260903.
  Rechnung  : Dimensionsscores und Memberships mit dem Original-Code der Pipeline
              (step02_indicators.compute_dimension_scores, step02_memberships.compute_memberships),
              d. h. StandardScaler auf den gestörten Indikatoren, robust-z auf Dimensionsebene, Sigmoid k = 1,0, λ = 0,5.
  Kennzahlen: Kipprate des Argmax (gesamt, je Margin-Klasse der Baseline: Δ < 0,05; 0,05 bis 0,10; ≥ 0,10, und je
              Baseline-Klasse), Übergangsmatrix Baseline-Klasse -> gestörte Klasse, Kippwahrscheinlichkeit und häufigste
              Ausweichklasse je Topic, mittlere absolute Membership-Verschiebung, Spearman je Membership.

Aufruf (im Pipeline-Ordner, nach run_all_phases):  python3 perturbation_experiment.py [--root PFAD] [--reps 1000]
         --code PFAD  falls die step02-Module nicht im --root liegen;  --figures-only  rendert nur die Abbildungen neu.
Liest  : output_phase{1,2}/indicators_16.csv, signal_memberships.csv (Baseline wird gegen die Datei geprüft)
Schreibt: output_phase{1,2}/perturbation_summary.csv, perturbation_topics.csv, perturbation_transitions.csv;
          perturbation_summary_both_phases.csv; figures_paper/perturbation_*.png/.pdf
Laufzeit: rund 4 Minuten für 1000 Replikate (beide Phasen). Keine Rohdaten (WoS/KATI) beteiligt; nur abgeleitete Indikatorwerte.
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

NOISE_LEVELS = [0.02, 0.05, 0.10, 0.20]
MEMB = ["m_ws", "m_trend", "m_ec", "m_latent"]
LABEL = {"m_ws": "Weak signal", "m_trend": "Trend", "m_ec": "Emerging topic", "m_latent": "Latent"}


def margin_class(m):
    return np.where(m < 0.05, "<0.05", np.where(m < 0.10, "0.05-0.10", ">=0.10"))


def run_phase(root: Path, phase: int, reps: int, seed: int):
    from step02_indicators import compute_dimension_scores
    from step02_memberships import compute_memberships
    from scipy.stats import spearmanr
    out_dir = root / f"output_phase{phase}"
    ind = pd.read_csv(out_dir / "indicators_16.csv", index_col=0)
    base_ref = pd.read_csv(out_dir / "signal_memberships.csv", index_col=0)
    base = compute_memberships(ind, compute_dimension_scores(ind))
    assert (base[MEMB].values - base_ref.loc[base.index, MEMB].values).__abs__().max() < 1e-6, "Baseline weicht ab"
    base_arg = base[MEMB].values.argmax(axis=1)
    base_margin = base["margin"].values
    cls = margin_class(base_margin)
    sd = ind.std(axis=0).values
    rng = np.random.default_rng(seed + phase)
    n = len(ind)
    summary, per_topic = [], pd.DataFrame({"topic": ind.index, "baseline_argmax": [MEMB[i] for i in base_arg],
                                           "baseline_margin": base_margin, "margin_class": cls})
    transitions = []
    for s in NOISE_LEVELS:
        flips = np.zeros((reps, n), dtype=bool)
        absdiff = np.zeros((reps, n))
        rho = {m: [] for m in MEMB}
        trans = np.zeros((4, 4), dtype=int)                 # Baseline-Klasse (Zeile) -> gestörte Klasse (Spalte)
        cnt = np.zeros((n, 4), dtype=int)                   # je Topic: Häufigkeit der gestörten Klassen
        for r in range(reps):
            noise = rng.normal(0.0, 1.0, size=ind.shape) * (s * sd)
            pert = pd.DataFrame(ind.values + noise, index=ind.index, columns=ind.columns)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                memb = compute_memberships(pert, compute_dimension_scores(pert))
            arg = memb[MEMB].values.argmax(axis=1)
            flips[r] = arg != base_arg
            absdiff[r] = np.abs(memb[MEMB].values - base[MEMB].values).mean(axis=1)
            np.add.at(trans, (base_arg, arg), 1)
            np.add.at(cnt, (np.arange(n), arg), 1)
            for m in MEMB:
                rho[m].append(spearmanr(base[m].values, memb[m].values).correlation)
        flip_rate = flips.mean(axis=1)                      # je Replikat
        p_topic = flips.mean(axis=0)                        # je Topic
        for i, mi in enumerate(MEMB):
            for j, mj in enumerate(MEMB):
                transitions.append({"phase": phase, "noise_sd_share": s, "from": mi, "to": mj, "count": int(trans[i, j]),
                                    "share_of_from": trans[i, j] / trans[i].sum() if trans[i].sum() else np.nan})
        # häufigste Ausweichklasse je Topic (nur unter den gestörten Zuordnungen, die von der Baseline abweichen)
        alt = cnt.copy(); alt[np.arange(n), base_arg] = -1
        per_topic[f"alt_class_s{s}"] = np.where(p_topic > 0, np.array(MEMB)[alt.argmax(axis=1)], "")
        row = {"phase": phase, "noise_sd_share": s, "reps": reps, "n_topics": n,
               "flip_rate_mean": flip_rate.mean(), "flip_rate_p2.5": np.percentile(flip_rate, 2.5),
               "flip_rate_p97.5": np.percentile(flip_rate, 97.5),
               "mean_abs_membership_shift": absdiff.mean(), "p95_abs_membership_shift": np.percentile(absdiff, 95),
               "share_topics_shift_le_0.05": (absdiff.mean(axis=0) <= 0.05).mean(),
               "share_topics_flip_prob_gt_0.5": (p_topic > 0.5).mean(),
               "share_topics_flip_prob_lt_0.05": (p_topic < 0.05).mean(),
               "spearman_flipprob_margin": spearmanr(p_topic, base_margin).correlation}
        for c in ["<0.05", "0.05-0.10", ">=0.10"]:
            mask = cls == c
            row[f"flip_rate_{c}"] = flips[:, mask].mean() if mask.any() else np.nan
            row[f"n_{c}"] = int(mask.sum())
        for m in MEMB:
            row[f"rho_{m}"] = float(np.mean(rho[m]))
        for i, m in enumerate(MEMB):                        # Kipprate je Baseline-Klasse
            row[f"flip_rate_from_{m}"] = 1.0 - trans[i, i] / trans[i].sum() if trans[i].sum() else np.nan
            row[f"n_from_{m}"] = int((base_arg == i).sum())
        summary.append(row)
        per_topic[f"flip_prob_s{s}"] = p_topic
        per_topic[f"mean_abs_shift_s{s}"] = absdiff.mean(axis=0)
        print(f"P{phase} s={s:.2f}: Kipprate {row['flip_rate_mean']:.3f} "
              f"[{row['flip_rate_p2.5']:.3f}, {row['flip_rate_p97.5']:.3f}] | "
              f"<0.05: {row['flip_rate_<0.05']:.3f} 0.05-0.10: {row['flip_rate_0.05-0.10']:.3f} >=0.10: {row['flip_rate_>=0.10']:.3f} | "
              f"mean |Δm| {row['mean_abs_membership_shift']:.4f}")
    summ = pd.DataFrame(summary)
    summ.to_csv(out_dir / "perturbation_summary.csv", index=False)
    per_topic.to_csv(out_dir / "perturbation_topics.csv", index=False)
    pd.DataFrame(transitions).to_csv(out_dir / "perturbation_transitions.csv", index=False)
    return summ, per_topic


def make_figures(root: Path, results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = root / "figures_paper"; fig_dir.mkdir(exist_ok=True)
    blues = {0.05: "#9ecae1", 0.10: "#4292c6", 0.20: "#08519c"}   # eine Farbe, hell nach dunkel = Rauschstärke
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a97a5", "axes.labelcolor": "#16202b", "xtick.color": "#5b6a7a", "ytick.color": "#5b6a7a"})
    # Abbildung 1: Kippwahrscheinlichkeit je Topic gegen Baseline-Margin, zwei Phasen
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    xmax = max(pt["baseline_margin"].max() for _, pt in results.values()) + 0.02
    for ax, (phase, (summ, pt)) in zip(axes, sorted(results.items())):
        for s in (0.05, 0.10, 0.20):
            ax.scatter(pt["baseline_margin"], pt[f"flip_prob_s{s}"], s=9, color=blues[s], alpha=0.55, linewidths=0,
                       label=f"noise {int(s*100)} % of SD")
        ax.axvline(0.05, color="#d8dee6", lw=1, zorder=0); ax.axvline(0.10, color="#d8dee6", lw=1, zorder=0)
        ax.set_xlabel("Baseline margin Δ (dominant minus second membership)" if ax is axes[0] else "Baseline margin Δ")
        ax.set_title(f"Phase {phase} (n = {len(pt)} topics)", fontsize=9, loc="left", color="#16202b")
        ax.grid(axis="y", color="#eef1f4", lw=0.8); ax.set_axisbelow(True)
        ax.set_xlim(-0.005, xmax); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Flip probability of the\ndominant configuration")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"perturbation_flip_vs_margin.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    # Abbildung 2: Kipprate je Margin-Klasse gegen mittlere Membership-Verschiebung
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    classes = ["<0.05", "0.05-0.10", ">=0.10"]
    for ax, (phase, (summ, pt)) in zip(axes, sorted(results.items())):
        x = np.arange(len(classes)); w = 0.26
        for i, s in enumerate((0.05, 0.10, 0.20)):
            row = summ[summ.noise_sd_share == s].iloc[0]
            vals = [row[f"flip_rate_{c}"] for c in classes]
            bars = ax.bar(x + (i - 1) * w, vals, width=w - 0.02, color=blues[s], label=f"noise {int(s*100)} % of SD")
            for b, v in zip(bars, vals):
                lab = "0" if v == 0 else ("<1" if v < 0.01 else f"{100*v:.0f}")   # Prozentwerte, ganzzahlig
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, lab, ha="center", va="bottom", fontsize=6.5, color="#16202b")
        ax.set_xticks(x); ax.set_xticklabels([f"{lab}\n(n = {int(summ.iloc[0][f'n_{c}'])})" for c, lab in zip(classes, ["Δ < 0.05", "0.05 ≤ Δ < 0.10", "Δ ≥ 0.10"])])
        ax.set_title(f"Phase {phase}", fontsize=9, loc="left", color="#16202b")
        ax.grid(axis="y", color="#eef1f4", lw=0.8); ax.set_axisbelow(True); ax.set_ylim(0, 0.45)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4]); ax.set_yticklabels(["0", "10", "20", "30", "40"])
    axes[0].set_ylabel("Flip rate of the\ndominant configuration (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"perturbation_flip_by_margin_class.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Pipeline-Ordner mit output_phase1/2 und den step02-Modulen")
    ap.add_argument("--code", default=None, help="Ordner mit step02_indicators.py/step02_memberships.py, falls abweichend")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    sys.path.insert(0, str(Path(a.code).resolve() if a.code else root))
    results = {}
    for phase in (1, 2):
        if a.figures_only:
            results[phase] = (pd.read_csv(root / f"output_phase{phase}" / "perturbation_summary.csv"),
                              pd.read_csv(root / f"output_phase{phase}" / "perturbation_topics.csv"))
        else:
            results[phase] = run_phase(root, phase, a.reps, a.seed)
    make_figures(root, results)
    both = pd.concat([r[0] for r in results.values()])
    both.to_csv(root / "perturbation_summary_both_phases.csv", index=False)
    print("fertig:", root / "perturbation_summary_both_phases.csv")


if __name__ == "__main__":
    main()
