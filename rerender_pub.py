"""
rerender_pub.py — Regeneriert ALLE F3-Abbildungen publikationsreif und konsistent.

- Wendet das einheitliche Styling (plot_style.apply_pub_style) an.
- Erzwingt 300 dpi und schreibt zusaetzlich eine Vektor-PDF zu jeder PNG.
- Nutzt die vorhandenen Artefakte in output_phase1/ output_phase2/ output_cross_phase/
  (kein erneutes SBERT/UMAP/HDBSCAN-Training noetig).
- Erzeugt zusaetzlich: topic_quality_boxplots (aus den per-Topic-CSVs) und
  framework_unified (idealtypisches Schema mit einheitlicher Palette).

Aufruf:  python3 rerender_pub.py      (aus diesem Verzeichnis)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from plot_style import apply_pub_style, save_fig
apply_pub_style()

# --- Dual-Save: jede .png-Ausgabe zusaetzlich als Vektor-.pdf, 300 dpi erzwungen ---
_orig_savefig = plt.savefig


def _dual_savefig(path, *a, **k):
    sp = str(path)
    k.setdefault("bbox_inches", "tight")
    if sp.lower().endswith(".png"):
        try:
            h_in = plt.gcf().get_size_inches()[1]
        except Exception:
            h_in = 0
        # sehr hohe Per-Topic-Heatmaps moderater rastern (Datei/Zeit), sonst 300 dpi
        k["dpi"] = 150 if h_in > 24 else 300
    _orig_savefig(path, *a, **k)
    if sp.lower().endswith(".png"):
        k2 = dict(k)
        k2.pop("dpi", None)
        _orig_savefig(sp[:-4] + ".pdf", *a, **k2)


plt.savefig = _dual_savefig

import config
import step03_efa_pca as s3
import step04_visualizations as s4
import step04c_cross_phase_viz as s4c
import rerender_loading_matrices as rl

PHASES = [("Phase 1", BASE / "output_phase1"), ("Phase 2", BASE / "output_phase2")]


def per_phase():
    for label, d in PHASES:
        print("\n" + "#" * 70)
        print(f"# {label}  ({d.name})")
        print("#" * 70)
        s3.OUTPUT_DIR = d
        s4.OUTPUT_DIR = d
        s3.run()      # scree, loading_matrix, correlation_matrix (+ pca_loadings.csv)
        s4.run()      # radar, extended_tem, dim_heatmap, ws_detail, temporal, membership_heatmap, margin
    # 4-PC-Ladungsmatrix konsistent (re)erzeugen
    rl.main()


def topic_quality_boxplots():
    p1 = pd.read_csv(BASE / "output_phase1" / "topic_quality_per_topic.csv")
    p2 = pd.read_csv(BASE / "output_phase2" / "topic_quality_per_topic.csv")
    navy = "#1f3b5c"
    orange = config.SIGNAL_COLORS.get("Emerging Concept", "#E67E22")
    specs = [("c_v", r"$C_v$ (top-10)"), ("c_npmi", r"$C_{NPMI}$ (top-10)")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (col, title) in zip(axes, specs):
        data = [p1[col].dropna().values, p2[col].dropna().values]
        ax.boxplot(
            data,
            tick_labels=[f"Phase 1\n(n={len(p1)})", f"Phase 2\n(n={len(p2)})"],
            widths=0.5, patch_artist=True, showmeans=True,
            medianprops=dict(color=navy, linewidth=2),
            meanprops=dict(marker="D", markerfacecolor=orange,
                           markeredgecolor=orange, markersize=8),
            flierprops=dict(marker="o", markerfacecolor="none",
                            markeredgecolor="#888888", markersize=5),
            boxprops=dict(facecolor="white", edgecolor=navy, linewidth=1.5),
            whiskerprops=dict(color=navy, linewidth=1.3),
            capprops=dict(color=navy, linewidth=1.3),
        )
        ax.set_title(title)
        ax.set_ylabel("Kohaerenz pro Topic")
        ax.axhline(0, color="#bbbbbb", linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, BASE / "output_cross_phase" / "topic_quality_boxplots.png")
    plt.close(fig)
    print("  topic_quality_boxplots erzeugt")


def framework_unified():
    """Idealtypisches Schema mit EINHEITLICHER Palette (Weak=rot, Emerging=orange, Trend=blau).

    Hinweis: Die Werte (1-5) sind idealtypisch und aus der bisherigen framework.png
    abgelesen. Bei Bedarf hier anpassen.
    """
    # (Label-Text, Basis-Dimension fuer Color-Coding wie in den uebrigen Radars)
    dim_specs = [
        ("Epistemische\nOffenheit",   "Epistemische Offenheit"),
        ("Niedrige\nWahrnehmbarkeit", "Wahrnehmbarkeit"),
        ("Frühe\nEntwicklungsphase", "Entwicklungsphase"),
        ("Niedrige\nDiffusion",       "Diffusion"),
        ("Wirkungspotenzial",         "Wirkungspotenzial"),
    ]
    labels = [d[0] for d in dim_specs]
    label_colors = [config.DIM_COLORS[d[1]] for d in dim_specs]
    profiles = {
        "Weak Signal":      [5.0, 4.7, 4.6, 4.2, 5.0],
        "Emerging Concept": [3.2, 3.2, 3.1, 2.7, 4.0],
        "Trend":            [1.2, 1.3, 1.3, 1.8, 2.7],
    }
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang += ang[:1]

    fig, ax = plt.subplots(figsize=(9.4, 9.4), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 5)

    # Profile (Zeichenflaeche) im Hintergrund
    for name, vals in profiles.items():
        c = config.SIGNAL_COLORS.get(name, "#777777")
        v = vals + vals[:1]
        ax.plot(ang, v, "o-", color=c, linewidth=2.6, markersize=6, label=name, zorder=3)
        ax.fill(ang, v, color=c, alpha=0.10, zorder=2)

    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color="#9aa0a6", fontsize=10)
    ax.set_rlabel_position(18)
    ax.set_xticks(ang[:-1])
    # Dimensions-Labels: farbcodiert, ausgeschrieben, OHNE Box (Stil wie radar_p2)
    ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
    for lab, col in zip(ax.get_xticklabels(), label_colors):
        lab.set_color(col)
    ax.tick_params(axis="x", pad=14)
    ax.grid(color="#cccccc", linewidth=0.8, alpha=0.7, zorder=1)
    ax.spines["polar"].set_color("#cccccc")

    fig.suptitle("Idealtypische Signalprofile im F2-Framework",
                 fontsize=15, fontweight="bold", y=1.04)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
              fontsize=11, frameon=False)
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.12, right=0.88)
    save_fig(fig, BASE / "framework_unified.png")
    plt.close(fig)
    print("  framework_unified erzeugt")


if __name__ == "__main__":
    per_phase()
    print("\n" + "#" * 70)
    print("# CROSS-PHASE")
    print("#" * 70)
    s4c.run()       # migration_sankey, membership_shift_heatmap, structure_compare_radar
    topic_quality_boxplots()
    framework_unified()
    print("\nALLES FERTIG.")
