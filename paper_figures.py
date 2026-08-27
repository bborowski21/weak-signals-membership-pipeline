"""paper_figures.py

Publikationsfassung ALLER Pipeline-Abbildungen als eigenstaendige Kopie.
Die Pipeline-Skripte (step03/step04/step04b, rerender_*) bleiben unangetastet;
dieses Skript liest ausschliesslich vorhandene Artefakte aus
output_phase1/, output_phase2/, output_cross_phase/ und schreibt nach
figures_paper/{phase1,phase2,cross_phase}/ jeweils PNG (300 dpi; 150 dpi bei
sehr hohen Per-Topic-Heatmaps) plus Vektor-PDF.

Designlinie (Referenz: migration_sankey aus step04b, v. 27.08.2026):
- englische Beschriftung, KEINE eingebetteten Titel (Titel gehoeren in die
  Bildunterschrift des Manuskripts); notwendige Panel-Kennungen bleiben
- keine Em-Dashes in Textelementen
- Farbzuordnung aus config.py (Signaltypen, Dimensionen) bleibt erhalten
- Zahlen im Bild stammen direkt aus den Artefakt-CSVs

HINWEIS TERMINOLOGIE: Die englischen Dimensionsnamen unten sind PROVISORISCH
und werden beim Kick-off mit den Co-Autoren final festgelegt. Nur DIM_EN
anpassen, alles andere zieht nach.

Aufruf:  python3 paper_figures.py            (alles)
         python3 paper_figures.py --only radar_profiles,scree_plot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import (
    BASE_DIR, SIGNAL_COLORS, DIM_COLORS, DIM_NAMES, INDICATOR_DIMENSIONS,
)

# ---------------------------------------------------------------- Verzeichnisse
P1_DIR = BASE_DIR / "output_phase1"
P2_DIR = BASE_DIR / "output_phase2"
CROSS_DIR = BASE_DIR / "output_cross_phase"
PAPER_DIR = BASE_DIR / "figures_paper"

# ------------------------------------------------------- Provisorische EN-Terme
DIM_EN = {  # deutsch -> (englischer Name, Kurzcode)  [beim Kick-off final]
    "Epistemische Offenheit": ("Epistemic openness", "EO"),
    "Wahrnehmbarkeit":        ("Perceptibility", "PE"),
    "Entwicklungsphase":      ("Developmental phase", "DP"),
    "Diffusion":              ("Diffusion", "DI"),
    "Wirkungspotenzial":      ("Impact potential", "IP"),
}
DIM_EN_NAMES = [DIM_EN[d][0] for d in DIM_NAMES]
DIM_EN_CODES = [DIM_EN[d][1] for d in DIM_NAMES]

MEMBERSHIP_COLUMNS = ["m_ws", "m_trend", "m_ec", "m_latent"]
MEMBERSHIP_LABELS = {
    "m_ws": "Weak Signal", "m_trend": "Trend",
    "m_ec": "Emerging Concept", "m_latent": "Latent/Mixed",
}
SIGNAL_ORDER = ["Weak Signal", "Emerging Concept", "Trend", "Latent/Mixed"]

PHASES = {
    1: {"dir": P1_DIR, "out": "phase1", "label": "Phase 1 (2000–2015)",
        "years": (2000, 2015)},
    2: {"dir": P2_DIR, "out": "phase2", "label": "Phase 2 (2016–2025)",
        "years": (2016, 2025)},
}

# ------------------------------------------------------------------- Styling
PUB_RCPARAMS = {
    "figure.dpi": 110,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
    "font.size": 12,
    "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8,
}
matplotlib.rcParams.update(PUB_RCPARAMS)

TEXT_DARK = "#222222"


def darken(hex_color: str, f: float = 0.72) -> str:
    """Mischt eine Hex-Farbe Richtung Schwarz (Textkontrast auf Weiss)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(int(r * f), int(g * f), int(b * f))


DIM_TEXT = {d: darken(DIM_COLORS[d]) for d in DIM_NAMES}
SIG_TEXT = {s: darken(c) for s, c in SIGNAL_COLORS.items()}


def save(fig, sub: str, name: str, tall: bool = False) -> None:
    out = PAPER_DIR / sub
    out.mkdir(parents=True, exist_ok=True)
    dpi = 150 if tall else 300
    fig.savefig(out / f"{name}.png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {sub}/{name}.png (+ PDF)")


# ------------------------------------------------------------------ Datenlader
def load_phase(ph: int) -> dict:
    d = PHASES[ph]["dir"]
    mem = pd.read_csv(d / "signal_memberships.csv", index_col=0)
    mem.index.name = "topic"
    mem["signal_type"] = mem[MEMBERSHIP_COLUMNS].idxmax(axis=1).map(
        MEMBERSHIP_LABELS)
    mem["ws_distance"] = 1.0 - mem["m_ws"]
    dim = pd.read_csv(d / "dimension_scores.csv", index_col=0)
    dim.index.name = "topic"
    classified = mem.join(dim, how="left")
    kw_df = pd.read_csv(d / "topic_keywords.csv")
    topic_kw = {
        tid: list(zip(g["keyword"].tolist(), g["score"].tolist()))
        for tid, g in kw_df.groupby("topic")
    }
    return {"classified": classified, "kw": topic_kw, "dir": d}


# ================================================================ Phasen-Figuren
def fig_radar_profiles(ph: int, data: dict) -> None:
    classified = data["classified"]
    fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(DIM_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    for st in SIGNAL_ORDER:
        sub = classified[classified["signal_type"] == st]
        if len(sub) == 0:
            continue
        vals = sub[DIM_NAMES].mean().tolist()
        vals += vals[:1]
        c = SIGNAL_COLORS[st]
        ax.plot(angles, vals, "o-", color=c, linewidth=2.2, markersize=5.5,
                label=f"{st} ($n$ = {len(sub)})")
        ax.fill(angles, vals, alpha=0.08, color=c)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([n.replace(" ", "\n") for n in DIM_EN_NAMES],
                       fontsize=12)
    for lab, d_ in zip(ax.get_xticklabels(), DIM_NAMES):
        lab.set_color(DIM_TEXT[d_])
        lab.set_fontweight("bold")
    ax.tick_params(axis="x", pad=16)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_rlabel_position(54)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2,
              fontsize=11)
    save(fig, PHASES[ph]["out"], "radar_profiles")


def fig_extended_tem(ph: int, data: dict) -> None:
    classified = data["classified"]
    tem = pd.read_csv(data["dir"] / "tem_metrics.csv").set_index("topic")
    merged = tem.join(
        classified[["signal_type", "Epistemische Offenheit", "margin"]],
        how="inner")

    fig, ax = plt.subplots(figsize=(12.5, 8.6))
    x_thresh = merged["avg_proportion"].median()
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.4, linewidth=1.2)
    ax.axvline(x_thresh, color="gray", linestyle="--", alpha=0.4,
               linewidth=1.2)

    props = dict(fontsize=10.5, alpha=0.42, fontweight="bold",
                 color="#555555", transform=ax.transAxes)
    ax.text(0.015, 0.985, "WEAK SIGNAL\nlow share, high growth",
            ha="left", va="top", **props)
    ax.text(0.985, 0.985, "STRONG SIGNAL\nhigh share, high growth",
            ha="right", va="top", **props)
    ax.text(0.015, 0.015, "LATENT\nlow share, decline",
            ha="left", va="bottom", **props)
    ax.text(0.985, 0.015, "ESTABLISHED\nhigh share, decline",
            ha="right", va="bottom", **props)

    for _, row in merged.iterrows():
        color = SIGNAL_COLORS.get(row["signal_type"], "#999999")
        size = 50 + abs(row["Epistemische Offenheit"]) * 300
        m_val = float(row["margin"]) if not pd.isna(row["margin"]) else 0.0
        if m_val >= 0.10:
            b_alpha, b_edge, b_lw = 0.70, "white", 0.5
        elif m_val >= 0.05:
            b_alpha, b_edge, b_lw = 0.50, "#333333", 0.8
        else:
            b_alpha, b_edge, b_lw = 0.30, "#666666", 0.8
        ax.scatter(row["avg_proportion"], row["growth_rate"], s=size,
                   c=color, alpha=b_alpha, edgecolors=b_edge, linewidth=b_lw)

    spacer = mpatches.Patch(color="none", label="")
    handles = [mpatches.Patch(color="none", label="$\\bf{Class}$")]
    handles += [mpatches.Patch(color=SIGNAL_COLORS[s], label=s, alpha=0.6)
                for s in SIGNAL_ORDER]
    handles += [spacer,
                mpatches.Patch(color="none",
                               label="$\\bf{Bubble\\ size}$: EO"),
                plt.scatter([], [], s=80, c="gray", alpha=0.5,
                            edgecolors="#666666", linewidth=0.4,
                            label="low epistemic openness"),
                plt.scatter([], [], s=500, c="gray", alpha=0.5,
                            edgecolors="#666666", linewidth=0.4,
                            label="high epistemic openness"),
                spacer,
                mpatches.Patch(color="none",
                               label="$\\bf{Outline/alpha}$: margin"),
                plt.scatter([], [], s=120, c="gray", alpha=0.70,
                            edgecolors="#444444", linewidth=0.8,
                            label="margin ≥ 0.10 (clear)"),
                plt.scatter([], [], s=120, c="gray", alpha=0.50,
                            edgecolors="#333333", linewidth=0.8,
                            label="0.05 ≤ margin < 0.10 (transition)"),
                plt.scatter([], [], s=120, c="gray", alpha=0.30,
                            edgecolors="#666666", linewidth=0.8,
                            label="margin < 0.05 (ambiguous)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=9.5, labelspacing=0.85, borderaxespad=0.0,
              handlelength=1.8)

    ax.set_xlabel("Average topic proportion $\\bar{p}$")
    ax.set_ylabel("Annualised growth rate $g$")
    ax.grid(True, alpha=0.2)
    save(fig, PHASES[ph]["out"], "extended_tem")


def fig_dimension_heatmap(ph: int, data: dict) -> None:
    classified, topic_kw = data["classified"], data["kw"]
    type_order = {s: i for i, s in enumerate(
        ["Weak Signal", "Emerging Concept", "Latent/Mixed", "Trend"])}
    sdf = classified.copy()
    sdf["type_order"] = sdf["signal_type"].map(type_order)
    sdf = sdf.sort_values(["type_order", "ws_distance"])

    labels = []
    for idx in sdf.index:
        kws = topic_kw.get(idx, [])
        kw_str = ", ".join(w for w, _ in kws[:2]) if kws else f"T{idx}"
        labels.append(f"T{idx}: {kw_str[:30]} (Δ={sdf.loc[idx, 'margin']:.2f})")

    fig, ax = plt.subplots(figsize=(11, max(16, len(sdf) * 0.15)))
    im = ax.imshow(sdf[DIM_NAMES].values, aspect="auto", cmap="RdYlBu_r",
                   vmin=-2, vmax=2)
    ax.set_xticks(range(len(DIM_NAMES)))
    ax.set_xticklabels(DIM_EN_CODES, fontsize=10)
    for lab, d_ in zip(ax.get_xticklabels(), DIM_NAMES):
        lab.set_color(DIM_TEXT[d_])
        lab.set_fontweight("bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for i, (_, row) in enumerate(sdf.iterrows()):
        ax.add_patch(plt.Rectangle(
            (-0.7, i - 0.5), 0.3, 1, clip_on=False,
            color=SIGNAL_COLORS.get(row["signal_type"], "#999999")))
    prev = None
    for i, (_, row) in enumerate(sdf.iterrows()):
        if row["signal_type"] != prev and prev is not None:
            ax.axhline(i - 0.5, color="black", linewidth=1.5)
        prev = row["signal_type"]
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Dimension score (z-standardised)")
    save(fig, PHASES[ph]["out"], "dimension_heatmap", tall=True)


def fig_ws_detail_radars(ph: int, data: dict, n_top: int = 6) -> None:
    classified, topic_kw = data["classified"], data["kw"]
    ws = classified[classified["signal_type"] == "Weak Signal"].sort_values(
        "ws_distance")
    top_ws = ws.head(n_top)
    if len(top_ws) == 0:
        return
    n_cols = min(3, len(top_ws))
    n_rows = (len(top_ws) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 4.9 * n_rows),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_1d(axes).flatten()
    angles = np.linspace(0, 2 * np.pi, len(DIM_NAMES), endpoint=False).tolist()
    angles += angles[:1]
    ws_med = ws[DIM_NAMES].median().tolist()
    ws_med += ws_med[:1]

    i = -1
    for i, (idx, row) in enumerate(top_ws.iterrows()):
        ax = axes[i]
        vals = row[DIM_NAMES].tolist()
        vals += vals[:1]
        ax.plot(angles, ws_med, "--", color="#999999", linewidth=1,
                alpha=0.6, label="Weak-signal median")
        ax.fill(angles, ws_med, alpha=0.05, color="gray")
        ax.plot(angles, vals, "o-", color=SIGNAL_COLORS["Weak Signal"],
                linewidth=2, markersize=4.5, label="Topic profile")
        ax.fill(angles, vals, alpha=0.14, color=SIGNAL_COLORS["Weak Signal"])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIM_EN_CODES, fontsize=10.5)
        for lab, d_ in zip(ax.get_xticklabels(), DIM_NAMES):
            lab.set_color(DIM_TEXT[d_])
            lab.set_fontweight("bold")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.set_rlabel_position(36)
        ax.tick_params(axis="y", labelsize=7)
        kws = topic_kw.get(idx, [])
        kw_str = ", ".join(w for w, _ in kws[:3])
        other_max = max(row["m_trend"], row["m_ec"], row["m_latent"])
        ax.set_title(f"T{idx}: {kw_str[:36]}\n"
                     f"WS distance {row['ws_distance']:.2f} · "
                     f"margin {row['m_ws'] - other_max:.2f}",
                     fontsize=10.5, pad=14)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    handles, lab_txt = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab_txt, loc="lower center", ncol=2, fontsize=10.5,
               bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(hspace=0.42, wspace=0.38, top=0.94, bottom=0.07)
    save(fig, PHASES[ph]["out"], "ws_detail_radars")


def fig_temporal_evolution(ph: int, data: dict) -> None:
    classified = data["classified"]
    assign = pd.read_csv(data["dir"] / "topic_assignments.csv",
                         usecols=["Year", "topic"])
    assign = assign[assign["topic"] >= 0]
    assign["signal_type"] = assign["topic"].map(classified["signal_type"])
    counts = (assign.groupby(["Year", "signal_type"]).size()
              .unstack(fill_value=0))
    props = counts.div(counts.sum(axis=1), axis=0)

    order = ["Weak Signal", "Emerging Concept", "Latent/Mixed", "Trend"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4))

    base = pd.Series(0.0, index=props.index)
    for st in order:
        if st in props.columns:
            new = base + props[st]
            ax1.fill_between(props.index, base, new, alpha=0.30,
                             color=SIGNAL_COLORS[st])
            ax1.plot(props.index, new, color=SIGNAL_COLORS[st],
                     linewidth=1.6, label=st)
            base = new
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Cumulative proportion of publications")
    ax1.set_xlim(props.index.min(), props.index.max())
    ax1.set_ylim(0, 1.0)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax1.legend(fontsize=10, loc="center left")
    ax1.grid(True, alpha=0.2)
    ax1.text(0.02, 1.02, "(a)", transform=ax1.transAxes, fontsize=13,
             fontweight="bold", va="bottom")

    bar_data = counts.reindex(columns=order, fill_value=0)
    bar_data.plot.bar(stacked=True, ax=ax2, width=0.8,
                      color=[SIGNAL_COLORS[c] for c in bar_data.columns],
                      alpha=0.85, legend=False)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Number of publications")
    ax2.tick_params(axis="x", rotation=60)
    ax2.grid(True, axis="y", alpha=0.2)
    ax2.text(0.02, 1.02, "(b)", transform=ax2.transAxes, fontsize=13,
             fontweight="bold", va="bottom")

    fig.tight_layout()
    save(fig, PHASES[ph]["out"], "temporal_evolution")


def fig_membership_heatmap(ph: int, data: dict) -> None:
    classified, topic_kw = data["classified"], data["kw"]
    sdf = classified.sort_values("margin", ascending=False)
    labels = []
    for idx in sdf.index:
        kws = topic_kw.get(idx, [])
        kw_str = ", ".join(w for w, _ in kws[:2]) if kws else f"T{idx}"
        labels.append(f"T{idx}: {kw_str[:30]} (Δ={sdf.loc[idx, 'margin']:.2f})")
    fig, ax = plt.subplots(figsize=(9.5, max(14, len(sdf) * 0.15)))
    im = ax.imshow(sdf[MEMBERSHIP_COLUMNS].values, aspect="auto",
                   cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(MEMBERSHIP_COLUMNS)))
    ax.set_xticklabels([MEMBERSHIP_LABELS[c] for c in MEMBERSHIP_COLUMNS],
                       fontsize=10, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    cbar = plt.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label("Membership score [0, 1]")
    save(fig, PHASES[ph]["out"], "membership_heatmap", tall=True)


def fig_margin_distribution(ph: int, data: dict) -> None:
    m = data["classified"]["margin"]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.hist(m, bins=40, color="#3498DB", alpha=0.75, edgecolor="white")
    ax.axvline(0.05, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"Δ < 0.05 (ambiguous): {(m < 0.05).sum()} topics")
    ax.axvline(0.10, color="#F39C12", linestyle="--", linewidth=1.5,
               label=f"Δ < 0.10 (transition zone): {(m < 0.10).sum()} topics")
    ax.set_xlabel("Margin $\\Delta = m_{(1)} - m_{(2)}$")
    ax.set_ylabel("Number of topics")
    ax.legend(fontsize=10.5)
    ax.grid(True, alpha=0.2)
    ax.text(0.985, 0.72, f"$n$ = {len(m)} topics\nmedian $\\Delta$ = {m.median():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
            color="#444444")
    save(fig, PHASES[ph]["out"], "margin_distribution")


# ================================================================= EFA-Figuren
def fig_scree_plot(ph: int) -> None:
    d = PHASES[ph]["dir"]
    s = json.loads((d / "efa_summary.json").read_text())
    ev = np.array(s["eigenvalues"], dtype=float)
    pa = np.array(s["pa_thresholds_mean"], dtype=float)
    n_keep = int(s["n_parallel"])
    x = np.arange(1, len(ev) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(x[:n_keep], ev[:n_keep], width=0.42, color="#3448DB",
           alpha=0.18, zorder=1)
    ax.plot(x, ev, "o-", color="#1a2ecc", linewidth=2.2, markersize=7,
            label="Observed eigenvalues", zorder=3)
    ax.plot(x[:len(pa)], pa, "--", color="#E74C3C", linewidth=2,
            label="Parallel-analysis threshold (mean)", zorder=2)
    ax.axhline(1.0, color="#aaaaaa", linestyle=":", linewidth=1.6,
               label="Kaiser criterion ($\\lambda$ = 1)", zorder=1)
    ax.set_xlabel("Factor number")
    ax.set_ylabel("Eigenvalue")
    ax.set_xticks(x)
    ax.legend(fontsize=10.5)
    ax.grid(True, alpha=0.2)
    ax.text(0.985, 0.72,
            f"{n_keep} factors retained\n(parallel analysis)",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5,
            color="#444444")
    fig.tight_layout()
    save(fig, PHASES[ph]["out"], "scree_plot")


def _indicator_groups() -> list:
    """[(dimension, [indikatoren])] in kanonischer Reihenfolge."""
    return [(d, INDICATOR_DIMENSIONS[d]) for d in DIM_NAMES]


def fig_loading_matrix(ph: int, k: int) -> None:
    d = PHASES[ph]["dir"]
    pat = pd.read_csv(d / f"efa_pattern_{k}f.csv", index_col=0)
    order = [i for _, inds in _indicator_groups() for i in inds
             if i in pat.index]
    pat = pat.reindex(order)

    fig, ax = plt.subplots(figsize=(7.8, 9.6))
    im = ax.imshow(pat.values, aspect="auto", cmap="RdBu_r",
                   vmin=-0.8, vmax=0.8)
    for i in range(pat.shape[0]):
        for j in range(pat.shape[1]):
            v = pat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.45 else "#222222")
    ax.set_xticks(range(pat.shape[1]))
    ax.set_xticklabels(pat.columns, fontsize=11)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10)
    ax.set_xlabel("Factors (pattern loadings)")

    cum = 0
    for dim, inds in _indicator_groups():
        present = [i for i in inds if i in order]
        if not present:
            continue
        if cum > 0:
            ax.axhline(cum - 0.5, color="black", linewidth=1.4)
        code = DIM_EN[dim][1]
        ax.text(-0.42, cum + len(present) / 2 - 0.5, code,
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=12, fontweight="bold", color=DIM_TEXT[dim])
        cum += len(present)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Pattern loading")
    save(fig, PHASES[ph]["out"], f"loading_matrix_{k}f")


def fig_phi_matrix(ph: int, k: int) -> None:
    d = PHASES[ph]["dir"]
    phi = pd.read_csv(d / f"efa_phi_{k}f.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(phi.values, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            v = phi.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10.5,
                    color="white" if abs(v) > 0.6 else "#222222")
    ax.set_xticks(range(phi.shape[1]))
    ax.set_xticklabels(phi.columns, fontsize=11)
    ax.set_yticks(range(phi.shape[0]))
    ax.set_yticklabels(phi.index, fontsize=11)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Factor correlation $\\varphi$")
    save(fig, PHASES[ph]["out"], f"phi_matrix_{k}f")


def fig_correlation_matrix(ph: int) -> None:
    d = PHASES[ph]["dir"]
    ind = pd.read_csv(d / "indicators_16.csv", index_col=0)
    order = [i for _, inds in _indicator_groups() for i in inds
             if i in ind.columns]
    corr = ind[order].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    data = np.ma.masked_array(corr.values, mask=mask)

    fig, ax = plt.subplots(figsize=(13.5, 11.5))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(order)):
        for j in range(i + 1):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5,
                    color="white" if abs(v) > 0.55 else "#222222")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=10.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10.5)
    cum = 0
    for dim, inds in _indicator_groups():
        n = len([i for i in inds if i in order])
        if cum > 0:
            ax.axhline(cum - 0.5, color="black", linewidth=1.3)
            ax.axvline(cum - 0.5, color="black", linewidth=1.3)
        cum += n
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Pearson $r$")
    save(fig, PHASES[ph]["out"], "correlation_matrix")


# ============================================================ Cross-Phase-Figuren
def load_cross() -> dict:
    p1 = load_phase(1)
    p2 = load_phase(2)
    matches = pd.read_csv(CROSS_DIR / "topic_matches_mutual.csv")
    return {"p1": p1, "p2": p2, "matches": matches}


def fig_migration_sankey(cross: dict) -> None:
    import step04b_cross_phase_viz as s4c
    out = PAPER_DIR / "cross_phase"
    out.mkdir(parents=True, exist_ok=True)
    s4c.plot_migration_sankey(cross["p1"]["classified"],
                              cross["p2"]["classified"],
                              cross["matches"],
                              out / "migration_sankey.png")


def fig_membership_shift_heatmap(cross: dict) -> None:
    m = cross["matches"].copy()
    c1, c2 = cross["p1"]["classified"], cross["p2"]["classified"]
    m["p1_class"] = m["phase1_topic"].map(c1["signal_type"])
    m["p2_class"] = m["phase2_topic"].map(c2["signal_type"])
    m["delta"] = (m["phase2_topic"].map(c2["m_ws"])
                  - m["phase1_topic"].map(c1["m_ws"]))
    m = m.dropna(subset=["p1_class", "p2_class", "delta"])

    cls = SIGNAL_ORDER
    mean_mat = np.full((4, 4), np.nan)
    n_mat = np.zeros((4, 4), dtype=int)
    for i, a in enumerate(cls):
        for j, b in enumerate(cls):
            sub = m[(m["p1_class"] == a) & (m["p2_class"] == b)]
            n_mat[i, j] = len(sub)
            if len(sub):
                mean_mat[i, j] = sub["delta"].mean()

    robust = np.abs(mean_mat[(n_mat >= 3) & np.isfinite(mean_mat)])
    vmax = max(float(robust.max()) if robust.size else 0.1, 0.1)

    fig, ax = plt.subplots(figsize=(9.6, 8))
    cmap = plt.get_cmap("RdBu_r")
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    for i in range(4):
        for j in range(4):
            if n_mat[i, j] == 0:
                fc, txt_c, txt = "#ffffff", "#bbbbbb", ""
            elif n_mat[i, j] < 3:
                fc, txt_c = "#f2f2f2", "#888888"
                txt = f"{mean_mat[i, j]:+.3f}\n($n$ = {n_mat[i, j]})"
            else:
                fc = cmap(norm(mean_mat[i, j]))
                txt_c = ("white" if abs(mean_mat[i, j]) > 0.55 * vmax
                         else "#222222")
                txt = f"{mean_mat[i, j]:+.3f}\n($n$ = {n_mat[i, j]})"
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=fc, edgecolor="white",
                                       linewidth=2))
            if txt:
                ax.text(j, i, txt, ha="center", va="center", fontsize=11.5,
                        color=txt_c)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(3.5, -0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(cls, fontsize=11.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels(cls, fontsize=11.5)
    for lab, s in zip(ax.get_xticklabels(), cls):
        lab.set_color(SIG_TEXT[s]); lab.set_fontweight("bold")
    for lab, s in zip(ax.get_yticklabels(), cls):
        lab.set_color(SIG_TEXT[s]); lab.set_fontweight("bold")
    ax.set_xlabel("Phase 2 class (argmax)")
    ax.set_ylabel("Phase 1 class (argmax)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label("Mean $\\Delta m_{\\mathrm{ws}}$ (Phase 2 − Phase 1)")
    ax.text(0.0, -0.115,
            "Cells with $n$ < 3 shown in grey (mean unstable).",
            transform=ax.transAxes, fontsize=9.5, color="#777777")
    save(fig, "cross_phase", "membership_shift_heatmap")


def fig_structure_compare_radar(cross: dict, q: float = 0.90) -> None:
    c1, c2 = cross["p1"]["classified"], cross["p2"]["classified"]
    angles = np.linspace(0, 2 * np.pi, len(DIM_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    def prof(df, fn):
        v = [fn(df[d].dropna()) for d in DIM_NAMES]
        return v + v[:1]

    fig, ax = plt.subplots(figsize=(8.8, 8.8), subplot_kw=dict(polar=True))
    P1C, P2C = "#2C3E50", "#E67E22"
    ax.plot(angles, prof(c1, lambda s: s.quantile(q)), "o-", color=P1C,
            linewidth=2.3, markersize=6,
            label=f"Phase 1, 90th percentile ($n$ = {len(c1)})")
    ax.fill(angles, prof(c1, lambda s: s.quantile(q)), alpha=0.06, color=P1C)
    ax.plot(angles, prof(c2, lambda s: s.quantile(q)), "s-", color=P2C,
            linewidth=2.3, markersize=6,
            label=f"Phase 2, 90th percentile ($n$ = {len(c2)})")
    ax.fill(angles, prof(c2, lambda s: s.quantile(q)), alpha=0.06, color=P2C)
    ax.plot(angles, prof(c1, lambda s: s.median()), "o--", color=P1C,
            linewidth=1.3, markersize=3.5, alpha=0.75, label="Phase 1, median")
    ax.plot(angles, prof(c2, lambda s: s.median()), "s--", color=P2C,
            linewidth=1.3, markersize=3.5, alpha=0.75, label="Phase 2, median")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([n.replace(" ", "\n") for n in DIM_EN_NAMES],
                       fontsize=12)
    for lab, d_ in zip(ax.get_xticklabels(), DIM_NAMES):
        lab.set_color(DIM_TEXT[d_])
        lab.set_fontweight("bold")
    ax.tick_params(axis="x", pad=16)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_rlabel_position(100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2,
              fontsize=10.5)
    save(fig, "cross_phase", "structure_compare_radar")


def fig_topic_quality_boxplots() -> None:
    p1 = pd.read_csv(P1_DIR / "topic_quality_per_topic.csv")
    p2 = pd.read_csv(P2_DIR / "topic_quality_per_topic.csv")
    navy, orange = "#1f3b5c", SIGNAL_COLORS.get("Emerging Concept", "#E67E22")
    specs = [("c_v", "$C_v$ (top-10)", "(a)"),
             ("c_npmi", "$C_{NPMI}$ (top-10)", "(b)")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, (col, mlabel, tag) in zip(axes, specs):
        data = [p1[col].dropna().values, p2[col].dropna().values]
        ax.boxplot(
            data,
            tick_labels=[f"Phase 1\n($n$ = {len(p1)})",
                         f"Phase 2\n($n$ = {len(p2)})"],
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
        ax.set_ylabel(f"Coherence per topic, {mlabel}")
        ax.axhline(0, color="#bbbbbb", linestyle="--", linewidth=1)
        ax.grid(axis="y", alpha=0.25)
        ax.text(0.02, 1.02, tag, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom")
    fig.tight_layout()
    save(fig, "cross_phase", "topic_quality_boxplots")


def fig_ws_membership_scatter(cross: dict) -> None:
    FLOOR = 0.25
    PH_COLORS = {1: "#2C3E50", 2: "#E67E22"}
    ANNOT_P2 = {0: "T0", 73: "T73", 141: "T141", 188: "T188", 153: "T153"}
    ro = {
        1: pd.read_csv(P1_DIR / "reference_overlap_p1.csv"
                       ).set_index("topic")["ratio_vs_global"],
        2: pd.read_csv(P2_DIR / "reference_overlap_p2.csv"
                       ).set_index("topic")["ratio_vs_global"],
    }
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.axhspan(FLOOR * 0.88, 1.0, color="#E74C3C", alpha=0.07, zorder=0)
    for ph in (1, 2):
        df = cross[f"p{ph}"]["classified"]
        ws = df[df["signal_type"] == "Weak Signal"]
        x = ws["m_ws"].values
        y = np.array([max(ro[ph].get(t, FLOOR), FLOOR) for t in ws.index])
        ax.scatter(x, y, c=PH_COLORS[ph], s=58, alpha=0.78,
                   edgecolors="white", linewidths=0.6, zorder=3,
                   label=f"{PHASES[ph]['label']}  ($n$ = {len(ws)})")
        if ph == 2:
            for t, lab in ANNOT_P2.items():
                if t in ws.index:
                    ax.annotate(lab, (ws.loc[t, "m_ws"],
                                      max(ro[ph].get(t, FLOOR), FLOOR)),
                                textcoords="offset points", xytext=(6, 4),
                                fontsize=9, fontweight="bold",
                                color=PH_COLORS[ph], zorder=4)
    ax.axhline(1.0, color="#555555", ls="--", lw=1.2, alpha=0.85, zorder=2)
    ax.text(0.503, 1.10,
            "$\\rho_t$ = 1  (corpus baseline; below: reference-heterogeneous)",
            fontsize=9, color="#555555")
    ax.set_yscale("log")
    ax.set_xlabel("Weak-signal membership  $m_{\\mathrm{ws}}$")
    ax.set_ylabel("Reference coherence  $\\rho_t$  (log scale)")
    ax.set_xlim(0.49, 0.97)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="both", alpha=0.22)
    fig.tight_layout()
    save(fig, "cross_phase", "ws_membership_scatter")


def fig_signature_scatter(cross: dict) -> None:
    """NEU: Cross-Phase-Signaturen auf der Zerlegung (sigma_sem, sigma_lex).

    Zonen nach empirischen Terzilen ueber die Mutual-Best-Paare; die drei
    Signaturen folgen Thesis-Abschnitt 3.3 (Befundtypisierung, keine
    Klassifikationsausgabe): stabil/stabil = Trend-Vorbedingung;
    sem hoch + lex niedrig = Concept Drift; sem niedrig = EC-Fruehphase.
    """
    m = cross["matches"].copy()
    kw2 = cross["p2"]["kw"]
    x = m["cosine"].values
    y = m["jaccard"].values
    x_lo, x_hi = np.quantile(x, [1 / 3, 2 / 3])
    y_lo, y_hi = np.quantile(y, [1 / 3, 2 / 3])

    # Konvention (konsistent mit Befund-Dokumentation): "hoch" = >= oberes
    # Terzil, "niedrig" = STRIKT < unteres Terzil. sigma_lex hat auf top-15
    # Jaccard massive Bindungen (18 Paare exakt bei 0.200 = q33 = Median);
    # die strikte Untergrenze reproduziert die dokumentierte Drift-Zelle
    # (n = 4). Der Randfall quantum annealing (0.971/0.200) liegt exakt AUF
    # der Grenze und gehoert nicht zur Zelle.
    stable = (x >= x_hi) & (y >= y_hi)
    drift = (x >= x_hi) & (y < y_lo)
    ec_pre = x < x_lo

    fig, ax = plt.subplots(figsize=(10.2, 7.2))
    x_min, x_max = x.min() - 0.02, x.max() + 0.012
    y_min, y_max = -0.03, y.max() + 0.06

    ax.add_patch(plt.Rectangle((x_hi, y_hi), x_max - x_hi, y_max - y_hi,
                               facecolor="#3498DB", alpha=0.07, zorder=0))
    ax.add_patch(plt.Rectangle((x_hi, y_min), x_max - x_hi, y_lo - y_min,
                               facecolor="#E67E22", alpha=0.10, zorder=0))
    ax.add_patch(plt.Rectangle((x_min, y_min), x_lo - x_min, y_max - y_min,
                               facecolor="#E74C3C", alpha=0.06, zorder=0))
    for v in (x_lo, x_hi):
        ax.axvline(v, color="#999999", linestyle=":", linewidth=1.1, zorder=1)
    for v in (y_lo, y_hi):
        ax.axhline(v, color="#999999", linestyle=":", linewidth=1.1, zorder=1)

    other = ~(stable | drift | ec_pre)
    ax.scatter(x[other], y[other], s=46, c="#8395a7", alpha=0.75,
               edgecolors="white", linewidths=0.5, zorder=3,
               label=f"Outside signature zones ($n$ = {other.sum()})")
    ax.scatter(x[stable], y[stable], s=52, c="#3498DB", alpha=0.85,
               edgecolors="white", linewidths=0.5, zorder=3,
               label=f"Stable core and vocabulary ($n$ = {stable.sum()})")
    ax.scatter(x[ec_pre], y[ec_pre], s=52, c="#E74C3C", alpha=0.85,
               edgecolors="white", linewidths=0.5, zorder=3,
               label=f"Semantically unstable ($n$ = {ec_pre.sum()})")
    ax.scatter(x[drift], y[drift], s=95, c="#E67E22", alpha=0.95,
               edgecolors="#7a4408", linewidths=1.0, zorder=4,
               label=f"Concept drift ($n$ = {drift.sum()})")

    # Drift-Faelle: nummerierte Callouts (Klartext in Caption und stdout),
    # innerhalb gleicher sigma_lex-Werte oben/unten versetzt
    drift_order = sorted(np.where(drift)[0],
                         key=lambda i: (-y[i], -x[i]))
    drift_key = []
    group_seen: dict = {}
    for rank, i in enumerate(drift_order):
        t2 = int(m.iloc[i]["phase2_topic"])
        kws = kw2.get(t2, [])
        lab = ", ".join(w for w, _ in kws[:2]) if kws else f"T{t2}"
        drift_key.append(f"({rank + 1}) P2#{t2}: {lab}")
        gk = round(y[i], 3)
        pos = group_seen.get(gk, 0)
        group_seen[gk] = pos + 1
        dx, dy = (8, -3.5) if pos == 0 else (-15, -3.5)
        ax.annotate(str(rank + 1), (x[i], y[i]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=10, color=darken("#E67E22", 0.55),
                    fontweight="bold", zorder=5)

    zone_props = dict(fontsize=10.5, fontweight="bold", alpha=0.85, zorder=2)
    ax.text(0.985, 0.975, "trend precondition",
            transform=ax.transAxes, ha="right", va="top",
            color=darken("#3498DB"), **zone_props)
    ax.text(0.985, 0.02, "concept drift",
            transform=ax.transAxes, ha="right", va="bottom",
            color=darken("#E67E22"), **zone_props)
    ax.text(0.015, 0.02, "EC early-phase precondition",
            transform=ax.transAxes, ha="left", va="bottom",
            color=darken("#E74C3C"), **zone_props)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Semantic similarity  $\\sigma^{\\mathrm{sem}}$"
                  "  (SBERT centroid cosine)")
    ax.set_ylabel("Lexical similarity  $\\sigma^{\\mathrm{lex}}$"
                  "  (Jaccard, top-15 keywords)")
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.93), fontsize=9.5)
    fig.tight_layout()
    save(fig, "cross_phase", "signature_scatter")
    print(f"    Signaturzellen (Terzile, hoch >= q67, niedrig < q33): "
          f"stable={stable.sum()}, drift={drift.sum()}, "
          f"ec_pre={ec_pre.sum()}, uebrige={other.sum()}, gesamt={len(m)}")
    for line in drift_key:
        print(f"    Drift {line}")


# ============================================================ Framework-Schema
def fig_framework_unified() -> None:
    dim_specs = [
        ("Epistemic\nopenness", "Epistemische Offenheit"),
        ("Low\nperceptibility", "Wahrnehmbarkeit"),
        ("Early developmental\nphase", "Entwicklungsphase"),
        ("Low\ndiffusion", "Diffusion"),
        ("Impact\npotential", "Wirkungspotenzial"),
    ]
    labels = [d[0] for d in dim_specs]
    label_colors = [DIM_TEXT[d[1]] for d in dim_specs]
    profiles = {
        "Weak Signal":      [5.0, 4.7, 4.6, 4.2, 5.0],
        "Emerging Concept": [3.2, 3.2, 3.1, 2.7, 4.0],
        "Trend":            [1.2, 1.3, 1.3, 1.8, 2.7],
    }
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(8.8, 8.8), subplot_kw=dict(polar=True))
    ax.set_ylim(0, 5)
    for name, vals in profiles.items():
        c = SIGNAL_COLORS.get(name, "#777777")
        v = vals + vals[:1]
        ax.plot(ang, v, "o-", color=c, linewidth=2.5, markersize=6,
                label=name, zorder=3)
        ax.fill(ang, v, color=c, alpha=0.10, zorder=2)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color="#9aa0a6", fontsize=9.5)
    ax.set_rlabel_position(18)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=11.5, fontweight="bold")
    for lab, col in zip(ax.get_xticklabels(), label_colors):
        lab.set_color(col)
    ax.tick_params(axis="x", pad=24)
    ax.grid(color="#cccccc", linewidth=0.8, alpha=0.7, zorder=1)
    ax.spines["polar"].set_color("#cccccc")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3,
              fontsize=11)
    save(fig, ".", "framework_unified")


# ======================================================================== main
PHASE_FIGS = {
    "radar_profiles": fig_radar_profiles,
    "extended_tem": fig_extended_tem,
    "dimension_heatmap": fig_dimension_heatmap,
    "ws_detail_radars": fig_ws_detail_radars,
    "temporal_evolution": fig_temporal_evolution,
    "membership_heatmap": fig_membership_heatmap,
    "margin_distribution": fig_margin_distribution,
}
EFA_FIGS = {
    "scree_plot": lambda ph: fig_scree_plot(ph),
    "loading_matrix_4f": lambda ph: fig_loading_matrix(ph, 4),
    "loading_matrix_5f": lambda ph: fig_loading_matrix(ph, 5),
    "phi_matrix_4f": lambda ph: fig_phi_matrix(ph, 4),
    "phi_matrix_5f": lambda ph: fig_phi_matrix(ph, 5),
    "correlation_matrix": lambda ph: fig_correlation_matrix(ph),
}
CROSS_FIGS = {
    "migration_sankey": fig_migration_sankey,
    "membership_shift_heatmap": fig_membership_shift_heatmap,
    "structure_compare_radar": fig_structure_compare_radar,
    "ws_membership_scatter": fig_ws_membership_scatter,
    "signature_scatter": fig_signature_scatter,
}
STANDALONE_FIGS = {
    "topic_quality_boxplots": fig_topic_quality_boxplots,
    "framework_unified": fig_framework_unified,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default=None,
                    help="Kommagetrennte Figurennamen, sonst alle")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    def want(name):
        return only is None or name in only

    print("=" * 70)
    print("PAPER-FIGUREN (eigenstaendige Publikationsfassung)")
    print(f"  Ziel: {PAPER_DIR}")
    print("=" * 70)

    phase_data = {}
    if any(want(n) for n in PHASE_FIGS):
        for ph in (1, 2):
            phase_data[ph] = load_phase(ph)
            print(f"\n{PHASES[ph]['label']}:")
            for name, fn in PHASE_FIGS.items():
                if want(name):
                    fn(ph, phase_data[ph])
    if any(want(n) for n in EFA_FIGS):
        for ph in (1, 2):
            print(f"\nEFA {PHASES[ph]['label']}:")
            for name, fn in EFA_FIGS.items():
                if want(name):
                    fn(ph)
    if any(want(n) for n in CROSS_FIGS):
        print("\nCross-Phase:")
        cross = load_cross()
        for name, fn in CROSS_FIGS.items():
            if want(name):
                fn(cross)
    for name, fn in STANDALONE_FIGS.items():
        if want(name):
            print(f"\n{name}:")
            fn()

    print("\n" + "=" * 70)
    print("FERTIG. Alle Ausgaben in figures_paper/ (PNG + PDF).")
    print("=" * 70)


if __name__ == "__main__":
    main()
