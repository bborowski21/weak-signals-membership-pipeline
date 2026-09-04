"""paper_figures_rp.py

Publikationsfassung aller Abbildungen im Stil fuer Research Policy (Elsevier).
Eigenstaendige Kopie neben paper_figures.py; die Pipeline-Skripte bleiben unangetastet.
Liest nur vorhandene Artefakte (output_phase1/, output_phase2/, output_cross_phase/,
optional perturbation_*.csv) und schreibt nach figures_rp/ je Abbildung
PDF (Vektor, exakte Endbreite), TIFF (RGB, LZW, 600 dpi), PNG (Sichtprobe, 300 dpi)
und eine Graustufenprobe. Stil, Palette und Zweitkodierung: rp_style.py.

Figure-Plan (provisorisch, Nummern folgen der Manuskriptreihenfolge; am 09.10. bestaetigen):
  Haupttext   Fig1 configuration_profiles, Fig2 margin_distribution, Fig3 class_profiles,
              Fig4 perturbation_flip_vs_margin, Fig5 ws_reference_coherence, Fig6 efa_pattern_loadings
  Anhang A    FigA1 temporal_evolution, FigA2 extended_tem, FigA3 scree, FigA4 factor_correlations,
              FigA5a/b indicator_correlations, FigA6 dimension_heatmap, FigA7 membership_heatmap,
              FigA8a/b ws_detail_radars, FigA9 structure_compare, FigA10 topic_quality,
              FigA11 perturbation_flip_by_margin_class
  Parkplatz   FigP1 migration_sankey, FigP2 membership_shift, FigP3 signature_scatter
              (Cross-Phase-Befunde, Option B; Entscheidung am 09.10.)

Terminologie: englische Dimensionsnamen sind PROVISORISCH (rp_style.DIM_EN).
Aufruf:  python3 paper_figures_rp.py [--only Fig2,FigA3] [--perturbation-dir PFAD] [--out figures_rp]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
from matplotlib.ticker import MaxNLocator, FixedLocator

warnings.filterwarnings("ignore")

import rp_style as rp
from config import BASE_DIR, DIM_NAMES, INDICATOR_DIMENSIONS

P1_DIR = BASE_DIR / "output_phase1"
P2_DIR = BASE_DIR / "output_phase2"
CROSS_DIR = BASE_DIR / "output_cross_phase"
OUT_DIR = BASE_DIR / "figures_rp"
PERT_DIR: Path | None = None

PHASES = {1: {"dir": P1_DIR, "years": (2000, 2015)}, 2: {"dir": P2_DIR, "years": (2016, 2025)}}
MEMB = ["m_ws", "m_trend", "m_ec", "m_latent"]
MEMB_LABEL = {"m_ws": "Weak Signal", "m_trend": "Trend", "m_ec": "Emerging Concept", "m_latent": "Latent/Mixed"}
CLS = rp.CLASS_ORDER
NOISE = [0.05, 0.10, 0.20]
NOISE_COLOR = {0.05: "#c6d7ea", 0.10: "#5b8ec4", 0.20: "#07519d"}
NOISE_MARKER = {0.05: "o", 0.10: "s", 0.20: "^"}
NOISE_GREY = {0.05: rp.STEP_GREY[0], 0.10: rp.STEP_GREY[1], 0.20: rp.STEP_GREY[2]}
NOISE_HATCH = {0.05: "", 0.10: "//", 0.20: "x"}

# Radarachsen: identische Beschriftung in Fig1, Fig3 und FigA9 (Richtung im Namen, s. rp_style)
RADAR_LABELS = ["Epistemic\nopenness", "Low\nperceptibility", "Early developmental\nstage",
                "Low\ndiffusion", "Impact\npotential"]                      # Zweispalter
RADAR_LABELS_NARROW = ["Epistemic\nopenness", "Low\nperceptibility", "Early\ndevelopmental\nstage",
                       "Low\ndiffusion", "Impact\npotential"]               # Einspalter
RADAR_A8_RANGE = (3.0, -1.5)   # (rmax, rmin) fuer FigA8a UND FigA8b, damit das Paar vergleichbar ist


# ------------------------------------------------------------------ Daten
def load_phase(ph: int) -> dict:
    d = PHASES[ph]["dir"]
    mem = pd.read_csv(d / "signal_memberships.csv", index_col=0)
    mem.index.name = "topic"
    mem["signal_type"] = mem[MEMB].idxmax(axis=1).map(MEMB_LABEL)
    mem["ws_distance"] = 1.0 - mem["m_ws"]
    dim = pd.read_csv(d / "dimension_scores.csv", index_col=0)
    dim.index.name = "topic"
    classified = mem.join(dim, how="left")
    kw_df = pd.read_csv(d / "topic_keywords.csv")
    topic_kw = {tid: list(zip(g["keyword"].tolist(), g["score"].tolist())) for tid, g in kw_df.groupby("topic")}
    return {"classified": classified, "kw": topic_kw, "dir": d}


def counts_by_class(df: pd.DataFrame) -> dict:
    return {c: int((df["signal_type"] == c).sum()) for c in CLS}


def pert_paths(ph: int):
    """perturbation_topics.csv und perturbation_summary.csv fuer eine Phase (Pipeline-Ordner oder --perturbation-dir)."""
    cands = [PHASES[ph]["dir"]]
    if PERT_DIR is not None:
        cands += [PERT_DIR / f"phase{ph}", PERT_DIR / f"output_phase{ph}", PERT_DIR]
    for c in cands:
        if (c / "perturbation_topics.csv").exists():
            return c / "perturbation_topics.csv", c / "perturbation_summary.csv"
    return None, None


def radar_axes(ax, labels, rmax=None, rmin=None, label_pad=4, codes=False):
    """Radarachse; die Achsenbeschriftung wird als eigene Textebene ueber alles gelegt.

    Bei einem Polarplot sitzen die Beschriftungen auf dem aeusseren Kreis. Als
    Achsenbeschriftung werden sie von Rahmen und Radialgitter durchschnitten, weil
    matplotlib beide nach der Achse zeichnet (betraf „Developmental stage" und
    „Impact potential"). Deshalb: Achsenbeschriftung abschalten und die Namen mit
    weisser Freistellflaeche und hoher Zeichenebene selbst setzen.
    """
    n = len(labels)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=rp.FS["small"], colors=rp.TEXT_MUTED, length=0)
    ax.grid(color=rp.GRID, linewidth=0.4)
    ax.spines["polar"].set_color(rp.GRID)
    ax.spines["polar"].set_linewidth(0.6)
    if rmax is not None:
        ax.set_ylim(rmin if rmin is not None else 0, rmax)
    r_out = ax.get_ylim()[1]
    off = label_pad + 3.0                      # Abstand in Punkten vom Aussenkreis
    for a, text in zip(ang, labels):
        phi = np.pi / 2 - a                    # Bildschirmwinkel (theta_offset pi/2, Richtung -1)
        cx, cy = np.cos(phi), np.sin(phi)
        ha = "left" if cx > 0.2 else ("right" if cx < -0.2 else "center")
        va = "bottom" if cy > 0.2 else ("top" if cy < -0.2 else "center")
        ax.annotate(text, xy=(a, r_out), xytext=(off * cx, off * cy), textcoords="offset points",
                    ha=ha, va=va, multialignment="center", clip_on=False, zorder=25,
                    fontsize=rp.FS["tick"] if codes else rp.FS["label"], color=rp.TEXT,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    for lab in ax.get_yticklabels():           # Radiuswerte ebenfalls freistellen
        lab.set_bbox(dict(facecolor="white", edgecolor="none", pad=1.0))
    return ang


def closed(vals):
    v = list(vals)
    return v + v[:1]


# ================================================================ HAUPTTEXT
def fig1_configuration_profiles() -> None:
    """Stilisierte Profile der Konfigurationen im Fuenf-Dimensionen-Raum (Konzeptabbildung, Schritt 2)."""
    labels = RADAR_LABELS_NARROW
    profiles = {  # ordinal 1 bis 5, stilisiert nach Thesis Kap. 3 (Framework-Schema)
        "Weak Signal":      [5.0, 4.7, 4.6, 4.2, 5.0],
        "Emerging Concept": [3.2, 3.2, 3.1, 2.7, 4.0],
        "Trend":            [1.2, 1.3, 1.3, 1.8, 2.7],
    }
    fig, ax = rp.figure("single", height_mm=88, polar=True)
    ang = radar_axes(ax, labels, rmax=5.2, label_pad=6)
    a = closed(ang)
    for c, vals in profiles.items():
        ax.plot(a, closed(vals), color=rp.CLASS_COLOR[c], ls=rp.CLASS_LS[c], lw=1.2, marker=rp.CLASS_MARKER[c],
                markersize=3.5, markerfacecolor=rp.CLASS_COLOR[c], markeredgecolor=rp.CLASS_EDGE[c], markeredgewidth=0.5,
                label=rp.CLASS_EN[c], zorder=3)
        ax.fill(a, closed(vals), color=rp.CLASS_COLOR[c], alpha=0.06, zorder=2)
    rp.radial_ticks(ax, [1, 2, 3, 4, 5], angle_deg=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, handlelength=2.2, columnspacing=1.0)
    fig.subplots_adjust(left=0.20, right=0.795, top=0.88, bottom=0.17)
    rp.save(fig, "Fig1_configuration_profiles", "single", OUT_DIR, note="Konzeptabbildung, stilisierte Profile")


def fig2_margin_distribution(data: dict) -> None:
    fig, axes = rp.figure("double", height_mm=62, ncols=2, sharey=False)
    bins = np.arange(0, 0.56, 0.01)
    for ax, ph, letter in zip(axes, (1, 2), "AB"):
        m = data[ph]["classified"]["margin"]
        n = len(m); n05 = int((m < 0.05).sum()); n10 = int((m < 0.10).sum())
        ax.axvspan(0, 0.05, color="#d9d9d9", alpha=0.6, lw=0, zorder=0)
        ax.axvspan(0.05, 0.10, color="#efefef", alpha=0.9, lw=0, zorder=0)
        ax.hist(m, bins=bins, color="#3a74b0", edgecolor="white", linewidth=0.3, zorder=2)
        ax.axvline(m.median(), color=rp.TEXT, ls="--", lw=0.8, zorder=3)
        import matplotlib.transforms as mtransforms
        blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(0.115, 0.985, "$\\leftarrow$ hybrid zone $\\Delta$ < 0.10", transform=blend, ha="left", va="top",
                fontsize=rp.FS["small"], color=rp.TEXT_MUTED)
        ax.text(0.98, 0.97, f"{rp.PHASE_LABEL[ph]}\n$n$ = {n} topics, median $\\Delta$ = {m.median():.3f}\n"
                f"$\\Delta$ < 0.05: {n05} topics ({100 * n05 / n:.0f} %)\n"
                f"0.05 $\\leq$ $\\Delta$ < 0.10: {n10 - n05} topics ({100 * (n10 - n05) / n:.0f} %)\n"
                f"$\\Delta$ $\\geq$ 0.10: {n - n10} topics ({100 * (n - n10) / n:.0f} %)",
                transform=ax.transAxes, ha="right", va="top", fontsize=rp.FS["annot"], color=rp.TEXT, linespacing=1.3)
        ax.set_xlim(0, 0.55)
        ax.set_xlabel("Margin $\\Delta$ = $m_{(1)}$ $-$ $m_{(2)}$")
        ax.set_ylabel("Number of topics")   # beide Panels: die y-Skalen sind verschieden (n unterschiedlich)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(5, integer=True))
        rp.grid(ax)
        rp.panel(ax, letter)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.19, wspace=0.14)
    rp.save(fig, "Fig2_margin_distribution", "double", OUT_DIR)


def fig3_class_profiles(data: dict) -> None:
    labels = RADAR_LABELS
    fig, axes = rp.figure("double", height_mm=92, ncols=2, polar=True)
    for ax, ph, letter in zip(axes, (1, 2), "AB"):
        df = data[ph]["classified"]
        ang = radar_axes(ax, labels, rmin=-1.2, rmax=1.0, label_pad=4)
        a = closed(ang)
        cnt = counts_by_class(df)
        for c in CLS:
            sub = df[df["signal_type"] == c]
            if not len(sub):
                continue
            vals = sub[DIM_NAMES].mean().tolist()
            ax.plot(a, closed(vals), color=rp.CLASS_COLOR[c], ls=rp.CLASS_LS[c], lw=1.1, marker=rp.CLASS_MARKER[c],
                    markersize=3.2, markerfacecolor=rp.CLASS_COLOR[c], markeredgecolor=rp.CLASS_EDGE[c],
                    markeredgewidth=0.4, zorder=3)
            ax.fill(a, closed(vals), color=rp.CLASS_COLOR[c], alpha=0.05, zorder=2)
        rp.radial_ticks(ax, [-1.0, -0.5, 0.0, 0.5, 1.0],
                        ["$-$1.0", "$-$0.5", "0.0", "0.5", "1.0"], angle_deg=54)
        # Nullring = Mittel ueber alle Topics. Frueher als Fuenfeck gezeichnet; das sah aus wie
        # eine fuenfte Konfiguration. Jetzt ein echter Kreis, also erkennbar als Bezugslinie.
        circ = np.linspace(0, 2 * np.pi, 240)
        ax.plot(circ, np.zeros_like(circ), color=rp.REF_LINE, lw=0.7, ls="-", zorder=1)
        ax.text(0.5, -0.13, f"{rp.PHASE_LABEL[ph]}, $n$ = {len(df)} topics\n" +
                ", ".join(f"{rp.CLASS_EN[c]} {cnt[c]}" for c in CLS),
                transform=ax.transAxes, ha="center", va="top", fontsize=rp.FS["annot"], color=rp.TEXT_MUTED, linespacing=1.3)
        rp.panel(ax, letter, polar=True)
    fig.legend(handles=rp.class_handles(kind="line"), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2, handlelength=2.4)
    fig.subplots_adjust(left=0.11, right=0.89, top=0.93, bottom=0.24, wspace=0.62)
    rp.save(fig, "Fig3_class_profiles", "double", OUT_DIR, note="Mittelwert der z-standardisierten Dimensionsscores je Konfiguration")


def fig4_perturbation_flip(data: dict) -> None:
    """Kippwahrscheinlichkeit gegen Baseline-Margin, nur die Berichtsstufe 10 % SD.

    Frueher drei Rauschstufen als Kreis / Quadrat / Dreieck uebereinander. Die Rauschstufe
    ist eine GEORDNETE Groesse; eine Formkodierung ist dafuer die falsche Kanalwahl und
    411 Punkte mal drei Stufen lagen nahe x = 0 uebereinander. Der Stufenvergleich steht
    als Balken in FigA11, hier zaehlt allein der Zusammenhang Margin -> Kipprisiko.
    Die Hybridzone ist wie in Fig2 hinterlegt (visueller Gleichklang), die frueheren
    Textmarken am oberen Rand entfallen und mit ihnen die Kollision.
    """
    import matplotlib.transforms as mtransforms
    S = 0.1
    fig, axes = rp.figure("double", height_mm=66, ncols=2, sharey=True)
    ok = True
    for ax, ph, letter in zip(axes, (1, 2), "AB"):
        pt_path, _ = pert_paths(ph)
        if pt_path is None:
            ok = False
            break
        pt = pd.read_csv(pt_path)
        y = pt[f"flip_prob_s{S}"].values
        x = pt["baseline_margin"].values
        ax.axvspan(0, 0.05, color="#d9d9d9", alpha=0.6, lw=0, zorder=0)
        ax.axvspan(0.05, 0.10, color="#efefef", alpha=0.9, lw=0, zorder=0)
        ax.scatter(x, y, s=11, marker="o", facecolor="#3a74b0", edgecolor="white",
                   linewidths=0.4, alpha=0.9, zorder=3)
        blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(0.115, 0.97, "$\\leftarrow$ hybrid zone $\\Delta$ < 0.10", transform=blend, ha="left", va="top",
                fontsize=rp.FS["small"], color=rp.TEXT_MUTED)
        n = len(pt); n_flip = int((y > 0).sum()); xmax = float(x[y > 0].max())
        ax.text(0.985, 0.90, f"{rp.PHASE_LABEL[ph]}\n$n$ = {n} topics\n"
                             f"{n_flip} topics with flip probability > 0\n"
                             f"highest margin with a flip: $\\Delta$ = {xmax:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=rp.FS["annot"],
                color=rp.TEXT, linespacing=1.35)
        ax.set_xlim(-0.006, 0.53); ax.set_ylim(-0.02, 0.80)
        ax.xaxis.set_major_locator(FixedLocator([0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]))
        ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8]))
        ax.set_xlabel("Baseline margin $\\Delta$")
        rp.grid(ax)
        rp.panel(ax, letter)
    if not ok:
        plt.close(fig)
        print("  Fig4 uebersprungen: perturbation_topics.csv nicht gefunden (--perturbation-dir setzen)")
        return
    axes[0].set_ylabel("Flip probability of the dominant\nconfiguration (noise = 10 % of SD)")
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.155, wspace=0.07)
    rp.save(fig, "Fig4_perturbation_flip_vs_margin", "double", OUT_DIR)


def fig5_ws_reference_coherence(data: dict) -> None:
    """Verteilung der Referenzkohaerenz der Weak-Signal-Topics, je Phase eine Zeile.

    Frueher ein Streudiagramm rho gegen m_ws mit Kreis gegen Quadrat. Zwei Gruende fuer die
    neue Form: (1) m_ws und rho sind unkorreliert (Spearman 0.023, p = 0.90 in Phase 1;
    -0.046, p = 0.71 in Phase 2), die x-Achse trug also keine Information und legte einen
    Zusammenhang nahe, den es nicht gibt; (2) die Aussage ist eine Verteilung gegen die
    Korpusbasislinie rho = 1, dafuer ist Box plus Punktstreifen die direkte Form (und sie
    passt zu FigA10). Phasen: gefuellt gegen offen, gleiche Form (nominale Zweiergruppe).
    Erweiterbar auf alle vier Konfigurationen, falls die Ko-Autoren das wollen.
    """
    FLOOR = 0.20
    ROW = {1: 1.0, 2: 0.0}
    fig, ax = rp.figure("single", height_mm=58)
    rng = np.random.default_rng(20260904)
    ax.axvspan(FLOOR * 0.86, 1.0, color="#efefef", lw=0, zorder=0)
    info = {}
    for ph in (1, 2):
        df = data[ph]["classified"]
        ro = pd.read_csv(PHASES[ph]["dir"] / f"reference_overlap_p{ph}.csv").set_index("topic")["ratio_vs_global"]
        ws = df[df["signal_type"] == "Weak Signal"]
        raw = np.array([float(ro.get(t, np.nan)) for t in ws.index])
        xv = np.clip(raw, FLOOR, None)
        yb = ROW[ph]
        bp = ax.boxplot([xv], positions=[yb], vert=False, widths=0.52, showfliers=False,
                        whis=(5, 95), manage_ticks=False, zorder=2)
        for k in ("boxes", "whiskers", "caps"):
            for art in bp[k]:
                art.set(color="#8a8a8a", linewidth=0.7)
        for art in bp["medians"]:
            art.set(color=rp.CLASS_EDGE["Weak Signal"], linewidth=1.4)
        yv = yb + rng.uniform(-0.17, 0.17, len(xv))
        # Einheitlicher Marker: die Phase steht schon an der Achse. Gefuellt gegen offen
        # kodierte nichts und warf beim Lesen die Frage auf, was der Unterschied bedeutet.
        # Offene Kreise, weil sich bei 68 ueberlappenden Punkten so die Dichte lesen laesst.
        ax.scatter(xv, yv, s=12, marker="o", facecolor="white", edgecolor=rp.TEXT,
                   linewidths=0.6, zorder=3)
        med = float(np.nanmedian(raw))
        ax.text(med * 1.13, yb + 0.30, f"median {med:.1f}", ha="left", va="bottom",
                fontsize=rp.FS["small"], color=rp.CLASS_EDGE["Weak Signal"])
        info[ph] = {"n": len(ws), "med": med, "below": [int(t) for t in ws.index[raw < 1.0]]}
        if ph == 2:
            j = int(np.argmin(raw))
            ax.annotate(f"T{ws.index[j]} ($\\rho_t$ = 0)", (xv[j], yv[j]), textcoords="offset points",
                        xytext=(-3, 9), ha="left", va="bottom", fontsize=rp.FS["small"], color=rp.TEXT, zorder=5)
    ax.axvline(1.0, color=rp.REF_LINE, ls="--", lw=0.8, zorder=1)
    ax.text(0.93, 1.52, "corpus baseline $\\rho_t$ = 1", ha="right", va="center",
            fontsize=rp.FS["small"], color=rp.TEXT_MUTED)
    ax.text(0.175, -0.40, "$\\leftarrow$ references more heterogeneous than the corpus", ha="left", va="center",
            fontsize=rp.FS["small"], color=rp.TEXT_MUTED)
    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.83, 140)
    ax.set_ylim(-0.48, 1.72)
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([ROW[1], ROW[2]])
    ax.set_yticklabels([f"Phase 1\n(2000 to 2015)\n$n$ = {info[1]['n']}",
                        f"Phase 2\n(2016 to 2025)\n$n$ = {info[2]['n']}"],
                       fontsize=rp.FS["tick"], multialignment="left", linespacing=1.3)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Reference coherence $\\rho_t$ (log scale)")
    ax.grid(True, axis="x", color=rp.GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=0.26, right=0.985, top=0.97, bottom=0.20)
    rp.save(fig, "Fig5_ws_reference_coherence", "single", OUT_DIR)
    print(f"    Fig5: unterhalb der Basislinie P1 {info[1]['below']}, P2 {info[2]['below']}")


def fmt2(v: float, lead: bool = True) -> str:
    """Zwei Nachkommastellen, echtes Minuszeichen, KEINE negative Null (-0.00 ist ein Rundungsartefakt)."""
    t = f"{v:.2f}"
    if t in ("-0.00", "0.00", "-0.0", "0.0"):
        t = "0.00"
    if not lead:
        t = t.replace("0.", ".").replace("-.", "$-$.") if t.startswith(("0.", "-0.")) else t
    return t.replace("-", "$-$") if not t.startswith("$") else t


def _indicator_groups():
    return [(d, INDICATOR_DIMENSIONS[d]) for d in DIM_NAMES]


def _loading_panel(ax, ph: int, k: int, show_ylabels: bool = True):
    d = PHASES[ph]["dir"]
    pat = pd.read_csv(d / f"efa_pattern_{k}f.csv", index_col=0)
    order = [i for _, inds in _indicator_groups() for i in inds if i in pat.index]
    pat = pat.reindex(order)
    # Skala bis +-1.0: die Ladungen reichen bis 1.00 / -0.99, bei +-0.8 sah 0.81 aus wie 1.00.
    im = ax.imshow(pat.values, aspect="auto", cmap=rp.CMAP_DIV, vmin=-1.0, vmax=1.0)
    for i in range(pat.shape[0]):
        for j in range(pat.shape[1]):
            v = pat.values[i, j]
            ax.text(j, i, fmt2(v), ha="center", va="center", fontsize=rp.FS["tick"],
                    color=rp.ink(rp.CMAP_DIV((v + 1.0) / 2.0)))
    ax.set_xticks(range(pat.shape[1])); ax.set_xticklabels(pat.columns)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o.replace("_", " ") for o in order] if show_ylabels else [""] * len(order))
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cum = 0
    for dim, inds in _indicator_groups():
        present = [i for i in inds if i in order]
        if cum > 0:
            ax.axhline(cum - 0.5, color=rp.TEXT, lw=0.6)
        if show_ylabels:
            ax.text(-0.62, cum + len(present) / 2 - 0.5, rp.DIM_EN[dim][1], transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=rp.FS["label"], fontweight="bold", color=rp.TEXT)
        cum += len(present)
    return im


def fig6_efa_loadings() -> None:
    fig, axes = rp.figure("double", height_mm=105, ncols=2, gridspec_kw={"width_ratios": [5, 4]})
    im = _loading_panel(axes[0], 1, 5, True)
    _loading_panel(axes[1], 2, 4, False)
    axes[0].set_xlabel(f"{rp.PHASE_LABEL[1]}: five factors")
    axes[1].set_xlabel(f"{rp.PHASE_LABEL[2]}: four factors")
    rp.panel(axes[0], "A", x=-0.55); rp.panel(axes[1], "B", x=-0.06)   # je linke Kante des eigenen Blocks (A hat die Zeilenbeschriftung davor)
    fig.subplots_adjust(left=0.26, right=0.855, top=0.925, bottom=0.08, wspace=0.06)
    cax = fig.add_axes([0.888, 0.20, 0.013, 0.60])   # Platz fuer die gedrehte Beschriftung rechts
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Pattern loading"); cb.ax.tick_params(labelsize=rp.FS["tick"], width=0.5)
    cb.outline.set_linewidth(0.5)
    rp.save(fig, "Fig6_efa_pattern_loadings", "double", OUT_DIR)


# ================================================================ ANHANG
def figA1_temporal_evolution(data: dict) -> None:
    fig, axes = rp.figure("double", height_mm=110, nrows=2, ncols=2)
    order = CLS   # gleiche Reihenfolge wie in allen anderen Abbildungen (rp.CLASS_ORDER)
    letters = iter("ABCD")
    for r, ph in enumerate((1, 2)):
        p = PHASES[ph]["dir"] / "topic_assignments.csv"
        if not p.exists():
            plt.close(fig); print("  FigA1 uebersprungen: topic_assignments.csv nicht vorhanden"); return
        classified = data[ph]["classified"]
        assign = pd.read_csv(p, usecols=["Year", "topic"])
        assign = assign[assign["topic"] >= 0]
        assign["signal_type"] = assign["topic"].map(classified["signal_type"])
        counts = assign.groupby(["Year", "signal_type"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
        props = counts.div(counts.sum(axis=1), axis=0)
        ax1, ax2 = axes[r]
        base = pd.Series(0.0, index=props.index)
        for st in order:
            new = base + props[st]
            ax1.fill_between(props.index, base, new, color=rp.CLASS_COLOR[st], alpha=0.85, lw=0,
                             hatch=rp.CLASS_HATCH[st], edgecolor="white")
            base = new
        ax1.set_xlim(props.index.min(), props.index.max()); ax1.set_ylim(0, 1)
        ax1.set_ylabel("Share of publications")
        ax1.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        bottom = np.zeros(len(counts))
        for st in order:
            ax2.bar(counts.index, counts[st], bottom=bottom, color=rp.CLASS_COLOR[st], width=0.8, lw=0,
                    hatch=rp.CLASS_HATCH[st], edgecolor="white")
            bottom = bottom + counts[st].values
        ax2.set_ylabel("Number of publications")
        ax2.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        for ax in (ax1, ax2):
            ax.set_xlabel("Year" if r == 1 else "")
            rp.grid(ax)
            # Freistellung statt schwarzem Kasten: Schwarz ist fuer Phase 1 reserviert und
            # stand vorher auch im Phase-2-Panel; ausserdem verdeckte der Kasten Daten.
            ax.text(0.02, 0.96, rp.PHASE_LABEL[ph], transform=ax.transAxes, ha="left", va="top",
                    fontsize=rp.FS["annot"], color=rp.TEXT, zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
            rp.panel(ax, next(letters))
    fig.legend(handles=rp.class_handles(order, kind="patch"), loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.15, hspace=0.35, wspace=0.22)
    rp.save(fig, "FigA1_temporal_evolution", "double", OUT_DIR, note="nutzt topic_assignments.csv (nur lokal)")


def figA2_extended_tem(data: dict) -> None:
    fig, axes = rp.figure("double", height_mm=80, ncols=2, sharey=True)
    # Gemeinsame x-Grenzen: die y-Achse ist geteilt, das Layout legt also einen direkten
    # Vergleich nahe. Vorher hatte Panel A 572 px pro Dekade, Panel B 505.
    xs = []
    for ph in (1, 2):
        t = pd.read_csv(PHASES[ph]["dir"] / "tem_metrics.csv")["avg_proportion"]
        xs += [float(t.min()), float(t.max())]
    xlim = (min(xs) * 0.75, max(xs) * 1.35)
    for ax, ph, letter in zip(axes, (1, 2), "AB"):
        df = data[ph]["classified"]
        tem = pd.read_csv(PHASES[ph]["dir"] / "tem_metrics.csv").set_index("topic")
        mg = tem.join(df[["signal_type", "Epistemische Offenheit", "margin"]], how="inner")
        ax.axhline(0, color=rp.REF_LINE, ls="--", lw=0.6, zorder=1)
        ax.axvline(mg["avg_proportion"].median(), color=rp.REF_LINE, ls="--", lw=0.6, zorder=1)
        for c in CLS:
            sub = mg[mg["signal_type"] == c]
            clear = sub["margin"] >= 0.10   # Groessenkodierung (epistemische Offenheit) entfernt:
            #  drei Kanaele (Form, Fuellung, Groesse) waren eine zu viel; EO steht in FigA6.
            ax.scatter(sub["avg_proportion"][clear], sub["growth_rate"][clear], s=15, marker=rp.CLASS_MARKER[c],
                       facecolor=rp.CLASS_COLOR[c], edgecolor=rp.CLASS_EDGE[c], linewidths=0.4, alpha=0.9, zorder=3)
            ax.scatter(sub["avg_proportion"][~clear], sub["growth_rate"][~clear], s=15, marker=rp.CLASS_MARKER[c],
                       facecolor="white", edgecolor=rp.CLASS_COLOR[c], linewidths=0.7, alpha=0.9, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        # Dezimalticks statt 10^-3: die Exponenten wurden mit 4.9 pt gesetzt, klar unter der
        # 7-pt-Grenze, und tragen gerade die Groessenordnung.
        ax.xaxis.set_major_locator(FixedLocator([0.001, 0.002, 0.005, 0.01, 0.02, 0.05]))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticklabels(["0.001", "0.002", "0.005", "0.01", "0.02", "0.05"])
        ax.set_xlabel("Average topic proportion $\\bar{p}$ (log scale)")
        ax.text(0.98, 0.97, f"{rp.PHASE_LABEL[ph]}\n$n$ = {len(mg)} topics", transform=ax.transAxes, ha="right", va="top",
                fontsize=rp.FS["annot"], color=rp.TEXT)
        rp.grid(ax, "both")
        rp.panel(ax, letter)
    axes[0].set_ylabel("Annualised growth rate $g$")
    from matplotlib.lines import Line2D
    extra = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="#555555", markersize=5,
                    label="filled: $\\Delta$ $\\geq$ 0.10"),
             Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", markersize=5,
                    label="open: $\\Delta$ < 0.10")]
    fig.legend(handles=rp.class_handles(kind="marker") + extra, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.0),
               columnspacing=0.9, handletextpad=0.35)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.27, wspace=0.08)
    rp.save(fig, "FigA2_extended_tem", "double", OUT_DIR)


def figA3_scree() -> None:
    fig, axes = rp.figure("double", height_mm=58, ncols=2, sharey=True)
    for ax, ph, letter in zip(axes, (1, 2), "AB"):
        s = json.loads((PHASES[ph]["dir"] / "efa_summary.json").read_text())
        ev = np.array(s["eigenvalues"], dtype=float); pa = np.array(s["pa_thresholds_mean"], dtype=float)
        n_keep = int(s["n_parallel"]); x = np.arange(1, len(ev) + 1)
        # Balken markieren nur "behalten/nicht behalten"; neutrales Grau, damit die Blau-Rampe
        # nicht mit der Klassenfarbe "Emerging concept" verwechselt wird.
        ax.bar(x[:n_keep], ev[:n_keep], width=0.6, color="#e2e2e2", zorder=1, label="Retained factors")
        ax.plot(x, ev, "o-", color=rp.TEXT, lw=1.0, markersize=3, label="Observed eigenvalues", zorder=3)
        # frueher blaues Quadrat gestrichelt: exakt die Kodierung von "Emerging concept".
        ax.plot(x[:len(pa)], pa, color=rp.REF_LINE, ls="--", marker="x", lw=0.9, markersize=3.2,
                markeredgewidth=0.8, label="Parallel analysis (mean)", zorder=2)
        ax.axhline(1.0, color=rp.REF_LINE, ls=":", lw=0.7, label="Kaiser criterion ($\\lambda$ = 1)", zorder=1)
        ax.set_xticks(x); ax.set_xlabel("Factor number")
        ax.text(0.98, 0.97, rp.PHASE_LABEL[ph], transform=ax.transAxes, ha="right", va="top", fontsize=rp.FS["annot"])
        rp.grid(ax); rp.panel(ax, letter)
    axes[0].set_ylabel("Eigenvalue")
    axes[0].legend(loc="upper right", bbox_to_anchor=(0.98, 0.88))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.20, wspace=0.08)
    rp.save(fig, "FigA3_scree_parallel_analysis", "double", OUT_DIR)


def figA4_factor_correlations() -> None:
    fig, axes = rp.figure("double", height_mm=62, ncols=2, gridspec_kw={"width_ratios": [4, 3]})
    # EINE Skala fuer beide Panels: es gibt nur eine Farbleiste, panelweise Grenzen waeren falsch.
    lim = 0.1
    for ph, k in ((1, 5), (2, 4)):
        v = pd.read_csv(PHASES[ph]["dir"] / f"efa_phi_{k}f.csv", index_col=0).values.astype(float)
        off = v[~np.triu(np.ones_like(v, dtype=bool))]
        lim = max(lim, float(np.nanmax(np.abs(off))))
    lim = float(np.ceil(lim * 10) / 10)
    for ax, ph, k, letter in zip(axes, (1, 2), (5, 4), "AB"):
        phi = pd.read_csv(PHASES[ph]["dir"] / f"efa_phi_{k}f.csv", index_col=0)
        # Unteres Dreieck ohne Diagonale, wie FigA5: die obere Haelfte druckte jeden Wert doppelt,
        # und die Diagonale (1.00) spannte allein die Skala auf, sodass alle echten Werte weiss blieben.
        vals = np.array(phi.values, dtype=float)
        mask = np.triu(np.ones_like(vals, dtype=bool))
        shown = np.where(mask, np.nan, vals)
        im = ax.imshow(np.ma.masked_invalid(shown), cmap=rp.CMAP_DIV, vmin=-lim, vmax=lim)
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                if mask[i, j]:
                    continue
                v = vals[i, j]
                ax.text(j, i, fmt2(v), ha="center", va="center", fontsize=rp.FS["tick"],
                        color=rp.ink(rp.CMAP_DIV((v / lim + 1.0) / 2.0)))
        # letzte Spalte und erste Zeile bleiben im unteren Dreieck leer, deshalb ohne Beschriftung
        ax.set_xticks(range(phi.shape[1] - 1)); ax.set_xticklabels(list(phi.columns)[:-1])
        ax.set_yticks(range(1, phi.shape[0])); ax.set_yticklabels(list(phi.index)[1:])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel(f"{rp.PHASE_LABEL[ph]}, {k} factors")
        rp.panel(ax, letter, x=-0.10)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03, aspect=25)
    cb.set_label("Factor correlation $\\varphi$"); cb.ax.tick_params(labelsize=rp.FS["tick"], width=0.5); cb.outline.set_linewidth(0.5)
    fig.subplots_adjust(left=0.06, right=0.88, top=0.90, bottom=0.16, wspace=0.25)
    rp.save(fig, "FigA4_factor_correlations", "double", OUT_DIR)


def figA5_indicator_correlations(ph: int) -> None:
    ind = pd.read_csv(PHASES[ph]["dir"] / "indicators_16.csv", index_col=0)
    order = [i for _, inds in _indicator_groups() for i in inds if i in ind.columns]
    corr = ind[order].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = rp.figure("double", height_mm=152)
    im = ax.imshow(np.ma.masked_array(corr.values, mask=mask), cmap=rp.CMAP_DIV, vmin=-1, vmax=1)
    for i in range(len(order)):
        for j in range(i + 1):
            v = corr.values[i, j]
            ax.text(j, i, fmt2(v, lead=False), ha="center", va="center",
                    fontsize=rp.FS["cell"], color=rp.ink(rp.CMAP_DIV((v + 1.0) / 2.0)))
    labs = [o.replace("_", " ") for o in order]
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labs, rotation=90)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labs)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cum = 0
    for dim, inds in _indicator_groups():
        n = len([i for i in inds if i in order])
        if cum > 0:
            ax.plot([-0.5, cum - 0.5], [cum - 0.5, cum - 0.5], color=rp.TEXT, lw=0.5)
            ax.plot([cum - 0.5, cum - 0.5], [cum - 0.5, len(order) - 0.5], color=rp.TEXT, lw=0.5)
        ax.text(len(order) - 0.3, cum + n / 2 - 0.5, rp.DIM_EN[dim][1], ha="left", va="center", fontsize=rp.FS["label"],
                fontweight="bold", color=rp.TEXT)
        cum += n
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.10, aspect=30)
    cb.set_label("Pearson $r$"); cb.ax.tick_params(labelsize=rp.FS["tick"], width=0.5); cb.outline.set_linewidth(0.5)
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.set_ticklabels(["$-$1", "$-$.5", "0", ".5", "1"])   # gleiche Schreibweise wie die Zellen (ohne fuehrende Null)
    fig.subplots_adjust(left=0.185, right=0.90, top=0.985, bottom=0.235)
    rp.save(fig, f"FigA5{'ab'[ph - 1]}_indicator_correlations_phase{ph}", "double", OUT_DIR,
            note="Zellwerte 6.5 pt ohne fuehrende Null; Zahlen zusaetzlich als CSV im Supplement")


def figA6_dimension_heatmap(data: dict) -> None:
    fig, axes = rp.figure("double", height_mm=86, ncols=2, gridspec_kw={"width_ratios": [146, 265]})
    order = CLS   # gleiche Reihenfolge wie ueberall (rp.CLASS_ORDER)
    for ax, ph, letter in zip(axes, (1, 2), "AB"):
        df = data[ph]["classified"].copy()
        df["o"] = df["signal_type"].map({c: i for i, c in enumerate(order)})
        df = df.sort_values(["o", "ws_distance"])
        im = ax.imshow(df[DIM_NAMES].values.T, aspect="auto", cmap=rp.CMAP_DIV, vmin=-2, vmax=2, interpolation="nearest")
        # Panel A nennt Name UND Kurzcode, damit das Kuerzelsystem der uebrigen Abbildungen
        # (Fig6, FigA5, FigA8) mindestens einmal im Abbildungssatz aufgeloest wird.
        ax.set_yticks(range(len(DIM_NAMES)))
        ax.set_yticklabels([f"{rp.DIM_EN[d][0]} ({rp.DIM_EN[d][1]})" for d in DIM_NAMES] if ph == 1
                           else rp.DIM_EN_CODES)
        ax.set_xticks([])
        start = 0
        for c in order:
            n = int((df["signal_type"] == c).sum())
            if n == 0:
                continue
            if start > 0:
                ax.axvline(start - 0.5, color=rp.TEXT, lw=0.8)
            ax.text(start + n / 2 - 0.5, -0.65, f"{rp.CLASS_EN[c].replace(' ', chr(10))}\n$n$ = {n}",
                    ha="center", va="bottom", fontsize=rp.FS["small"], color=rp.TEXT, linespacing=1.25)
            start += n
        ax.set_xlabel(f"{rp.PHASE_LABEL[ph]}: topics sorted by\nconfiguration and weak-signal distance")
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(length=0)
        rp.panel(ax, letter, y=1.34)   # ueber den Gruppenbeschriftungen, sonst stoesst (B) an "Latent/mixed" von Panel A
    fig.subplots_adjust(left=0.20, right=0.985, top=0.775, bottom=0.32, wspace=0.075)
    cax = fig.add_axes([0.35, 0.12, 0.40, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Dimension score (z-standardised)"); cb.ax.tick_params(labelsize=rp.FS["tick"], width=0.5); cb.outline.set_linewidth(0.5)
    rp.save(fig, "FigA6_dimension_heatmap", "double", OUT_DIR, note="ohne Topic-Beschriftung; Werte im Supplement-CSV")


def figA7_membership_heatmap(data: dict) -> None:
    fig, axes = rp.figure("double", height_mm=72, ncols=2, gridspec_kw={"width_ratios": [146, 265]})
    for ax, ph, letter in zip(axes, (1, 2), "AB"):
        df = data[ph]["classified"].sort_values("margin", ascending=False)
        cols = ["m_ws", "m_ec", "m_trend", "m_latent"]
        im = ax.imshow(df[cols].values.T, aspect="auto", cmap=rp.CMAP_SEQ, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(4)); ax.set_yticklabels([rp.CLASS_EN[MEMB_LABEL[c]] for c in cols] if ph == 1 else ["", "", "", ""])
        n = len(df); n05 = int((df["margin"] < 0.05).sum()); n10 = int((df["margin"] < 0.10).sum())
        for k, lab in ((n - n10, "$\\Delta$ = 0.10"), (n - n05, "$\\Delta$ = 0.05")):
            ax.axvline(k - 0.5, color=rp.TEXT, lw=0.7, ls="--")
            ax.text(k - 0.5, -0.7, lab, ha="center", va="bottom", fontsize=rp.FS["small"], color=rp.TEXT)
        ax.set_xticks([0, n - 1]); ax.set_xticklabels(["1", str(n)])
        ax.set_xlabel(f"{rp.PHASE_LABEL[ph]}: topics ranked by margin (high to low)")
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(length=0)
        rp.panel(ax, letter, y=1.12, x=-0.012)
    fig.subplots_adjust(left=0.12, right=0.978, top=0.86, bottom=0.38, wspace=0.075)
    cax = fig.add_axes([0.35, 0.16, 0.40, 0.04])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("Membership value"); cb.ax.tick_params(labelsize=rp.FS["tick"], width=0.5); cb.outline.set_linewidth(0.5)
    rp.save(fig, "FigA7_membership_heatmap", "double", OUT_DIR)


def figA8_ws_detail_radars(data: dict, ph: int, n_top: int = 6) -> None:
    df, kw = data[ph]["classified"], data[ph]["kw"]
    ws = df[df["signal_type"] == "Weak Signal"].sort_values("ws_distance")
    top = ws.head(n_top)
    fig, axes = rp.figure("double", height_mm=118, nrows=2, ncols=3, polar=True)
    axes = axes.flatten()
    med = ws[DIM_NAMES].median().tolist()
    # Feste, fuer beide Phasen gleiche Skala: FigA8a und FigA8b werden als Paar gelesen.
    rmax, rmin = RADAR_A8_RANGE
    assert top[DIM_NAMES].values.max() <= rmax and top[DIM_NAMES].values.min() >= rmin, "FigA8: Wert ausserhalb der festen Skala"
    for i, (idx, row) in enumerate(top.iterrows()):
        ax = axes[i]
        ang = radar_axes(ax, rp.DIM_EN_CODES, rmin=rmin, rmax=rmax, label_pad=1, codes=True)
        a = closed(ang)
        ax.plot(a, closed(med), color=rp.REF_LINE, ls="--", lw=0.7, zorder=2)
        ax.plot(a, closed(row[DIM_NAMES].tolist()), color=rp.CLASS_COLOR["Weak Signal"], lw=1.0, marker="o", markersize=2.5,
                markeredgecolor=rp.CLASS_EDGE["Weak Signal"], markeredgewidth=0.4, zorder=3)
        ax.fill(a, closed(row[DIM_NAMES].tolist()), color=rp.CLASS_COLOR["Weak Signal"], alpha=0.10, zorder=2)
        rp.radial_ticks(ax, [-1.5, 0.0, 1.5, 3.0], ["$-$1.5", "0.0", "1.5", "3.0"], angle_deg=36)
        kws = ", ".join(w for w, _ in kw.get(idx, [])[:2])
        other = max(row["m_trend"], row["m_ec"], row["m_latent"])
        # pad 6 war zu wenig: die EO-Beschriftung am oberen Rand des Radars lag auf der
        # zweiten Titelzeile und hat den m_ws-Wert ueberdeckt.
        ax.set_title(f"T{idx}: {kws[:30]}\n$m_{{\\mathrm{{ws}}}}$ = {row['m_ws']:.2f}, $\\Delta$ = {row['m_ws'] - other:.2f}",
                     fontsize=rp.FS["annot"], pad=17, color=rp.TEXT, linespacing=1.3)
        rp.panel(ax, "ABCDEF"[i], polar=True)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], color=rp.CLASS_COLOR["Weak Signal"], marker="o", markersize=3, lw=1.0, label="Topic profile"),
                        Line2D([0], [0], color=rp.REF_LINE, ls="--", lw=0.7, label="Weak-signal median of the phase")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.84, bottom=0.10, hspace=0.72, wspace=0.35)
    rp.save(fig, f"FigA8{'ab'[ph - 1]}_ws_detail_radars_phase{ph}", "double", OUT_DIR)


def figA9_structure_compare(data: dict, q: float = 0.90) -> None:
    labels = RADAR_LABELS_NARROW
    fig, ax = rp.figure("single", height_mm=90, polar=True)
    ang = radar_axes(ax, labels, rmin=-0.4, rmax=1.0, label_pad=4)
    a = closed(ang)

    def prof(df, fn):
        return closed([fn(df[d].dropna()) for d in DIM_NAMES])
    # Phase 2 breit und zuerst, Phase 1 schmal darueber: an drei von fuenf Ecken liegen die
    # Werte fast aufeinander, vorher war Phase 1 dort vollstaendig verdeckt.
    for ph, lw_p, ms in ((2, 2.2, 5.0), (1, 1.0, 2.8)):
        df = data[ph]["classified"]
        ax.plot(a, prof(df, lambda s: s.quantile(q)), color=rp.PHASE_COLOR[ph], ls="-", lw=lw_p,
                marker=rp.PHASE_MARKER[ph], markersize=ms, zorder=3 + (ph == 1))
        ax.plot(a, prof(df, lambda s: s.median()), color=rp.PHASE_COLOR[ph], ls=":", lw=max(0.9, lw_p - 0.9),
                marker=rp.PHASE_MARKER[ph], markersize=max(2.2, ms - 1.6), markerfacecolor="white",
                zorder=3 + (ph == 1))
    rp.radial_ticks(ax, [-0.4, 0.0, 0.4, 0.8], ["$-$0.4", "0.0", "0.4", "0.8"], angle_deg=100)
    # Legende in zwei Bloecken: Phase (Farbe und Marker) und Statistik (Linientyp). Vorher stand
    # die Phasenbezeichnung viermal in voller Laenge.
    from matplotlib.lines import Line2D
    hs = [Line2D([0], [0], color=rp.PHASE_COLOR[1], lw=1.0, marker="o", markersize=2.8,
                 label=f"{rp.PHASE_LABEL[1]}, $n$ = 146"),
          Line2D([0], [0], color=rp.PHASE_COLOR[2], lw=2.2, marker="s", markersize=5.0,
                 label=f"{rp.PHASE_LABEL[2]}, $n$ = 265"),
          Line2D([0], [0], color=rp.TEXT_MUTED, lw=1.2, ls="-", label="90th percentile"),
          Line2D([0], [0], color=rp.TEXT_MUTED, lw=1.0, ls=":", marker="o", markersize=2.4,
                 markerfacecolor="white", label="Median")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, columnspacing=1.0)
    fig.subplots_adjust(left=0.22, right=0.78, top=0.95, bottom=0.24)
    rp.save(fig, "FigA9_structure_compare_phases", "single", OUT_DIR)


def figA10_topic_quality() -> None:
    p1 = pd.read_csv(P1_DIR / "topic_quality_per_topic.csv"); p2 = pd.read_csv(P2_DIR / "topic_quality_per_topic.csv")
    specs = [("c_v", "Coherence per topic, $C_{\\mathrm{v}}$ (top-10)"),
             ("c_npmi", "Coherence per topic, $C_{\\mathrm{NPMI}}$ (top-10)")]
    fig, axes = rp.figure("double", height_mm=58, ncols=2)
    for ax, (col, lab), letter in zip(axes, specs, "AB"):
        dat = [p1[col].dropna().values, p2[col].dropna().values]
        ax.boxplot(dat, tick_labels=[f"Phase 1\n(2000 to 2015)\n$n$ = {len(p1)}",
                                     f"Phase 2\n(2016 to 2025)\n$n$ = {len(p2)}"], widths=0.45,
                   patch_artist=True, showmeans=True,
                   medianprops=dict(color=rp.TEXT, lw=1.0),
                   meanprops=dict(marker="D", markerfacecolor=rp.TEXT_MUTED, markeredgecolor=rp.TEXT_MUTED, markersize=3.5),
                   flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="#888888", markersize=2.5, markeredgewidth=0.5),
                   boxprops=dict(facecolor="white", edgecolor=rp.TEXT, lw=0.8), whiskerprops=dict(color=rp.TEXT, lw=0.7),
                   capprops=dict(color=rp.TEXT, lw=0.7))
        ax.set_ylabel(lab)
        if col == "c_npmi":       # nur hier liegt die Null im Wertebereich; in Panel A war sie eine zweite Grundlinie
            ax.axhline(0, color=rp.REF_LINE, ls="--", lw=0.5)
        rp.grid(ax); rp.panel(ax, letter)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.26, wspace=0.30)
    rp.save(fig, "FigA10_topic_quality_boxplots", "double", OUT_DIR)


def figA11_perturbation_by_class() -> None:
    fig, axes = rp.figure("double", height_mm=58, ncols=2, sharey=True)
    classes = ["<0.05", "0.05-0.10", ">=0.10"]; labs = ["$\\Delta$ < 0.05", "0.05 $\\leq$ $\\Delta$ < 0.10", "$\\Delta$ $\\geq$ 0.10"]
    for ax, ph, letter in zip(axes, (1, 2), "AB"):
        _, sp = pert_paths(ph)
        if sp is None:
            plt.close(fig); print("  FigA11 uebersprungen: perturbation_summary.csv nicht gefunden"); return
        summ = pd.read_csv(sp)
        x = np.arange(3); w = 0.26
        for i, s in enumerate(NOISE):
            row = summ[np.isclose(summ.noise_sd_share, s)].iloc[0]
            vals = [100 * row[f"flip_rate_{c}"] for c in classes]
            bars = ax.bar(x + (i - 1) * w, vals, width=w - 0.03, color=NOISE_GREY[s], edgecolor=rp.TEXT, linewidth=0.4,
                          hatch=NOISE_HATCH[s], label=f"Noise = {int(s * 100)} % of SD")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.6, "0" if v == 0 else ("<1" if v < 1 else f"{v:.0f}"),
                        ha="center", va="bottom", fontsize=rp.FS["small"], color=rp.TEXT)
        ax.set_xticks(x); ax.set_xticklabels([f"{l}\n($n$ = {int(summ.iloc[0][f'n_{c}'])})" for c, l in zip(classes, labs)])
        ax.set_ylim(0, 45)
        ax.text(0.98, 0.97, rp.PHASE_LABEL[ph], transform=ax.transAxes, ha="right", va="top", fontsize=rp.FS["annot"])
        rp.grid(ax); rp.panel(ax, letter)
    axes[0].set_ylabel("Flip rate of the dominant\nconfiguration (%)")
    axes[0].legend(loc="upper right", bbox_to_anchor=(0.98, 0.88))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.24, wspace=0.08)
    rp.save(fig, "FigA11_perturbation_flip_by_margin_class", "double", OUT_DIR)


# ================================================================ PARKPLATZ (Option B)
def load_cross(data: dict) -> dict:
    matches = pd.read_csv(CROSS_DIR / "topic_matches_mutual.csv")
    return {"p1": data[1], "p2": data[2], "matches": matches}


def _ribbon(x0, y0t, y0b, x1, y1t, y1b, c=0.5):
    cx0 = x0 + c * (x1 - x0); cx1 = x1 - c * (x1 - x0)
    verts = [(x0, y0t), (cx0, y0t), (cx1, y1t), (x1, y1t), (x1, y1b), (cx1, y1b), (cx0, y0b), (x0, y0b), (x0, y0t)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    return MPath(verts, codes)


def figP1_migration_sankey(cross: dict) -> None:
    m = cross["matches"].copy()
    c1, c2 = cross["p1"]["classified"], cross["p2"]["classified"]
    m["p1_class"] = m["phase1_topic"].map(c1["signal_type"]); m["p2_class"] = m["phase2_topic"].map(c2["signal_type"])
    m = m.dropna(subset=["p1_class", "p2_class"])
    cell = m.groupby(["p1_class", "p2_class"]).size().reset_index(name="count")
    idx = {c: i for i, c in enumerate(CLS)}
    cell["i"] = cell["p1_class"].map(idx); cell["j"] = cell["p2_class"].map(idx)
    t1 = m.groupby("p1_class").size().reindex(CLS, fill_value=0); t2 = m.groupby("p2_class").size().reindex(CLS, fill_value=0)
    total = max(int(t1.sum()), int(t2.sum()), 1)
    gap = 0.035; usable = 1.0 - gap * (len(CLS) - 1); unit = usable / total

    def stack(tot):
        out, cur = [], 1.0
        for h in (tot / total).values * usable:
            out.append((cur, cur - h)); cur -= h + gap
        return out
    s1, s2 = stack(t1), stack(t2)
    src, cur = {}, [t for t, _ in s1]
    for _, r in cell.sort_values(["i", "j"]).iterrows():
        h = r["count"] * unit; src[(r["i"], r["j"])] = (cur[r["i"]], cur[r["i"]] - h); cur[r["i"]] -= h
    tgt, cur = {}, [t for t, _ in s2]
    for _, r in cell.sort_values(["j", "i"]).iterrows():
        h = r["count"] * unit; tgt[(r["i"], r["j"])] = (cur[r["j"]], cur[r["j"]] - h); cur[r["j"]] -= h
    fig, ax = rp.figure("double", height_mm=92)
    xl, xr, bw = 0.17, 0.83, 0.025
    for _, r in cell.sort_values("count", ascending=False).iterrows():
        k = (r["i"], r["j"]); y0t, y0b = src[k]; y1t, y1b = tgt[k]
        # Schraffur je Quellklasse: ohne sie war die Farbe alleiniger Traeger und die
        # Graustufenfassung nicht mehr auflesbar (alle Baender lagen zwischen Grauwert 140 und 200).
        ax.add_patch(mpatches.PathPatch(_ribbon(xl, y0t, y0b, xr, y1t, y1b), facecolor=rp.CLASS_COLOR[r["p1_class"]],
                                        alpha=0.55, edgecolor="white", linewidth=0.4,
                                        hatch=rp.CLASS_HATCH[r["p1_class"]], zorder=2))
    for _, r in cell.iterrows():
        if r["count"] >= 3:
            k = (r["i"], r["j"])
            ax.text(xl + 0.008, sum(src[k]) / 2, str(int(r["count"])), ha="left", va="center", fontsize=rp.FS["small"], color=rp.TEXT, zorder=4)
            ax.text(xr - 0.008, sum(tgt[k]) / 2, str(int(r["count"])), ha="right", va="center", fontsize=rp.FS["small"], color=rp.TEXT, zorder=4)
    for st, tot, x, ha, dx in ((s1, t1, xl - bw, "right", -0.010), (s2, t2, xr, "left", bw + 0.010)):
        for k, (top, bot) in enumerate(st):
            ax.add_patch(mpatches.Rectangle((x, bot), bw, top - bot, facecolor=rp.CLASS_COLOR[CLS[k]], edgecolor=rp.CLASS_EDGE[CLS[k]],
                                            linewidth=0.4, zorder=3))
            ax.text(x + dx, (top + bot) / 2, f"{rp.CLASS_EN[CLS[k]]}\n$n$ = {int(tot.iloc[k])}", ha=ha, va="center",
                    fontsize=rp.FS["annot"], color=rp.TEXT, linespacing=1.2)
    ax.text(xl - bw / 2, 1.03, rp.PHASE_LABEL[1], ha="center", va="bottom", fontsize=rp.FS["label"], fontweight="bold")
    ax.text(xr + bw / 2, 1.03, rp.PHASE_LABEL[2], ha="center", va="bottom", fontsize=rp.FS["label"], fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(-0.01, 1.10); ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    rp.save(fig, "FigP1_migration_sankey", "double", OUT_DIR, note=f"{len(m)} mutual-best pairs")


def figP2_membership_shift(cross: dict) -> None:
    m = cross["matches"].copy(); c1, c2 = cross["p1"]["classified"], cross["p2"]["classified"]
    m["p1_class"] = m["phase1_topic"].map(c1["signal_type"]); m["p2_class"] = m["phase2_topic"].map(c2["signal_type"])
    m["delta"] = m["phase2_topic"].map(c2["m_ws"]) - m["phase1_topic"].map(c1["m_ws"])
    m = m.dropna(subset=["p1_class", "p2_class", "delta"])
    mean = np.full((4, 4), np.nan); n = np.zeros((4, 4), dtype=int)
    for i, a in enumerate(CLS):
        for j, b in enumerate(CLS):
            sub = m[(m["p1_class"] == a) & (m["p2_class"] == b)]; n[i, j] = len(sub)
            if len(sub):
                mean[i, j] = sub["delta"].mean()
    robust = np.abs(mean[(n >= 3) & np.isfinite(mean)]); vmax = max(float(robust.max()) if robust.size else 0.1, 0.1)
    fig, ax = rp.figure("single", height_mm=78)
    norm = plt.Normalize(-vmax, vmax)
    for i in range(4):
        for j in range(4):
            if n[i, j] == 0:
                fc, tc, txt = "white", "#bbbbbb", ""
            elif n[i, j] < 3:
                # frueher flach grau: die beiden groessten Positivwerte sahen dadurch aus wie Null.
                # Jetzt normale Farbe plus Schraffur als Vorbehalt "zu wenige Paare".
                fc = rp.CMAP_DIV(norm(np.clip(mean[i, j], -vmax, vmax))); tc = rp.TEXT
                txt = f"{mean[i, j]:+.2f}\n($n$ = {n[i, j]})"
            else:
                fc = rp.CMAP_DIV(norm(mean[i, j])); tc = "white" if abs(mean[i, j]) > 0.55 * vmax else rp.TEXT
                txt = f"{mean[i, j]:+.2f}\n($n$ = {n[i, j]})"
            hat = "///" if 0 < n[i, j] < 3 else ""
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=fc, edgecolor="white", linewidth=1.2,
                                       hatch=hat))
            if txt:
                bb = dict(facecolor="white", edgecolor="none", pad=0.8) if hat else None
                ax.text(j, i, txt.replace("-", "$-$"), ha="center", va="center", fontsize=rp.FS["tick"],
                        color=tc, bbox=bb, zorder=4)
            elif n[i, j] == 0:
                ax.text(j, i, "$n$ = 0", ha="center", va="center", fontsize=rp.FS["tick"], color="#aaaaaa", zorder=4)
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5)
    ax.set_xticks(range(4)); ax.set_xticklabels([rp.CLASS_EN[c].replace(" ", "\n") for c in CLS])
    ax.set_yticks(range(4)); ax.set_yticklabels([rp.CLASS_EN[c].replace(" ", "\n") for c in CLS])
    ax.set_xlabel("Phase 2 configuration"); ax.set_ylabel("Phase 1 configuration")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=rp.CMAP_DIV), ax=ax, fraction=0.045, pad=0.04)
    cb.set_label("Mean $\\Delta m_{\\mathrm{ws}}$ (Phase 2 $-$ Phase 1)")
    cb.ax.tick_params(labelsize=rp.FS["tick"], width=0.5); cb.outline.set_linewidth(0.5)
    cb.set_ticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    cb.set_ticklabels(["$-$0.30", "$-$0.20", "$-$0.10", "0.00", "+0.10", "+0.20", "+0.30"])  # Vorzeichen wie in den Zellen
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="white", edgecolor="#999999", hatch="///", linewidth=0.5,
                             label="fewer than three matched pairs")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.subplots_adjust(left=0.24, right=0.845, top=0.97, bottom=0.28)
    rp.save(fig, "FigP2_membership_shift_heatmap", "single", OUT_DIR)


def figP3_signature_scatter(cross: dict) -> None:
    m = cross["matches"].copy(); kw2 = cross["p2"]["kw"]
    x = m["cosine"].values; y = m["jaccard"].values
    x_lo, x_hi = np.quantile(x, [1 / 3, 2 / 3]); y_lo, y_hi = np.quantile(y, [1 / 3, 2 / 3])
    stable = (x >= x_hi) & (y >= y_hi); drift = (x >= x_hi) & (y < y_lo); ec_pre = x < x_lo
    other = ~(stable | drift | ec_pre)
    fig, ax = rp.figure("mid", height_mm=88)
    x_min, x_max = x.min() - 0.02, x.max() + 0.012; y_min, y_max = -0.03, y.max() + 0.06
    # Zonen in drei unterscheidbaren Grautoenen statt in Klassen-Pastelltoenen: die alten
    # Toene lagen im Graustufendruck 3 von 255 auseinander und liehen sich Klassenfarben aus.
    ax.add_patch(plt.Rectangle((x_hi, y_hi), x_max - x_hi, y_max - y_hi, facecolor="#e6e6e6", lw=0, zorder=0))
    ax.add_patch(plt.Rectangle((x_hi, y_min), x_max - x_hi, y_lo - y_min, facecolor="#f0f0f0", lw=0, zorder=0))
    ax.add_patch(plt.Rectangle((x_min, y_min), x_lo - x_min, y_max - y_min, facecolor="#f7f7f7", lw=0, zorder=0))
    for v in (x_lo, x_hi):
        ax.axvline(v, color=rp.REF_LINE, ls=":", lw=0.6, zorder=1)
    for v in (y_lo, y_hi):
        ax.axhline(v, color=rp.REF_LINE, ls=":", lw=0.6, zorder=1)
    ax.scatter(x[stable], y[stable], s=14, marker="s", facecolor=rp.TEXT, edgecolor=rp.TEXT, linewidths=0.4, zorder=3,
               label=f"Stable core and vocabulary ($n$ = {stable.sum()})")
    ax.scatter(x[ec_pre], y[ec_pre], s=14, marker="v", facecolor="white", edgecolor=rp.TEXT, linewidths=0.7, zorder=3,
               label=f"Semantically unstable ($n$ = {ec_pre.sum()})")
    ax.scatter(x[drift], y[drift], s=30, marker="*", facecolor=rp.TEXT, edgecolor=rp.TEXT, linewidths=0.4, zorder=4,
               label=f"Concept drift ($n$ = {drift.sum()})")
    ax.scatter(x[other], y[other], s=9, marker=".", facecolor="#9a9a9a", edgecolor="none", zorder=2,
               label=f"Outside signature zones ($n$ = {other.sum()})")   # Restklasse zuletzt, wie ueberall
    order = sorted(np.where(drift)[0], key=lambda i: (-y[i], -x[i]))
    # feste, paarweise verschiedene Versaetze: die vier Punkte liegen als zwei Paare fast
    # aufeinander, mit nur zwei Versaetzen ueberdeckten sich Ziffer 2 und 3.
    # Bezugslinie, weil die vier Punkte als zwei fast deckungsgleiche Paare liegen und
    # eine blosse Ziffer daneben nicht sagt, zu welchem Stern sie gehoert.
    offs = [(17, 12), (-20, 12), (17, -14), (-20, -14)]
    for rank, i in enumerate(order):
        ax.annotate(str(rank + 1), (x[i], y[i]), textcoords="offset points", xytext=offs[rank % 4],
                    fontsize=rp.FS["annot"], color=rp.TEXT, zorder=6, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=rp.TEXT_MUTED, shrinkA=2, shrinkB=3))
        t2 = int(m.iloc[i]["phase2_topic"]); print(f"    drift ({rank + 1}) P2#{t2}: {', '.join(w for w, _ in kw2.get(t2, [])[:2])}")
    zp = dict(fontsize=rp.FS["annot"], color=rp.TEXT_MUTED, zorder=2)
    ax.text(0.985, 0.975, "trend precondition", transform=ax.transAxes, ha="right", va="top", **zp)
    ax.text(0.985, 0.02, "concept drift", transform=ax.transAxes, ha="right", va="bottom", **zp)
    # „phase" ist im Paper fuer die beiden Korpusphasen belegt, deshalb hier ausgeschrieben.
    ax.text(0.015, 0.02, "emerging-concept precondition", transform=ax.transAxes, ha="left", va="bottom", **zp)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Semantic similarity $\\sigma_{\\mathrm{sem}}$ (SBERT centroid cosine)")
    ax.set_ylabel("Lexical similarity $\\sigma_{\\mathrm{lex}}$ (Jaccard, top-15 keywords)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.93))
    rp.grid(ax, "both")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.14)
    rp.save(fig, "FigP3_signature_scatter", "mid", OUT_DIR)


# ================================================================ main
def main() -> None:
    global OUT_DIR, PERT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="kommagetrennte Praefixe, z. B. Fig2,FigA3")
    ap.add_argument("--perturbation-dir", default=None, help="Ordner mit phase1/ und phase2/ (perturbation_*.csv), falls nicht im Output")
    ap.add_argument("--out", default=None, help="Zielordner (Standard figures_rp/ im Pipeline-Ordner)")
    a = ap.parse_args()
    if a.out:
        OUT_DIR = Path(a.out).resolve()
    if a.perturbation_dir:
        PERT_DIR = Path(a.perturbation_dir).resolve()
    only = [s.strip() for s in a.only.split(",")] if a.only else None

    def want(name):
        return only is None or any(name.startswith(o) for o in only)

    print(f"Schrift: {rp.active_font()} | Ziel: {OUT_DIR}")
    data = {1: load_phase(1), 2: load_phase(2)}
    jobs = [
        ("Fig1", lambda: fig1_configuration_profiles()),
        ("Fig2", lambda: fig2_margin_distribution(data)),
        ("Fig3", lambda: fig3_class_profiles(data)),
        ("Fig4", lambda: fig4_perturbation_flip(data)),
        ("Fig5", lambda: fig5_ws_reference_coherence(data)),
        ("Fig6", lambda: fig6_efa_loadings()),
        ("FigA1", lambda: figA1_temporal_evolution(data)),
        ("FigA2", lambda: figA2_extended_tem(data)),
        ("FigA3", lambda: figA3_scree()),
        ("FigA4", lambda: figA4_factor_correlations()),
        ("FigA5a", lambda: figA5_indicator_correlations(1)),
        ("FigA5b", lambda: figA5_indicator_correlations(2)),
        ("FigA6", lambda: figA6_dimension_heatmap(data)),
        ("FigA7", lambda: figA7_membership_heatmap(data)),
        ("FigA8a", lambda: figA8_ws_detail_radars(data, 1)),
        ("FigA8b", lambda: figA8_ws_detail_radars(data, 2)),
        ("FigA9", lambda: figA9_structure_compare(data)),
        ("FigA10", lambda: figA10_topic_quality()),
        ("FigA11", lambda: figA11_perturbation_by_class()),
    ]
    cross = None
    for name, fn in jobs:
        if want(name):
            fn()
    if any(want(n) for n in ("FigP1", "FigP2", "FigP3")) and (CROSS_DIR / "topic_matches_mutual.csv").exists():
        cross = load_cross(data)
        if want("FigP1"):
            figP1_migration_sankey(cross)
        if want("FigP2"):
            figP2_membership_shift(cross)
        if want("FigP3"):
            figP3_signature_scatter(cross)
    rp.write_manifest(OUT_DIR)
    print("fertig:", OUT_DIR)


if __name__ == "__main__":
    main()
