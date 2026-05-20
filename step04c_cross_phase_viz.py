"""
SCHRITT 4c: Cross-Phase Visualisierungen (Pipeline V2)
==========================================================

Aufgabe
-------
Vergleicht die beiden Phasen-Outputs (P1: 2000-2015, P2: 2016-2025) auf drei
komplementaeren Ebenen:

  1. Membership-Migrations-Sankey
     - Topic-genaue Sicht ueber Schritt-01c-Matches.
     - P1-Argmax-Klasse -> P2-Argmax-Klasse.
     - Bandbreite ~ Anzahl gematchter Topics.
     - Farbintensitaet ~ mittlere Match-Qualitaet (Cosine).

  2. Membership-Verschiebungs-Heatmap (4x4)
     - Topic-genaue Sicht, aber kontinuierlich.
     - Zelle (i,j) = mittlere Delta-m_ws (P2 - P1) ueber Migrationspfad i->j.

  3. Strukturvergleich-Radar (matching-frei)
     - Aggregat-Sicht ohne Matching-Annahme.
     - Mittlere Dimensionsscores P1 vs. P2 auf den 5 Dimensions-Achsen.
     - Dient als methodischer Anker: Identische Struktur auch ohne Topic-Bezug.

V2-Hinweis
----------
Die Sankey- und Heatmap-Sicht reduziert Memberships argmax-basiert (Klassen-
Repraesentant). Das ist eine bewusste Reduktion fuer Migrationsdarstellungen.
Die kontinuierliche Primaergroesse (Membership-Vektor) bleibt in der
Delta-m_ws-Heatmap erhalten.

Autor: Ben Borowski
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from config import (
    BASE_DIR, SIGNAL_COLORS, DIM_COLORS, DIM_NAMES,
    DIM_SHORT_CODES, DIM_SHORT_LIST, FIG_DPI,
)


# =============================================================================
# Pfade
# =============================================================================

P1_DIR = BASE_DIR / "output_phase1"
P2_DIR = BASE_DIR / "output_phase2"
CROSS_DIR = BASE_DIR / "output_cross_phase"


# =============================================================================
# Klassen-Mapping
# =============================================================================

MEMBERSHIP_COLUMNS = ["m_ws", "m_trend", "m_ec", "m_latent"]
MEMBERSHIP_LABELS = {
    "m_ws":     "Weak Signal",
    "m_trend":  "Trend",
    "m_ec":     "Emerging Concept",
    "m_latent": "Latent/Mixed",
}
SIGNAL_ORDER = ["Weak Signal", "Emerging Concept", "Trend", "Latent/Mixed"]


# =============================================================================
# Hilfsfunktionen
# =============================================================================

def _derive_argmax(memberships: pd.DataFrame) -> pd.DataFrame:
    """Argmax-Reduktion ueber die vier Memberships -> signal_type."""
    df = memberships.copy()
    argmax_col = df[MEMBERSHIP_COLUMNS].idxmax(axis=1)
    df["signal_type"] = argmax_col.map(MEMBERSHIP_LABELS)
    return df


def _load_phase(phase_dir, label: str) -> pd.DataFrame:
    """Laedt signal_memberships + dimension_scores einer Phase.
    Gibt Dataframe indiziert auf topic mit Spalten:
      m_ws, m_trend, m_ec, m_latent, margin, signal_type,
      <5 Dimensionsspalten>
    Fehlende Dimensionsspalten werden mit NaN aufgefuellt (Robustheit).
    """
    mem_path = phase_dir / "signal_memberships.csv"
    dim_path = phase_dir / "dimension_scores.csv"
    if not mem_path.exists():
        raise FileNotFoundError(f"Phase {label}: {mem_path} fehlt.")

    mem = pd.read_csv(mem_path)
    mem = _derive_argmax(mem)

    if dim_path.exists():
        dim = pd.read_csv(dim_path)
        # Fehlende Dimensionsspalten mit NaN auffuellen (Robustheit)
        for dim_name in DIM_NAMES:
            if dim_name not in dim.columns:
                dim[dim_name] = np.nan
        keep_cols = ["topic"] + DIM_NAMES
        dim = dim[keep_cols]
        df = mem.merge(dim, on="topic", how="left")
    else:
        # Fallback ohne Dimensionsscores
        for dim_name in DIM_NAMES:
            mem[dim_name] = np.nan
        df = mem

    df = df.set_index("topic")
    df.attrs["phase_label"] = label
    return df


def _load_matches(prefer_mutual: bool = True) -> pd.DataFrame:
    """Laedt Cross-Phase-Matches.
    Bevorzugt topic_matches_mutual.csv (hoechste Qualitaet);
    Fallback topic_matches_best_p1_to_p2.csv (groessere Abdeckung).
    """
    mutual = CROSS_DIR / "topic_matches_mutual.csv"
    best_p1 = CROSS_DIR / "topic_matches_best_p1_to_p2.csv"
    if prefer_mutual and mutual.exists():
        return pd.read_csv(mutual)
    if best_p1.exists():
        return pd.read_csv(best_p1)
    raise FileNotFoundError(
        f"Keine Match-Datei in {CROSS_DIR}. Erforderlich: "
        f"topic_matches_mutual.csv oder topic_matches_best_p1_to_p2.csv."
    )


# =============================================================================
# 1. MEMBERSHIP-MIGRATIONS-SANKEY
# =============================================================================

def _bezier_ribbon(ax, x0, y0_top, y0_bot, x1, y1_top, y1_bot,
                   color, alpha, hatch=None):
    """Zeichnet ein Sankey-Band zwischen zwei vertikalen Schlitzen.

    (x0, [y0_bot, y0_top]) -> (x1, [y1_bot, y1_top])
    Beide Kanten als kubische Bezier-Kurven.

    Parameter
    ---------
    hatch : str | None
        Optionales matplotlib-Hatchmuster (z.B. "//") zur Kennzeichnung
        knapper Migrationen (Margin < 0.10 in mindestens einer Phase). None
        rendert ein vollflaechiges Band ohne Mustering.
    """
    mid = (x0 + x1) / 2.0
    verts = [
        (x0, y0_top),
        (mid, y0_top), (mid, y1_top), (x1, y1_top),
        (x1, y1_bot),
        (mid, y1_bot), (mid, y0_bot), (x0, y0_bot),
        (x0, y0_top),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    if hatch is None:
        patch = mpatches.PathPatch(
            Path(verts, codes),
            facecolor=color, edgecolor="none", alpha=alpha,
        )
    else:
        # Gehatchtes Band: feiner weisser Rand erhoeht Kontrast zum Muster.
        patch = mpatches.PathPatch(
            Path(verts, codes),
            facecolor=color, edgecolor="white", alpha=alpha,
            hatch=hatch, linewidth=0.0,
        )
    ax.add_patch(patch)


def plot_migration_sankey(df_p1: pd.DataFrame, df_p2: pd.DataFrame,
                          matches: pd.DataFrame, output_path) -> dict:
    """Sankey: Argmax-Klassen-Migration P1 -> P2 entlang Matches.

    Bandbreite ~ Anzahl gematchter Topics in der Zelle.
    Alpha ~ mittlere Match-Cosine in der Zelle.
    """
    # Match-Tabelle mit P1- und P2-Klasse + Margin anreichern.
    # Eine Migration gilt als "knapp", wenn in mindestens einer der beiden
    # Phasen die Margin < 0.10 ist (operative Schwelle aus
    # step02b_memberships.py). Ohne diese Differenzierung wuerde die
    # argmax-basierte Sankey-Darstellung knappe Membership-Wechsel optisch
    # identisch zu eindeutigen Klassenwechseln darstellen — was der in V2
    # zurueckgewiesenen kategorialen Reifizierung entspraeche.
    m = matches.copy()
    m["p1_class"] = m["phase1_topic"].map(df_p1["signal_type"])
    m["p2_class"] = m["phase2_topic"].map(df_p2["signal_type"])
    m["margin_p1"] = m["phase1_topic"].map(df_p1["margin"])
    m["margin_p2"] = m["phase2_topic"].map(df_p2["margin"])
    m["is_knapp"] = (
        (m["margin_p1"].fillna(0.0) < 0.10)
        | (m["margin_p2"].fillna(0.0) < 0.10)
    )
    m = m.dropna(subset=["p1_class", "p2_class"])

    # Aggregation pro Zelle — getrennt nach klar / knapp.
    cell = (
        m.groupby(["p1_class", "p2_class"])
         .agg(count=("phase1_topic", "size"),
              n_knapp=("is_knapp", "sum"),
              mean_cos=("cosine", "mean"))
         .reset_index()
    )
    cell["n_clear"] = cell["count"] - cell["n_knapp"]

    # Klassen-Layout
    cls = SIGNAL_ORDER
    n_classes = len(cls)

    p1_totals = m.groupby("p1_class")["phase1_topic"].size().reindex(cls, fill_value=0)
    p2_totals = m.groupby("p2_class")["phase2_topic"].size().reindex(cls, fill_value=0)
    total = max(p1_totals.sum(), p2_totals.sum(), 1)

    # Vertikale Lage der Klassenboxen (oberer Rand abwaerts).
    # Skalierung so, dass alle Boxen + Gaps in y in [0, 1] passen.
    n_cls = len(cls)
    gap = 0.03
    usable_height = 1.0 - gap * (n_cls - 1)
    height_p1 = (p1_totals / total).values * usable_height
    height_p2 = (p2_totals / total).values * usable_height

    def _stack(heights):
        # Rueckgabe (top, bottom) pro Klasse, gestapelt von oben nach unten
        out = []
        cursor = 1.0
        for h in heights:
            top = cursor
            bot = cursor - h
            out.append((top, bot))
            cursor = bot - gap
        return out

    p1_stack = _stack(height_p1)
    p2_stack = _stack(height_p2)

    # Akkumulator-Cursor pro Klassenbox (fuer Sub-Bandzuordnung)
    p1_cursor = [t for (t, _) in p1_stack]
    p2_cursor = [t for (t, _) in p2_stack]

    # Zellen nach (p1_class-Index, count desc) sortieren, damit
    # breite Baender oben in jeder Box liegen
    cell["i"] = cell["p1_class"].map({c: i for i, c in enumerate(cls)})
    cell["j"] = cell["p2_class"].map({c: i for i, c in enumerate(cls)})
    cell = cell.sort_values(["i", "count"], ascending=[True, False])

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    x_left = 0.05
    x_right = 0.95
    box_w = 0.04

    # Klassenboxen P1
    for i, (top, bot) in enumerate(p1_stack):
        cls_name = cls[i]
        color = SIGNAL_COLORS.get(cls_name, "#999999")
        ax.add_patch(mpatches.Rectangle(
            (x_left - box_w, bot), box_w, top - bot,
            facecolor=color, edgecolor="white", linewidth=0.8,
        ))
        n = int(p1_totals.iloc[i])
        ax.text(x_left - box_w - 0.01, (top + bot) / 2.0,
                f"{cls_name}\n(n={n})", ha="right", va="center",
                fontsize=10, fontweight="bold")

    # Klassenboxen P2
    for j, (top, bot) in enumerate(p2_stack):
        cls_name = cls[j]
        color = SIGNAL_COLORS.get(cls_name, "#999999")
        ax.add_patch(mpatches.Rectangle(
            (x_right, bot), box_w, top - bot,
            facecolor=color, edgecolor="white", linewidth=0.8,
        ))
        n = int(p2_totals.iloc[j])
        ax.text(x_right + box_w + 0.01, (top + bot) / 2.0,
                f"{cls_name}\n(n={n})", ha="left", va="center",
                fontsize=10, fontweight="bold")

    # Verbindungsbaender — pro Zelle wird zuerst der klare Anteil
    # (Margin >= 0.10 in beiden Phasen) als Vollband gerendert, anschliessend
    # der knappe Anteil als gehatchtes Band. So bleibt die Bandbreite-Summe
    # informationserhaltend, waehrend die Margin-Codierung pro Zelle direkt
    # ablesbar wird.
    for _, row in cell.iterrows():
        i, j = int(row["i"]), int(row["j"])
        n_clear = int(row["n_clear"])
        n_knapp = int(row["n_knapp"])
        color = SIGNAL_COLORS.get(row["p1_class"], "#999999")
        # Alpha aus Cosine (0.3 .. 0.85)
        cos = float(row["mean_cos"]) if not pd.isna(row["mean_cos"]) else 0.5
        alpha = 0.30 + 0.55 * np.clip(cos, 0.0, 1.0)

        if n_clear > 0:
            h_clear = n_clear / total
            y0_top = p1_cursor[i]
            y0_bot = y0_top - h_clear
            p1_cursor[i] = y0_bot
            y1_top = p2_cursor[j]
            y1_bot = y1_top - h_clear
            p2_cursor[j] = y1_bot
            _bezier_ribbon(ax, x_left, y0_top, y0_bot,
                           x_right, y1_top, y1_bot,
                           color, alpha)

        if n_knapp > 0:
            h_knapp = n_knapp / total
            y0_top = p1_cursor[i]
            y0_bot = y0_top - h_knapp
            p1_cursor[i] = y0_bot
            y1_top = p2_cursor[j]
            y1_bot = y1_top - h_knapp
            p2_cursor[j] = y1_bot
            _bezier_ribbon(ax, x_left, y0_top, y0_bot,
                           x_right, y1_top, y1_bot,
                           color, alpha, hatch="//")

    # Cosmetics
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.10, 1.10)
    ax.set_aspect("auto")
    ax.axis("off")
    n_total_knapp = int(m["is_knapp"].sum())
    n_total_clear = int(len(m) - n_total_knapp)
    ax.set_title(
        f"Membership-Migrations-Sankey: Phase 1 -> Phase 2\n"
        f"({len(m)} gematchte Topics; Alpha ~ Match-Cosine; "
        f"Hatch ~ knappe Migration, Margin < 0.10 in P1 oder P2)",
        fontsize=13, pad=15,
    )
    # Beschriftung der Achsen-Endpunkte
    ax.text(x_left - box_w / 2.0, 1.06, "Phase 1\n(2000-2015)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(x_right + box_w / 2.0, 1.06, "Phase 2\n(2016-2025)",
            ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Legende: klare vs. knappe Migration (Margin-Codierung)
    legend_clear = mpatches.Patch(
        facecolor="#888888", alpha=0.6, edgecolor="none",
        label=f"Klare Migration (Margin ≥ 0.10 in beiden Phasen): "
              f"n={n_total_clear}",
    )
    legend_knapp = mpatches.Patch(
        facecolor="#888888", alpha=0.6, edgecolor="white", hatch="//",
        label=f"Knappe Migration (Margin < 0.10 in P1 oder P2): "
              f"n={n_total_knapp}",
    )
    ax.legend(
        handles=[legend_clear, legend_knapp],
        loc="lower center", bbox_to_anchor=(0.5, -0.06),
        ncol=2, fontsize=9, frameon=False,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Sankey gespeichert: {output_path}")
    return {
        "n_matches": int(len(m)),
        "n_clear": n_total_clear,
        "n_knapp": n_total_knapp,
        "cells": cell.to_dict(orient="records"),
    }


# =============================================================================
# 2. MEMBERSHIP-VERSCHIEBUNGS-HEATMAP (Delta m_ws)
# =============================================================================

def plot_membership_shift_heatmap(df_p1: pd.DataFrame, df_p2: pd.DataFrame,
                                  matches: pd.DataFrame, output_path) -> dict:
    """4x4-Heatmap: mittlere Delta-m_ws ueber Migrationspfade."""
    m = matches.copy()
    m["p1_class"] = m["phase1_topic"].map(df_p1["signal_type"])
    m["p2_class"] = m["phase2_topic"].map(df_p2["signal_type"])
    m["m_ws_p1"] = m["phase1_topic"].map(df_p1["m_ws"])
    m["m_ws_p2"] = m["phase2_topic"].map(df_p2["m_ws"])
    m["delta_m_ws"] = m["m_ws_p2"] - m["m_ws_p1"]
    m = m.dropna(subset=["p1_class", "p2_class", "delta_m_ws"])

    cls = SIGNAL_ORDER
    mean_mat = np.full((len(cls), len(cls)), np.nan)
    count_mat = np.zeros((len(cls), len(cls)), dtype=int)

    for i, c1 in enumerate(cls):
        for j, c2 in enumerate(cls):
            sub = m[(m["p1_class"] == c1) & (m["p2_class"] == c2)]
            if len(sub) > 0:
                mean_mat[i, j] = sub["delta_m_ws"].mean()
                count_mat[i, j] = len(sub)

    # Annotationen: "Delta m_ws\n(n)"
    annot = np.empty(mean_mat.shape, dtype=object)
    for i in range(len(cls)):
        for j in range(len(cls)):
            if count_mat[i, j] == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = f"{mean_mat[i, j]:+.3f}\n(n={count_mat[i, j]})"

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    vmax = np.nanmax(np.abs(mean_mat)) if np.isfinite(np.nanmax(np.abs(mean_mat))) else 0.5
    vmax = max(vmax, 0.1)
    sns.heatmap(
        mean_mat, annot=annot, fmt="",
        cmap="RdBu_r", center=0.0, vmin=-vmax, vmax=vmax,
        xticklabels=cls, yticklabels=cls,
        cbar_kws={"label": "Mittlere $\\Delta m_{ws}$ (P2 − P1)"},
        linewidths=0.5, linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Phase 2: Argmax-Klasse", fontsize=11)
    ax.set_ylabel("Phase 1: Argmax-Klasse", fontsize=11)
    ax.set_title(
        "Membership-Verschiebung der Weak-Signal-Dimension\n"
        "ueber Argmax-Migrationspfade (Delta m_ws, P2 − P1)",
        fontsize=13, pad=12,
    )
    # x-Tick-Farben analog SIGNAL_COLORS
    for tlab, c in zip(ax.get_xticklabels(), cls):
        tlab.set_color(SIGNAL_COLORS.get(c, "#333333"))
        tlab.set_fontweight("bold")
    for tlab, c in zip(ax.get_yticklabels(), cls):
        tlab.set_color(SIGNAL_COLORS.get(c, "#333333"))
        tlab.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Shift-Heatmap gespeichert: {output_path}")
    return {
        "n_matches": int(len(m)),
        "mean_delta_m_ws": float(np.nanmean(mean_mat)),
    }


# =============================================================================
# 3. STRUKTURVERGLEICH-RADAR (matching-frei)
# =============================================================================

def plot_structure_compare_radar(df_p1: pd.DataFrame, df_p2: pd.DataFrame,
                                 output_path, q: float = 0.90) -> dict:
    """Aggregat-Radar P1 vs. P2 ueber alle Topics einer Phase.

    Matching-frei: dient als methodischer Anker, um den strukturellen
    Phasenunterschied auch ohne Topic-Bezug sichtbar zu machen.

    Methodischer Hinweis
    --------------------
    Die Dimensionsscores werden in step03_efa_pca *pro Phase* z-standardisiert.
    Ein Mittelwert-Vergleich der Phasen-Aggregate ist damit per Konstruktion
    nahe Null und uninformativ. Stattdessen wird das q-Quantil (default q=0.90)
    pro Dimension verwendet: Es beschreibt die strukturelle "Spitze" der jeweiligen
    Phase und bleibt auch unter phaseninterner z-Standardisierung aussagekraeftig.
    """
    def _profile(df, dims, q):
        out = {}
        for d in dims:
            if d in df.columns and df[d].notna().any():
                out[d] = (float(df[d].quantile(q)),
                          float(df[d].median()))
            else:
                out[d] = (np.nan, np.nan)
        return out

    s_p1 = _profile(df_p1, DIM_NAMES, q)
    s_p2 = _profile(df_p2, DIM_NAMES, q)

    angles = np.linspace(0, 2 * np.pi, len(DIM_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(polar=True))

    # Phase 1 — q-Quantil (Hauptlinie) + Median (gestrichelt)
    vals_p1_q = [s_p1[d][0] for d in DIM_NAMES]; vals_p1_q += vals_p1_q[:1]
    vals_p1_m = [s_p1[d][1] for d in DIM_NAMES]; vals_p1_m += vals_p1_m[:1]
    ax.plot(angles, vals_p1_q, "o-", color="#34495E", linewidth=2.2,
            label=f"Phase 1, q={int(q*100)} (n={len(df_p1)})")
    ax.fill(angles, vals_p1_q, alpha=0.10, color="#34495E")
    ax.plot(angles, vals_p1_m, ":", color="#34495E", linewidth=1.2, alpha=0.7,
            label="Phase 1, Median")

    # Phase 2
    vals_p2_q = [s_p2[d][0] for d in DIM_NAMES]; vals_p2_q += vals_p2_q[:1]
    vals_p2_m = [s_p2[d][1] for d in DIM_NAMES]; vals_p2_m += vals_p2_m[:1]
    ax.plot(angles, vals_p2_q, "s-", color="#E67E22", linewidth=2.2,
            label=f"Phase 2, q={int(q*100)} (n={len(df_p2)})")
    ax.fill(angles, vals_p2_q, alpha=0.10, color="#E67E22")
    ax.plot(angles, vals_p2_m, ":", color="#E67E22", linewidth=1.2, alpha=0.7,
            label="Phase 2, Median")

    # Achsen
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_SHORT_LIST, fontsize=12)
    for tlab, dim in zip(ax.get_xticklabels(), DIM_NAMES):
        tlab.set_color(DIM_COLORS[dim])
        tlab.set_fontweight("bold")

    ax.set_title(
        "Strukturvergleich (matching-frei)\n"
        f"Profile pro Phase: q{int(q*100)}-Quantil (Spitzenstruktur) und Median",
        fontsize=13, pad=22,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Strukturvergleich-Radar gespeichert: {output_path}")

    delta_q = {
        d: float(s_p2[d][0] - s_p1[d][0])
        if not (np.isnan(s_p1[d][0]) or np.isnan(s_p2[d][0])) else np.nan
        for d in DIM_NAMES
    }
    return {"q": q, "delta_q_quantile": delta_q}


# =============================================================================
# 4. RUN
# =============================================================================

def run() -> dict:
    print("\n" + "=" * 70)
    print("CROSS-PHASE VISUALISIERUNGEN (Pipeline V2)")
    print("=" * 70)

    if not P1_DIR.exists() or not P2_DIR.exists():
        raise FileNotFoundError(
            f"Phasen-Outputs fehlen: {P1_DIR} oder {P2_DIR}."
        )
    if not CROSS_DIR.exists():
        raise FileNotFoundError(
            f"Cross-Phase-Output {CROSS_DIR} fehlt. "
            f"Bitte zuerst step01c_cross_phase_matching.py ausfuehren."
        )

    print(f"  P1-Output    : {P1_DIR}")
    print(f"  P2-Output    : {P2_DIR}")
    print(f"  Cross-Output : {CROSS_DIR}")

    df_p1 = _load_phase(P1_DIR, "Phase 1")
    df_p2 = _load_phase(P2_DIR, "Phase 2")
    matches = _load_matches(prefer_mutual=True)
    print(f"  Topics P1    : {len(df_p1)}")
    print(f"  Topics P2    : {len(df_p2)}")
    print(f"  Match-Paare  : {len(matches)}")

    out_sankey  = CROSS_DIR / "migration_sankey.png"
    out_heat    = CROSS_DIR / "membership_shift_heatmap.png"
    out_radar   = CROSS_DIR / "structure_compare_radar.png"

    print("\n1. Membership-Migrations-Sankey...")
    info_sankey = plot_migration_sankey(df_p1, df_p2, matches, out_sankey)

    print("2. Membership-Verschiebungs-Heatmap...")
    info_heat = plot_membership_shift_heatmap(df_p1, df_p2, matches, out_heat)

    print("3. Strukturvergleich-Radar (matching-frei)...")
    info_radar = plot_structure_compare_radar(df_p1, df_p2, out_radar)

    print("\n" + "=" * 70)
    print("FERTIG: Cross-Phase-Visualisierungen")
    print(f"  Ausgaben in: {CROSS_DIR}")
    print("=" * 70)

    return {
        "sankey": info_sankey,
        "shift_heatmap": info_heat,
        "structure_radar": info_radar,
    }


if __name__ == "__main__":
    run()
