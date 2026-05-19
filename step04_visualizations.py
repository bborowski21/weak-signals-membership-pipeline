"""
SCHRITT 4: Visualisierungen — Signalprofile und Ergebnisdarstellung (Pipeline V2)
==================================================================================

Generiert alle Visualisierungen für den F3-Prototyp V2:
  1. Radar-Profile: Mittlere Dimensionsscores pro Signaltyp (argmax-Repräsentant)
  2. Extended TEM: Ebadi-Quadranten + Epistemische Offenheit als Bubble
  3. Dimensionsheatmap: Alle Topics × 5 Dimensionen
  4. Top Weak Signals: Individuelle Detail-Radars
  5. Temporale Evolution: Signal-Typ-Komposition über Zeit
  6. Membership-Heatmap: kontinuierliche Klassenzugehörigkeiten (V2-Mehrwert)

V2-Hinweis: Der „signal_type" pro Topic wird hier on-demand als argmax über die
vier Memberships {m_ws, m_trend, m_ec, m_latent} berechnet. Das ist eine
deterministische Reduktion der kontinuierlichen Membership-Vektoren zu einem
Repräsentanten — ausschließlich für die nicht-kontinuierlichen Plot-Typen
(Stacked Area, Heatmap-Sortierung). Die Primärgröße der Pipeline V2 bleibt
der Membership-Vektor selbst.

Autor: Ben Borowski
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from config import (
    OUTPUT_DIR, SIGNAL_COLORS, DIM_COLORS, DIM_NAMES,
    DIM_SHORT_CODES, DIM_SHORT_LIST, FIG_DPI,
)


# Achsenbeschriftungen der Radar-/Heatmap-Plots — Dimensions-Kuerzel.
# Interne DataFrame-Spaltennamen (DIM_NAMES) bleiben die langen Bezeichner.
DIM_LABELS_RADAR = DIM_SHORT_LIST


# Mapping Membership-Spalte → Signaltyp-Label (für argmax-Repräsentation)
MEMBERSHIP_COLUMNS = ["m_ws", "m_trend", "m_ec", "m_latent"]
MEMBERSHIP_LABELS = {
    "m_ws":     "Weak Signal",
    "m_trend":  "Trend",
    "m_ec":     "Emerging Concept",
    "m_latent": "Latent/Mixed",
}


def derive_argmax_representation(memberships: pd.DataFrame) -> pd.DataFrame:
    """V2-Helfer: Reduziert Membership-Vektoren auf argmax-Repräsentant.

    Erzeugt die Spalten:
      - signal_type:  argmax über {m_ws, m_trend, m_ec, m_latent}
      - ws_distance:  1 − m_ws (Kompatibilitäts-Proxy für Top-WS-Ranking)

    Verwendet ausschließlich für deterministische Plot-Typen (Sortierung,
    Stacked-Area). Die Primärgröße der V2-Pipeline bleibt der kontinuierliche
    Membership-Vektor — diese Reduktion ist sekundär und nur visualisierungs-
    bedingt.
    """
    df = memberships.copy()
    argmax_col = df[MEMBERSHIP_COLUMNS].idxmax(axis=1)
    df["signal_type"]  = argmax_col.map(MEMBERSHIP_LABELS)
    df["ws_distance"]  = 1.0 - df["m_ws"]
    return df


# =============================================================================
# 1. RADAR-PROFILE
# =============================================================================

def plot_radar_profiles(classified: pd.DataFrame, output_path: str):
    """Mittlere Radar-Profile pro Signaltyp."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 9), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(DIM_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    for signal_type, color in SIGNAL_COLORS.items():
        subset = classified[classified["signal_type"] == signal_type]
        if len(subset) == 0:
            continue

        values = subset[DIM_NAMES].mean().tolist()
        values += values[:1]

        ax.plot(angles, values, "o-", color=color, linewidth=2,
                label=f"{signal_type} (n={len(subset)})")
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_LABELS_RADAR, fontsize=11)
    # Dimensions-Kuerzel in Dimensionsfarbe einfaerben (vgl. Abb. 3.2)
    for label, dim in zip(ax.get_xticklabels(), DIM_NAMES):
        label.set_color(DIM_COLORS[dim])
        label.set_fontweight("bold")
    ax.set_title("Signalprofil-Vergleich\n(Mittlere Dimensionsscores pro Signaltyp)",
                 fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Radar-Profile gespeichert: {output_path}")


# =============================================================================
# 2. EXTENDED TEM (Topic Emergence Map)
# =============================================================================

def plot_extended_tem(classified: pd.DataFrame, tem_metrics: pd.DataFrame,
                      topic_keywords: dict, output_path: str):
    """
    Erweiterte Topic Emergence Map nach Ebadi (2026):
    - x-Achse: Average Topic Proportion
    - y-Achse: Annualized Growth Rate
    - Farbe: Signaltyp
    - Bubble-Größe: Epistemische Offenheit
    """
    tem = tem_metrics.copy().set_index("topic")
    merged = tem.join(
        classified[["signal_type", "Epistemische Offenheit"]], how="inner"
    )

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    x_thresh = merged["avg_proportion"].median()
    y_thresh = 0.0

    ax.axhline(y=y_thresh, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=x_thresh, color="gray", linestyle="--", alpha=0.4)

    # Quadrantenbeschriftung
    props = dict(fontsize=11, alpha=0.3, fontweight="bold")
    xlim_r = merged["avg_proportion"].max() * 1.1
    ylim_t = merged["growth_rate"].max() * 1.1
    ylim_b = merged["growth_rate"].min() * 1.1

    ax.text(x_thresh * 0.3, ylim_t * 0.85,
            "WEAK SIGNAL\n(geringe Proportion, hohes Wachstum)", ha="center", **props)
    ax.text(xlim_r * 0.7, ylim_t * 0.85,
            "STRONG SIGNAL\n(hohe Proportion, hohes Wachstum)", ha="center", **props)
    ax.text(x_thresh * 0.3, ylim_b * 0.7,
            "LATENT\n(geringe Proportion, Rückgang)", ha="center", **props)
    ax.text(xlim_r * 0.7, ylim_b * 0.7,
            "ESTABLISHED\n(hohe Proportion, Rückgang)", ha="center", **props)

    # Bubbles plotten
    for idx, row in merged.iterrows():
        color = SIGNAL_COLORS.get(row["signal_type"], "#999999")
        eo = row["Epistemische Offenheit"]
        size = 50 + abs(eo) * 300

        ax.scatter(row["avg_proportion"], row["growth_rate"],
                   s=size, c=color, alpha=0.6, edgecolors="white", linewidth=0.5)

        # Annotation für auffällige Topics
        if abs(row["growth_rate"]) > 0.2 or row["avg_proportion"] > x_thresh * 1.5:
            kws = topic_keywords.get(idx, [])
            label = ", ".join([w for w, _ in kws[:2]]) if kws else f"T{idx}"
            ax.annotate(label, (row["avg_proportion"], row["growth_rate"]),
                        fontsize=7, alpha=0.8, xytext=(5, 5),
                        textcoords="offset points")

    # Legende
    handles = [mpatches.Patch(color=c, label=l, alpha=0.6)
               for l, c in SIGNAL_COLORS.items()]
    handles.append(plt.scatter([], [], s=50, c="gray", alpha=0.5,
                               label="Niedrige EO"))
    handles.append(plt.scatter([], [], s=300, c="gray", alpha=0.5,
                               label="Hohe EO"))
    ax.legend(handles=handles, loc="upper right", fontsize=9)

    ax.set_xlabel("Average Topic Proportion (p̄)", fontsize=12)
    ax.set_ylabel("Annualized Growth Rate (g)", fontsize=12)
    ax.set_title("Extended Topic Emergence Map\n"
                 "(nach Ebadi et al. 2026, erweitert um EO)",
                 fontsize=13)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Extended TEM gespeichert: {output_path}")


# =============================================================================
# 3. DIMENSIONSHEATMAP
# =============================================================================

def plot_dimension_heatmap(classified: pd.DataFrame, topic_keywords: dict,
                           output_path: str):
    """Heatmap: alle Topics × 5 Dimensionen, sortiert nach Signaltyp."""
    type_order = {"Weak Signal": 0, "Emerging Concept": 1,
                  "Latent/Mixed": 2, "Trend": 3}
    sorted_df = classified.copy()
    sorted_df["type_order"] = sorted_df["signal_type"].map(type_order)
    sorted_df = sorted_df.sort_values(["type_order", "ws_distance"])

    # Labels erstellen
    labels = []
    for idx in sorted_df.index:
        kws = topic_keywords.get(idx, [])
        kw_str = ", ".join([w for w, _ in kws[:2]]) if kws else f"T{idx}"
        labels.append(f"T{idx}: {kw_str[:30]}")

    fig, ax = plt.subplots(1, 1,
                           figsize=(12, max(16, len(sorted_df) * 0.15)))

    data = sorted_df[DIM_NAMES].values
    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", vmin=-2, vmax=2)

    ax.set_xticks(range(len(DIM_NAMES)))
    ax.set_xticklabels(DIM_LABELS_RADAR, fontsize=10, rotation=0)
    # Dimensions-Kuerzel in Dimensionsfarbe einfaerben (vgl. Abb. 3.2)
    for label, dim in zip(ax.get_xticklabels(), DIM_NAMES):
        label.set_color(DIM_COLORS[dim])
        label.set_fontweight("bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)

    # Signaltyp-Farbe am linken Rand
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        color = SIGNAL_COLORS.get(row["signal_type"], "#999999")
        ax.add_patch(plt.Rectangle((-0.7, i - 0.5), 0.3, 1,
                                    color=color, clip_on=False))

    # Trennlinien zwischen Signaltypen
    prev_type = None
    for i, (_, row) in enumerate(sorted_df.iterrows()):
        if row["signal_type"] != prev_type and prev_type is not None:
            ax.axhline(y=i - 0.5, color="black", linewidth=1.5)
        prev_type = row["signal_type"]

    plt.colorbar(im, ax=ax, label="Dimensionsscore (z-standardisiert)", shrink=0.5)
    ax.set_title("Topic × Dimension Heatmap\n"
                 "(sortiert nach Signaltyp, rot = hoher Score)", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Dimensionsheatmap gespeichert: {output_path}")


# =============================================================================
# 4. TOP WEAK SIGNALS DETAIL-RADARS
# =============================================================================

def plot_ws_detail_radars(classified: pd.DataFrame, topic_keywords: dict,
                          output_path: str, n_top: int = 6):
    """Individuelle Radar-Plots für die Top Weak Signals."""
    ws = classified[classified["signal_type"] == "Weak Signal"].sort_values("ws_distance")
    top_ws = ws.head(n_top)

    if len(top_ws) == 0:
        print("  Keine Weak Signals gefunden — überspringe Detail-Radars.")
        return

    n_cols = min(3, len(top_ws))
    n_rows = (len(top_ws) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows),
                              subplot_kw=dict(polar=True))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    angles = np.linspace(0, 2 * np.pi, len(DIM_NAMES), endpoint=False).tolist()
    angles += angles[:1]

    # Ideal-WS-Profil (Median aller WS)
    ws_ideal = ws[DIM_NAMES].median().tolist()
    ws_ideal += ws_ideal[:1]

    for i, (idx, row) in enumerate(top_ws.iterrows()):
        if i >= len(axes):
            break

        ax = axes[i]
        values = row[DIM_NAMES].tolist()
        values += values[:1]

        ax.plot(angles, ws_ideal, "--", color="#999999", linewidth=1,
                alpha=0.5, label="WS-Median")
        ax.fill(angles, ws_ideal, alpha=0.05, color="gray")

        ax.plot(angles, values, "o-", color="#e74c3c", linewidth=2)
        ax.fill(angles, values, alpha=0.15, color="#e74c3c")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIM_LABELS_RADAR, fontsize=8)
        # Dimensions-Kuerzel in Dimensionsfarbe einfaerben (vgl. Abb. 3.2)
        for label, dim in zip(ax.get_xticklabels(), DIM_NAMES):
            label.set_color(DIM_COLORS[dim])
            label.set_fontweight("bold")

        kws = topic_keywords.get(idx, [])
        kw_str = ", ".join([w for w, _ in kws[:3]])
        ax.set_title(f"T{idx}: {kw_str[:35]}\nWS-dist={row['ws_distance']:.2f}",
                      fontsize=10, pad=15)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Top Weak Signal Profile — Individuelle Dimensionsanalyse",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  WS Detail-Radars gespeichert: {output_path}")


# =============================================================================
# 5. TEMPORALE EVOLUTION
# =============================================================================

def plot_temporal_evolution(df_topics: pd.DataFrame, classified: pd.DataFrame,
                            output_path: str):
    """Signal-Typ-Komposition über Zeit (Proportionen + absolute Counts)."""
    df = df_topics.copy()
    df = df[df["topic"] >= 0]

    topic_type = classified["signal_type"].to_dict()
    df["signal_type"] = df["topic"].map(topic_type)

    counts = df.groupby(["Year", "signal_type"]).size().unstack(fill_value=0)
    props = counts.div(counts.sum(axis=1), axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked Area: Proportionen
    order = ["Weak Signal", "Emerging Concept", "Latent/Mixed", "Trend"]
    cumsum = pd.DataFrame(0, index=props.index, columns=["base"])
    for st in order:
        if st in props.columns:
            new_cum = cumsum["base"] + props[st]
            ax1.fill_between(props.index, cumsum["base"], new_cum,
                             alpha=0.3, color=SIGNAL_COLORS[st])
            ax1.plot(props.index, new_cum, color=SIGNAL_COLORS[st],
                     linewidth=1.5, label=st)
            cumsum["base"] = new_cum

    ax1.set_xlabel("Jahr", fontsize=12)
    ax1.set_ylabel("Kumulative Proportion", fontsize=12)
    ax1.set_title("Signaltyp-Proportionen über Zeit", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2)

    # Balkendiagramm: absolute Counts
    bar_data = counts.reindex(columns=order, fill_value=0)
    bar_data.plot.bar(stacked=True, ax=ax2,
                       color=[SIGNAL_COLORS[c] for c in bar_data.columns],
                       alpha=0.8)

    ax2.set_xlabel("Jahr", fontsize=12)
    ax2.set_ylabel("Anzahl Publikationen", fontsize=12)
    ax2.set_title("Signaltyp-Counts über Zeit", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Temporale Evolution gespeichert: {output_path}")


# =============================================================================
# 6. MEMBERSHIP-HEATMAP (V2-spezifisch: kontinuierliche Klassenzugehörigkeiten)
# =============================================================================

def plot_membership_heatmap(classified: pd.DataFrame, topic_keywords: dict,
                            output_path: str):
    """Heatmap: Topics × 4 Memberships (V2-Kernartefakt).

    Visualisiert die kontinuierlichen Klassenzugehörigkeiten ohne argmax-
    Reduktion. Sortiert nach absteigender Margin (Top = klar zuordenbare
    Topics, Boden = Übergangsfälle).
    """
    sorted_df = classified.copy().sort_values("margin", ascending=False)

    labels = []
    for idx in sorted_df.index:
        kws = topic_keywords.get(idx, [])
        kw_str = ", ".join([w for w, _ in kws[:2]]) if kws else f"T{idx}"
        labels.append(f"T{idx}: {kw_str[:30]} (Δ={sorted_df.loc[idx, 'margin']:.2f})")

    fig, ax = plt.subplots(1, 1,
                           figsize=(10, max(14, len(sorted_df) * 0.15)))

    data = sorted_df[MEMBERSHIP_COLUMNS].values
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(len(MEMBERSHIP_COLUMNS)))
    ax.set_xticklabels(
        ["Weak Signal", "Trend", "Emerging Concept", "Latent"],
        fontsize=10, rotation=20, ha="right"
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)

    plt.colorbar(im, ax=ax, label="Membership-Score [0, 1]", shrink=0.5)
    ax.set_title(
        "Membership-Heatmap (Pipeline V2)\n"
        "Topics × 4 kontinuierliche Klassenzugehörigkeiten\n"
        "Sortiert nach Margin (Top: klar, Boden: Übergangsfälle)",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Membership-Heatmap gespeichert: {output_path}")


def plot_margin_distribution(classified: pd.DataFrame, output_path: str):
    """Diagnostik: Verteilung der Margin (Top-1 minus Top-2 Membership).

    Niedrige Margins markieren Übergangsfälle. Histogramm + zwei Schwellen-
    Markierungen (0.05 sehr unklar; 0.10 unklar) für visuelle Inspektion.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    m = classified["margin"]
    ax.hist(m, bins=40, color="#3498db", alpha=0.7, edgecolor="white")

    ax.axvline(x=0.05, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Δ < 0.05 (sehr unklar): {(m < 0.05).sum()} Topics")
    ax.axvline(x=0.10, color="#f39c12", linestyle="--", linewidth=1.5,
               label=f"Δ < 0.10 (unklar): {(m < 0.10).sum()} Topics")

    ax.set_xlabel("Margin Δ = m_(1) − m_(2)", fontsize=12)
    ax.set_ylabel("Anzahl Topics", fontsize=12)
    ax.set_title(
        "Margin-Verteilung — Diagnostik für Übergangsfälle\n"
        f"n={len(m)} Topics, Median Δ={m.median():.3f}",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Margin-Verteilung gespeichert: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def run():
    print("=" * 70)
    print("SCHRITT 4: VISUALISIERUNGEN — Pipeline V2")
    print("=" * 70)

    # --- Daten laden ---
    # V2: Memberships statt Klassifikation
    memberships = pd.read_csv(
        OUTPUT_DIR / "signal_memberships.csv", index_col=0
    )
    memberships.index.name = "topic"

    # Argmax-Repräsentation für deterministische Plots ableiten
    classified = derive_argmax_representation(memberships)

    tem_metrics = pd.read_csv(OUTPUT_DIR / "tem_metrics.csv")
    dim_scores = pd.read_csv(
        OUTPUT_DIR / "dimension_scores.csv", index_col=0
    )
    dim_scores.index.name = "topic"

    # Dimensionsscores in classified-DF mergen (für Radar/Heatmap)
    for col in DIM_NAMES:
        if col not in classified.columns:
            classified[col] = dim_scores[col]

    # Topic Keywords laden
    kw_df = pd.read_csv(OUTPUT_DIR / "topic_keywords.csv")
    topic_keywords = {}
    for tid in classified.index:
        kws = kw_df[kw_df["topic"] == tid].head(5)
        topic_keywords[tid] = list(
            zip(kws["keyword"].tolist(), kws["score"].tolist())
        )

    # Topic Assignments für temporale Analyse
    topic_assignments = pd.read_csv(OUTPUT_DIR / "topic_assignments.csv")

    # --- Visualisierungen generieren ---
    out = str(OUTPUT_DIR)

    print("\n1. Radar-Profile (argmax-basierte Repräsentation)...")
    plot_radar_profiles(classified, f"{out}/radar_profiles.png")

    print("\n2. Extended TEM...")
    plot_extended_tem(classified, tem_metrics, topic_keywords,
                      f"{out}/extended_tem.png")

    print("\n3. Dimensionsheatmap...")
    plot_dimension_heatmap(classified, topic_keywords,
                           f"{out}/dimension_heatmap.png")

    print("\n4. Top WS Detail-Radars...")
    plot_ws_detail_radars(classified, topic_keywords,
                          f"{out}/ws_detail_radars.png")

    print("\n5. Temporale Evolution...")
    plot_temporal_evolution(topic_assignments, classified,
                           f"{out}/temporal_evolution.png")

    print("\n6. Membership-Heatmap (V2-Kernartefakt)...")
    plot_membership_heatmap(classified, topic_keywords,
                            f"{out}/membership_heatmap.png")

    print("\n7. Margin-Verteilung (Diagnostik)...")
    plot_margin_distribution(classified, f"{out}/margin_distribution.png")

    print(f"\n{'=' * 70}")
    print("Alle Visualisierungen generiert (Pipeline V2).")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run()
