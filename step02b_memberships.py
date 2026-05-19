"""
SCHRITT 2b: Kontinuierliche Klassen-Memberships (Pipeline V2)
==============================================================

Ersetzt die schwellenbasierte classify_signals() aus V1 durch ein
Membership-Paradigma. Für jedes Topic werden vier kontinuierliche Scores
in [0, 1] berechnet — Weak Signal, Trend, Emerging Concept, Latent — die
sich nicht gegenseitig ausschließen.

Theoretischer Anker:
  - Hiltunen (2008): Weak Signals zeichnen sich durch hohe Unsicherheit
    über alle Kerndimensionen aus (EO, WN, EP, DI). Wirkungspotenzial (WP)
    ist im WS-Begriff KONSTITUTIV NICHT enthalten.
  - Rotolo et al. (2015): Emerging Concepts werden positiv über Novelty
    (EP1), Growth (WP1), Impact (WP2) und Field-Breadth (WP3) definiert
    — nicht residual als „Nicht-WS und Nicht-Trend".

Mathematische Spezifikation:
  Pro Topic t und Dimension d ∈ {EO, WN, EP, DI, WP}:
      z_d(t) = (ind_d(t) − Median_d) / max(IQR_d / 2, eps)        (robust)
      σ(x)   = 1 / (1 + exp(−k · x))                              (Sigmoid)

  Memberships (alle in [0, 1], nicht-exklusiv):
      m_WS    = σ( mean(z_EO, z_WN, z_EP, z_DI) )                 (Hiltunen)
      m_Trend = σ( −mean(z_EO, z_WN, z_EP, z_DI) + λ · z_WP )
      m_EC    = σ( mean(z_EP1, z_WP1, z_WP2, z_WP3) )             (Rotolo)
      m_Lat   = σ( −mean(z_EO, z_WN, z_EP, z_DI, z_WP) )

  Margin (Diagnostik): m_(1) − m_(2) — Differenz Top-1 vs. Top-2 Membership.
  Niedrige Margin = Übergangsfall (nicht-deterministisches Verständnis).

Eingabe:
  output/indicators_16.csv      (aus step02_indicators.py)
  output/dimension_scores.csv   (aus step02_indicators.py)

Ausgabe:
  output/signal_memberships.csv mit Spalten:
      topic_id, m_ws, m_trend, m_ec, m_latent, margin

Autor: Ben Borowski
"""

import pandas as pd
import numpy as np

from config import (
    OUTPUT_DIR,
    MEMBERSHIP_SIGMOID_K,
    MEMBERSHIP_LAMBDA_WP,
    INDICATOR_DIMENSIONS,
    DIM_NAMES,
)

# Dimensions-Konstanten
CORE_DIMS = ["Epistemische Offenheit", "Wahrnehmbarkeit",
             "Entwicklungsphase", "Diffusion"]
WP_DIM = "Wirkungspotenzial"

# EC nach Rotolo et al. 2015: Novelty (EP1) + Growth (WP1) + Impact (WP2)
# + Field-Breadth (WP3). Subindikator-Namen wie in INDICATOR_DIMENSIONS.
EC_SUBINDICATORS = ["temporal_novelty", "growth_rate",
                    "citation_momentum", "field_breadth"]


def robust_z(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    """Robuster z-Score: (x − median) / max(IQR / 2, eps).

    Eingangsgröße: kontinuierliche Werte pro Topic.
    Ergebnis: gleichlange Series mit dimensionsneutralen Distanzen vom Median.
    eps schützt gegen IQR = 0 (degeneriert bei kleinem Topic-Set).
    """
    med = series.median()
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    iqr = q75 - q25
    return (series - med) / max(iqr / 2.0, eps)


def sigmoid(x: pd.Series, k: float = 1.0) -> pd.Series:
    """Logistic mit Skalenparameter k.

    k = 1.0 → sanfte Trennung; σ(0)=0.5, σ(1)≈0.73, σ(2)≈0.88.
    Größere k → schärfer; kleinere → flacher (näher an linear).
    """
    return 1.0 / (1.0 + np.exp(-k * x))


def compute_memberships(
    indicator_df: pd.DataFrame,
    dim_scores: pd.DataFrame,
    lambda_wp: float = MEMBERSHIP_LAMBDA_WP,
    k: float = MEMBERSHIP_SIGMOID_K,
) -> pd.DataFrame:
    """Berechnet die vier Klassen-Memberships pro Topic.

    Args:
        indicator_df: DataFrame [topic_id × 16 Subindikatoren] mit
                      kontinuierlichen Indikatorwerten (aus step02).
        dim_scores:   DataFrame [topic_id × 5 Dimensionen] mit z-standardi-
                      sierten Dimensionsmittelwerten (aus step02).
        lambda_wp:    Gewichtung von WP im Trend-Score (Default 0.5).
        k:            Sigmoid-Skalenparameter (Default 1.0).

    Returns:
        DataFrame [topic_id × {m_ws, m_trend, m_ec, m_latent, margin}].
    """
    # Robust z-Scores pro Dimension (über alle Topics)
    z_dim = dim_scores.apply(robust_z, axis=0)

    # Robust z-Scores pro EC-Subindikator
    missing_ec = [c for c in EC_SUBINDICATORS if c not in indicator_df.columns]
    if missing_ec:
        raise KeyError(
            f"EC-Subindikatoren fehlen in indicator_df: {missing_ec}. "
            f"Verfügbare Spalten: {list(indicator_df.columns)}"
        )
    z_ec = indicator_df[EC_SUBINDICATORS].apply(robust_z, axis=0)

    # Aggregations-Mittelwerte
    z_core_mean = z_dim[CORE_DIMS].mean(axis=1)
    z_wp        = z_dim[WP_DIM]
    z_ec_mean   = z_ec.mean(axis=1)
    z_all_mean  = z_dim[CORE_DIMS + [WP_DIM]].mean(axis=1)

    # Memberships
    m_ws     = sigmoid(z_core_mean, k=k)
    m_trend  = sigmoid(-z_core_mean + lambda_wp * z_wp, k=k)
    m_ec     = sigmoid(z_ec_mean, k=k)
    m_latent = sigmoid(-z_all_mean, k=k)

    memberships = pd.DataFrame({
        "m_ws":     m_ws,
        "m_trend":  m_trend,
        "m_ec":     m_ec,
        "m_latent": m_latent,
    }, index=indicator_df.index)

    # Margin: Differenz zwischen Top-1 und Top-2 Membership pro Topic
    sorted_vals = np.sort(memberships.values, axis=1)
    memberships["margin"] = sorted_vals[:, -1] - sorted_vals[:, -2]

    return memberships


def run():
    """Lädt Step-2-Artefakte, berechnet Memberships, schreibt CSV."""
    print("=" * 70)
    print("SCHRITT 2b: MEMBERSHIP-SCORES — Pipeline V2")
    print("=" * 70)
    print(f"Sigmoid-Skalenparameter k = {MEMBERSHIP_SIGMOID_K}")
    print(f"WP-Gewicht λ            = {MEMBERSHIP_LAMBDA_WP}")
    print(f"EC-Subindikatoren        = {EC_SUBINDICATORS}")

    print("\nLade Indikator- und Dimensionsscores...")
    indicator_df = pd.read_csv(OUTPUT_DIR / "indicators_16.csv", index_col=0)
    dim_scores   = pd.read_csv(OUTPUT_DIR / "dimension_scores.csv", index_col=0)

    print(f"  Indikator-Matrix: {indicator_df.shape}")
    print(f"  Dimensionsscores: {dim_scores.shape}")

    memberships = compute_memberships(
        indicator_df=indicator_df,
        dim_scores=dim_scores,
        lambda_wp=MEMBERSHIP_LAMBDA_WP,
        k=MEMBERSHIP_SIGMOID_K,
    )

    # Speichern
    out_path = OUTPUT_DIR / "signal_memberships.csv"
    memberships.to_csv(out_path)
    print(f"\nMemberships gespeichert: {out_path}")

    # Zusammenfassung
    print(f"\n{'=' * 70}")
    print("MEMBERSHIP-VERTEILUNG")
    print(f"{'=' * 70}")
    for col in ["m_ws", "m_trend", "m_ec", "m_latent"]:
        v = memberships[col]
        print(f"  {col:10s}: mean={v.mean():.3f}, std={v.std():.3f}, "
              f"min={v.min():.3f}, max={v.max():.3f}")

    print(f"\nMargin-Verteilung (Diagnostik Übergangsfälle):")
    m = memberships["margin"]
    print(f"  mean={m.mean():.3f}, median={m.median():.3f}, "
          f"min={m.min():.3f}, max={m.max():.3f}")
    print(f"  Topics mit margin < 0.10 (unklar): "
          f"{(m < 0.10).sum()} von {len(m)}")
    print(f"  Topics mit margin < 0.05 (sehr unklar): "
          f"{(m < 0.05).sum()} von {len(m)}")

    return memberships


if __name__ == "__main__":
    run()
