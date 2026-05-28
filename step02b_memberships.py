
import pandas as pd
import numpy as np

from config import (
    OUTPUT_DIR,
    MEMBERSHIP_SIGMOID_K,
    MEMBERSHIP_LAMBDA_WP,
    INDICATOR_DIMENSIONS,
    DIM_NAMES,
)

CORE_DIMS = ["Epistemische Offenheit", "Wahrnehmbarkeit",
             "Entwicklungsphase", "Diffusion"]
WP_DIM = "Wirkungspotenzial"

EC_SUBINDICATORS = ["temporal_novelty", "growth_rate",
                    "citation_momentum", "field_breadth"]


def robust_z(series: pd.Series, eps: float = 1e-9) -> pd.Series:
    med = series.median()
    q75 = series.quantile(0.75)
    q25 = series.quantile(0.25)
    iqr = q75 - q25
    return (series - med) / max(iqr / 2.0, eps)


def sigmoid(x: pd.Series, k: float = 1.0) -> pd.Series:
    return 1.0 / (1.0 + np.exp(-k * x))


def compute_memberships(
    indicator_df: pd.DataFrame,
    dim_scores: pd.DataFrame,
    lambda_wp: float = MEMBERSHIP_LAMBDA_WP,
    k: float = MEMBERSHIP_SIGMOID_K,
) -> pd.DataFrame:
    z_dim = dim_scores.apply(robust_z, axis=0)

    missing_ec = [c for c in EC_SUBINDICATORS if c not in indicator_df.columns]
    if missing_ec:
        raise KeyError(
            f"EC-Subindikatoren fehlen in indicator_df: {missing_ec}. "
            f"Verfügbare Spalten: {list(indicator_df.columns)}"
        )
    z_ec = indicator_df[EC_SUBINDICATORS].apply(robust_z, axis=0)

    z_core_mean = z_dim[CORE_DIMS].mean(axis=1)
    z_wp        = z_dim[WP_DIM]
    z_ec_mean   = z_ec.mean(axis=1)
    z_all_mean  = z_dim[CORE_DIMS + [WP_DIM]].mean(axis=1)

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

    sorted_vals = np.sort(memberships.values, axis=1)
    memberships["margin"] = sorted_vals[:, -1] - sorted_vals[:, -2]

    return memberships


def run():
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

    out_path = OUTPUT_DIR / "signal_memberships.csv"
    memberships.to_csv(out_path)
    print(f"\nMemberships gespeichert: {out_path}")

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
