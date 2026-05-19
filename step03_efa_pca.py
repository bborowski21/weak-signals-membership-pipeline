"""
SCHRITT 3: Strukturentdeckung — EFA/PCA für interne Kohärenzprüfung
=====================================================================

Prüft, ob sich die 5 theoretischen Dimensionen aus F2 empirisch
in den 16 (bzw. 15) Indikatoren wiederfinden.

Tests:
  1. KMO & Bartlett → Eignung der Daten für Faktorenanalyse
  2. Scree-Plot + Parallelanalyse → Anzahl der Faktoren
  3. PCA-Ladungsmatrix → Welche Indikatoren laden auf welche Faktoren?
  4. Kohärenz-Assessment → Theoretische vs. empirische Struktur

Hinweis: Indikatoren mit Varianz ≈ 0 (z.B. review_absence bei reinem
Article/Conference-Paper-Datensatz) werden automatisch ausgeschlossen.

Autor: Ben Borowski
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from config import (
    OUTPUT_DIR, EFA_MIN_VARIANCE, EFA_PARALLEL_N_ITER, DIM_COLORS,
    DIM_SHORT_CODES,
)


# =============================================================================
# THEORETISCHES MAPPING (Indikator → Dimension)
# =============================================================================

THEORETICAL_MAPPING = {
    "keyword_volatility": "Epistemische Offenheit",
    "disciplinary_entropy": "Epistemische Offenheit",
    "semantic_incoherence": "Epistemische Offenheit",
    "relative_proportion_inv": "Wahrnehmbarkeit",
    "noise_ratio": "Wahrnehmbarkeit",
    "journal_specificity": "Wahrnehmbarkeit",
    "temporal_novelty": "Entwicklungsphase",
    "terminological_instability": "Entwicklungsphase",
    "review_absence": "Entwicklungsphase",
    "author_concentration": "Diffusion",
    "institutional_concentration": "Diffusion",
    "geographic_concentration": "Diffusion",
    "citation_concentration": "Diffusion",
    "growth_rate": "Wirkungspotenzial",
    "citation_momentum": "Wirkungspotenzial",
    "field_breadth": "Wirkungspotenzial",
}

DIM_ORDER = [
    "Epistemische Offenheit", "Wahrnehmbarkeit",
    "Entwicklungsphase", "Diffusion", "Wirkungspotenzial",
]


# =============================================================================
# 1. KMO & BARTLETT
# =============================================================================

def compute_kmo(corr_matrix: np.ndarray) -> float:
    """
    Kaiser-Meyer-Olkin Maß der Stichprobeneignung.
    KMO > 0.6 = akzeptabel, > 0.7 = gut, > 0.8 = sehr gut.
    """
    n = corr_matrix.shape[0]
    try:
        inv_corr = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr_matrix)

    d = np.diag(1.0 / np.sqrt(np.diag(inv_corr)))
    partial_corr = -d @ inv_corr @ d
    np.fill_diagonal(partial_corr, 0)

    sum_r2 = np.sum(corr_matrix ** 2) - n
    sum_p2 = np.sum(partial_corr ** 2)

    return sum_r2 / (sum_r2 + sum_p2)


def bartlett_test(corr_matrix: np.ndarray, n_obs: int) -> tuple:
    """
    Bartletts Sphärizitätstest.
    H0: Korrelationsmatrix = Einheitsmatrix (keine Faktorstruktur).
    """
    p = corr_matrix.shape[0]
    det = np.linalg.det(corr_matrix)

    if det <= 0:
        return 999.0, 0.0

    chi_sq = -(n_obs - 1 - (2 * p + 5) / 6) * np.log(det)
    df = p * (p - 1) / 2
    p_value = 1 - chi2.cdf(chi_sq, df)

    return chi_sq, p_value


# =============================================================================
# 2. PARALLELANALYSE
# =============================================================================

def parallel_analysis(data: np.ndarray, n_iter: int = 200) -> np.ndarray:
    """
    Horns Parallelanalyse: Generiert Random-Eigenwerte aus zufälligen
    Daten gleicher Dimension. Faktoren werden beibehalten, wenn
    der reale Eigenwert den zufälligen übersteigt.
    """
    n_obs, n_vars = data.shape
    random_eigenvalues = np.zeros((n_iter, n_vars))

    rng = np.random.RandomState(42)
    for i in range(n_iter):
        random_data = rng.normal(size=(n_obs, n_vars))
        random_corr = np.corrcoef(random_data, rowvar=False)
        random_eigenvalues[i] = np.sort(np.linalg.eigvalsh(random_corr))[::-1]

    return random_eigenvalues.mean(axis=0)


# =============================================================================
# 3. PCA
# =============================================================================

def run_pca(z_data: np.ndarray, n_components: int,
            feature_names: list) -> tuple:
    """PCA durchführen und Ladungsmatrix zurückgeben."""
    pca = PCA(n_components=n_components)
    pca.fit(z_data)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    return loadings, pca


# =============================================================================
# 4. KOHÄRENZ-ASSESSMENT
# =============================================================================

def assess_coherence(loadings: pd.DataFrame, valid_indicators: list,
                     threshold: float = 0.4) -> dict:
    """
    Prüft, wie gut die PCA-Komponenten mit den theoretischen
    Dimensionen übereinstimmen.

    Für jede Dimension:
    - Auf welche PC laden die zugehörigen Indikatoren am stärksten?
    - Kohärenz = Anteil der Indikatoren, die auf dieselbe PC laden
    """
    results = {}
    mapping = {k: v for k, v in THEORETICAL_MAPPING.items() if k in valid_indicators}

    for dim in DIM_ORDER:
        dim_indicators = [k for k, v in mapping.items() if v == dim]
        if not dim_indicators:
            results[dim] = {
                "dominant_pc": "N/A",
                "coherence": 0.0,
                "mean_loading": 0.0,
                "indicator_pcs": {},
                "strong_loading": False,
                "note": "Keine validen Indikatoren",
            }
            continue

        primary_pcs = {}
        for ind in dim_indicators:
            abs_loadings = loadings.loc[ind].abs()
            primary_pcs[ind] = abs_loadings.idxmax()

        pc_counts = pd.Series(list(primary_pcs.values())).value_counts()
        dominant_pc = pc_counts.index[0]
        coherence = pc_counts.iloc[0] / len(dim_indicators)
        mean_loading = loadings.loc[dim_indicators, dominant_pc].abs().mean()

        results[dim] = {
            "dominant_pc": dominant_pc,
            "coherence": coherence,
            "mean_loading": mean_loading,
            "indicator_pcs": primary_pcs,
            "strong_loading": mean_loading > threshold,
        }

    return results


# =============================================================================
# VISUALISIERUNGEN
# =============================================================================

def plot_scree(eigenvalues: np.ndarray, parallel_eigs: np.ndarray,
               output_path: str):
    """Scree-Plot mit Parallelanalyse-Schwellenwert."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    n = len(eigenvalues)
    x = range(1, n + 1)

    ax.plot(x, eigenvalues, "bo-", label="Eigenwerte (real)", markersize=8)
    ax.plot(x, parallel_eigs[:n], "r--", label="Parallelanalyse-Schwelle", alpha=0.7)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Kaiser-Kriterium (λ=1)")

    n_factors = sum(eigenvalues > parallel_eigs[:n])
    for i in range(n_factors):
        ax.fill_between([i + 0.8, i + 1.2], 0, eigenvalues[i], alpha=0.2, color="blue")

    ax.set_xlabel("Faktor-Nummer", fontsize=12)
    ax.set_ylabel("Eigenwert", fontsize=12)
    ax.set_title(f"Scree-Plot mit Parallelanalyse — {n_factors} Faktoren beibehalten",
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(list(x))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scree-Plot gespeichert: {output_path}")


def plot_loading_matrix(loadings: pd.DataFrame, valid_indicators: list,
                        output_path: str):
    """Heatmap der Faktorladungen, gruppiert nach theoretischen Dimensionen."""
    mapping = {k: v for k, v in THEORETICAL_MAPPING.items() if k in valid_indicators}
    sorted_indicators = sorted(
        loadings.index,
        key=lambda x: DIM_ORDER.index(mapping.get(x, "Wirkungspotenzial"))
    )
    loadings_sorted = loadings.loc[sorted_indicators]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    sns.heatmap(
        loadings_sorted, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-0.8, vmax=0.8,
        ax=ax, linewidths=0.5,
    )

    # Dimensionsgruppen markieren — Kuerzel in Dimensionsfarbe (vgl. Abb. 3.2)
    current_dim = None
    dim_block_start = 0
    for i, indicator in enumerate(sorted_indicators):
        dim = mapping.get(indicator, "")
        if dim != current_dim:
            # Annotation des vorhergehenden Blocks (zentriert ueber die Zeilen)
            if current_dim is not None:
                ax.axhline(y=i, color="black", linewidth=2)
                center_y = (dim_block_start + i) / 2.0
                color = DIM_COLORS.get(current_dim, "#999999")
                ax.annotate(
                    DIM_SHORT_CODES.get(current_dim, current_dim[:12]),
                    xy=(-0.34, center_y),
                    xycoords=("axes fraction", "data"),
                    fontsize=16, fontweight="bold", color=color,
                    va="center", ha="center",
                )
            current_dim = dim
            dim_block_start = i

    # Letzten Block annotieren
    if current_dim is not None:
        center_y = (dim_block_start + len(sorted_indicators)) / 2.0
        color = DIM_COLORS.get(current_dim, "#999999")
        ax.annotate(
            DIM_SHORT_CODES.get(current_dim, current_dim[:12]),
            xy=(-0.34, center_y),
            xycoords=("axes fraction", "data"),
            fontsize=16, fontweight="bold", color=color,
            va="center", ha="center",
        )

    ax.set_title("PCA-Ladungsmatrix — Indikatoren × Hauptkomponenten\n"
                 "(Theoretische Dimensionszuordnung links)", fontsize=13)
    ax.set_ylabel("Indikatoren (gruppiert nach F2-Dimension)")
    ax.set_xlabel("Hauptkomponenten")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Ladungsmatrix gespeichert: {output_path}")


def plot_correlation_matrix(indicator_df: pd.DataFrame, valid_indicators: list,
                            output_path: str):
    """Korrelationsmatrix, gruppiert nach theoretischen Dimensionen."""
    mapping = {k: v for k, v in THEORETICAL_MAPPING.items() if k in valid_indicators}
    sorted_cols = sorted(
        indicator_df.columns,
        key=lambda x: DIM_ORDER.index(mapping.get(x, "Wirkungspotenzial"))
    )

    corr = indicator_df[sorted_cols].corr()
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax, linewidths=0.5, square=True,
    )

    # Dimensionsgrenzen einzeichnen
    boundaries = []
    current_dim = mapping.get(sorted_cols[0], "")
    for i, col in enumerate(sorted_cols):
        if mapping.get(col, "") != current_dim:
            boundaries.append(i)
            current_dim = mapping.get(col, "")

    for b in boundaries:
        ax.axhline(y=b, color="black", linewidth=2)
        ax.axvline(x=b, color="black", linewidth=2)

    ax.set_title("Indikator-Korrelationsmatrix\n"
                 "(gruppiert nach F2-Dimensionen)", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Korrelationsmatrix gespeichert: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def run():
    print("=" * 70)
    print("SCHRITT 3: STRUKTURENTDECKUNG — EFA/PCA")
    print("=" * 70)

    # --- Daten laden ---
    indicator_df = pd.read_csv(OUTPUT_DIR / "indicators_16.csv", index_col="topic")
    print(f"Indikator-Matrix geladen: {indicator_df.shape}")

    # --- Zero-Varianz-Indikatoren entfernen ---
    variances = indicator_df.std()
    low_var = variances[variances < EFA_MIN_VARIANCE].index.tolist()
    if low_var:
        print(f"\n  Ausgeschlossene Indikatoren (Varianz ≈ 0): {low_var}")
        indicator_df = indicator_df.drop(columns=low_var)
        print(f"  Verbleibende Indikatoren: {indicator_df.shape[1]}")

    valid_indicators = indicator_df.columns.tolist()

    # --- z-Standardisierung ---
    scaler = StandardScaler()
    z_data = scaler.fit_transform(indicator_df)
    z_df = pd.DataFrame(z_data, index=indicator_df.index, columns=indicator_df.columns)

    # --- 1. KMO & Bartlett ---
    print("\n--- Stichprobeneignung ---")
    corr = np.corrcoef(z_data, rowvar=False)
    kmo = compute_kmo(corr)
    chi_sq, p_val = bartlett_test(corr, n_obs=len(indicator_df))
    kmo_label = ("sehr gut" if kmo > 0.8 else "gut" if kmo > 0.7
                 else "akzeptabel" if kmo > 0.6 else "problematisch")
    print(f"  KMO: {kmo:.3f} ({kmo_label})")
    print(f"  Bartlett: χ²={chi_sq:.1f}, p={p_val:.6f} "
          f"({'signifikant' if p_val < 0.05 else 'nicht signifikant'})")

    # --- 2. Eigenwerte & Scree ---
    print("\n--- Eigenwertanalyse ---")
    eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]
    parallel_eigs = parallel_analysis(z_data, n_iter=EFA_PARALLEL_N_ITER)

    n_kaiser = int(sum(eigenvalues > 1.0))
    n_parallel = int(sum(eigenvalues > parallel_eigs[:len(eigenvalues)]))

    print(f"  Eigenwerte: {', '.join(f'{e:.2f}' for e in eigenvalues[:8])}")
    print(f"  Kaiser-Kriterium (λ>1): {n_kaiser} Faktoren")
    print(f"  Parallelanalyse: {n_parallel} Faktoren")

    var_total = sum(eigenvalues)
    var_5 = sum(eigenvalues[:5]) / var_total * 100
    print(f"  Varianz erklärt (erste 5): {var_5:.1f}%")

    plot_scree(eigenvalues, parallel_eigs, str(OUTPUT_DIR / "scree_plot.png"))

    # --- 3. PCA mit 5 Komponenten (theoriegetrieben) ---
    print("\n--- PCA mit 5 Komponenten (theoriegetrieben) ---")
    loadings_5, pca_5 = run_pca(z_data, n_components=5,
                                 feature_names=valid_indicators)
    var_explained = pca_5.explained_variance_ratio_
    print(f"  Varianz pro PC: {', '.join(f'{v:.1%}' for v in var_explained)}")
    print(f"  Kumulativ: {sum(var_explained):.1%}")

    plot_loading_matrix(loadings_5, valid_indicators,
                        str(OUTPUT_DIR / "loading_matrix.png"))

    # --- Auch datengetrieben ---
    n_retain = max(n_parallel, 3)
    if n_retain != 5:
        print(f"\n--- PCA mit {n_retain} Komponenten (datengetrieben) ---")
        loadings_n, pca_n = run_pca(z_data, n_components=n_retain,
                                     feature_names=valid_indicators)
        var_n = pca_n.explained_variance_ratio_
        print(f"  Varianz pro PC: {', '.join(f'{v:.1%}' for v in var_n)}")
        print(f"  Kumulativ: {sum(var_n):.1%}")
        plot_loading_matrix(loadings_n, valid_indicators,
                            str(OUTPUT_DIR / f"loading_matrix_{n_retain}pc.png"))

    # --- 4. Kohärenz-Assessment ---
    print("\n--- Theorie-Empirie-Kohärenz ---")
    coherence = assess_coherence(loadings_5, valid_indicators)

    for dim, info in coherence.items():
        if info["dominant_pc"] == "N/A":
            print(f"  - {dim:25s}: {info.get('note', 'keine Daten')}")
            continue

        status = ("✓" if info["coherence"] >= 0.67 else
                  "~" if info["coherence"] >= 0.5 else "✗")
        print(f"  {status} {dim:25s} → {info['dominant_pc']} "
              f"(Kohärenz={info['coherence']:.0%}, "
              f"|Ladung|={info['mean_loading']:.2f})")

        for ind, pc in info["indicator_pcs"].items():
            loading_val = loadings_5.loc[ind, info["dominant_pc"]]
            match = "  " if pc == info["dominant_pc"] else f"→{pc}"
            print(f"      {ind:35s}: {loading_val:+.3f} {match}")

    # --- 5. Korrelationsmatrix ---
    print("\n--- Korrelationsanalyse ---")
    plot_correlation_matrix(indicator_df, valid_indicators,
                            str(OUTPUT_DIR / "correlation_matrix.png"))

    # Intra-Dimensions-Korrelationen
    for dim in DIM_ORDER:
        dim_inds = [k for k, v in THEORETICAL_MAPPING.items()
                    if v == dim and k in valid_indicators]
        if len(dim_inds) < 2:
            continue
        sub_corr = indicator_df[dim_inds].corr()
        mask = ~np.eye(len(dim_inds), dtype=bool)
        mean_corr = sub_corr.values[mask].mean()
        print(f"  {dim:25s}: r̄ intra-dim = {mean_corr:+.3f}")

    # ===== ZUSAMMENFASSUNG =====
    n_coherent = sum(1 for v in coherence.values()
                     if v.get("coherence", 0) >= 0.67)
    n_partial = sum(1 for v in coherence.values()
                    if 0.5 <= v.get("coherence", 0) < 0.67)

    print(f"\n{'=' * 70}")
    print("STRUKTURBEWERTUNG")
    print(f"{'=' * 70}")
    print(f"  Dimensionen mit starker Kohärenz (≥67%): {n_coherent}/5")
    print(f"  Dimensionen mit partieller Kohärenz (≥50%): {n_partial}/5")
    print(f"  KMO: {kmo:.3f}")
    print(f"  Faktoren (Parallelanalyse): {n_parallel}")
    print(f"  Faktoren (Kaiser): {n_kaiser}")
    print(f"  Theoretisches Modell: 5 Dimensionen")

    if n_parallel < 5:
        print(f"\n  → Empirisch genügen {n_parallel} Faktoren. "
              "Einige Dimensionen verschmelzen ggf. empirisch.")
    elif n_parallel > 5:
        print(f"\n  → Empirisch entstehen {n_parallel} Faktoren. "
              "Einige Dimensionen differenzieren sich weiter aus.")

    # ===== SPEICHERN =====
    summary = {
        "kmo": float(kmo),
        "bartlett_chi2": float(chi_sq),
        "bartlett_p": float(p_val),
        "n_kaiser": n_kaiser,
        "n_parallel": n_parallel,
        "var_explained_5pc": float(sum(var_explained)),
        "eigenvalues": [float(e) for e in eigenvalues],
        "excluded_indicators": low_var,
        "n_valid_indicators": len(valid_indicators),
    }

    with open(OUTPUT_DIR / "efa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    loadings_5.to_csv(OUTPUT_DIR / "pca_loadings.csv")

    print(f"\nAlles gespeichert in {OUTPUT_DIR}/")
    return summary, loadings_5, coherence


if __name__ == "__main__":
    run()
