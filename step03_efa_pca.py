"""
step03_efa_pca.py — Schritt 3: Strukturentdeckung durch explorative Faktorenanalyse (EFA).

Pipeline V3 (EFA-Umstellung, Juni 2026):
- Hauptverfahren: gemeinsame Faktorenanalyse (Common Factor), Extraktion Minimum-Residual
  (minres), Rotation oblique (Oblimin). Begruendung: Das Framework beschreibt die fuenf
  Dimensionen als konstitutives Wechselverhaeltnis; orthogonale Rotation widerspraeche
  dieser Annahme. Die Faktorkorrelationsmatrix (Phi) wird als eigenstaendiger Befund
  dokumentiert.
- Faktorzahlbestimmung: Horns Parallelanalyse auf den Eigenwerten der unreduzierten
  Korrelationsmatrix (klassische Spezifikation; Mittelwert-Schwelle, 200 Replikationen,
  konsekutive Zaehlung). Eine alternative Spezifikation auf SMC-reduzierten
  Common-Factor-Eigenwerten wird diagnostisch mitberechnet und im Summary dokumentiert;
  sie degeneriert in der vorliegenden Datenlage (Ueberextraktion bei grossem n) und
  traegt daher nicht die Faktorzahlentscheidung.
- PCA: nur noch als ausdruecklich etikettierter Robustheitscheck (pca_loadings.csv),
  nicht als Hauptverfahren.

Frueheres Verhalten (V2): PCA als Realisierung der "konzeptionell als EFA angelegten"
Pruefung. Der V2-Stand liegt im Backup sbert_pipeline_membership_BACKUP_2026-06-09.
"""

import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2
from factor_analyzer import FactorAnalyzer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from config import (
    OUTPUT_DIR, EFA_MIN_VARIANCE, EFA_PARALLEL_N_ITER, DIM_COLORS,
    DIM_SHORT_CODES, FIG_DPI,
)


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

EFA_METHOD = "minres"      # Common-Factor-Extraktion (Minimum-Residual)
EFA_ROTATION = "oblimin"   # oblique Rotation, theoriekonsistent (Interdependenz)


# ---------------------------------------------------------------------------
# Stichprobeneignung
# ---------------------------------------------------------------------------

def compute_kmo(corr_matrix: np.ndarray) -> float:
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
    p = corr_matrix.shape[0]
    det = np.linalg.det(corr_matrix)

    if det <= 0:
        return 999.0, 0.0

    chi_sq = -(n_obs - 1 - (2 * p + 5) / 6) * np.log(det)
    df = p * (p - 1) / 2
    p_value = 1 - chi2.cdf(chi_sq, df)

    return chi_sq, p_value


# ---------------------------------------------------------------------------
# Faktorzahlbestimmung
# ---------------------------------------------------------------------------

def smc(corr: np.ndarray) -> np.ndarray:
    """Squared Multiple Correlations (Communality-Startschaetzung)."""
    try:
        inv_corr = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr)
    return 1.0 - 1.0 / np.diag(inv_corr)


def common_factor_eigenvalues(corr: np.ndarray) -> np.ndarray:
    """Eigenwerte der SMC-reduzierten Korrelationsmatrix."""
    reduced = corr.copy()
    np.fill_diagonal(reduced, smc(corr))
    return np.sort(np.linalg.eigvalsh(reduced))[::-1]


def parallel_analysis(n_obs: int, n_vars: int, basis: str = "full",
                      n_iter: int = 200, seed: int = 42) -> tuple:
    """Horn-Parallelanalyse.

    basis='full': Eigenwerte der unreduzierten Korrelationsmatrix
                  (klassische Spezifikation; traegt die Faktorzahlentscheidung).
    basis='cf':   Eigenwerte der SMC-reduzierten Matrix (nur Diagnostik).
    Rueckgabe: (Mittelwert-Schwellen, P95-Schwellen).
    """
    rng = np.random.RandomState(seed)
    eig = np.zeros((n_iter, n_vars))
    for i in range(n_iter):
        rnd = rng.normal(size=(n_obs, n_vars))
        rc = np.corrcoef(rnd, rowvar=False)
        if basis == "cf":
            eig[i] = common_factor_eigenvalues(rc)
        else:
            eig[i] = np.sort(np.linalg.eigvalsh(rc))[::-1]
    return eig.mean(axis=0), np.percentile(eig, 95, axis=0)


def n_factors_consecutive(eigenvalues: np.ndarray, thresholds: np.ndarray) -> int:
    """Konsekutive Zaehlung: Abbruch beim ersten Eigenwert <= Schwelle."""
    n = 0
    for e, t in zip(eigenvalues, thresholds):
        if e > t:
            n += 1
        else:
            break
    return n


# ---------------------------------------------------------------------------
# Extraktion
# ---------------------------------------------------------------------------

def run_efa(z_df: pd.DataFrame, n_factors: int) -> dict:
    """Gemeinsame Faktorenanalyse: minres-Extraktion, Oblimin-Rotation."""
    fa = FactorAnalyzer(n_factors=n_factors, method=EFA_METHOD, rotation=EFA_ROTATION)
    fa.fit(z_df.values)

    cols = [f"F{i+1}" for i in range(n_factors)]
    pattern = pd.DataFrame(fa.loadings_, index=z_df.columns, columns=cols)
    phi = pd.DataFrame(getattr(fa, "phi_", np.eye(n_factors)), index=cols, columns=cols)
    communalities = pd.Series(fa.get_communalities(), index=z_df.columns,
                              name="communality")
    # Gemeinsame Varianz: rotationsinvariant ueber Communalities
    var_common = float(communalities.sum() / len(z_df.columns))

    return {"fa": fa, "pattern": pattern, "phi": phi,
            "communalities": communalities, "var_common": var_common}


def run_pca(z_data: np.ndarray, n_components: int,
            feature_names: list) -> tuple:
    """PCA — nur Robustheitscheck, kein Hauptverfahren (vgl. Modul-Docstring)."""
    pca = PCA(n_components=n_components)
    pca.fit(z_data)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    return loadings, pca


def assess_coherence(loadings: pd.DataFrame, valid_indicators: list,
                     threshold: float = 0.4) -> dict:
    """Theorie-Empirie-Kohaerenz: Anteil der Indikatoren einer Dimension mit
    Primaerladung auf demselben Faktor."""
    results = {}
    mapping = {k: v for k, v in THEORETICAL_MAPPING.items() if k in valid_indicators}

    for dim in DIM_ORDER:
        dim_indicators = [k for k, v in mapping.items() if v == dim]
        if not dim_indicators:
            results[dim] = {
                "dominant_factor": "N/A",
                "coherence": 0.0,
                "mean_loading": 0.0,
                "indicator_factors": {},
                "strong_loading": False,
                "note": "Keine validen Indikatoren",
            }
            continue

        primary = {}
        for ind in dim_indicators:
            abs_loadings = loadings.loc[ind].abs()
            primary[ind] = abs_loadings.idxmax()

        counts = pd.Series(list(primary.values())).value_counts()
        dominant = counts.index[0]
        coherence = counts.iloc[0] / len(dim_indicators)
        mean_loading = loadings.loc[dim_indicators, dominant].abs().mean()

        results[dim] = {
            "dominant_factor": dominant,
            "coherence": coherence,
            "mean_loading": mean_loading,
            "indicator_factors": primary,
            "strong_loading": mean_loading > threshold,
        }

    return results


# ---------------------------------------------------------------------------
# Visualisierung
# ---------------------------------------------------------------------------

def plot_scree(eigenvalues: np.ndarray, parallel_eigs: np.ndarray,
               output_path: str):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    n = len(eigenvalues)
    x = range(1, n + 1)

    ax.plot(x, eigenvalues, "bo-", label="Eigenwerte (Korrelationsmatrix)",
            markersize=8)
    ax.plot(x, parallel_eigs[:n], "r--",
            label="Parallelanalyse-Schwelle (Mittelwert)", alpha=0.7)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5,
               label="Kaiser-Kriterium (λ=1)")

    n_factors = n_factors_consecutive(eigenvalues, parallel_eigs[:n])
    for i in range(n_factors):
        ax.fill_between([i + 0.8, i + 1.2], 0, eigenvalues[i], alpha=0.2,
                        color="blue")

    ax.set_xlabel("Faktor-Nummer", fontsize=12)
    ax.set_ylabel("Eigenwert", fontsize=12)
    ax.set_title(f"Scree-Plot mit Parallelanalyse — {n_factors} Faktoren beibehalten",
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(list(x))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Scree-Plot gespeichert: {output_path}")


def plot_loading_matrix(loadings: pd.DataFrame, valid_indicators: list,
                        output_path: str, title: str = None):
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

    current_dim = None
    dim_block_start = 0
    for i, indicator in enumerate(sorted_indicators):
        dim = mapping.get(indicator, "")
        if dim != current_dim:
            if current_dim is not None:
                ax.axhline(y=i, color="black", linewidth=2)
                center_y = (dim_block_start + i) / 2.0
                color = DIM_COLORS.get(current_dim, "#999999")
                ax.annotate(
                    DIM_SHORT_CODES.get(current_dim, current_dim[:12]),
                    xy=(-0.43, center_y),
                    xycoords=("axes fraction", "data"),
                    fontsize=16, fontweight="bold", color=color,
                    va="center", ha="center",
                )
            current_dim = dim
            dim_block_start = i

    if current_dim is not None:
        center_y = (dim_block_start + len(sorted_indicators)) / 2.0
        color = DIM_COLORS.get(current_dim, "#999999")
        ax.annotate(
            DIM_SHORT_CODES.get(current_dim, current_dim[:12]),
            xy=(-0.43, center_y),
            xycoords=("axes fraction", "data"),
            fontsize=16, fontweight="bold", color=color,
            va="center", ha="center",
        )

    if title is None:
        title = ("EFA-Pattern-Matrix — Indikatoren × Faktoren\n"
                 "(minres, Oblimin; theoretische Dimensionszuordnung links)")
    ax.set_title(title, fontsize=13)
    # labelpad schiebt das ylabel links an der Dimensionskuerzel-Spalte
    # (Annotationen bei x=-0.34, axes fraction) vorbei — sonst schneiden
    # sich Beschriftung und Kuerzel.
    ax.set_ylabel("Indikatoren (gruppiert nach F2-Dimension)", labelpad=165)
    ax.set_xlabel("Faktoren (Pattern-Ladungen)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Pattern-Matrix gespeichert: {output_path}")


def plot_phi_matrix(phi: pd.DataFrame, output_path: str):
    """Faktorkorrelationsmatrix (Oblimin) als kompakte Heatmap."""
    k = phi.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(1.4 * k + 2, 1.2 * k + 1.5))

    mask = np.triu(np.ones_like(phi.values, dtype=bool), k=1)
    sns.heatmap(
        phi, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-0.6, vmax=0.6,
        ax=ax, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Faktorkorrelationsmatrix $\\Phi$ (Oblimin)", fontsize=13)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Phi-Matrix gespeichert: {output_path}")


def plot_correlation_matrix(indicator_df: pd.DataFrame, valid_indicators: list,
                            output_path: str):
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
    plt.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Korrelationsmatrix gespeichert: {output_path}")


# ---------------------------------------------------------------------------
# Hauptlauf
# ---------------------------------------------------------------------------

def run():
    print("=" * 70)
    print("SCHRITT 3: STRUKTURENTDECKUNG — EFA (minres, Oblimin)")
    print("=" * 70)

    indicator_df = pd.read_csv(OUTPUT_DIR / "indicators_16.csv", index_col="topic")
    print(f"Indikator-Matrix geladen: {indicator_df.shape}")

    variances = indicator_df.std()
    low_var = variances[variances < EFA_MIN_VARIANCE].index.tolist()
    if low_var:
        print(f"\n  Ausgeschlossene Indikatoren (Varianz ≈ 0): {low_var}")
        indicator_df = indicator_df.drop(columns=low_var)
        print(f"  Verbleibende Indikatoren: {indicator_df.shape[1]}")

    valid_indicators = indicator_df.columns.tolist()

    scaler = StandardScaler()
    z_data = scaler.fit_transform(indicator_df)
    z_df = pd.DataFrame(z_data, index=indicator_df.index, columns=indicator_df.columns)
    n_obs = len(indicator_df)

    # --- Stichprobeneignung (extraktionsunabhaengig) ---
    print("\n--- Stichprobeneignung ---")
    corr = np.corrcoef(z_data, rowvar=False)
    kmo = compute_kmo(corr)
    chi_sq, p_val = bartlett_test(corr, n_obs=n_obs)
    kmo_label = ("sehr gut" if kmo > 0.8 else "gut" if kmo > 0.7
                 else "akzeptabel" if kmo > 0.6 else "problematisch")
    print(f"  KMO: {kmo:.3f} ({kmo_label})")
    print(f"  Bartlett: χ²={chi_sq:.1f}, p={p_val:.6f} "
          f"({'signifikant' if p_val < 0.05 else 'nicht signifikant'})")

    # --- Faktorzahlbestimmung ---
    print("\n--- Eigenwertanalyse & Parallelanalyse ---")
    eigenvalues = np.sort(np.linalg.eigvalsh(corr))[::-1]
    pa_mean, pa_p95 = parallel_analysis(n_obs, len(valid_indicators),
                                        basis="full", n_iter=EFA_PARALLEL_N_ITER)

    n_kaiser = int(sum(eigenvalues > 1.0))
    n_parallel = n_factors_consecutive(eigenvalues, pa_mean)
    n_parallel_p95 = n_factors_consecutive(eigenvalues, pa_p95)

    print(f"  Eigenwerte: {', '.join(f'{e:.2f}' for e in eigenvalues[:8])}")
    print(f"  Kaiser-Kriterium (λ>1): {n_kaiser} Faktoren")
    print(f"  Parallelanalyse (Mittelwert): {n_parallel} Faktoren "
          f"(P95: {n_parallel_p95})")

    # Diagnostik: SMC-reduzierte PA-Variante (traegt NICHT die Entscheidung;
    # degeneriert bei grossem n — Dokumentation fuer Methoden-Fussnote)
    ev_cf = common_factor_eigenvalues(corr)
    pa_cf_mean, pa_cf_p95 = parallel_analysis(n_obs, len(valid_indicators),
                                              basis="cf",
                                              n_iter=EFA_PARALLEL_N_ITER)
    n_pa_cf_mean = n_factors_consecutive(ev_cf, pa_cf_mean)
    n_pa_cf_p95 = n_factors_consecutive(ev_cf, pa_cf_p95)
    print(f"  [Diagnostik] PA auf SMC-reduzierten Eigenwerten: "
          f"{n_pa_cf_mean} (Mittelwert) / {n_pa_cf_p95} (P95)"
          + ("  → degeneriert, nicht entscheidungstragend"
             if n_pa_cf_mean > 8 else ""))

    plot_scree(eigenvalues, pa_mean, str(OUTPUT_DIR / "scree_plot.png"))

    # --- EFA: Referenzloesung (PA) + symmetrische Vergleichsloesung ---
    n_reference = max(n_parallel, 3)
    n_compare = 4 if n_reference == 5 else 5

    solutions = {}
    for k, role in [(n_reference, "Referenz"), (n_compare, "Vergleich")]:
        print(f"\n--- EFA mit {k} Faktoren ({role}; {EFA_METHOD}, {EFA_ROTATION}) ---")
        res = run_efa(z_df, n_factors=k)
        solutions[k] = res

        print(f"  Gemeinsame Varianz (Communalities/p): {res['var_common']:.1%}")
        off = res["phi"].values[~np.eye(k, dtype=bool)]
        print(f"  Faktorkorrelationen: |r| max={np.max(np.abs(off)):.3f}, "
              f"mean={np.mean(np.abs(off)):.3f}")

        res["pattern"].to_csv(OUTPUT_DIR / f"efa_pattern_{k}f.csv")
        res["phi"].to_csv(OUTPUT_DIR / f"efa_phi_{k}f.csv")
        res["communalities"].to_csv(OUTPUT_DIR / f"efa_communalities_{k}f.csv")

        plot_loading_matrix(
            res["pattern"], valid_indicators,
            str(OUTPUT_DIR / f"loading_matrix_{k}f.png"),
            title=(f"EFA-Pattern-Matrix — {k}-Faktoren-Lösung ({role})\n"
                   f"(minres, Oblimin; theoretische Dimensionszuordnung links)"),
        )
        plot_phi_matrix(res["phi"], str(OUTPUT_DIR / f"phi_matrix_{k}f.png"))

    ref = solutions[n_reference]

    # --- Theorie-Empirie-Kohaerenz (auf der Referenzloesung) ---
    print("\n--- Theorie-Empirie-Kohärenz (Referenzlösung) ---")
    coherence = assess_coherence(ref["pattern"], valid_indicators)

    for dim, info in coherence.items():
        if info["dominant_factor"] == "N/A":
            print(f"  - {dim:25s}: {info.get('note', 'keine Daten')}")
            continue

        status = ("✓" if info["coherence"] >= 0.67 else
                  "~" if info["coherence"] >= 0.5 else "✗")
        print(f"  {status} {dim:25s} → {info['dominant_factor']} "
              f"(Kohärenz={info['coherence']:.0%}, "
              f"|Ladung|={info['mean_loading']:.2f})")

        for ind, fac in info["indicator_factors"].items():
            loading_val = ref["pattern"].loc[ind, info["dominant_factor"]]
            match = "  " if fac == info["dominant_factor"] else f"→{fac}"
            print(f"      {ind:35s}: {loading_val:+.3f} {match}")

    # --- Schwach determinierte Indikatoren ---
    weak = ref["communalities"][ref["communalities"] < 0.25]
    if len(weak):
        print("\n--- Schwach determinierte Indikatoren (h² < 0,25) ---")
        for ind, h2 in weak.items():
            print(f"  {ind:35s}: h² = {h2:.3f}")

    # --- Korrelationsanalyse (von der Extraktion unabhaengig) ---
    print("\n--- Korrelationsanalyse ---")
    plot_correlation_matrix(indicator_df, valid_indicators,
                            str(OUTPUT_DIR / "correlation_matrix.png"))

    for dim in DIM_ORDER:
        dim_inds = [k for k, v in THEORETICAL_MAPPING.items()
                    if v == dim and k in valid_indicators]
        if len(dim_inds) < 2:
            continue
        sub_corr = indicator_df[dim_inds].corr()
        mask = ~np.eye(len(dim_inds), dtype=bool)
        mean_corr = sub_corr.values[mask].mean()
        print(f"  {dim:25s}: r̄ intra-dim = {mean_corr:+.3f}")

    # --- PCA-Robustheitscheck (etikettiert, kein Hauptverfahren) ---
    print("\n--- PCA-Robustheitscheck (nur Vergleichszwecke) ---")
    loadings_pca, pca_5 = run_pca(z_data, n_components=5,
                                  feature_names=valid_indicators)
    var_explained = pca_5.explained_variance_ratio_
    print(f"  PCA-Totalvarianz (5 PC, kumulativ): {sum(var_explained):.1%} "
          f"— nicht mit der gemeinsamen Varianz der EFA vergleichbar")
    loadings_pca.to_csv(OUTPUT_DIR / "pca_loadings.csv")

    # --- Strukturbewertung ---
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
    print(f"  Referenzlösung: {n_reference} Faktoren "
          f"({EFA_METHOD}, {EFA_ROTATION})")
    print(f"  Theoretisches Modell: 5 Dimensionen")

    if n_parallel < 5:
        print(f"\n  → Empirisch genügen {n_parallel} Faktoren. "
              "Einige Dimensionen verschmelzen ggf. empirisch.")
    elif n_parallel > 5:
        print(f"\n  → Empirisch entstehen {n_parallel} Faktoren. "
              "Einige Dimensionen differenzieren sich weiter aus.")

    summary = {
        "method": f"EFA ({EFA_METHOD}, {EFA_ROTATION})",
        "kmo": float(kmo),
        "bartlett_chi2": float(chi_sq),
        "bartlett_p": float(p_val),
        "n_kaiser": n_kaiser,
        "n_parallel": n_parallel,
        "n_parallel_p95": n_parallel_p95,
        "n_parallel_cf_mean_diagnostic": n_pa_cf_mean,
        "n_parallel_cf_p95_diagnostic": n_pa_cf_p95,
        "n_reference": int(n_reference),
        "n_compare": int(n_compare),
        "var_common_reference": ref["var_common"],
        "var_common_compare": solutions[n_compare]["var_common"],
        "phi_reference": ref["phi"].values.tolist(),
        "phi_max_abs_offdiag_reference": float(np.max(np.abs(
            ref["phi"].values[~np.eye(n_reference, dtype=bool)]))),
        "communalities_reference": {k: float(v) for k, v
                                    in ref["communalities"].items()},
        "pca_var_5pc_robustness_check": float(sum(var_explained)),
        "eigenvalues": [float(e) for e in eigenvalues],
        "eigenvalues_cf_diagnostic": [float(e) for e in ev_cf],
        "pa_thresholds_mean": [float(e) for e in pa_mean],
        "pa_thresholds_p95": [float(e) for e in pa_p95],
        "excluded_indicators": low_var,
        "n_valid_indicators": len(valid_indicators),
    }

    with open(OUTPUT_DIR / "efa_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAlles gespeichert in {OUTPUT_DIR}/")
    return summary, ref["pattern"], coherence


if __name__ == "__main__":
    run()
