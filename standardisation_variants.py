# -*- coding: utf-8 -*-
"""Zusatzprüfung: Wie stark hängen die Memberships von der Standardisierung der 16 Indikatoren ab?

Hintergrund: Die Thesis (Gl. 3.6) beschreibt eine robuste Standardisierung (Median, IQR/2) aller 16 Indikatoren
vor der Dimensionsaggregation. Der Pipeline-Code (step02_indicators.compute_dimension_scores) standardisiert die
Indikatoren klassisch (StandardScaler: Mittelwert, SD) und wendet robust-z erst auf die Dimensionsscores und die
vier EC-Subindikatoren an (step02_memberships.compute_memberships). Hier werden drei Varianten verglichen:

  A  Code wie implementiert (Referenz, reproduziert signal_memberships.csv)
  B  robust-z auf den 16 Indikatoren, Dimensionsmittel, dann wie im Code robust-z auf Dimensionsebene und Sigmoid
  C  robust-z auf den 16 Indikatoren, Dimensionsmittel, Sigmoid direkt (ohne zweite Standardisierung)

Aufruf: python3 standardisation_variants.py --root work --code /pfad/zur/pipeline
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd

MEMB = ["m_ws", "m_trend", "m_ec", "m_latent"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--code", default=None)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    sys.path.insert(0, str(Path(a.code).resolve() if a.code else root))
    from step02_indicators import compute_dimension_scores
    from step02_memberships import compute_memberships, robust_z, sigmoid, CORE_DIMS, WP_DIM, EC_SUBINDICATORS
    from config import INDICATOR_DIMENSIONS, MEMBERSHIP_LAMBDA_WP, MEMBERSHIP_SIGMOID_K
    from scipy.stats import spearmanr

    def dim_scores_robust(ind):
        z = ind.apply(robust_z, axis=0)
        out = pd.DataFrame(index=ind.index)
        for d, cols in INDICATOR_DIMENSIONS.items():
            valid = [c for c in cols if c in z.columns and z[c].std() > 0.01]
            out[d] = z[valid].mean(axis=1) if valid else 0.0
        return out

    def memberships_direct(ind, dims):        # Variante C: keine zweite Standardisierung auf Dimensionsebene
        z_ec = ind[EC_SUBINDICATORS].apply(robust_z, axis=0)
        core = dims[CORE_DIMS].mean(axis=1); wp = dims[WP_DIM]; allm = dims[CORE_DIMS + [WP_DIM]].mean(axis=1)
        k, lam = MEMBERSHIP_SIGMOID_K, MEMBERSHIP_LAMBDA_WP
        m = pd.DataFrame({"m_ws": sigmoid(core, k), "m_trend": sigmoid(-core + lam * wp, k),
                          "m_ec": sigmoid(z_ec.mean(axis=1), k), "m_latent": sigmoid(-allm, k)}, index=ind.index)
        sv = np.sort(m.values, axis=1); m["margin"] = sv[:, -1] - sv[:, -2]
        return m

    rows = []
    for phase in (1, 2):
        ind = pd.read_csv(root / f"output_phase{phase}" / "indicators_16.csv", index_col=0)
        ref = pd.read_csv(root / f"output_phase{phase}" / "signal_memberships.csv", index_col=0)
        A = compute_memberships(ind, compute_dimension_scores(ind))
        assert np.abs(A[MEMB].values - ref.loc[A.index, MEMB].values).max() < 1e-6
        B = compute_memberships(ind, dim_scores_robust(ind))
        C = memberships_direct(ind, dim_scores_robust(ind))
        argA = A[MEMB].values.argmax(axis=1)
        for name, V in (("B robust-z Indikatoren + robust-z Dimensionen", B), ("C robust-z Indikatoren, Sigmoid direkt", C)):
            argV = V[MEMB].values.argmax(axis=1)
            row = {"phase": phase, "variant": name, "n": len(ind),
                   "argmax_agreement": (argA == argV).mean(),
                   "mean_abs_shift": np.abs(V[MEMB].values - A[MEMB].values).mean(),
                   "p95_abs_shift": np.percentile(np.abs(V[MEMB].values - A[MEMB].values), 95),
                   "share_margin_lt_0.05": (V["margin"] < 0.05).mean(), "share_margin_lt_0.10": (V["margin"] < 0.10).mean(),
                   "median_margin": V["margin"].median()}
            for m in MEMB:
                row[f"rho_{m}"] = spearmanr(A[m], V[m]).correlation
            for i, m in enumerate(MEMB):
                row[f"share_{m}"] = (argV == i).mean()
            rows.append(row)
        rows.append({"phase": phase, "variant": "A Code (Referenz)", "n": len(ind), "argmax_agreement": 1.0, "mean_abs_shift": 0.0,
                     "p95_abs_shift": 0.0, "share_margin_lt_0.05": (A["margin"] < 0.05).mean(),
                     "share_margin_lt_0.10": (A["margin"] < 0.10).mean(), "median_margin": A["margin"].median(),
                     **{f"rho_{m}": 1.0 for m in MEMB}, **{f"share_{m}": (argA == i).mean() for i, m in enumerate(MEMB)}})
    out = pd.DataFrame(rows)
    out.to_csv(root / "standardisation_variants.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
