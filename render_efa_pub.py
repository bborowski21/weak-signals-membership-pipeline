"""
render_efa_pub.py — Publikationsreifes Rendering der EFA-Abbildungen (V3).

Erzeugt fuer beide Phasen: Scree-Plot (PA-Schwelle), Pattern-Matrizen (5F/4F),
Phi-Matrizen (5F/4F) — jeweils als 300-dpi-PNG plus Vektor-PDF, im einheitlichen
Stil der uebrigen F3-Abbildungen (plot_style).

Aufruf:  python3 render_efa_pub.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from plot_style import apply_pub_style
apply_pub_style()

# Dual-Save: jede .png-Ausgabe zusaetzlich als Vektor-.pdf (vgl. rerender_pub.py)
_orig_savefig = plt.savefig


def _dual_savefig(path, *a, **k):
    sp = str(path)
    k.setdefault("bbox_inches", "tight")
    if sp.lower().endswith(".png"):
        k["dpi"] = 300
    _orig_savefig(path, *a, **k)
    if sp.lower().endswith(".png"):
        k2 = dict(k)
        k2.pop("dpi", None)
        _orig_savefig(sp[:-4] + ".pdf", *a, **k2)


plt.savefig = _dual_savefig

import step03_efa_pca as s3

PHASES = [("Phase 1", BASE / "output_phase1"), ("Phase 2", BASE / "output_phase2")]
SOLUTIONS = (5, 4)


def main() -> None:
    for label, d in PHASES:
        print(f"\n== {label} ({d.name}) ==")
        summary = json.loads((d / "efa_summary.json").read_text())

        # Scree mit PA-Schwelle (aus dem Summary, kein Re-Fit noetig)
        ev = np.array(summary["eigenvalues"])
        pa = np.array(summary["pa_thresholds_mean"])
        s3.plot_scree(ev, pa, str(d / "scree_plot.png"))

        n_ref = summary["n_reference"]
        for k in SOLUTIONS:
            pattern = pd.read_csv(d / f"efa_pattern_{k}f.csv", index_col=0)
            role = "Referenz" if k == n_ref else "Vergleich"
            s3.plot_loading_matrix(
                pattern, pattern.index.tolist(),
                str(d / f"loading_matrix_{k}f.png"),
                title=(f"EFA-Pattern-Matrix — {k}-Faktoren-Lösung ({role})\n"
                       f"(minres, Oblimin; theoretische Dimensionszuordnung links)"),
            )
            phi = pd.read_csv(d / f"efa_phi_{k}f.csv", index_col=0)
            s3.plot_phi_matrix(phi, str(d / f"phi_matrix_{k}f.png"))

    print("\nFERTIG.")


if __name__ == "__main__":
    main()
