"""
rerender_loading_matrices.py — Re-Rendering der EFA-Pattern-Matrizen (V3).

Liest die von step03_efa_pca.py erzeugten Pattern-CSVs (efa_pattern_{k}f.csv)
und rendert die Heatmaps neu, ohne Schritt 3 vollstaendig auszufuehren.
Anhang-Schema der Arbeit: pro Phase eine 5- und eine 4-Faktoren-Loesung.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

import step03_efa_pca as s3

BASE_DIR = Path(__file__).parent
PHASES = [
    ("Phase 1", BASE_DIR / "output_phase1"),
    ("Phase 2", BASE_DIR / "output_phase2"),
]
SOLUTIONS = (5, 4)


def rerender(label: str, output_dir: Path) -> None:
    print(f"\n== {label} ({output_dir.name}) ==")
    for k in SOLUTIONS:
        pattern_path = output_dir / f"efa_pattern_{k}f.csv"
        if not pattern_path.exists():
            print(f"  SKIP: {pattern_path.name} fehlt "
                  f"(zuerst step03_efa_pca.py ausführen).")
            continue

        pattern = pd.read_csv(pattern_path, index_col=0)
        valid_indicators = pattern.index.tolist()

        out = output_dir / f"loading_matrix_{k}f.png"
        s3.plot_loading_matrix(
            pattern, valid_indicators, str(out),
            title=(f"EFA-Pattern-Matrix — {k}-Faktoren-Lösung\n"
                   f"(minres, Oblimin; theoretische Dimensionszuordnung links)"),
        )

        phi_path = output_dir / f"efa_phi_{k}f.csv"
        if phi_path.exists():
            phi = pd.read_csv(phi_path, index_col=0)
            s3.plot_phi_matrix(phi, str(output_dir / f"phi_matrix_{k}f.png"))


def main() -> None:
    print("=" * 70)
    print("RE-RENDER: EFA-Pattern- und Phi-Matrizen (V3)")
    print("=" * 70)
    for label, out_dir in PHASES:
        rerender(label, out_dir)
    print("\nFERTIG.")


if __name__ == "__main__":
    main()
