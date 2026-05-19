"""
Re-Render der Loading-Matrices (5PC und 4PC) fuer beide Phasen.

Nutzt die bereits berechneten pca_loadings.csv-Dateien und ruft
step03_efa_pca.plot_loading_matrix auf — mit aktualisierter
Dimensions-Kuerzel-+Farb-Annotation.

Aufruf:
    python rerender_loading_matrices.py
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


def rerender(label: str, output_dir: Path) -> None:
    print(f"\n== {label} ({output_dir.name}) ==")
    loadings_path = output_dir / "pca_loadings.csv"
    if not loadings_path.exists():
        print(f"  SKIP: {loadings_path} fehlt.")
        return

    loadings = pd.read_csv(loadings_path, index_col=0)
    valid_indicators = loadings.index.tolist()

    # 5-PC-Version
    out_5 = output_dir / "loading_matrix.png"
    s3.plot_loading_matrix(loadings, valid_indicators, str(out_5))

    # 4-PC-Version (auf PC1..PC4 zuschneiden)
    pc_cols = [c for c in loadings.columns if c.startswith("PC")]
    if len(pc_cols) >= 4:
        loadings_4 = loadings[pc_cols[:4]]
        out_4 = output_dir / "loading_matrix_4pc.png"
        s3.plot_loading_matrix(loadings_4, valid_indicators, str(out_4))


def main() -> None:
    print("=" * 70)
    print("RE-RENDER: Loading-Matrices mit Dimensions-Kuerzel + Farben")
    print("=" * 70)
    for label, out_dir in PHASES:
        rerender(label, out_dir)
    print("\nFERTIG.")


if __name__ == "__main__":
    main()
