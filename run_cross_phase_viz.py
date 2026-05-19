"""
Ein-Befehl-Wrapper: Cross-Phase-Visualisierungen (Pipeline V2)
================================================================

Aufruf:
    python run_cross_phase_viz.py

Voraussetzungen:
    output_phase1/signal_memberships.csv
    output_phase1/dimension_scores.csv
    output_phase2/signal_memberships.csv
    output_phase2/dimension_scores.csv
    output_cross_phase/topic_matches_mutual.csv
        (oder topic_matches_best_p1_to_p2.csv als Fallback)

Schreibt nach output_cross_phase/:
    migration_sankey.png
    membership_shift_heatmap.png
    structure_compare_radar.png

Autor: Ben Borowski
"""

from __future__ import annotations

import sys

import step04c_cross_phase_viz


def main() -> None:
    print("=" * 70)
    print("CROSS-PHASE VISUALISIERUNGEN — Wrapper")
    print("=" * 70)
    try:
        info = step04c_cross_phase_viz.run()
    except FileNotFoundError as e:
        print(f"FEHLER: {e}")
        sys.exit(2)

    sankey = info.get("sankey", {})
    heat = info.get("shift_heatmap", {})
    print()
    print(f"  Sankey-Matches  : n={sankey.get('n_matches', 0)}")
    print(f"  Heatmap-Matches : n={heat.get('n_matches', 0)}")
    print(f"  Mittlere Delta-m_ws (Heatmap): {heat.get('mean_delta_m_ws', float('nan')):+.4f}")


if __name__ == "__main__":
    main()
