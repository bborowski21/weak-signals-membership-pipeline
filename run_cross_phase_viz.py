
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
