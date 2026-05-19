"""
Ein-Befehl-Wrapper: Externe Validierung (Schritt 3b) für eine Phase
====================================================================

Aufruf:
    python run_phase_validation.py 1     # Phase 1 (2000-2015)
    python run_phase_validation.py 2     # Phase 2 (2016-2025)

Voraussetzungen:
    output_phaseX/indicators_16.csv
    output_phaseX/topic_assignments.csv  (mit RTW / CTW Spalten)
    Eingabe-CSV mit RTW / CTW pro Dokument

Schreibt nach output_phaseX/:
    external_validation_paper.csv   - Paper-Ebene RTW/CTW + Topic
    external_validation_topic.csv   - Topic-Aggregate
    external_validation_corr.csv    - Korrelationsmatrix
    external_validation_mtmm.csv    - MTMM-Tabelle nach Campbell & Fiske

Autor: Ben Borowski (analog zu run_phase_efa.py)
"""

from __future__ import annotations

import sys
from pathlib import Path


BASE_DIR = Path(__file__).parent
PARENT_DIR = BASE_DIR.parent

PHASES = {
    "1": {
        "data_csv": "wos_qc_phase1_2000_2015.csv",
        "output_dir": "output_phase1",
        "label": "Phase 1 (2000-2015)",
        "year_min": 2000,
        "year_max": 2015,
    },
    "2": {
        "data_csv": "wos_qc_phase2_2016_2025.csv",
        "output_dir": "output_phase2",
        "label": "Phase 2 (2016-2025)",
        "year_min": 2016,
        "year_max": 2025,
    },
}


def usage_and_exit() -> None:
    print(__doc__)
    sys.exit(1)


def parse_phase_arg() -> str:
    if len(sys.argv) < 2:
        usage_and_exit()
    arg = sys.argv[1].strip().lower().replace("phase", "").replace("p", "").strip()
    if arg not in PHASES:
        print(f"Unbekanntes Phasen-Argument: {sys.argv[1]!r}")
        usage_and_exit()
    return arg


def ensure_inputs_ready(output_dir: Path) -> None:
    required = ["indicators_16.csv", "topic_assignments.csv"]
    missing = [f for f in required if not (output_dir / f).exists()]
    if missing:
        print(f"FEHLER: Voraussetzungen in {output_dir} unvollständig.")
        print(f"  Fehlend: {', '.join(missing)}")
        sys.exit(2)


def main() -> None:
    phase_key = parse_phase_arg()
    cfg = PHASES[phase_key]
    data_path = PARENT_DIR / cfg["data_csv"]
    output_dir = BASE_DIR / cfg["output_dir"]

    print("=" * 70)
    print(f"EXTERNE VALIDIERUNG (RTW/CTW + MTMM) — {cfg['label']}")
    print("=" * 70)
    print(f"  Eingabe     : {data_path}")
    print(f"  Output      : {output_dir}")
    print()

    ensure_inputs_ready(output_dir)

    import config
    config.DATA_PATH = data_path
    config.OUTPUT_DIR = output_dir
    config.PHASE_YEAR_MIN = cfg["year_min"]
    config.PHASE_YEAR_MAX = cfg["year_max"]

    import step03b_external_validation
    # run() akzeptiert data_path/output_dir explizit
    step03b_external_validation.run(data_path=data_path, output_dir=output_dir)

    print()
    print("=" * 70)
    print(f"FERTIG: Externe Validierung — {cfg['label']}")
    print(f"  Ergebnisse in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
