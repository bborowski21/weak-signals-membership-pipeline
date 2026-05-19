"""
Ein-Befehl-Wrapper: Sensitivitätsanalyse (Schritt 5) für eine Phase (V2)
========================================================================

Aufruf:
    python run_phase_sensitivity.py 1     # Phase 1 (2000-2015)
    python run_phase_sensitivity.py 2     # Phase 2 (2016-2025)

Voraussetzungen:
    output_phaseX/step1_artifacts.pkl   (aus build_step5_artifacts.py)
    output_phaseX/indicators_16.csv     (aus step02_indicators.py)
    output_phaseX/dimension_scores.csv  (aus step02_indicators.py)
    output_phaseX/signal_memberships.csv (aus step02b_memberships.py)

Falls step1_artifacts.pkl fehlt, wird automatisch build_step5_artifacts.main()
aufgerufen (idempotent). step2_artifacts.pkl wird in V2 nicht mehr benötigt
— step05 V2 liest Indikatoren/Dimensionen/Memberships direkt aus den CSVs.

Schreibt nach output_phaseX/:
    sensitivity_membership_kl.csv         - 2D-Grid k×λ (V2)
    sensitivity_parameter_alpha.csv       - Bayes-Prior α-Variation
    sensitivity_parameter_hparam.csv      - BERTopic-Hyperparameter
    sensitivity_ablation.csv              - Indikator-Ablation (V2: Spearman)
    sensitivity_fields.csv                - Feld-Variation
    sensitivity_seeds.csv                 - Seed-Stabilität
    sensitivity_phase.csv                 - Phasen-Splitjahr-Variation
    sensitivity_latency.csv               - Indexierungslatenz
    sensitivity_report.md                 - Konsolidierter Markdown-Bericht

Die Pipeline-Module (config.py, step05_sensitivity.py) bleiben unverändert.

Autor: Ben Borowski (analog zu run_phase_indicators.py)
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


def ensure_artifacts_ready(output_dir: Path, phase_key: str) -> None:
    """Stellt sicher, dass step1_artifacts.pkl + Membership-CSVs vorliegen.

    V2: step2_artifacts.pkl ist nicht mehr nötig; step05 V2 liest Indikatoren,
    Dimensionen und Memberships direkt aus den CSV-Outputs. Falls
    step1_artifacts.pkl fehlt, wird build_step5_artifacts automatisch
    aufgerufen (in-Memory).
    """
    needed_csvs = [
        "indicators_16.csv",
        "dimension_scores.csv",
        "signal_memberships.csv",
    ]
    missing_csvs = [f for f in needed_csvs if not (output_dir / f).exists()]
    if missing_csvs:
        print(f"FEHLER: V2-Voraussetzungen in {output_dir} unvollständig.")
        print(f"  Fehlend: {', '.join(missing_csvs)}")
        print(f"  Bitte zuerst: python run_phase_indicators.py {phase_key}")
        sys.exit(2)

    if not (output_dir / "step1_artifacts.pkl").exists():
        print(f"  Hinweis: step1_artifacts.pkl fehlt — "
              f"baue jetzt automatisch via build_step5_artifacts.py")
        old_argv = sys.argv
        try:
            sys.argv = ["build_step5_artifacts.py", phase_key]
            import build_step5_artifacts
            build_step5_artifacts.main()
        finally:
            sys.argv = old_argv


def main() -> None:
    phase_key = parse_phase_arg()
    cfg = PHASES[phase_key]
    data_path = PARENT_DIR / cfg["data_csv"]
    output_dir = BASE_DIR / cfg["output_dir"]

    print("=" * 70)
    print(f"SENSITIVITÄTSANALYSE — {cfg['label']} (Pipeline V2)")
    print("=" * 70)
    print(f"  Output : {output_dir}")
    print()

    ensure_artifacts_ready(output_dir, phase_key)

    import config
    config.DATA_PATH = data_path
    config.OUTPUT_DIR = output_dir
    config.PHASE_YEAR_MIN = cfg["year_min"]
    config.PHASE_YEAR_MAX = cfg["year_max"]

    import step05_sensitivity
    step05_sensitivity.OUTPUT_DIR = output_dir

    step05_sensitivity.run(output_dir=output_dir)

    print()
    print("=" * 70)
    print(f"FERTIG: Sensitivitätsanalyse — {cfg['label']}")
    print(f"  Ergebnisse in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
