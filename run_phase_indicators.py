"""
Ein-Befehl-Wrapper: Indikatorberechnung + Memberships pro Phase (V2)
=====================================================================

Aufruf:
    python run_phase_indicators.py 1     # Phase 1 (2000-2015)
    python run_phase_indicators.py 2     # Phase 2 (2016-2025)

Was passiert:
  1. config.DATA_PATH und config.OUTPUT_DIR werden für diese Phase auf die
     entsprechenden Werte gesetzt (in-memory, keine Datei-Modifikation).
  2. config.PHASE_YEAR_MIN / PHASE_YEAR_MAX werden auf die Phasengrenzen
     gesetzt.
  3. step02_indicators.run() berechnet 16 Indikatoren und Dimensionsscores.
  4. step02b_memberships.run() berechnet die vier kontinuierlichen
     Memberships (m_ws, m_trend, m_ec, m_latent) und Margin.

Voraussetzungen:
  Topic Modeling der entsprechenden Phase muss bereits gelaufen sein
  (output_phaseX/model_results.pkl, tem_metrics.csv, topic_keywords.csv,
   topic_proportions_yearly.csv).

Ergebnisse landen in:
    output_phase1/   bzw.   output_phase2/

Geschrieben werden:
    indicators_16.csv        - 16 Indikatoren pro Topic
    dimension_scores.csv     - Aggregierte Dimensions-Scores
    signal_memberships.csv   - Kontinuierliche Memberships + Margin (V2)

Die Pipeline-Module (config.py, step02_indicators.py, step02b_memberships.py)
bleiben unverändert.

Autor: Ben Borowski (analog zu run_phase.py)
"""

from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# Phasen-Konfiguration (identisch zu run_phase.py)
# =============================================================================

BASE_DIR = Path(__file__).parent           # → sbert_pipeline/
PARENT_DIR = BASE_DIR.parent               # → F3_Prototyp/

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
    arg = sys.argv[1].strip().lower()
    arg = arg.replace("phase", "").replace("p", "").strip()
    if arg not in PHASES:
        print(f"Unbekanntes Phasen-Argument: {sys.argv[1]!r}")
        usage_and_exit()
    return arg


def ensure_topic_modeling_done(output_dir: Path) -> None:
    """Prüft, dass Schritt 1 für diese Phase vorliegt."""
    required = [
        "model_results.pkl",
        "tem_metrics.csv",
        "topic_keywords.csv",
        "topic_proportions_yearly.csv",
    ]
    missing = [f for f in required if not (output_dir / f).exists()]
    if missing:
        print(f"FEHLER: Topic-Modeling-Outputs in {output_dir} unvollständig.")
        print(f"  Fehlend: {', '.join(missing)}")
        print(f"  Bitte zuerst: python run_phase.py {output_dir.name[-1]}")
        sys.exit(2)


def main() -> None:
    phase_key = parse_phase_arg()
    cfg = PHASES[phase_key]
    data_path = PARENT_DIR / cfg["data_csv"]
    output_dir = BASE_DIR / cfg["output_dir"]
    year_min = cfg["year_min"]
    year_max = cfg["year_max"]

    print("=" * 70)
    print(f"INDIKATORBERECHNUNG + MEMBERSHIPS — {cfg['label']} (Pipeline V2)")
    print("=" * 70)
    print(f"  Eingabe     : {data_path}")
    print(f"  Output      : {output_dir}")
    print(f"  Year-Filter : [{year_min}, {year_max}]")
    print()

    # 1. Topic Modeling muss vorliegen
    ensure_topic_modeling_done(output_dir)

    # 2. Konfiguration in-memory umschalten — config + alle abhängigen Module
    #    (step02 und step02b importieren Konstanten per `from config import …`
    #    direkt in ihren Namespace, deshalb alle patchen).
    import config
    config.DATA_PATH = data_path
    config.OUTPUT_DIR = output_dir
    config.PHASE_YEAR_MIN = year_min
    config.PHASE_YEAR_MAX = year_max

    import step02_indicators
    step02_indicators.DATA_PATH = data_path
    step02_indicators.OUTPUT_DIR = output_dir
    step02_indicators.PHASE_YEAR_MIN = year_min
    step02_indicators.PHASE_YEAR_MAX = year_max

    import step02b_memberships
    step02b_memberships.OUTPUT_DIR = output_dir

    # 3. Indikatorberechnung
    step02_indicators.run()

    # 4. Membership-Berechnung (V2-Kernartefakt)
    print()
    step02b_memberships.run()

    print()
    print("=" * 70)
    print(f"FERTIG: {cfg['label']} — Indikatoren + Memberships")
    print(f"  Ergebnisse in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
