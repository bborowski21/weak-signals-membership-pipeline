"""
Ein-Befehl-Wrapper: Topic Modeling für eine spezifische Phase
==============================================================

Aufruf:
    python run_phase.py 1     # Phase 1 (2000-2015)
    python run_phase.py 2     # Phase 2 (2016-2025)

Was passiert:
  1. Falls die phasen-spezifische Eingabe-CSV noch nicht existiert, wird
     prepare_kati_data.main() automatisch ausgeführt.
  2. config.DATA_PATH und config.OUTPUT_DIR werden für diese Phase auf die
     entsprechenden Werte gesetzt (in-memory, keine Datei-Modifikation).
  3. step01_topic_modeling.run() wird aufgerufen.

Ergebnisse landen in:
    output_phase1/   bzw.   output_phase2/

Die Pipeline-Module (config.py, step01_topic_modeling.py) bleiben unverändert.

Autor: Ben Borowski
"""

from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# Phasen-Konfiguration
# =============================================================================

BASE_DIR = Path(__file__).parent           # → sbert_pipeline/
PARENT_DIR = BASE_DIR.parent               # → F3_Prototyp/

PHASES = {
    "1": {
        # _clean.csv: textbereinigte Variante (clean_pipeline_data.py).
        # Originale wos_qc_phase1_2000_2015.csv bleibt als Rohmapping erhalten.
        "data_csv": "wos_qc_phase1_2000_2015_clean.csv",
        "output_dir": "output_phase1",
        "label": "Phase 1 (2000-2015)",
        "year_min": 2000,
        "year_max": 2015,
    },
    "2": {
        "data_csv": "wos_qc_phase2_2016_2025_clean.csv",
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
    # Toleranz: '1', 'phase1', 'p1' etc. werden auf '1' normalisiert
    arg = arg.replace("phase", "").replace("p", "").strip()
    if arg not in PHASES:
        print(f"Unbekanntes Phasen-Argument: {sys.argv[1]!r}")
        usage_and_exit()
    return arg


def ensure_data_prepared(data_path: Path) -> None:
    """Erzeugt die Pipeline-CSVs aus den KATI-Quelldateien, wenn noch nicht da.

    Kettet zwei Stufen:
      (a) prepare_kati_data → wos_qc_phaseN_YYYY_YYYY.csv (Field-Mapping)
      (b) clean_pipeline_data → ..._clean.csv (Text-Cleaning Title/Abstract)
    """
    if data_path.exists():
        return

    # (a) Rohmapping: Original wos_qc_phaseN.csv (ohne _clean-Suffix)
    raw_path = data_path.with_name(data_path.name.replace("_clean.csv", ".csv"))
    if not raw_path.exists():
        print(f"Rohmapping fehlt ({raw_path.name}) — starte prepare_kati_data …")
        import prepare_kati_data
        prepare_kati_data.main()

    # (b) Text-Cleaning: ..._clean.csv
    if not data_path.exists():
        print(f"Cleaned-CSV fehlt ({data_path.name}) — starte "
              f"clean_pipeline_data …")
        import clean_pipeline_data
        clean_pipeline_data.main()


def main() -> None:
    phase_key = parse_phase_arg()
    cfg = PHASES[phase_key]
    data_path = PARENT_DIR / cfg["data_csv"]
    output_dir = BASE_DIR / cfg["output_dir"]
    year_min = cfg["year_min"]
    year_max = cfg["year_max"]

    print("=" * 70)
    print(f"TOPIC MODELING — {cfg['label']}")
    print("=" * 70)
    print(f"  Eingabe     : {data_path}")
    print(f"  Output      : {output_dir}")
    print(f"  Year-Filter : [{year_min}, {year_max}]")
    print()

    # 1. Datenaufbereitung (idempotent)
    ensure_data_prepared(data_path)

    # 2. Konfiguration in-memory umschalten (ohne Datei-Modifikation).
    #    Wichtig: DATA_PATH, OUTPUT_DIR und die Phasen-Jahresgrenzen müssen
    #    sowohl in config als auch im step01-Modul gepatcht werden, weil
    #    step01 die Konstanten per `from config import …` direkt in seinen
    #    Namespace zieht.
    import config
    config.DATA_PATH = data_path
    config.OUTPUT_DIR = output_dir
    config.PHASE_YEAR_MIN = year_min
    config.PHASE_YEAR_MAX = year_max

    import step01_topic_modeling
    step01_topic_modeling.DATA_PATH = data_path
    step01_topic_modeling.OUTPUT_DIR = output_dir
    step01_topic_modeling.PHASE_YEAR_MIN = year_min
    step01_topic_modeling.PHASE_YEAR_MAX = year_max

    # 3. Pipeline starten
    step01_topic_modeling.run()

    print()
    print("=" * 70)
    print(f"FERTIG: {cfg['label']}")
    print(f"  Ergebnisse in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
