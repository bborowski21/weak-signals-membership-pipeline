
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd


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


def ensure_indicators_done(output_dir: Path) -> None:
    required = ["indicators_16.csv"]
    missing = [f for f in required if not (output_dir / f).exists()]
    if missing:
        print(f"FEHLER: Indikator-Outputs in {output_dir} unvollständig.")
        print(f"  Fehlend: {', '.join(missing)}")
        print(f"  Bitte zuerst: python run_phase_indicators.py {output_dir.name[-1]}")
        sys.exit(2)


def main() -> None:
    phase_key = parse_phase_arg()
    cfg = PHASES[phase_key]
    data_path = PARENT_DIR / cfg["data_csv"]
    output_dir = BASE_DIR / cfg["output_dir"]
    year_min = cfg["year_min"]
    year_max = cfg["year_max"]

    print("=" * 70)
    print(f"EFA/PCA — {cfg['label']}")
    print("=" * 70)
    print(f"  Indikatoren : {output_dir / 'indicators_16.csv'}")
    print(f"  Output      : {output_dir}")
    print()

    ensure_indicators_done(output_dir)

    ind_path = output_dir / "indicators_16.csv"
    ind_backup_path = output_dir / "indicators_16_full.csv"

    ind_df = pd.read_csv(ind_path, index_col="topic")
    all_nan_cols = ind_df.columns[ind_df.isna().all()].tolist()

    backup_created = False
    if all_nan_cols:
        print(f"  Hinweis: Ausschluss vollflächiger NaN-Spalten vor EFA: {all_nan_cols}")
        shutil.copy2(ind_path, ind_backup_path)
        backup_created = True
        ind_df.drop(columns=all_nan_cols).to_csv(ind_path)
        print(f"  Indikator-Matrix für EFA: {ind_df.shape[1] - len(all_nan_cols)} Spalten")
        print(f"  Backup der Vollmatrix: {ind_backup_path.name}")
        print()

    try:
        import config
        config.DATA_PATH = data_path
        config.OUTPUT_DIR = output_dir
        config.PHASE_YEAR_MIN = year_min
        config.PHASE_YEAR_MAX = year_max

        import step03_efa_pca
        step03_efa_pca.OUTPUT_DIR = output_dir

        step03_efa_pca.run()
    finally:
        if backup_created and ind_backup_path.exists():
            shutil.move(str(ind_backup_path), str(ind_path))
            print(f"\n  Vollmatrix in {ind_path.name} wiederhergestellt.")

    print()
    print("=" * 70)
    print(f"FERTIG: EFA/PCA — {cfg['label']}")
    print(f"  Ergebnisse in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
