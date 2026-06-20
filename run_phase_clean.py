
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent

PHASES = {
    "1": {
        "raw_csv":   "wos_qc_phase1_2000_2015.csv",
        "clean_csv": "wos_qc_phase1_2000_2015_clean.csv",
        "output_dir": "output_phase1",
        "label": "Phase 1 (2000-2015)",
        "year_min": 2000,
        "year_max": 2015,
    },
    "2": {
        "raw_csv":   "wos_qc_phase2_2016_2025.csv",
        "clean_csv": "wos_qc_phase2_2016_2025_clean.csv",
        "output_dir": "output_phase2",
        "label": "Phase 2 (2016-2025)",
        "year_min": 2016,
        "year_max": 2025,
    },
}



def parse_phase_arg() -> str:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    raw = sys.argv[1].strip().lower()
    for prefix in ("phase", "p"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    if raw not in PHASES:
        print(f"Unbekanntes Phasen-Argument: {sys.argv[1]!r}")
        print(__doc__)
        sys.exit(1)
    return raw



def stage1_prepare(raw_path: Path, label: str) -> None:
    if raw_path.exists():
        print(f"[Stufe 1/4] Field-Mapping übersprungen — vorhanden: "
              f"{raw_path.name}")
        return
    print(f"[Stufe 1/4] Field-Mapping (KATI → Pipeline-Format): {label}")
    import prepare_kati_data
    prepare_kati_data.main()


def stage2_clean(clean_path: Path, label: str) -> None:
    if clean_path.exists():
        print(f"[Stufe 2/4] Text-Cleaning übersprungen — vorhanden: "
              f"{clean_path.name}")
        return
    print(f"[Stufe 2/4] Text-Cleaning (LaTeX/HTML/Copyright): {label}")
    import clean_pipeline_data
    clean_pipeline_data.main()


def stage3_topic_modeling(
    clean_path: Path, output_dir: Path, label: str,
    year_min: int, year_max: int,
) -> None:
    print(f"[Stufe 3/4] Topic Modeling (SBERT + UMAP + HDBSCAN + c-TF-IDF): {label}")
    print(f"   Phasen-Year-Filter: [{year_min}, {year_max}]")
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"   ⚠  Output-Verzeichnis enthält bereits Dateien: {output_dir}")
        print(f"   → Bestehende Dateien werden überschrieben.")

    import config
    config.DATA_PATH = clean_path
    config.OUTPUT_DIR = output_dir
    config.PHASE_YEAR_MIN = year_min
    config.PHASE_YEAR_MAX = year_max

    import step01_topic_modeling
    step01_topic_modeling.DATA_PATH = clean_path
    step01_topic_modeling.OUTPUT_DIR = output_dir
    step01_topic_modeling.PHASE_YEAR_MIN = year_min
    step01_topic_modeling.PHASE_YEAR_MAX = year_max

    step01_topic_modeling.run()


def stage4_tem_robustness(phase_key: str, label: str) -> None:
    print(f"[Stufe 4/4] TEM-Robustheit (Partial-Year-Detektion + Auto-Trim): {label}")
    cmd = [
        sys.executable,
        str(BASE_DIR / "step01c_tem_robustness.py"),
        "--phase", phase_key,
        "--auto-trim",
    ]
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"   ⚠  TEM-Robustheit endete mit returncode={result.returncode}")
        print(f"   (Topic Modeling ist bereits abgeschlossen — Stufe 4 manuell "
              f"nachholbar mit: python step01c_tem_robustness.py --phase "
              f"{phase_key} --auto-trim)")



def main() -> None:
    phase_key = parse_phase_arg()
    cfg = PHASES[phase_key]

    raw_path   = PARENT_DIR / cfg["raw_csv"]
    clean_path = PARENT_DIR / cfg["clean_csv"]
    output_dir = BASE_DIR / cfg["output_dir"]
    label      = cfg["label"]

    print("=" * 70)
    print(f"Vier-Stufen-Pipeline: {label}")
    print(f"  Rohdaten   : {raw_path.name}")
    print(f"  Cleaned    : {clean_path.name}")
    print(f"  Output-Dir : {output_dir.name}/")
    print("=" * 70)

    stage1_prepare(raw_path, label)
    print()
    stage2_clean(clean_path, label)
    print()
    stage3_topic_modeling(
        clean_path, output_dir, label,
        year_min=cfg["year_min"], year_max=cfg["year_max"],
    )
    print()
    stage4_tem_robustness(phase_key, label)
    print()

    print("=" * 70)
    print(f"Fertig: {label}")
    print(f"  → {output_dir}/")
    print(f"  → {output_dir}/tem_robust/")
    print()
    print("Nach Abschluss BEIDER Phasen: Cross-Phase-Matching")
    print("  python step01b_cross_phase_matching.py --with-sbert")


if __name__ == "__main__":
    main()
