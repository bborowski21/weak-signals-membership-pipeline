
import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR


def run_step(step_id: str, name: str, module_name: str, optional: bool = False):
    print(f"\n{'#' * 70}")
    print(f"# SCHRITT {step_id}: {name}")
    print(f"{'#' * 70}\n")

    t0 = time.time()

    try:
        module = __import__(module_name)
        module.run()
    except (FileNotFoundError, ValueError) as e:
        if optional:
            print(f"   ⚠  Schritt {step_id} übersprungen — Voraussetzung "
                  f"nicht erfüllt: {e}")
            return
        raise

    elapsed = time.time() - t0
    mins, secs = divmod(elapsed, 60)
    print(f"\n→ Schritt {step_id} abgeschlossen in {int(mins)}m {secs:.0f}s")


def _normalize_step_id(raw: str) -> str:
    return str(raw).strip().lower()


def main():
    parser = argparse.ArgumentParser(
        description="F3 SBERT-Pipeline — Weak Signal Operationalisierung"
    )
    parser.add_argument(
        "--from", type=str, default="1", dest="from_step",
        help="Ab welchem Schritt starten (1, 2, 2b, 3, 3b, 4, 5)"
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Nur diesen einen Schritt ausführen (1, 2, 2b, 3, 3b, 4, 5)"
    )
    parser.add_argument(
        "--skip", type=str, default=None,
        help="Diese Schritte überspringen (komma-getrennt: '5' oder '2b,5')"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    steps = [
        ("1",  "Topic Modeling (SBERT + UMAP + HDBSCAN)",
                                    "step01_topic_modeling",      False),
        ("2",  "Indikatorberechnung (16 × 5 Dimensionen)",
                                    "step02_indicators",          False),
        ("2b", "Referenz-Overlap (Cited References)",
                                    "step02b_reference_overlap",  True),
        ("3",  "Strukturentdeckung (EFA/PCA)",
                                    "step03_efa_pca",             False),
        ("3b", "Externe Validierung (RTW/CTW, MTMM)",
                                    "step03b_external_validation", True),
        ("4",  "Visualisierungen",
                                    "step04_visualizations",      False),
        ("5",  "Sensitivitätsanalyse",
                                    "step05_sensitivity",         True),
    ]
    step_order = [s[0] for s in steps]

    skip = set()
    if args.skip:
        skip = {_normalize_step_id(s) for s in args.skip.split(",") if s.strip()}

    print("=" * 70)
    print("F3 SBERT-PIPELINE — Weak Signal Operationalisierungsframework")
    print("=" * 70)
    print(f"Output-Verzeichnis: {OUTPUT_DIR}")

    t_total = time.time()

    if args.only:
        only = _normalize_step_id(args.only)
        matching = [s for s in steps if s[0] == only]
        if not matching:
            print(f"Fehler: Schritt {only!r} existiert nicht "
                  f"({', '.join(step_order)}).")
            sys.exit(1)
        run_step(*matching[0])
    else:
        from_id = _normalize_step_id(args.from_step)
        if from_id not in step_order:
            print(f"Fehler: --from={from_id!r} existiert nicht "
                  f"({', '.join(step_order)}).")
            sys.exit(1)
        from_idx = step_order.index(from_id)
        for step_id, name, module, optional in steps[from_idx:]:
            if step_id in skip:
                print(f"\n# SCHRITT {step_id}: {name} — übersprungen (--skip)")
                continue
            run_step(step_id, name, module, optional=optional)

    total_elapsed = time.time() - t_total
    mins, secs = divmod(total_elapsed, 60)

    print(f"\n{'=' * 70}")
    print(f"PIPELINE ABGESCHLOSSEN — Gesamtdauer: {int(mins)}m {secs:.0f}s")
    print(f"{'=' * 70}")
    print(f"\nAlle Ergebnisse gespeichert in: {OUTPUT_DIR}/")
    print("\nGenerierte Dateien:")

    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.iterdir()):
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.0f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f.name:40s} {size_str:>10s}")


if __name__ == "__main__":
    main()
