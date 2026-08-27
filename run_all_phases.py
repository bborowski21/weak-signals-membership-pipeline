
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

PHASES = (1, 2)



def build_steps(only_phase: int | None = None,
                skip_cross: bool = False) -> list[tuple[str, str, list[str]]]:
    steps: list[tuple[str, str, list[str]]] = []

    def add_phase_step(step_prefix: str, label_prefix: str, script: str) -> None:
        for p in PHASES:
            if only_phase is not None and p != only_phase:
                continue
            steps.append((
                f"{step_prefix}.{p}",
                f"{label_prefix} — Phase {p}",
                [PYTHON, script, str(p)],
            ))

    add_phase_step("1", "Topic Modeling", "run_phase.py")

    if not skip_cross and only_phase is None:
        steps.append((
            "1b",
            "Cross-Phase-Matching",
            [PYTHON, "step01b_cross_phase_matching.py", "--with-sbert"],
        ))

    add_phase_step("2", "Indikatoren + Memberships", "run_phase_indicators.py")

    add_phase_step("3", "EFA/PCA", "run_phase_efa.py")

    add_phase_step("3b", "Externe Validierung", "run_phase_validation.py")

    add_phase_step("4", "Visualisierungen", "run_phase_viz.py")

    add_phase_step("5b", "Sensitivitäts-Artefakte", "step05b_artifacts.py")

    add_phase_step("5", "Sensitivitätsanalyse", "run_phase_sensitivity.py")

    if not skip_cross and only_phase is None:
        steps.append((
            "5c",
            "Cross-Phase-Sensitivität (Hybrid-α)",
            [PYTHON, "step05c_cross_phase_sensitivity.py"],
        ))

    return steps



def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.2f}h"


def run_step(step_id: str, label: str, cmd: list[str], dry_run: bool = False) -> None:
    bar = "=" * 78
    print()
    print(bar)
    print(f"  STEP {step_id}  —  {label}")
    print(bar)
    print(f"  CMD: {' '.join(cmd)}")
    if dry_run:
        print("  (dry-run, übersprungen)")
        return

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.time() - t0

    if result.returncode != 0:
        print()
        print(f"  ✗ STEP {step_id} FEHLGESCHLAGEN nach {fmt_duration(dt)} "
              f"(returncode={result.returncode})")
        print(f"    Wiederaufnahme:  python run_all_phases.py --from-step {step_id}")
        sys.exit(result.returncode)

    print()
    print(f"  ✓ STEP {step_id} OK  ({fmt_duration(dt)})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Vollständige F3-Pipeline-Orchestrierung über beide Phasen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--from-step",
        default=None,
        metavar="ID",
        help="Wiederaufnahme ab Step-ID (z.B. '3.1', '5a.2'). "
             "Mit --dry-run vorher alle IDs anzeigen.",
    )
    ap.add_argument(
        "--only-phase",
        type=int,
        choices=[1, 2],
        default=None,
        help="Nur diese Phase ausführen (Cross-Phase-Matching wird übersprungen).",
    )
    ap.add_argument(
        "--skip-cross",
        action="store_true",
        help="Cross-Phase-Matching überspringen.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Schritte nur anzeigen, nicht ausführen.",
    )
    args = ap.parse_args()

    steps = build_steps(only_phase=args.only_phase, skip_cross=args.skip_cross)

    if args.from_step:
        ids = [s[0] for s in steps]
        if args.from_step not in ids:
            print(f"  ✗ Unbekannte Step-ID: {args.from_step}")
            print(f"    Verfügbar: {ids}")
            sys.exit(2)
        steps = steps[ids.index(args.from_step):]

    print()
    print("#" * 78)
    print(f"  F3-PIPELINE-ORCHESTRIERUNG  —  {len(steps)} Schritte")
    if args.only_phase:
        print(f"  Modus: nur Phase {args.only_phase}")
    if args.dry_run:
        print("  Modus: dry-run")
    print("#" * 78)
    for sid, lbl, cmd in steps:
        print(f"    {sid:6}  {lbl:40}  {' '.join(cmd[1:])}")

    t_total = time.time()
    for sid, lbl, cmd in steps:
        run_step(sid, lbl, cmd, dry_run=args.dry_run)
    dt_total = time.time() - t_total

    print()
    print("#" * 78)
    print(f"  PIPELINE FERTIG  —  Gesamtzeit: {fmt_duration(dt_total)}")
    print("#" * 78)


if __name__ == "__main__":
    main()
