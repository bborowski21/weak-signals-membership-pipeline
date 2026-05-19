"""
run_all_phases.py — Vollständige F3-Pipeline-Orchestrierung (V2)
==============================================================================

Führt die komplette F3-Pipeline V2 für Phase 1 (2000–2015) und Phase 2
(2016–2025) sequentiell aus, indem die phasen-bewussten Wrapper-Scripts als
Subprozesse gestartet werden. Ein einziger Aufruf, ein einziger Endpunkt.

V2-Unterschied zur V1:
  Step 2 berechnet jetzt zusätzlich kontinuierliche Memberships
  (step02b_memberships) — das ist innerhalb von run_phase_indicators.py
  integriert und benötigt keinen separaten Pipeline-Schritt im Orchestrator.

Verwendung:
    python run_all_phases.py                  # Alles ausführen (Default)
    python run_all_phases.py --from-step 3.1  # Wiederaufnahme ab einem Step
    python run_all_phases.py --only-phase 1   # Nur Phase 1
    python run_all_phases.py --skip-cross     # Cross-Phase-Matching überspringen
    python run_all_phases.py --dry-run        # Schritte nur anzeigen

Pipeline-Reihenfolge (Step-IDs):
    1.1, 1.2     Topic Modeling                 — run_phase.py
    1c           Cross-Phase-Matching           — step01c_cross_phase_matching.py
    2.1, 2.2     Indikatoren + Memberships (V2) — run_phase_indicators.py
    3.1, 3.2     EFA/PCA                        — run_phase_efa.py
    3b.1, 3b.2   Externe Validierung            — run_phase_validation.py
    4.1, 4.2     Visualisierungen               — run_phase_viz.py
    5a.1, 5a.2   Step5-Artefakt-Build           — build_step5_artifacts.py
    5b.1, 5b.2   Sensitivitätsanalyse (V2)      — run_phase_sensitivity.py
    5c           Cross-Phase-Sensitivität       — run_cross_phase_sensitivity.py

Bei Fehler stoppt das Script und nennt die Step-ID für den Wiederaufnahme-Aufruf.

Autor: Ben Borowski
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable  # nutzt das aktive (venv-)Python

PHASES = (1, 2)


# =============================================================================
# Pipeline-Definition
# =============================================================================

def build_steps(only_phase: int | None = None,
                skip_cross: bool = False) -> list[tuple[str, str, list[str]]]:
    """Erzeugt die geordnete Liste (step_id, label, cmd)."""
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

    # Step 1: Topic Modeling
    add_phase_step("1", "Topic Modeling", "run_phase.py")

    # Step 1c: Cross-Phase-Matching (nur einmal, wenn beide Phasen laufen)
    if not skip_cross and only_phase is None:
        steps.append((
            "1c",
            "Cross-Phase-Matching",
            [PYTHON, "step01c_cross_phase_matching.py"],
        ))

    # Step 2: Indikatoren + Memberships (V2 — step02 und step02b im Wrapper)
    add_phase_step("2", "Indikatoren + Memberships", "run_phase_indicators.py")

    # Step 3: EFA/PCA
    add_phase_step("3", "EFA/PCA", "run_phase_efa.py")

    # Step 3b: Externe Validierung (RTW/CTW + MTMM)
    add_phase_step("3b", "Externe Validierung", "run_phase_validation.py")

    # Step 4: Visualisierungen
    add_phase_step("4", "Visualisierungen", "run_phase_viz.py")

    # Step 5a: Artefakt-Build für Sensitivität
    add_phase_step("5a", "Step5-Artefakte", "build_step5_artifacts.py")

    # Step 5b: Sensitivitätsanalyse
    add_phase_step("5b", "Sensitivitätsanalyse", "run_phase_sensitivity.py")

    # Step 5c: Cross-Phase-Sensitivität (nur einmal, wenn beide Phasen laufen)
    if not skip_cross and only_phase is None:
        steps.append((
            "5c",
            "Cross-Phase-Sensitivität (Hybrid-α)",
            [PYTHON, "run_cross_phase_sensitivity.py"],
        ))

    return steps


# =============================================================================
# Ausführung
# =============================================================================

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

    # Vorab-Übersicht
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

    # Ausführung
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
