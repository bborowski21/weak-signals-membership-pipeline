
from __future__ import annotations

import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase1-dir", type=Path,
                    default=BASE_DIR / "output_phase1")
    ap.add_argument("--phase2-dir", type=Path,
                    default=BASE_DIR / "output_phase2")
    ap.add_argument("--out-dir", type=Path,
                    default=BASE_DIR / "output_cross_phase")
    args = ap.parse_args()

    print("=" * 70)
    print("CROSS-PHASE SENSITIVITÄT — Hybrid-α (Topic-Matching)")
    print("=" * 70)
    print(f"  Phase 1: {args.phase1_dir}")
    print(f"  Phase 2: {args.phase2_dir}")
    print(f"  Output : {args.out_dir}")
    print()

    import step05_sensitivity
    step05_sensitivity.run_cross_phase_sensitivity(
        phase1_dir=args.phase1_dir,
        phase2_dir=args.phase2_dir,
        output_dir=args.out_dir,
    )

    print()
    print("=" * 70)
    print(f"FERTIG: Cross-Phase-Sensitivität → {args.out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
