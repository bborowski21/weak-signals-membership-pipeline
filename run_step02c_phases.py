"""
run_step02c_phases.py — Citation-Topic-Profil (Schritt 2c) für beide Phasen.

Führt das Diagnostikmodul step02c für Phase 1 und Phase 2 in einem Aufruf aus.
Je Phase wird gegen die phasenspezifischen Pipeline-Outputs
(output_phase{1,2}/topic_assignments.csv, signal_memberships.csv) und den
zugehörigen Korpus (wos_qc_phase{1,2}_*.csv im Repository-Parent) gerechnet.

Ausgabe je Phase:
  output_phase{N}/citation_topic_profile.csv         (alle Topics)
  output_phase{N}/citation_topic_profile_rows.tex    (WS-dominante Topics)

Zusätzlich, einfügefertig für die Anhang-Tabelle tab:app_citation_topic_profile
(appendix.tex, Schritt 2c) — beide Phasen kombiniert, mit vorangestellter
Phasenspalte (P1/P2):
  citation_topic_profile_appendix_rows.tex

Aufruf:  python run_step02c_phases.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import step02c_citation_topic_profile as step02c

BASE = Path(__file__).resolve().parent

PHASES = {
    1: "wos_qc_phase1_*.csv",
    2: "wos_qc_phase2_*.csv",
}


def _find_corpus(pattern: str) -> Path | None:
    hits = sorted(glob.glob(str(BASE.parent / pattern)))
    return Path(hits[0]) if hits else None


def _prefix_rows(rows_file: Path, phase_label: str) -> list[str]:
    """Liest die von step02c erzeugten Datenzeilen und stellt die Phasenspalte
    voran. Kommentarzeilen (%) werden übersprungen."""
    if not rows_file.exists():
        return []
    out = []
    for line in rows_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        out.append(f"{phase_label} & {s}")
    return out


def main() -> None:
    combined: list[str] = []

    for phase, pattern in PHASES.items():
        out_dir = BASE / f"output_phase{phase}"
        corpus = _find_corpus(pattern)

        print("=" * 64)
        print(f"Schritt 2c — Phase {phase}")
        print(f"  Output:  {out_dir}")
        print(f"  Korpus:  {corpus if corpus else '— NICHT GEFUNDEN'}")

        if not (out_dir / "topic_assignments.csv").exists():
            print(f"  UEBERSPRUNGEN: {out_dir}/topic_assignments.csv fehlt "
                  f"(zuerst Phase {phase} der Pipeline laufen lassen).")
            continue
        if corpus is None:
            print(f"  UEBERSPRUNGEN: kein Korpus '{pattern}' im Repository-Parent "
                  f"({BASE.parent}). Citation-Topics liegen nur dort.")
            continue

        step02c.DATA_PATH = corpus
        step02c.run(output_dir=out_dir)
        combined += _prefix_rows(
            out_dir / "citation_topic_profile_rows.tex", f"P{phase}")

    print("=" * 64)
    if combined:
        appendix_rows = BASE / "citation_topic_profile_appendix_rows.tex"
        header = [
            "% Auto-generiert durch run_step02c_phases.py.",
            "% Diese Zeilen ersetzen die Platzhalterzeile in der Tabelle",
            "% tab:app_citation_topic_profile (appendix.tex, Schritt 2c).",
        ]
        appendix_rows.write_text("\n".join(header + combined) + "\n",
                                 encoding="utf-8")
        print(f"Einfügefertig (beide Phasen): {appendix_rows}")
        print(f"  -> {len(combined)} WS-dominante Topic-Zeilen")
        print("\nDiese Zeilen in die Anhang-Tabelle tab:app_citation_topic_profile")
        print("(appendix.tex, Schritt 2c) anstelle der Platzhalterzeile einsetzen.")
    else:
        print("Keine Datenzeilen erzeugt — bitte obige Hinweise prüfen "
              "(Pipeline-Outputs und/oder Korpus fehlen).")


if __name__ == "__main__":
    main()
