"""
Standalone-Post-Processor: Text-Cleaning für vorbereitete Pipeline-CSVs
========================================================================

Liest die von prepare_kati_data.py erzeugten Pipeline-Eingabe-CSVs
(wos_qc_phase1_2000_2015.csv, wos_qc_phase2_2016_2025.csv) und schreibt
*neue* Cleaned-Versionen mit dem Suffix "_clean":
    wos_qc_phase1_2000_2015_clean.csv
    wos_qc_phase2_2016_2025_clean.csv

Designentscheidung — separation of concerns:
  prepare_kati_data.py : Field-Mapping (KATI-Schema → Pipeline-Schema)
  clean_pipeline_data.py: Text-Normalisierung (LaTeX, HTML, Copyright, Unicode)

Originaldateien werden NICHT überschrieben. Damit bleibt die Field-Mapping-
Stufe vollständig nachvollziehbar und idempotent.

Aufruf:
    python clean_pipeline_data.py             # idempotent
    python clean_pipeline_data.py --force     # neu erzeugen
    python clean_pipeline_data.py --diagnose  # nur Artefakt-Statistik

Schalter zum produktiven Einsatz:
  Nach erfolgreichem Lauf in run_phase.py die PHASES-Datenpfade auf die
  *_clean.csv-Versionen umstellen (1 Zeile pro Phase).

Autor: Ben Borowski
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

from text_preprocessing import clean_text, diagnose_artifacts

# =============================================================================
# KONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
PARENT_DIR = BASE_DIR.parent  # F3_Prototyp/

PHASES = [
    {
        "src": "wos_qc_phase1_2000_2015.csv",
        "dst": "wos_qc_phase1_2000_2015_clean.csv",
        "label": "Phase 1 (2000-2015)",
    },
    {
        "src": "wos_qc_phase2_2016_2025.csv",
        "dst": "wos_qc_phase2_2016_2025_clean.csv",
        "label": "Phase 2 (2016-2025)",
    },
]

# Spalten, die Text-Cleaning erhalten. Strikt auf semantischen Freitext
# beschränkt — kategorische Felder (Document Type, Source title) bleiben unberührt.
TEXT_COLUMNS = ["Title", "Abstract"]


# =============================================================================
# KERN-FUNKTIONEN
# =============================================================================

def diagnose_phase(src_path: Path, label: str) -> dict:
    """Zählt Artefakte vor dem Cleaning. Reine Inspektion, kein Schreiben."""
    print(f"\n=== Diagnose: {label} ===")
    print(f"Quelle: {src_path.name}")
    df = pd.read_csv(src_path)
    print(f"Records: {len(df):,}")
    out = {}
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            print(f"  [warnung] Spalte fehlt: {col}")
            continue
        c = diagnose_artifacts(df[col].astype(str))
        out[col] = c
        n_total = c["n_total"]
        n_any = c["n_with_any"]
        pct = (n_any / n_total * 100) if n_total else 0
        print(f"  {col}: {n_any}/{n_total} ({pct:.2f}%) mit Artefakt")
        for k in ("latex_cmd", "inline_math", "html_tag", "copyright_tail"):
            if c[k]:
                print(f"    {k:>15s}: {c[k]:>6,}")
    return out


def clean_phase(src_path: Path, dst_path: Path, label: str) -> dict:
    """Wendet clean_text auf TEXT_COLUMNS an, schreibt *_clean.csv.
    Liefert Diagnostik-Counts vor/nach für QC-Reporting."""
    print(f"\n=== Cleaning: {label} ===")
    print(f"Quelle:  {src_path.name}")

    df = pd.read_csv(src_path)
    n_in = len(df)

    counts_before, counts_after = {}, {}
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            print(f"  [warnung] Spalte fehlt, übersprungen: {col}")
            continue
        before = diagnose_artifacts(df[col].astype(str))
        # NaN-sicher: clean_text behandelt None/NaN selbst zu ""
        df[col] = df[col].apply(clean_text)
        after = diagnose_artifacts(df[col].astype(str))
        counts_before[col] = before
        counts_after[col] = after

        # Falls Cleaning Title oder Abstract leert: Record droppen
        # (kein Embedding möglich)
        if col in ("Title", "Abstract"):
            n_pre_drop = len(df)
            df = df[df[col].astype(str).str.strip() != ""].copy()
            n_dropped = n_pre_drop - len(df)
            if n_dropped:
                print(f"  {col}: {n_dropped} Records mit leerem Feld nach "
                      f"Cleaning entfernt")

    df.to_csv(dst_path, index=False)
    print(f"Ziel:    {dst_path.name}  ({len(df):,} Records, {n_in - len(df)} verloren)")

    # Reduktionsstatistik
    print("Reduktion (Records mit Artefakt vor/nach Cleaning):")
    for col in counts_before:
        b, a = counts_before[col], counts_after[col]
        print(f"  {col}: {b['n_with_any']:>6,}{a['n_with_any']:>6,}  "
              f"(latex {b['latex_cmd']}{a['latex_cmd']}, "
              f"math {b['inline_math']}{a['inline_math']}, "
              f"html {b['html_tag']}{a['html_tag']}, "
              f"copyr {b['copyright_tail']}{a['copyright_tail']})")
    return dict(before=counts_before, after=counts_after, n_out=len(df))


def main(force: bool = False, diagnose_only: bool = False) -> None:
    print("Pipeline-Daten-Cleaning (LaTeX / HTML / Copyright / Whitespace)")
    print("=" * 70)
    if diagnose_only:
        print("[Modus: Diagnose-only — keine Dateien werden geschrieben]\n")

    for phase in PHASES:
        src = PARENT_DIR / phase["src"]
        dst = PARENT_DIR / phase["dst"]
        if not src.exists():
            print(f"\n=== {phase['label']} ===  [QUELLE FEHLT: {src}]")
            continue

        if diagnose_only:
            diagnose_phase(src, phase["label"])
            continue

        if dst.exists() and not force:
            print(f"\n=== {phase['label']} ===  (übersprungen — Datei vorhanden: {dst.name})")
            print(f"   --force für Neuerzeugung verwenden")
            continue

        clean_phase(src, dst, phase["label"])

    print("\n" + "=" * 70)
    if not diagnose_only:
        print("Fertig. Nächste Schritte:")
        print("  1. In run_phase.py die PHASES-Datenpfade von")
        print("       'wos_qc_phaseN_YYYY_YYYY.csv'")
        print("     auf")
        print("       'wos_qc_phaseN_YYYY_YYYY_clean.csv'")
        print("     umstellen.")
        print("  2. python run_phase.py 1   und   python run_phase.py 2")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        force="--force" in args,
        diagnose_only="--diagnose" in args,
    )
