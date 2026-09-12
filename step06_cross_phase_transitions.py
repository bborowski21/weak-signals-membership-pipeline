"""step06_cross_phase_transitions.py

Uebergangsmatrix der dominanten Konfiguration zwischen den beiden Phasen.

Liest die Mutual-Best-Paare aus step01b_cross_phase_matching.py und die
Memberships beider Phasen, bestimmt je Topic die dominante Konfiguration
(Argmax ueber m_ws, m_trend, m_ec, m_latent) und zaehlt aus, wie sich die
Zuordnung zwischen Phase 1 und Phase 2 verschiebt.

Erzeugt in output_cross_phase/:
  cross_phase_transitions.csv   die Matrix, Zeilen Phase 1, Spalten Phase 2
  cross_phase_transitions.json  dieselben Zahlen plus Kennziffern

Die beiden Groessen, die im Manuskript stehen koennen:
  changed_share   Anteil der Paare mit gewechselter Konfiguration (Abschnitt 4.1)
  ws_to_latent    Paare, die von Weak Signal nach Latent wechseln

WICHTIG zur Lesart: Die Paarmenge ist keine Zufallsstichprobe. Sie deckt
101 der 146 Topics aus Phase 1 und 101 der 265 aus Phase 2 ab; Anteile
daraus sind Aussagen ueber die verknuepfbaren Topics, nicht ueber das Feld.

Aufruf:  python3 step06_cross_phase_transitions.py
         python3 step06_cross_phase_transitions.py --cross-dir output_cross_phase_sbert
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_PHASE1_DIR = BASE_DIR / "output_phase1"
DEFAULT_PHASE2_DIR = BASE_DIR / "output_phase2"
DEFAULT_CROSS_DIR = BASE_DIR / "output_cross_phase"

MEMBERSHIP_COLS = ["m_ws", "m_trend", "m_ec", "m_latent"]
LABELS = {"m_ws": "WS", "m_trend": "Trend", "m_ec": "EC", "m_latent": "Latent"}
ORDER = ["WS", "EC", "Trend", "Latent"]


def dominant(memberships_csv: Path) -> pd.Series:
    """Dominante Konfiguration je Topic, Argmax ueber die vier Memberships."""
    df = pd.read_csv(memberships_csv)
    missing = [c for c in MEMBERSHIP_COLS + ["topic"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{memberships_csv}: Spalten fehlen: {missing}")
    dom = df[MEMBERSHIP_COLS].idxmax(axis=1).map(LABELS)
    return pd.Series(dom.values, index=df["topic"].astype(str), name="dominant")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-dir", type=Path, default=DEFAULT_PHASE1_DIR)
    ap.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2_DIR)
    ap.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS_DIR)
    args = ap.parse_args()

    pairs_csv = args.cross_dir / "topic_matches_mutual.csv"
    for p in (pairs_csv, args.phase1_dir / "signal_memberships.csv",
              args.phase2_dir / "signal_memberships.csv"):
        if not p.exists():
            raise SystemExit(f"fehlt: {p}")

    d1 = dominant(args.phase1_dir / "signal_memberships.csv")
    d2 = dominant(args.phase2_dir / "signal_memberships.csv")
    pairs = pd.read_csv(pairs_csv)

    a = pairs["phase1_topic"].astype(str).map(d1)
    b = pairs["phase2_topic"].astype(str).map(d2)
    if a.isna().any() or b.isna().any():
        raise SystemExit("Paare ohne Membership-Eintrag: "
                         f"{int(a.isna().sum())} in Phase 1, {int(b.isna().sum())} in Phase 2")

    matrix = (pd.crosstab(a, b)
                .reindex(index=ORDER, columns=ORDER, fill_value=0))
    matrix.index.name = "phase1"
    matrix.columns.name = "phase2"

    n = len(pairs)
    changed = int((a != b).sum())
    out = matrix.copy()
    out["Summe"] = out.sum(axis=1)
    out.loc["Summe"] = out.sum(axis=0)
    out.to_csv(args.cross_dir / "cross_phase_transitions.csv")

    kennziffern = {
        "n_pairs": n,
        "changed": changed,
        "changed_share": round(changed / n, 4),
        "ws_to_latent": int(matrix.loc["WS", "Latent"]),
        "latent_to_ws": int(matrix.loc["Latent", "WS"]),
        "n_topics_phase1": int(len(d1)),
        "n_topics_phase2": int(len(d2)),
        "coverage_phase1": round(n / len(d1), 4),
        "coverage_phase2": round(n / len(d2), 4),
        # orient="index": aeussere Schluessel sind die Phase-1-Zustaende,
        # innere die Phase-2-Zustaende, wie in der CSV. Das blanke
        # to_dict() liefert spaltenweise und liest sich transponiert.
        "matrix": matrix.to_dict(orient="index"),
    }
    (args.cross_dir / "cross_phase_transitions.json").write_text(
        json.dumps(kennziffern, indent=2, ensure_ascii=False), encoding="utf-8")

    print(out.to_string())
    print()
    print(f"Paare                      {n}")
    print(f"Konfiguration gewechselt   {changed}  ({100 * changed / n:.1f} %)")
    print(f"Weak Signal -> Latent      {kennziffern['ws_to_latent']}")
    print(f"Latent -> Weak Signal      {kennziffern['latent_to_ws']}")
    print(f"Abdeckung Phase 1          {n} von {len(d1)} Topics ({100 * n / len(d1):.1f} %)")
    print(f"Abdeckung Phase 2          {n} von {len(d2)} Topics ({100 * n / len(d2):.1f} %)")
    print()
    print(f"geschrieben: {args.cross_dir / 'cross_phase_transitions.csv'}")
    print(f"geschrieben: {args.cross_dir / 'cross_phase_transitions.json'}")


if __name__ == "__main__":
    main()
