#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabelle S3 des Supplements: die drei führenden Begriffe je Topic mit der dominanten Konfiguration.

Liest je Phase topic_keywords.csv (c-TF-IDF-Begriffe mit Gewicht) und signal_memberships.csv eines Laufordners
und schreibt die LaTeX-Tabelle (longtable mit booktabs), wie sie im Supplement steht. Regel: je Topic die drei
Begriffe mit dem höchsten Gewicht (stabile Sortierung), dominante Konfiguration = Argmax der vier Memberships.
Nur abgeleitete Größen, keine Rohdaten.

Aufruf:  python3 make_table_s3.py --run-dir LAUFORDNER [--out supplement_S3_topic_terms.tex]
   LAUFORDNER enthält output_phase1/ und output_phase2/, z. B. output_neulauf_voll_2026-09-29/prior.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

# Reihenfolge wie beim Bau von Tabelle S3 für das Manuskript; sie zählt nur bei exakt gleichen Memberships.
MEMB = ["m_ws", "m_ec", "m_trend", "m_latent"]
NAME = {"m_ws": "Weak Signal", "m_ec": "Emerging Concept", "m_trend": "Trend", "m_latent": "Latent"}
KOPF = r"""\begin{longtable}{@{}rlp{0.44\textwidth}l@{}}
\caption{The three leading terms of every topic, by phase, with the dominant configuration. Terms are the highest-weighted c-TF-IDF terms of the topic. Phase~1 comprises @N1@ topics, Phase~2 comprises @N2@. Topic~0 of Phase~2 collects records from outside the field (notes to Table~\ref{tab:s_query}).}\label{tab:s_terms} \\
\toprule
Topic & Phase & Leading terms & Dominant configuration \\
\midrule
\endfirsthead
\multicolumn{4}{@{}l}{\textit{Table \thetable\ continued}} \\
\toprule
Topic & Phase & Leading terms & Dominant configuration \\
\midrule
\endhead
\midrule \multicolumn{4}{r@{}}{\textit{continued on the next page}} \\
\endfoot
\bottomrule
\endlastfoot
"""


def zeilen_der_phase(d: Path, ph: int) -> list:
    kw = pd.read_csv(d / "topic_keywords.csv")
    mem = pd.read_csv(d / "signal_memberships.csv", index_col=0)
    dominant = mem[MEMB].idxmax(axis=1)
    terme = {int(t): [str(w) for w in g.sort_values("score", ascending=False, kind="stable").keyword.head(3)]
             for t, g in kw.groupby("topic") if t >= 0}
    if set(terme) != {int(t) for t in dominant.index}:
        raise SystemExit(f"Phase {ph}: Topics in topic_keywords.csv und signal_memberships.csv verschieden")
    zeilen = []
    for t in sorted(terme):
        text = ", ".join(terme[t])
        if re.search(r"[&%$#_{}~^\\]", text):
            raise SystemExit(f"Phase {ph}, Topic {t}: Sonderzeichen für LaTeX in {text!r}")
        zeilen.append(f"{t} & {ph} & {text} & {NAME[dominant[t]]} \\\\")
    return zeilen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="supplement_S3_topic_terms.tex")
    a = ap.parse_args()
    run = Path(a.run_dir).resolve()
    z1, z2 = zeilen_der_phase(run / "output_phase1", 1), zeilen_der_phase(run / "output_phase2", 2)
    text = KOPF.replace("@N1@", str(len(z1))).replace("@N2@", str(len(z2)))
    text += "\n".join(z1 + z2) + "\n\\end{longtable}\n"
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"{a.out}: {len(z1)} + {len(z2)} Zeilen")


if __name__ == "__main__":
    main()
