# -*- coding: utf-8 -*-
"""build_supplement.py

Erzeugt die Datendateien des Supplements zum Manuskript (Indikatorkorrelationen,
Dimensionswerte, Memberships, Indikatorwerte je Topic).
Enthalten sind ausschliesslich abgeleitete Groessen (Indikatorwerte, Dimensionsscores,
Memberships, Margins, c-TF-IDF-Schluesselwoerter); keine Rohdaten und keine
bibliographischen Angaben aus Web of Science.

Aufruf im Pipeline-Ordner:  python3 build_supplement.py [--out supplement_rp] [--run-dir LAUFORDNER]
Schreibt:
  indicator_correlations_phase1.csv / _phase2.csv   (16 x 16, Pearson r)
  topic_dimension_scores.csv                        (Topic, Phase, 5 Dimensionen, Konfiguration, Margin)
  topic_memberships.csv                             (Topic, Phase, 4 Memberships, Margin, Margin-Klasse)
  topic_indicators.csv                              (Topic, Phase, 16 Indikatoren)
  README_supplement.md
Die Dateinamen tragen seit v2.4 keine S-Nummern mehr; die S-Nummern des Supplements
bezeichnen dort Tabellen und Abbildungen.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import BASE_DIR, DIM_NAMES, INDICATOR_DIMENSIONS
import rp_style as rp

MEMB = ["m_ws", "m_trend", "m_ec", "m_latent"]
# 30.09.: Spalte configuration traegt die Namen des Manuskripts (Weak Signal, Emerging Concept, Trend, Latent;
# Entscheidung vom 06.09.), nicht mehr "Weak signal", "Emerging concept", "Latent/mixed".
MEMB_LABEL = {"m_ws": "Weak Signal", "m_trend": "Trend", "m_ec": "Emerging Concept", "m_latent": "Latent"}
PHASES = {1: BASE_DIR / "output_phase1", 2: BASE_DIR / "output_phase2"}
PHASE_YEARS = {1: "2000-2015", 2: "2016-2025"}
IND_ORDER = [i for d in DIM_NAMES for i in INDICATOR_DIMENSIONS[d]]
DIM_EN_COL = {d: rp.DIM_EN[d][0].lower().replace(" ", "_") for d in DIM_NAMES}


def margin_class(m):
    return np.where(m < 0.05, "<0.05", np.where(m < 0.10, "0.05-0.10", ">=0.10"))


def keywords(d: Path, n: int = 3) -> dict:
    kw = pd.read_csv(d / "topic_keywords.csv")
    return {t: ", ".join(g.sort_values("score", ascending=False)["keyword"].head(n)) for t, g in kw.groupby("topic")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="supplement_rp")
    ap.add_argument("--run-dir", default=None,
                    help="Laufordner mit output_phase1/2 (Standard: Pipeline-Ordner)")
    a = ap.parse_args()
    if a.run_dir:
        run = Path(a.run_dir).resolve()
        PHASES[1], PHASES[2] = run / "output_phase1", run / "output_phase2"
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dims, membs, inds = [], [], []
    for ph, d in PHASES.items():
        mem = pd.read_csv(d / "signal_memberships.csv", index_col=0)
        dim = pd.read_csv(d / "dimension_scores.csv", index_col=0)
        ind = pd.read_csv(d / "indicators_16.csv", index_col=0)[IND_ORDER]
        kws = keywords(d)
        conf = mem[MEMB].idxmax(axis=1).map(MEMB_LABEL)
        cls = margin_class(mem["margin"].values)

        corr = ind.corr().round(4)
        corr.index.name = "indicator"
        corr.to_csv(out / f"indicator_correlations_phase{ph}.csv")

        base = pd.DataFrame({"topic": mem.index, "phase": ph, "period": PHASE_YEARS[ph],
                             "keywords_top3": [kws.get(t, "") for t in mem.index],
                             "configuration": conf.values, "margin": mem["margin"].round(4).values,
                             "margin_class": cls})
        dims.append(pd.concat([base, dim.rename(columns=DIM_EN_COL).round(4).reset_index(drop=True)], axis=1))
        membs.append(pd.concat([base, mem[MEMB].round(4).reset_index(drop=True)], axis=1))
        inds.append(pd.concat([base[["topic", "phase", "period", "keywords_top3"]],
                               ind.round(6).reset_index(drop=True)], axis=1))

    pd.concat(dims).to_csv(out / "topic_dimension_scores.csv", index=False)
    pd.concat(membs).to_csv(out / "topic_memberships.csv", index=False)
    pd.concat(inds).to_csv(out / "topic_indicators.csv", index=False)

    n1, n2 = len(dims[0]), len(dims[1])
    (out / "README_supplement.md").write_text(f"""# Supplementary data

All files contain derived quantities only: indicator values computed from the topic model,
dimension scores, membership values and margins. No bibliographic records, no raw data.

| File | Content | Rows |
|---|---|---|
| `indicator_correlations_phase1.csv` | Pearson correlations between the 16 indicators, Phase 1 ({PHASE_YEARS[1]}) | 16 |
| `indicator_correlations_phase2.csv` | Pearson correlations between the 16 indicators, Phase 2 ({PHASE_YEARS[2]}) | 16 |
| `topic_dimension_scores.csv` | Dimension scores (means of z-standardised indicators) per topic, both phases, with dominant configuration and margin | {n1 + n2} |
| `topic_memberships.csv` | The four membership values per topic, both phases, with margin and margin class | {n1 + n2} |
| `topic_indicators.csv` | The 16 indicator values per topic, both phases | {n1 + n2} |

Columns common to the three topic-level files: `topic` (identifier within the phase), `phase`, `period`,
`keywords_top3` (three leading c-TF-IDF terms of the topic).
The dimension-score and membership files additionally carry `configuration` (argmax of the four memberships), `margin`
(difference between the highest and the second-highest membership) and `margin_class`
(< 0.05, 0.05 to 0.10, >= 0.10).

These tables underlie the results reported in Section 4 of the manuscript and the corresponding supplementary figures.
Phase 1 contains {n1} topics, Phase 2 contains {n2} topics.
""", encoding="utf-8")

    print(f"geschrieben nach {out}:")
    for f in sorted(out.iterdir()):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
