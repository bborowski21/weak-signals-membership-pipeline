#!/usr/bin/env python3
"""
Baut Figure S16 des Supplements: die Phasengrenze.

Panel A zaehlt den Analysekorpus je Publikationsjahr und markiert die
berichtete Grenze sowie die beiden Alternativen ein Jahr davor und danach.
Panel B traegt die niedrigste der vier Membership-Rangkorrelationen gegen
das alternative Splitjahr auf, beide Phasen, mit der vorab gesetzten
Schwelle 0,7.

Das Skript steht eigenstaendig neben paper_figures_rp.py, weil es seine
Zahlen aus dem Stabilitaetslauf zur Phasengrenze zieht und nicht aus den
Phasenordnern. Stil und Speicherfunktion kommen aus rp_style, damit die
Abbildung zum uebrigen Satz passt (Arial, exakte Breite, PDF plus TIFF plus
Graustufenprobe).

Aufruf:
    python make_figS16.py [--out figures_rp]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
import rp_style as rp  # noqa: E402

PHASE_CSV = {1: ("wos_qc_phase1_2000_2015_clean.csv", 2000, 2015),
             2: ("wos_qc_phase2_2016_2025_clean.csv", 2016, 2025)}
REFERENZ = 2015


def jahre(csv: str, ymin: int, ymax: int) -> np.ndarray:
    """Publikationsjahre des Analysekorpus, mit demselben Filter wie step01."""
    p = BASE_DIR.parent / csv
    df = pd.read_csv(p, usecols=lambda c: c in ("Title", "Abstract", "Year"))
    text = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")
    clean = text.apply(lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", t.lower())).strip()
                       if isinstance(t, str) else "")
    year = pd.to_numeric(df["Year"], errors="coerce")
    keep = (year >= ymin) & (year <= ymax) & (clean.str.split().str.len() >= 10)
    return year[keep].to_numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BASE_DIR / "figures_rp"))
    args = ap.parse_args()
    out_dir = Path(args.out)

    print(f"Schrift: {rp.active_font()} | Ziel: {out_dir}")

    y = np.concatenate([jahre(*PHASE_CSV[1]), jahre(*PHASE_CSV[2])])
    cnt = pd.Series(y).value_counts().sort_index()
    print(f"  Analysekorpus: {len(y)} Dokumente, {int(cnt.index.min())} bis {int(cnt.index.max())}")

    res = pd.read_csv(BASE_DIR / "output_phase_boundary" / "phase_boundary_vs_control.csv")
    rho = {ph: dict(zip(res[res.phase == ph].alt_split_year.astype(int),
                        res[res.phase == ph].rho_min)) for ph in (1, 2)}
    for ph in (1, 2):
        rho[ph][REFERENZ] = 1.0            # Referenzschnitt gegen sich selbst
    print(f"  Splitjahre: {sorted(rho[1])}")

    fig, (axA, axB) = rp.figure(width="double", height_mm=68.0, nrows=1, ncols=2)

    xs = cnt.index.to_numpy().astype(int)
    vals = cnt.to_numpy()
    p1 = xs <= REFERENZ
    axA.bar(xs[p1], vals[p1], color=rp.PHASE_COLOR[1], edgecolor="none", width=0.78, label="Phase 1")
    axA.bar(xs[~p1], vals[~p1], color=rp.PHASE_COLOR[2], edgecolor="none", width=0.78, label="Phase 2")
    axA.axvline(REFERENZ + 0.5, color=rp.TEXT, linewidth=1.1)
    for xv in (REFERENZ - 0.5, REFERENZ + 1.5):
        axA.axvline(xv, color=rp.REF_LINE, linewidth=0.8, linestyle=(0, (2, 2)))
    ymax = vals.max()
    axA.annotate("reported boundary", xy=(REFERENZ + 0.4, ymax * 0.755),
                 xytext=(xs.min() + 4.3, ymax * 0.755), fontsize=rp.FS["legend"],
                 color=rp.TEXT, va="center",
                 arrowprops=dict(arrowstyle="-", color=rp.TEXT, linewidth=0.7))
    axA.text(REFERENZ + 2.0, ymax * 0.545, "varied by\none year",
             fontsize=rp.FS["legend"], color=rp.TEXT_MUTED, va="center")
    axA.set_xlabel("Publication year")
    axA.set_ylabel("Publications in the analysis corpus")
    axA.set_xlim(xs.min() - 0.8, xs.max() + 0.9)
    axA.set_xticks([t for t in range(2000, 2026, 5)])
    rp.grid(axA, axis="y")
    axA.legend(frameon=False, fontsize=rp.FS["legend"], loc="upper left", handlelength=1.1)
    rp.panel(axA, "A")

    axB.axvspan(REFERENZ - 1, REFERENZ + 1, color="#000000", alpha=0.055, linewidth=0)
    for ph in (1, 2):
        xv = np.array(sorted(rho[ph]))
        yv = np.array([rho[ph][x] for x in xv])
        axB.plot(xv, yv, color=rp.PHASE_COLOR[ph], linestyle=rp.PHASE_LS[ph], linewidth=1.2,
                 marker=rp.PHASE_MARKER[ph], markersize=3.4, label=f"Phase {ph}")
        axB.plot([REFERENZ], [1.0], marker=rp.PHASE_MARKER[ph], markersize=4.8,
                 markerfacecolor="white", markeredgecolor=rp.PHASE_COLOR[ph],
                 markeredgewidth=1.0, linestyle="none")
    axB.axhline(0.7, color=rp.REF_LINE, linewidth=0.9, linestyle=(0, (4, 2.5)))
    xlo, xhi = min(rho[1]) - 0.4, max(rho[1]) + 0.5
    axB.text(xhi - 0.08, 0.712, "criterion 0.7", fontsize=rp.FS["legend"],
             color=rp.TEXT_MUTED, ha="right", va="bottom")
    axB.text(xlo + 0.12, 1.052, "boundary varied by one year", fontsize=rp.FS["legend"],
             color=rp.TEXT_MUTED, ha="left", va="top")
    axB.annotate("reference cut,\none by construction", xy=(REFERENZ + 0.14, 0.995),
                 xytext=(REFERENZ + 0.95, 0.978), fontsize=rp.FS["legend"],
                 color=rp.TEXT_MUTED, va="top")
    axB.set_xlabel("Alternative phase boundary")
    axB.set_ylabel("Lowest membership rank correlation")
    axB.set_xlim(xlo, xhi)
    axB.set_ylim(0.46, 1.075)
    axB.set_xticks(sorted(rho[1]))
    rp.grid(axB, axis="y")
    axB.legend(frameon=False, fontsize=rp.FS["legend"], loc="lower left", handlelength=1.9)
    rp.panel(axB, "B")

    fig.subplots_adjust(left=0.075, right=0.995, top=0.93, bottom=0.135, wspace=0.26)
    rec = rp.save(fig, "figS16_phase_boundary", width="double", out_dir=out_dir,
                  note="Phase boundary: corpus over time and stability of the memberships")
    print(f"  {rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
