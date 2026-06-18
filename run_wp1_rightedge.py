#!/usr/bin/env python3
"""Lokaler Runner fuer die WP1-Rechtsrand-Bias-Diagnostik.

Fuehrt NUR das WP1-Rechtsrand-Submodul der Sensitivitaetsanalyse aus (ohne die
teuren Seed-/Hyperparameter-Re-Runs der vollen step05-Analyse) und ruft dafuer
die in die Pipeline integrierte Funktion `growth_rate_rightedge` aus
`step05_sensitivity.py` auf -- es gibt also keinen Doppelcode, sondern eine
einzige Quelle der Wahrheit.

Schreibt je Phase nach output_phaseX/:
    growth_rate_comparison.csv        (CAGR/OLS/Half-Split, full vs. Endjahr-Trim)
    wp1_membership_stability.csv      (argmax baseline vs. de-biased WP1)
    sensitivity_growth_rightedge.csv  (Kennzahl-Zusammenfassung)

Aufruf (aus beliebigem Verzeichnis -- der Runner wechselt selbst ins
Pipeline-Verzeichnis):
    python run_wp1_rightedge.py        # beide Phasen
    python run_wp1_rightedge.py 2      # nur Phase 2

Voraussetzung: Die Pipeline wurde bis mindestens Schritt 2b gelaufen, d. h. je
Phase liegen step1_artifacts.pkl, indicators_16.csv und signal_memberships.csv
im jeweiligen output_phaseX/-Ordner.
"""
import os
import sys
import pickle
from pathlib import Path

import pandas as pd

# Ins Pipeline-Verzeichnis wechseln und es importierbar machen, damit der
# Runner aus jedem Arbeitsverzeichnis heraus funktioniert.
BASE = Path(__file__).resolve().parent
os.chdir(BASE)
sys.path.insert(0, str(BASE))

from step05_sensitivity import growth_rate_rightedge


def run_phase(phase: str) -> None:
    od = BASE / f"output_phase{phase}"
    if not od.is_dir():
        print(f"[skip] {od} existiert nicht")
        return

    with open(od / "step1_artifacts.pkl", "rb") as f:
        art = pickle.load(f)
    df, labels = art["df"], art["labels"]

    icsv = od / "indicators_16.csv"
    if not icsv.exists():
        icsv = od / "indicators_16 2.csv"
    indicator_df = pd.read_csv(icsv, index_col=0)
    baseline = pd.read_csv(od / "signal_memberships.csv", index_col=0)

    summary = growth_rate_rightedge(od, baseline, indicator_df, df, labels)
    summary.to_csv(od / "sensitivity_growth_rightedge.csv", index=False)

    r = summary.iloc[0]
    print(f"\n=== Phase {phase}  (Endjahr {r['end_year']}, {r['n_topics']} Topics) ===")
    print(f"  rho(OLS, Half-Split)    = {r['rho_ols_vs_halfsplit']}   ->  {r['decision']}")
    print(f"  rho(CAGR full vs trim)  = {r['rho_cagr_full_vs_trim']}")
    print(f"  Median-Shift trim-full  = {r['median_shift_trim_minus_full']:+}"
          f"   | nach unten verzerrt: {r['topics_biased_down']}/{r['n_topics']}")
    print(f"  argmax stabil           = {r['argmax_stable_pct']}%"
          f"   | WS-Klasse {r['ws_baseline']} -> {r['ws_detrend']}")
    print(f"  -> geschrieben: {od.name}/growth_rate_comparison.csv, "
          f"wp1_membership_stability.csv, sensitivity_growth_rightedge.csv")


def main() -> None:
    phases = [sys.argv[1]] if len(sys.argv) > 1 else ["1", "2"]
    for ph in phases:
        run_phase(ph)


if __name__ == "__main__":
    main()
