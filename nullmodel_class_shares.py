#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Klassenanteile (Argmax der vier Memberships) gegen die Nullmodell-Replikate.

Ergänzt nullmodel_experiment.py um die noch fehlenden p-Werte und 95-Prozent-Intervalle
der vier Klassenanteile. Liest nur nullmodel_replicates.csv und signal_memberships.csv
(abgeleitete Größen, keine Rohdaten).

Aufruf:  python3 nullmodel_class_shares.py --root /pfad/zur/pipeline [--out klassenanteile_null.csv]
   root enthält output_phase1/ und output_phase2/ mit nullmodel_replicates.csv und signal_memberships.csv.

p-Wert: zweiseitig, Anteil der Replikate, deren Abstand vom Nullmittel mindestens so groß ist wie der
des realen Werts. Intervall: Perzentile 2,5 und 97,5 der Replikate.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

COLS = ['m_ws', 'm_trend', 'm_ec', 'm_latent']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default='klassenanteile_null.csv')
    a = ap.parse_args()
    rows = []
    for ph in (1, 2):
        d = Path(a.root) / f'output_phase{ph}'
        sm = pd.read_csv(d / 'signal_memberships.csv')
        am = sm[COLS].idxmax(axis=1)
        real = {c: float((am == c).mean()) for c in COLS}
        rep = pd.read_csv(d / 'nullmodel_replicates.csv')
        for model, sub in rep.groupby('model'):
            for c in COLS:
                x = sub[f'share_{c}'].values
                mu = float(x.mean())
                v = real[c]
                p = float(np.mean(np.abs(x - mu) >= abs(v - mu)))
                lo, hi = np.percentile(x, [2.5, 97.5])
                rows.append(dict(phase=ph, model=model, klasse=c, real=round(v, 4), null_mean=round(mu, 4),
                                 null_p2_5=round(float(lo), 4), null_p97_5=round(float(hi), 4),
                                 p_two_sided=round(p, 3), outside_95=bool(v < lo or v > hi), reps=len(sub)))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
