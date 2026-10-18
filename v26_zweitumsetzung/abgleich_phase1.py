"""Abgleich der Zweitumsetzung mit der Erstumsetzung auf Phase 1 (Protokoll 3.2 mit der Praezisierung vom 17.10.).
Liest die Topicwerte der Erstumsetzung aus output_v26_entwicklung/optimismus_phase1/daten.pkl (Schluessel ref) und die
Zweitumsetzung aus output_v26_zweitumsetzung/zweitumsetzung_phase1.csv. Ausgabe nur Kennzahlen, keine Topicwerte."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

V26 = Path(__file__).resolve().parent.parent
D = pickle.load(open(V26 / "output_v26_entwicklung" / "optimismus_phase1" / "daten.pkl", "rb"))
erst = D["ref"]
zweit = pd.read_csv(V26 / "output_v26_zweitumsetzung" / "zweitumsetzung_phase1.csv", index_col="topic")
SPALTEN = ["EO1_A1", "EO2_A1", "EO3_A1", "EO4_A1B", "EO4_A2B", "EO4_R3B", "EO5_A1", "DS2_A3", "DI1_A1", "DI2_A1",
           "DI4_A4", "IP2_A4", "IP3_A1"]
assert sorted(map(int, erst.index)) == sorted(map(int, zweit.index)) and len(zweit) == 146
erst.index = erst.index.astype(int)
zweit.index = zweit.index.astype(int)
zeilen = []
for c in SPALTEN:
    a = erst[c].astype(float).reindex(zweit.index)
    b = zweit[c].astype(float)
    na_a, na_b = a.isna(), b.isna()
    beide = ~(na_a | na_b)
    sd = float(a[beide].std(ddof=0))
    d = (a[beide] - b[beide]).abs()
    z = {"fassung": c, "nan_erst": int(na_a.sum()), "nan_zweit": int(na_b.sum()), "nan_ungleich": int((na_a != na_b).sum()),
         "spearman": round(float(spearmanr(a[beide], b[beide]).statistic), 6),
         "max_abw_in_sd": float(f"{(d.max() / sd if sd > 0 else float('nan')):.3g}"),
         "median_abw_in_sd": float(f"{(d.median() / sd if sd > 0 else float('nan')):.3g}"),
         "verhaeltnis_mittel": float(f"{(b[beide].mean() / a[beide].mean()):.6g}") if a[beide].mean() != 0 else None}
    if c == "DI1_A1":
        se = zweit.loc[beide[beide].index, "DI1_A1_SD"].astype(float) / np.sqrt(20)
        fest = se < 1e-12  # Topics mit genau m Dokumenten: eine einzige, feste Ziehung; dort muss der Wert gleich sein
        zz = (a[beide] - b[beide])[~fest] / se[~fest]
        z["di1_feste_topics"] = int(fest.sum())
        z["di1_feste_ungleich"] = int((d[fest] > 1e-12).sum())
        z["di1_ueber_3_se"] = int((zz.abs() > 3).sum())
        z["di1_ueber_5_se"] = int((zz.abs() > 5).sum())
        z["di1_max_abs_z"] = round(float(zz.abs().max()), 2)
        z["urteil"] = "gleich" if (z["di1_ueber_3_se"] <= 2 and z["di1_ueber_5_se"] == 0 and z["di1_feste_ungleich"] == 0
                                   and z["spearman"] >= 0.98 and z["nan_ungleich"] == 0) else "ABWEICHUNG"
    else:
        z["urteil"] = "gleich" if (z["spearman"] >= 0.999 and z["nan_ungleich"] == 0 and (d.max() / sd if sd > 0 else 0) <= 1e-6) else "ABWEICHUNG"
    zeilen.append(z)
tab = pd.DataFrame(zeilen)
print(tab.to_string(index=False))
ziel = V26 / "output_v26_zweitumsetzung" / "abgleich_phase1.json"
ziel.write_text(json.dumps(zeilen, ensure_ascii=False, indent=1), encoding="utf-8")
