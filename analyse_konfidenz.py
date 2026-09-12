#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beantwortet die drei Kippbedingungen der Rat-Beratung zur Match-Konfidenz.

Braucht nur pandas und numpy. Kein Re-Run von step01b noetig: die vollstaendige
Score-Matrix liegt bereits in topic_matches_full.csv beider Ausgabeordner.

Aufruf (aus dem Pipeline-Ordner heraus):
    python3 analyse_konfidenz.py
oder mit explizitem Pfad:
    python3 analyse_konfidenz.py /pfad/zu/sbert_pipeline_membership_v3_efa
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
DIR_TFIDF = BASE / "output_cross_phase_v2.2_ctfidf_backup"
DIR_SBERT = BASE / "output_cross_phase_sbert"

SPALTEN = ["phase1_topic", "phase2_topic", "cosine", "jaccard", "hybrid"]


def laden(d: Path, name: str) -> pd.DataFrame:
    p = d / "topic_matches_full.csv"
    if not p.exists():
        sys.exit(f"FEHLT: {p}\nBitte Pfad als Argument uebergeben.")
    df = pd.read_csv(p, usecols=lambda c: c in SPALTEN + ["rank_p1_to_p2", "mutual_best"])
    print(f"  {name:9s} {len(df):6d} Paare aus {p.parent.name}/")
    return df


def titel(t: str) -> None:
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def tail_p(werte: np.ndarray, h: float) -> float:
    """P(H > h) auf der empirischen Verteilung aller Paare."""
    return float((werte > h).sum()) / len(werte)


def spearman(a, b) -> float:
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


print("Lade Score-Matrizen ...")
tf = laden(DIR_TFIDF, "c-TF-IDF")
sb = laden(DIR_SBERT, "SBERT")

# Zeilenmaxima: bester Treffer je Phase-1-Topic
best = {}
for name, df in (("c-TF-IDF", tf), ("SBERT", sb)):
    g = df.loc[df.groupby("phase1_topic")["hybrid"].idxmax()].copy()
    best[name] = g.sort_values("phase1_topic").reset_index(drop=True)

N_KAND = tf["phase2_topic"].nunique()   # 265 Kandidaten je P1-Topic
M = len(best["c-TF-IDF"])               # 146 P1-Topics

titel("1 | E-WERTE: Streuen sie ueber Dekaden oder stauen sie sich am Boden?")
print(f"Kandidaten je Topic n = {N_KAND}, Zeilenmaxima m = {M}")
print("E = n * P(H > h_max), also die erwartete Zahl zufaelliger Treffer dieser Guete.")
print(f"Aufloesungsgrenze: kleinstes darstellbares P(H>h) = 1/{len(tf)} "
      f"=> E_min = {N_KAND / len(tf):.2e}")

E = {}
for name, df in (("c-TF-IDF", tf), ("SBERT", sb)):
    alle = df["hybrid"].to_numpy()
    hmax = best[name]["hybrid"].to_numpy()
    p = np.array([tail_p(alle, h) for h in hmax])
    e = N_KAND * p
    E[name] = e
    am_boden = int((p == 0).sum())
    e_pos = e[e > 0]
    dekaden = (np.log10(e_pos.max()) - np.log10(e_pos.min())) if len(e_pos) > 1 else 0.0
    print(f"\n  {name}")
    print(f"    E-Wert  min {e.min():.3e}   median {np.median(e):.3e}   max {e.max():.3e}")
    print(f"    Spannweite ueber {dekaden:.1f} Dekaden (nur E > 0)")
    print(f"    am Aufloesungsboden (P = 0, nicht weiter aufloesbar): {am_boden} von {M}")
    for schwelle in (0.01, 0.1, 1.0, 10.0):
        print(f"    E < {schwelle:<5}: {int((e < schwelle).sum()):3d} von {M}")

print("\n  Lesart: Streuen die E-Werte ueber mehrere Dekaden und liegen wenige am Boden,")
print("  liefert Pfad C eine abgestufte Groesse. Stauen sie sich am Boden, ist C nur")
print("  Pfad A mit mehr Text.")

titel("2 | EFFEKTIVE KANDIDATENZAHL: Ist der Korrekturfaktor 265 richtig?")
print("Unter der Null ist der Rang des Maximums von n Kandidaten Beta(n,1)-verteilt.")
print("MLE aus den beobachteten Raengen:  n = -m / sum(ln r_i)")
print("ACHTUNG: Diese Schaetzung vermischt zwei Effekte. Echte Treffer heben die")
print("Maxima ueber das Zufallsniveau und treiben n nach oben; Abhaengigkeit zwischen")
print("den Kandidaten druecken es nach unten. Der Wert taugt zum Vergleich der beiden")
print("Skalen und als Groessenordnung, NICHT als saubere Kandidatenzahl.")

for name, df in (("c-TF-IDF", tf), ("SBERT", sb)):
    alle = df["hybrid"].to_numpy()
    hmax = best[name]["hybrid"].to_numpy()
    r = np.array([1.0 - tail_p(alle, h) for h in hmax])
    r = np.clip(r, 1e-12, 1 - 1e-12)
    n_all = -M / np.log(r).sum()
    # Robustheitsvariante: nur die unteren 50 % der Maxima. Das ist eine
    # abgeschnittene Stichprobe, also braucht es die TRUNKIERTE Likelihood
    # n = k / (k*ln c - sum ln r) mit c = Abschneidepunkt. Die naive Formel
    # waere hier verzerrt.
    c = float(np.median(r))
    unten = r <= c
    k = int(unten.sum())
    n_low = k / (k * np.log(c) - np.log(r[unten]).sum())
    bindung = float((df["hybrid"].to_numpy() == 0).mean())
    print(f"\n  {name}")
    print(f"    Bindungsmasse bei H = 0         : {bindung * 100:5.1f} %")
    print(f"    n aus allen {M} Maxima          : {n_all:9.1f}   (nominal {N_KAND})")
    print(f"    n untere Haelfte, trunkierte MLE: {n_low:9.1f}")
    print(f"    Faktor gegenueber nominal       : {n_all / N_KAND:6.2f} x")
    if bindung > 0.5:
        print(f"    ACHTUNG: Beta(n,1) setzt eine stetige Nullverteilung voraus.")
        print(f"    Bei {bindung * 100:.1f} % Punktmasse ist diese Schaetzung NICHT interpretierbar.")

titel("3 | DAS UNTERE ENDE: Sind schwache Treffer wirklich falsche Treffer?")
print("Achtung, das ist die wichtigste Pruefung. Ein Konfidenzmass behauptet, dass")
print("niedrige Werte auf Fehler hindeuten. Ob das stimmt, entscheidet nicht die")
print("Formel, sondern der Blick auf die schwaechsten Treffer. Ein einzelner")
print("Beispielfall genuegt dafuer nicht.")
print()
print("  Die 10 schwaechsten Zeilenmaxima auf der SBERT-Skala:")
_b = best["SBERT"].nsmallest(10, "hybrid")
for _, _r in _b.iterrows():
    print(f"    H={_r['hybrid']:.3f}  P1#{int(_r['phase1_topic']):3d} "
          f"[{str(_r.get('phase1_keywords', ''))[:34]}]")
    print(f"                -> P2#{int(_r['phase2_topic']):3d} "
          f"[{str(_r.get('phase2_keywords', ''))[:34]}]")
print()
print("  Pruefe diese Liste fachlich durch. Sind darunter Paare, die inhaltlich")
print("  richtig sind und nur wenig Vokabular teilen, dann misst H Aehnlichkeit")
print("  und nicht Korrektheit. Dann darf die Groesse im Manuskript nicht")
print("  'Konfidenz' heissen, egal wie sie berechnet wird.")

titel("3b | EINZELFALL P1#56 (brain/microtubules/conscious) zum Vergleich")
for name, df in (("c-TF-IDF", tf), ("SBERT", sb)):
    b = best[name]
    zeile = b[b["phase1_topic"] == 56]
    if zeile.empty:
        print(f"  {name}: P1#56 nicht gefunden")
        continue
    h = float(zeile["hybrid"].iloc[0])
    p2 = int(zeile["phase2_topic"].iloc[0])
    hm = b["hybrid"].to_numpy()
    rang = int((hm < h).sum()) + 1
    z = (h - hm.mean()) / hm.std(ddof=0)
    alle = df["hybrid"].to_numpy()
    q99 = float(np.quantile(alle, 0.99))
    e = N_KAND * tail_p(alle, h)
    print(f"\n  {name}")
    print(f"    bester Partner P2#{p2}, H = {h:.5f}")
    print(f"    Rang unter den {M} Zeilenmaxima: {rang}  (1 = schlechtester)")
    print(f"    z-Wert gegen die Maxima        : {z:+.2f}")
    print(f"    E-Wert                         : {e:.3e}")
    print(f"    q99 der Gesamtverteilung       : {q99:.5f}  "
          f"=> {'GEFLAGGT' if h < q99 else 'NICHT geflaggt'}")

titel("4 | KIPPBEDINGUNG PFAD A: Ueberlappen sich die geflaggten Mengen?")
flags = {}
for name, df in (("c-TF-IDF", tf), ("SBERT", sb)):
    alle = df["hybrid"].to_numpy()
    q99 = float(np.quantile(alle, 0.99))
    b = best[name]
    s = set(b.loc[b["hybrid"] < q99, "phase1_topic"].astype(int))
    flags[name] = s
    print(f"  {name:9s} q99 = {q99:.5f}  geflaggt: {len(s)} von {M}  {sorted(s)}")
a, bb = flags["c-TF-IDF"], flags["SBERT"]
inter = a & bb
print(f"\n  Schnittmenge: {len(inter)}  {sorted(inter) if inter else '(leer)'}")
if a | bb:
    print(f"  Jaccard der beiden Mengen: {len(inter) / len(a | bb):.3f}")
print("  Lesart: Eine kleine Schnittmenge heisst, dass die Schwelle je Skala etwas")
print("  anderes markiert. Dann traegt Pfad A den Skalenvergleich nicht.")

titel("5 | KIPPBEDINGUNG PFAD D: Ueberlappen sich die beiden Quartilsmengen?")
print("Thesis-Kriterium der EC-Fruehphase: H_max je P2-Topic unter dem ersten Quartil.")
qs = {}
for name, df in (("c-TF-IDF", tf), ("SBERT", sb)):
    hm2 = df.groupby("phase2_topic")["hybrid"].max()
    q25 = float(hm2.quantile(0.25))
    s = set(hm2[hm2 < q25].index.astype(int))
    qs[name] = s
    print(f"  {name:9s} q0,25 = {q25:.5f}   {len(s)} von {len(hm2)} P2-Topics")
a2, b2 = qs["c-TF-IDF"], qs["SBERT"]
i2 = a2 & b2
print(f"\n  Schnittmenge: {len(i2)} von je {len(a2)} bzw. {len(b2)}")
if a2 | b2:
    print(f"  Jaccard: {len(i2) / len(a2 | b2):.3f}")
print("  Lesart: Ist die Ueberlappung hoch, ist das Quartilskriterium skalenrobust")
print("  und Pfad D wird stark. Ist sie niedrig, markiert es je Skala anderes.")

titel("6 | ZERLEGUNG: Messen sigma_sem und sigma_lex ueberhaupt Verschiedenes?")
print("Der Einwand des Kontrarians. 40 % des Scores (Jaccard) sind ueber beide Laeufe")
print("per Konstruktion identisch; entscheidend ist die Cosinus-Komponente.")
tf_s = tf.sort_values(["phase1_topic", "phase2_topic"]).reset_index(drop=True)
sb_s = sb.sort_values(["phase1_topic", "phase2_topic"]).reset_index(drop=True)
print(f"\n  Jaccard ueber beide Laeufe identisch: "
      f"{np.array_equal(tf_s['jaccard'].to_numpy(), sb_s['jaccard'].to_numpy())}")
print(f"  Spearman(cos_ctfidf, cos_sbert)     : {spearman(tf_s['cosine'], sb_s['cosine']):.4f}")
print()
for name, df in (("c-TF-IDF", tf), ("SBERT", sb)):
    alle_sp = spearman(df["cosine"], df["jaccard"])
    nz = df[df["cosine"] != 0]
    nz_sp = spearman(nz["cosine"], nz["jaccard"]) if len(nz) > 2 else float("nan")
    mu = df[df["mutual_best"] == True] if "mutual_best" in df.columns else df.iloc[0:0]
    mu_sp = spearman(mu["cosine"], mu["jaccard"]) if len(mu) > 2 else float("nan")
    n0 = int((df["cosine"] == 0).sum())
    j0 = int((df["jaccard"] == 0).sum())
    beide0 = int(((df["cosine"] == 0) & (df["jaccard"] == 0)).sum())
    print(f"  {name}")
    print(f"    Spearman(cos, jac) alle Paare      : {alle_sp:.4f}")
    print(f"    Spearman(cos, jac) nur cos != 0    : {nz_sp:.4f}   (n = {len(nz)})")
    print(f"    Spearman(cos, jac) nur Mutual-Best : {mu_sp:.4f}   (n = {len(mu)})")
    print(f"    cos == 0: {n0}   jac == 0: {j0}   beides zugleich: {beide0}")
    print(f"    => Traegermengen identisch: {n0 == j0 == beide0}")
print("\n  Lesart: Faellt die Rangkorrelation auf den Mutual-Paaren nahe 1, sind die")
print("  beiden Komponenten dort dieselbe Groesse und die Gewichtung alpha_H ist")
print("  ohne Wirkung. Dann gehoert der Skalenvergleich auf die Cosinus-Komponente")
print("  allein, nicht auf den Hybrid-Score.")

titel("ZUSAMMENFASSUNG IN EINEM SATZ JE FRAGE")
e_sb = E["SBERT"]
e_pos = e_sb[e_sb > 0]
print(f"1 E-Werte SBERT: {len(e_pos)} von {M} aufloesbar, "
      f"Spannweite {np.log10(e_pos.max()) - np.log10(e_pos.min()):.1f} Dekaden"
      if len(e_pos) > 1 else "1 E-Werte SBERT: nicht aufloesbar")
print(f"2 n-Schaetzung: siehe Abschnitt 2, mit dem dort genannten Vorbehalt")
print(f"3 P1#56 wird von der stetigen Groesse abgetrennt: siehe Abschnitt 3")
print(f"4 Schnittmenge der q99-Flags ueber die Skalen: {len(inter)}")
print(f"5 Schnittmenge der Quartilsmengen: {len(i2)} von je {len(a2)}/{len(b2)}")
print("6 Zerlegung: siehe Abschnitt 6")
print()
print("Fertig. Nichts geschrieben, nur gelesen und gerechnet.")
