#!/usr/bin/env python3
"""V2.6, Phase 1, Schritt 1.4: Optimismusschaetzung der Auswahl ueber eine 5 x 5-fache Kreuzvalidierung ueber Topics.

Vorgaben laut Protokoll, Eintrag vom 17.10.2026 vor dem Rechnen (Optimismusschaetzung, Punkte a bis f). Gelesen werden nur
gespeicherte Topicwerte aus 1a bis 1h (backtest_phase1.py, backtest_kandidaten.py, auswertung_eo4_k3.py; deren Funktionen
werden unveraendert verwendet) und fuer K1 die Zahl der Dokumente mit Feld je Topic aus dem Zwischenspeicher von 1b.
Ausgegeben werden nur Guetemasse und Auswahlen, keine Zugehoerigkeiten und keine Konfigurationen. In der Bestaetigung
(3.4, Phasenschalter phase.py) laufen nur die Teile daten und kriterien, auf der Phase des Schalters.

Aufruf (im Ordner v26_entwicklung):
  python3 optimismus_phase1.py daten               Topicwerte aller Fassungen sammeln (optimismus_phase1/daten.pkl)
  python3 optimismus_phase1.py regress             Kennzahlen und Auswahl auf allen 146 Topics gegen 1a bis 1h und E9
  python3 optimismus_phase1.py lauf --budget 150   Wiederholungen rechnen, bis das Zeitbudget erreicht ist (setzt fort)
  python3 optimismus_phase1.py report              Optimismus je Kriterium und Platz, Auswahlquoten, Spanne fuer B2
  python3 optimismus_phase1.py kriterien           K1 bis K6 und K3 aller Fassungen auf allen Topics der Phase, mit
                                                   Status und Abgleich gegen 1a bis 1h derselben Phase (fuer B1 und B2)
"""
import argparse
import hashlib
import json
import pickle
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import auswertung_eo4_k3 as ak  # noqa: E402  (laedt backtest_kandidaten und backtest_phase1, beide auf Phase 1)

bk = ak.bk
bt = bk.bt
phs = bt.phs
OUT = phs.ordner("output_v26_entwicklung/optimismus_phase1", "optimismus")

# Vor dem Rechnen festgelegt (Protokoll, Eintrag vom 17.10.2026, Optimismusschaetzung); nach Ergebnissen nicht veraendern.
WIEDERHOLUNGEN = 5
FOLDS = 5
SEED_BASIS = 1400          # Wiederholung r: Seed 1400 + r; Teilmengen des Trainingsteils: Seed [1400 + r, Fold]
TEILMENGEN = 20            # Zufallsteilmengen des Trainingsteils in Groesse und Schichtung des Testteils
K1_MIN_DOK, K1_ANTEIL = 10, 0.9
K2_AUS, K2_GUT = 0.5, 0.75
K3_SCHWELLE = ak.K3_SCHWELLE
K4_GRENZE, K5_GRENZE, K6_GRENZE = 0.2, 0.5, 0.9
KRITERIEN = ["K1", "K2", "K3", "K4", "K5", "K6"]
RICHTUNG = {"K1": 1, "K2": 1, "K3": 1, "K4": -1, "K5": -1, "K6": -1}  # Guete: hoeher besser (+1), niedriger (-1); K4 als Betrag
OHNE_PE1 = ("K4", "K5")  # E4 und Kriterium K4: fuer PE1 nicht anwendbar

PLAETZE = ["EO1", "EO2", "EO3", "EO4", "EO5", "PE1", "PE2", "PE3", "DS1", "DS2", "DS3", "DI1", "DI2", "DI3", "DI4",
           "IP1", "IP2", "IP3"]
R4_WAHL = {"EO3": "EO3_A1", "PE2": "PE2_A1", "DS1": "DS1_BEST", "DI1": "DI1_A1", "DI2": "DI2_A1",  # E2
           "DS2": "DS2_A3", "IP2": "IP2_A4"}                                                      # D1, E7
R2_VORRANG = {"IP2": ["IP2_A4", "IP2_A5"]}                                                       # E7
AUSWAHL_E9 = {"EO1": ("EO1_A1", "voll"), "EO2": ("EO2_A1", "voll"), "EO3": ("EO3_A1", "eingeschraenkt"),
              "EO4": ("EO4_R3B", "voll"), "EO5": ("EO5_A1", "voll"), "PE1": ("PE1_BEST", "voll"),
              "PE2": ("PE2_BEST", "eingeschraenkt"), "PE3": ("PE3_BEST", "voll"),
              "DS1": ("DS1_BEST", "eingeschraenkt"), "DS2": ("DS2_A3", "eingeschraenkt"), "DS3": ("DS3_BEST", "voll"),
              "DI1": ("DI1_A1", "eingeschraenkt"), "DI2": ("DI2_A1", "eingeschraenkt"), "DI3": ("DI3_BEST", "voll"),
              "DI4": ("DI4_A4", "voll"), "IP1": ("IP1_BEST", "voll"), "IP2": ("IP2_A4", "voll"),
              "IP3": ("IP3_A1", "voll")}

BEST = [bt.CODE[c] + "_BEST" for c in bt.IND]                     # bestehende Indikatoren (Werte aus 1a)
KAND = [c for c in bk.KCODES if c != "DIAG_A1"]                   # Kandidaten aus 1b bis 1f (EO4_A2 nur als Vergleich)
EO4_EXTRA = ["EO4_A1B", "EO4_A2B", "EO4_R3B"]                     # bereinigte Fassungen (Eintrag vom 17.10., Bereinigung)
FASS = BEST + KAND + EO4_EXTRA
PLATZ_VON = {**{b: b[:3] for b in BEST}, **{c: bk.PLATZ[c] for c in KAND}, **{c: "EO4" for c in EO4_EXTRA}}
KAND_PLATZ = {p: [c for c in KAND if bk.PLATZ[c] == p] for p in PLAETZE if p not in ("EO4", "EO5")}


def uhr(t0, text):
    print(f"[{time.time() - t0:7.1f} s] {text}", flush=True)


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def runde(x, k=6):
    return None if x is None or x != x else round(float(x), k)


def bevorzugt(p):
    b = bk.BEVORZUGT.get(p)
    if b == "bestehend":
        return p + "_BEST"
    return b if b in KAND else None


# ---------------------------------------------------------------- Korrelationen je Spalte (wie scipy, paarweise vollstaendig)

def _korr_spalten(A, B, raenge):
    n, k = A.shape
    res = np.full(k, np.nan)
    if n == 0:
        return res
    fehlt = np.isnan(A) | np.isnan(B)
    voll = ~fehlt.any(axis=0)
    if voll.any() and n > 10:
        a, b = A[:, voll], B[:, voll]
        if raenge:
            a, b = rankdata(a, axis=0), rankdata(b, axis=0)
        a = a - a.mean(axis=0)
        b = b - b.mean(axis=0)
        nenner = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
        with np.errstate(invalid="ignore", divide="ignore"):
            res[voll] = np.where(nenner > 0, (a * b).sum(axis=0) / nenner, np.nan)
    for j in np.where(~voll)[0]:
        ok = ~fehlt[:, j]
        if ok.sum() > 10:
            a, b = A[ok, j], B[ok, j]
            if raenge:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res[j] = spearmanr(a, b)[0]
            elif a.std() > 0 and b.std() > 0:
                res[j] = np.corrcoef(a, b)[0, 1]
    return res


def spearman_spalten(A, B):
    return _korr_spalten(A, B, True)


def pearson_spalten(A, B):
    return _korr_spalten(A, B, False)


# ---------------------------------------------------------------- Daten

def tabelle(name):
    """Bestehende Indikatoren (wie bt.lies, fuer Guetemasse nur bestimmbare Werte) und Kandidaten mit EO4 (wie in 1g)."""
    best = bt.lies(name)
    best.columns = [bt.CODE[c] + "_BEST" for c in best.columns]
    kand = ak.lies_mit_eo4(name)
    return pd.concat([best, kand[KAND + EO4_EXTRA]], axis=1)[FASS]


K1_ABWEICHUNGEN = {}


def k1_zaehlung(labels, tids, x_eo4):
    """Dokumente mit dem fuer K1 benoetigten Feld je Topic (Felder wie Korpus.abdeckung in 1b und repro in 1a)."""
    f = np.load(bk.CACHE / "felder.npz")
    L = {m: np.asarray(sparse.load_npz(bk.CACHE / f"{m}.npz").sum(axis=1)).ravel()
         for m in ("phrasen", "keywords", "kategorien", "kwplus", "institutionen", "laender", "autoren")}
    jahr = f["jahr"]
    feld = {"phrasen": L["phrasen"] > 0, "keywords": L["keywords"] > 0, "kategorien": L["kategorien"] > 0,
            "kwplus": L["kwplus"] > 0, "institutionen": L["institutionen"] > 0, "laender": L["laender"] > 0,
            "autoren": L["autoren"] > 0, "abstract": f["woerter_abstract"] >= bk.MIN_ABSTRACT,
            "journal": f["journal"] >= 0, "journal_kat": (f["journal"] >= 0) & (f["erste_kat"] >= 0),
            "typ": f["hat_typ"].astype(bool), "alle": np.ones(len(labels), dtype=bool),
            "ab2003": (jahr >= bk.NEU_AB) & (L["phrasen"] > 0), "ab2011": jahr >= bk.KOHORTE_AB}
    zug = labels >= 0

    def zaehle(maske):
        return pd.Series(np.asarray(maske)[zug].astype(int)).groupby(labels[zug]).sum().reindex(tids).fillna(0)

    abd = json.loads((bk.OUT / "repro_kandidaten.json").read_text(encoding="utf-8"))["abdeckung"]
    gross = 10 ** 9
    k1 = pd.DataFrame(gross, index=pd.Index(tids, name="topic"), columns=FASS, dtype=float)
    pruef = {}
    # Zaehlungen je Topic aus 1a und 1b derselben Phase (repro schreibt sie seit dem Phasenschalter, Protokoll 3, Punkt b)
    f_bk, f_bt = bk.OUT / "k1_je_topic.csv", bt.OUT / "k1_je_topic.csv"
    je_bk = pd.read_csv(f_bk, index_col=0) if f_bk.exists() else None
    je_bt = pd.read_csv(f_bt, index_col=0) if f_bt.exists() else None
    for c in KAND:
        fe = bk.FELD[c]
        if fe == "eo5":  # braucht das Unigrammvokabular; in 1f ohne Topic unter 10 Dokumenten
            if je_bk is not None and c in je_bk.columns:
                k1[c] = je_bk[c].reindex(tids).fillna(0).astype(float)
                pruef[c] = (int((k1[c] < K1_MIN_DOK).sum()), abd[c]["topics_unter_10_dok"])
            elif abd[c]["topics_unter_10_dok"] != 0:
                raise SystemExit("EO5 A1: K1 je Topic nicht aus dem Zwischenspeicher bestimmbar")
            continue
        k1[c] = zaehle(feld[fe])
        pruef[c] = (int((k1[c] < K1_MIN_DOK).sum()), abd[c]["topics_unter_10_dok"])
    if je_bt is not None:  # bestehende Indikatoren mit der Zaehlung aus 1a (Feld vorhanden und nicht leer)
        for c in BEST:
            if c not in ("EO1_BEST", "DS2_BEST", "IP3_BEST") and c[:3] in je_bt.columns:
                k1[c] = je_bt[c[:3]].reindex(tids).fillna(0).astype(float)
    for c in ("EO1_BEST", "DS2_BEST"):  # 1a: Author Keywords; alle uebrigen bestehenden ohne Topic unter 10 Dokumenten
        k1[c] = zaehle(feld["keywords"])
    k1["IP3_BEST"] = zaehle(feld["kwplus"])
    for c in ("EO4_A1B", "EO4_A2B"):
        k1[c] = zaehle(~np.isnan(x_eo4[c]))
    k1["EO4_R3B"] = np.minimum(k1["EO4_A1B"], k1["EO4_A2B"])
    rep = json.loads((bt.OUT / "repro.json").read_text(encoding="utf-8"))["abdeckung"]
    for c in BEST:
        soll = rep[c[:3]]["topics_unter_10_dok"]
        pruef[c] = (int((k1[c] < K1_MIN_DOK).sum()), soll)
    abw = {c: v for c, v in pruef.items() if v[0] != v[1]}
    if abw and (phs.ENTWICKLUNG or not set(abw) <= {"EO1_BEST", "DS2_BEST", "IP3_BEST"}):
        raise SystemExit(f"K1-Zaehlung weicht von 1a/1b ab: {abw}")
    K1_ABWEICHUNGEN.clear()
    K1_ABWEICHUNGEN.update(abw)  # Bestaetigung: nur bei Fassungen ausserhalb der Auswahl zulaessig, wird berichtet
    return k1


def teil_daten(t0):
    labels = bk.lade_labels()
    tids = sorted(set(labels[labels >= 0].tolist()))
    n = pd.Series(labels[labels >= 0]).value_counts().reindex(tids)
    splits = sorted({p.stem.split("_")[1] for p in bk.OUT.glob("split_r*_B.csv")})
    nulls = sorted(p.stem for p in bk.OUT.glob("null_r*.csv"))
    if (len(splits), len(nulls)) != (50, 50):
        raise SystemExit(f"Pseudokorpora unvollstaendig: {len(splits)} Teilungen, {len(nulls)} Permutationen")
    ref = tabelle("referenz")
    ref_voll = pd.read_csv(bt.OUT / "referenz.csv", index_col=0)[bt.IND]
    ref_voll.columns = BEST
    A = {s: tabelle(f"split_{s}_A") for s in splits}
    B = {s: tabelle(f"split_{s}_B") for s in splits}
    uhr(t0, "Referenz und 50 Teilungen gelesen")
    e0 = pd.concat([tabelle(p) for p in nulls], keys=range(len(nulls))).groupby(level=1).mean()[FASS]
    uhr(t0, "Nullerwartung aus 50 Permutationen")
    x_eo4 = ak.eo4_dokumente(len(labels))[0]
    k1 = k1_zaehlung(labels, tids, x_eo4)
    uhr(t0, "K1 je Topic gezaehlt und gegen 1a und 1b geprueft")
    # K3: angebundene Topics und Werte je Seed (wie teil_k3report). Referenz wie dort und nach 1h: die berichteten Labels,
    # wenn der Kontrolllauf mit Seed 42 sie reproduziert, sonst der Kontrolllauf (Protokoll 3, Eintrag zur Umsetzung, c)
    ref_ist_bericht = ak.ref_gleich()
    ref_lab = labels if ref_ist_bericht else np.load(ak.seed_dateien(42)[0]).astype(np.int64)
    ref_k, ref_b = ak.tabellen_fuer(ref_lab, "referenz" if ref_ist_bericht else "seed42", x_eo4)
    ref_b.columns = [bt.CODE[c] + "_BEST" for c in ref_b.columns]
    ref_k3 = pd.concat([ref_b, ref_k[KAND + EO4_EXTRA]], axis=1)[FASS]
    if ref_ist_bericht:
        d_k3 = float(np.nanmax(np.abs(ref_k3.reindex(tids).to_numpy(float) - ref.reindex(tids).to_numpy(float))))
        if d_k3 > 1e-9:
            raise SystemExit(f"Referenzwerte fuer K3 weichen ab ({d_k3})")
    paare, seedwerte = {}, {}
    for s in ak.K3_SEEDS:
        lab = np.load(ak.seed_dateien(s)[0]).astype(np.int64)
        p = pd.read_csv(ak.OUT_K3 / f"anbindung_seed{s}.csv")
        p2, _, _ = ak.anbindung(ref_lab, lab)
        if len(p) != len(p2):
            raise SystemExit(f"Anbindung Seed {s} weicht von k3report ab")
        if not (p[["ref", "seed"]].values == p2[["ref", "seed"]].values).all():
            raise SystemExit(f"Anbindung Seed {s} weicht von k3report ab")
        kd, bs = ak.tabellen_fuer(lab, f"seed{s}", x_eo4)
        bs.columns = [bt.CODE[c] + "_BEST" for c in bs.columns]
        paare[s] = p[["ref", "seed"]]
        seedwerte[s] = pd.concat([bs, kd[KAND + EO4_EXTRA]], axis=1)[FASS]
    uhr(t0, f"K3: {len(ak.K3_SEEDS)} Seeds mit Anbindung")
    terzil = pd.Series(pd.qcut(n.rank(method="first"), 3, labels=False).astype(int).values, index=n.index)
    D = {"tids": tids, "n": n, "logn": np.log(n.astype(float)), "terzil": terzil, "ref": ref, "ref_voll": ref_voll,
         "splits": splits, "A": A, "B": B, "e0": e0, "k1": k1, "seeds": list(ak.K3_SEEDS), "paare": paare,
         "seedwerte": seedwerte,
         "quellen": {f"{bt.KENNZAHLEN}.csv": sha16(bt.OUT / f"{bt.KENNZAHLEN}.csv"),
                     "kandidaten_kennzahlen.csv": sha16(bk.OUT / "kandidaten_kennzahlen.csv"),
                     "eo4_kennzahlen.csv": sha16(ak.AUS / "eo4_kennzahlen.csv"),
                     "k3_ergebnis.json": sha16(ak.AUS / "k3_ergebnis.json"),
                     "model_results.pkl": sha16(phs.lies(bt.REF / "model_results.pkl", "Pruefsumme des berichteten Laufs"))}}
    if K1_ABWEICHUNGEN:
        D["k1_abweichungen"] = {c: list(v) for c, v in K1_ABWEICHUNGEN.items()}
    if not ref_ist_bericht:  # K3 gegen den Kontrolllauf: dessen Werte, Topicnummern des Kontrolllaufs
        D["ref_k3"] = ref_k3
        D["k3_referenz"] = "Kontrolllauf Seed 42"
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "daten.pkl.tmp", "wb") as f:
        pickle.dump(D, f)
    Path(OUT / "daten.pkl.tmp").replace(OUT / "daten.pkl")
    uhr(t0, f"daten.pkl geschrieben: {len(tids)} Topics, {len(FASS)} Fassungen, Terzile {terzil.value_counts().to_dict()}")


def lade_daten():
    with open(OUT / "daten.pkl", "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- Kriterien auf einer Topicmenge

def kriterien(D, U, fass=None):
    """K1 bis K6 je Fassung auf der Topicmenge U, mit denselben Schaetzern wie 1a bis 1h, nur ueber die Topics in U."""
    fass = list(fass or FASS)
    U = list(U)
    z = pd.DataFrame(index=fass, columns=KRITERIEN, dtype=float)
    ref = D["ref"].loc[U, fass].to_numpy(float)
    z["K1"] = (D["k1"].loc[U, fass] >= K1_MIN_DOK).mean(axis=0).values
    rs = np.vstack([spearman_spalten(D["A"][s].loc[U, fass].to_numpy(float), D["B"][s].loc[U, fass].to_numpy(float))
                    for s in D["splits"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rh = np.nanmean(rs, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z["K2"] = np.where(rh > -1, 2 * rh / (1 + rh), np.nan)
    logn = np.repeat(D["logn"].loc[U].to_numpy(float)[:, None], len(fass), axis=1)
    z["K4"] = spearman_spalten(ref, logn)
    z["K5"] = pearson_spalten(ref, D["e0"].loc[U, fass].to_numpy(float)) ** 2
    # K6: Rangkorrelationen ueber U; bestehende wie 1a (volle Werte, untereinander), Kandidaten wie 1b, EO4 wie 1g
    R = D["ref"].loc[U].rank().corr().abs()
    Rv = D["ref_voll"].loc[U].rank().corr().abs()
    k6 = {}
    for f in fass:
        p = PLATZ_VON[f]
        if f in BEST:
            k6[f] = Rv[f].drop(f).max()
        elif f in EO4_EXTRA:
            k6[f] = max(R.loc[BEST, f].max(), R.loc[[c for c in KAND if bk.PLATZ[c] != "EO4"], f].max())
        else:
            k6[f] = max(R.loc[[b for b in BEST if b[:3] != p], f].max(), R.loc[[c for c in KAND if bk.PLATZ[c] != p], f].max())
    z["K6"] = pd.Series(k6)
    # K3: Mittel der Seed-Korrelationen ueber die angebundenen Topics in U (je Seed mehr als 10)
    Us = set(U)
    ref_k3 = D.get("ref_k3")  # nur gesetzt, wenn der Kontrolllauf die Referenz ist
    if ref_k3 is not None and Us != set(D["tids"]):
        raise SystemExit("K3 mit dem Kontrolllauf als Referenz ist nur auf allen Topics der Phase bestimmbar")
    k3 = []
    for s in D["seeds"]:
        pr = D["paare"][s]
        if ref_k3 is None:
            pr = pr[pr["ref"].isin(Us)]
        a = (D["ref"] if ref_k3 is None else ref_k3).reindex(pr["ref"].values)[fass].to_numpy(float)
        b = D["seedwerte"][s].reindex(pr["seed"].values)[fass].to_numpy(float)
        k3.append(spearman_spalten(a, b))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z["K3"] = np.nanmean(np.vstack(k3), axis=0)
    return z


def status(z):
    st = {}
    for f, r in z.iterrows():
        pe1 = PLATZ_VON[f] == "PE1"
        k1 = bool(r["K1"] >= K1_ANTEIL)
        k2 = "verfehlt" if not r["K2"] >= K2_AUS else ("bedingt" if r["K2"] < K2_GUT else "gut")
        k4 = True if pe1 else bool(abs(r["K4"]) < K4_GRENZE)
        k5 = True if pe1 else bool(r["K5"] <= K5_GRENZE)
        k6 = bool(r["K6"] <= K6_GRENZE)
        k3 = None if np.isnan(r["K3"]) else bool(r["K3"] >= K3_SCHWELLE)
        r1 = k1 and k2 != "verfehlt" and k4 and k5 and k6
        st[f] = {"K1": k1, "K2": k2, "K3": k3, "K4": k4, "K5": k5, "K6": k6, "r1": r1, "alle": r1 and k3 is not False,
                 "r2_ausloeser": (not k1) or k2 == "verfehlt" or (not k4) or (not k5)}
    return st


def waehle(st, z):
    """Mechanische Auswahl nach R1 bis R4 mit E1, E2, E4, E6, E7, E8, D1 und den Regeln fuer EO4 und EO5."""
    wahl = {}
    for p in PLAETZE:
        if p == "EO4":
            a1, a2 = st["EO4_A1B"]["alle"], st["EO4_A2B"]["alle"]
            wahl[p] = ("EO4_R3B", "voll") if (a1 and a2) else ("EO4_A2B", "voll" if a2 else "eingeschraenkt")
            continue
        if p == "EO5":
            wahl[p] = ("EO5_A1", "voll" if st["EO5_A1"]["alle"] else "eingeschraenkt")
            continue
        best, alts = p + "_BEST", KAND_PLATZ[p]
        b = st[best]
        if b["r1"]:
            wahl[p] = (best, "voll" if b["K3"] is not False else "eingeschraenkt")
            continue
        ok = [c for c in alts if st[c]["alle"]]
        if b["r2_ausloeser"] and ok:
            vor = R2_VORRANG.get(p, []) + [x for x in [bevorzugt(p)] if x in alts] + alts
            wahl[p] = (next(c for c in vor if c in ok), "voll")
            continue
        c4 = R4_WAHL.get(p) or bevorzugt(p) or best
        if st[c4]["K3"] is False:
            k3_ok = [c for c in [best] + alts if st[c]["K3"] is True]
            if k3_ok:
                c4 = min(k3_ok, key=lambda c: abs(z.loc[c, "K4"]))
        wahl[p] = (c4, "eingeschraenkt")
    return wahl


# ---------------------------------------------------------------- Pruefung auf allen Topics

def abgleich_berichte(z, st):
    """K2 bis K6 und K3 sowie die Status gegen 1a (bestehende), 1b (Kandidaten), 1g (EO4) und 1h (K3) derselben Phase."""
    b1a = pd.read_csv(bt.OUT / f"{bt.KENNZAHLEN}.csv").set_index("Indikator")
    kk = pd.read_csv(bk.OUT / "kandidaten_kennzahlen.csv").set_index("Code")
    e4 = pd.read_csv(ak.AUS / "eo4_kennzahlen.csv").set_index("Code")
    k3j = json.loads((ak.AUS / "k3_ergebnis.json").read_text(encoding="utf-8"))["k3"]
    spalte = {"K2": "K2_spearman_brown", "K4": "K4_rho_logn", "K5": "K5_null_r2", "K6": "K6_max_abs_rho"}
    abw = {}
    for f in FASS:
        if f in BEST:
            soll = {k: b1a.loc[f[:3], v] for k, v in spalte.items()}
            soll["K3"] = k3j[bt.IND[BEST.index(f)]]["mittel"]
            tol = {"K2": 6e-4, "K4": 6e-4, "K5": 6e-4, "K6": 6e-4, "K3": 2e-6}
        else:
            tab = e4 if f in EO4_EXTRA else kk
            soll = {k: tab.loc[f, v] for k, v in spalte.items()}
            soll["K3"] = k3j[f]["mittel"]
            tol = {"K2": 2e-6, "K4": 2e-6, "K5": 2e-6, "K6": 2e-6, "K3": 2e-6}
        for k in ("K2", "K3", "K4", "K5", "K6"):
            ist, ziel = float(z.loc[f, k]), (np.nan if soll[k] is None else float(soll[k]))
            if ist != ist and ziel != ziel:  # beide nicht bestimmbar (etwa K3 ohne genug angebundene Topics)
                continue
            d = abs(ist - ziel)
            if not d <= tol[k]:
                abw[f"{f} {k}"] = [runde(z.loc[f, k]), runde(soll[k])]
    # Status gegen 1b (Kandidaten), 1g (EO4) und die Entscheidungsvorlage (bestehende)
    vorlage = json.loads((bk.OUT / "entscheidungsvorlage_regeln.json").read_text(encoding="utf-8"))["bestehend"]
    norm = {"erfuellt": True, "verfehlt": False, "entfaellt": True}
    s_abw = {}
    for f in FASS:
        if f in BEST:
            v = vorlage[f[:3]]
            soll = {"K1": norm[v["K1"]], "K2": v["K2"], "K4": norm[v["K4"]], "K5": norm[v["K5"]], "K6": norm[v["K6"]]}
        else:
            r = (e4 if f in EO4_EXTRA else kk).loc[f]
            soll = {"K1": norm[r["S_K1"]], "K2": r["S_K2"], "K4": norm[r["S_K4"]], "K5": norm[r["S_K5"]],
                    "K6": norm[r["S_K6"]]}
        for k, v in soll.items():
            if st[f][k] != v and not _an_der_schwelle(z.loc[f, k], k, 6e-4 if f in BEST else 2e-6):
                s_abw[f"{f} {k}"] = [st[f][k], v]
    return abw, s_abw


SCHWELLEN = {"K1": [K1_ANTEIL], "K2": [K2_AUS, K2_GUT], "K4": [K4_GRENZE], "K5": [K5_GRENZE], "K6": [K6_GRENZE]}


def _an_der_schwelle(wert, k, tol):
    """Ein Status aus gerundeten Kennzahlen (1a dreistellig, 1b und 1g sechsstellig, K1 vierstellig) kann an der Schwelle
    vom ungerundeten Wert abweichen; das ist keine Abweichung der Rechnung."""
    w = abs(float(wert)) if k == "K4" else float(wert)
    return w == w and any(abs(w - t) <= (5e-5 if k == "K1" else tol) for t in SCHWELLEN[k])


def teil_regress(t0):
    D = lade_daten()
    z = kriterien(D, D["tids"])
    uhr(t0, "Kriterien auf allen 146 Topics gerechnet")
    st = status(z)
    abw, s_abw = abgleich_berichte(z, st)
    wahl = waehle(st, z)
    w_abw = {p: [list(wahl[p]), list(AUSWAHL_E9[p])] for p in PLAETZE if wahl[p] != AUSWAHL_E9[p]}
    erg = {"kennzahlen_abweichungen": abw, "status_abweichungen": s_abw, "auswahl_abweichungen": w_abw,
           "auswahl": {p: list(w) for p, w in wahl.items()}, "quellen": D["quellen"],
           "bestanden": not (abw or s_abw or w_abw)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "regression.json").write_text(json.dumps(erg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(erg, ensure_ascii=False))
    if not erg["bestanden"]:
        raise SystemExit("Regressionspruefung nicht bestanden; kein Lauf")


def teil_kriterien(t0):
    """K1 bis K6 und K3 aller 51 Fassungen auf allen Topics der Phase, mit Status nach 0.5, E1 und E4 (Grundlage fuer B1
    und B2 in 3.4) und dem Abgleich gegen die Berichte aus 1a bis 1h derselben Phase. Keine Auswahl, keine Zugehoerigkeit."""
    if not phs.ENTWICKLUNG:
        phs.aktiver_lauf()  # im Bestaetigungsmodus nur in einem offenen Lauf (nach dem Abschluss wird nichts geschrieben)
    D = lade_daten()
    z = kriterien(D, D["tids"])
    uhr(t0, f"Kriterien auf allen {len(D['tids'])} Topics gerechnet")
    st = status(z)
    abw, s_abw = abgleich_berichte(z, st)
    OUT.mkdir(parents=True, exist_ok=True)
    z.round(12).to_csv(OUT / "kriterien_alle_topics.csv")
    erg = {"phase": phs.PHASE, "topics": len(D["tids"]), "fassungen": len(z), "kennzahlen_abweichungen": abw,
           "status_abweichungen": s_abw, "abgleich_bestanden": not (abw or s_abw),
           "k1_abweichungen": D.get("k1_abweichungen", {}), "status": st, "quellen": D["quellen"]}
    (OUT / "kriterien_alle_topics.json").write_text(json.dumps(erg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: erg[k] for k in ("phase", "topics", "fassungen", "abgleich_bestanden", "kennzahlen_abweichungen",
                                          "status_abweichungen", "k1_abweichungen")}, ensure_ascii=False))


# ---------------------------------------------------------------- Kreuzvalidierung

def folds(D, r):
    """Fold-Nummer je Topic fuer Wiederholung r, geschichtet nach Groessenterzil."""
    rng = np.random.default_rng(SEED_BASIS + r)
    fold = pd.Series(-1, index=D["n"].index)
    for t in sorted(D["terzil"].unique()):
        ids = D["terzil"].index[D["terzil"] == t].to_numpy()
        ids = ids[rng.permutation(len(ids))]
        fold.loc[ids] = np.arange(len(ids)) % FOLDS
    return fold


def gute(z, codes):
    """Kriterienwerte der Fassungen; K4 als Betrag."""
    w = z.loc[codes, KRITERIEN].copy()
    w["K4"] = w["K4"].abs()
    return w


def erfuellt_anteil(st, codes):
    return float(np.mean([st[c]["alle"] for c in codes]))


def ein_fold(D, r, f, fold):
    V = fold.index[fold == f].tolist()
    T = fold.index[fold != f].tolist()
    z_T = kriterien(D, T)
    st_T = status(z_T)
    wahl = waehle(st_T, z_T)
    sel = [wahl[p][0] for p in PLAETZE]
    fest = [AUSWAHL_E9[p][0] for p in PLAETZE]
    codes = list(dict.fromkeys(sel + fest))
    z_V = kriterien(D, V, codes)
    st_V = status(z_V)
    rng = np.random.default_rng([SEED_BASIS + r, f])
    terz = D["terzil"]
    soll = terz.loc[V].value_counts()
    teil_w, teil_sel, teil_fest = [], [], []
    for _ in range(TEILMENGEN):
        U = []
        for t, anz in soll.items():
            pool = [i for i in T if terz.loc[i] == t]
            U += list(rng.choice(pool, size=int(anz), replace=False))
        z_U = kriterien(D, U, codes)
        st_U = status(z_U)
        teil_w.append(gute(z_U, codes))
        teil_sel.append(erfuellt_anteil(st_U, sel))
        teil_fest.append(erfuellt_anteil(st_U, fest))
    in_w = pd.concat(teil_w).groupby(level=0).mean().reindex(codes)
    return {"wdh": r, "fold": f, "n_train": len(T), "n_test": len(V),
            "wahl": {p: list(w) for p, w in wahl.items()},
            "train": gute(z_T, codes).round(6).to_dict(orient="index"),
            "test": gute(z_V, codes).round(6).to_dict(orient="index"),
            "train_teil": in_w.round(6).to_dict(orient="index"),
            "erfuellt": {"sel_in": round(float(np.mean(teil_sel)), 6), "sel_out": erfuellt_anteil(st_V, sel),
                         "fest_in": round(float(np.mean(teil_fest)), 6), "fest_out": erfuellt_anteil(st_V, fest)}}


def teil_lauf(t0, budget):
    D = lade_daten()
    reg = json.loads((OUT / "regression.json").read_text(encoding="utf-8")) if (OUT / "regression.json").exists() else {}
    if not reg.get("bestanden"):
        raise SystemExit("Regressionspruefung fehlt oder nicht bestanden")
    for r in range(WIEDERHOLUNGEN):
        ziel = OUT / f"lauf_wdh{r}.json"
        if ziel.exists():
            continue
        if time.time() - t0 > budget:
            break
        fold = folds(D, r)
        res = [ein_fold(D, r, f, fold) for f in range(FOLDS)]
        ziel.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        uhr(t0, f"Wiederholung {r} fertig: Foldgroessen {fold.value_counts().sort_index().tolist()}")
    fertig = sum((OUT / f"lauf_wdh{r}.json").exists() for r in range(WIEDERHOLUNGEN))
    print(f"STAND: {fertig} von {WIEDERHOLUNGEN} Wiederholungen", flush=True)


# ---------------------------------------------------------------- Report

def optimismus(rec, codes_je_platz, oben, unten):
    """Mittel ueber die Plaetze von Richtung x (oben minus unten) je Kriterium; K4 und K5 ohne PE1, K3 ohne fehlende Werte."""
    o = {}
    je_platz = {}
    for k in KRITERIEN:
        werte = []
        for p, c in codes_je_platz.items():
            if k in OHNE_PE1 and p == "PE1":
                continue
            a, b = rec[oben][c][k], rec[unten][c][k]
            if a is None or b is None or a != a or b != b:
                continue
            d = RICHTUNG[k] * (a - b)
            werte.append(d)
            je_platz.setdefault(p, {})[k] = d
        o[k] = float(np.mean(werte)) if werte else float("nan")
    return o, je_platz


def kennwerte(x):
    x = np.asarray([v for v in x if v == v], dtype=float)
    if not len(x):
        return None
    return {"mittel": runde(x.mean()), "median": runde(np.median(x)), "p5": runde(np.percentile(x, 5)),
            "p95": runde(np.percentile(x, 95)), "min": runde(x.min()), "max": runde(x.max()), "folds": int(len(x))}


def teil_report():
    recs = []
    for r in range(WIEDERHOLUNGEN):
        ziel = OUT / f"lauf_wdh{r}.json"
        if not ziel.exists():
            raise SystemExit(f"Wiederholung {r} fehlt")
        recs += json.loads(ziel.read_text(encoding="utf-8"))
    fest = {p: AUSWAHL_E9[p][0] for p in PLAETZE}
    zeilen, plaetze = [], {}
    for rec in recs:
        sel = {p: rec["wahl"][p][0] for p in PLAETZE}
        o_in, jp = optimismus(rec, sel, "train_teil", "test")
        o_voll, _ = optimismus(rec, sel, "train", "test")
        o_null, _ = optimismus(rec, fest, "train_teil", "test")
        z = {"wdh": rec["wdh"], "fold": rec["fold"], "n_train": rec["n_train"], "n_test": rec["n_test"]}
        for k in KRITERIEN:
            z[f"opt_{k}"], z[f"opt_ohne_angleich_{k}"], z[f"null_{k}"] = o_in[k], o_voll[k], o_null[k]
        e = rec["erfuellt"]
        z["opt_erfuellt"], z["null_erfuellt"] = e["sel_in"] - e["sel_out"], e["fest_in"] - e["fest_out"]
        z["erfuellt_in"], z["erfuellt_out"] = e["sel_in"], e["sel_out"]
        z["gleiche_wahl"] = int(sum(rec["wahl"][p] == list(AUSWAHL_E9[p]) for p in PLAETZE))
        zeilen.append(z)
        for p in PLAETZE:
            q = plaetze.setdefault(p, {"wahl": Counter(), "opt": {k: [] for k in KRITERIEN}})
            q["wahl"][" ".join(rec["wahl"][p])] += 1
            for k, v in jp.get(p, {}).items():
                q["opt"][k].append(v)
    tab = pd.DataFrame(zeilen)
    tab.round(6).to_csv(OUT / "optimismus_folds.csv", index=False)
    erg = {"folds": len(recs), "kriterien": {}, "ohne_angleich": {}, "nullprobe": {}, "spanne_b2": {}, "plaetze": {}}
    for k in KRITERIEN:
        erg["kriterien"][k] = kennwerte(tab[f"opt_{k}"])
        erg["ohne_angleich"][k] = kennwerte(tab[f"opt_ohne_angleich_{k}"])
        erg["nullprobe"][k] = kennwerte(tab[f"null_{k}"])
        kw = erg["kriterien"][k]
        erg["spanne_b2"][k] = None if kw is None else [kw["p5"], kw["p95"]]
    erg["erfuellt"] = {"optimismus": kennwerte(tab["opt_erfuellt"]), "nullprobe": kennwerte(tab["null_erfuellt"]),
                       "in_sample": kennwerte(tab["erfuellt_in"]), "out_of_fold": kennwerte(tab["erfuellt_out"])}
    erg["gleiche_wahl_je_fold"] = kennwerte(tab["gleiche_wahl"])
    zp = []
    for p in PLAETZE:
        q = plaetze[p]
        soll = " ".join(AUSWAHL_E9[p])
        anteil_code = sum(v for w, v in q["wahl"].items() if w.split()[0] == AUSWAHL_E9[p][0]) / len(recs)
        erg["plaetze"][p] = {"wahl": dict(q["wahl"]), "anteil_wie_e9": runde(q["wahl"][soll] / len(recs)),
                             "anteil_gleiche_fassung": runde(anteil_code),
                             "optimismus": {k: runde(np.mean(v)) if v else None for k, v in q["opt"].items()}}
        zp.append({"platz": p, "e9": soll, "anteil_wie_e9": erg["plaetze"][p]["anteil_wie_e9"],
                   "anteil_gleiche_fassung": runde(anteil_code), "wahlen": "; ".join(f"{w}: {v}" for w, v in q["wahl"].most_common()),
                   **{f"opt_{k}": erg["plaetze"][p]["optimismus"][k] for k in KRITERIEN}})
    pd.DataFrame(zp).to_csv(OUT / "optimismus_plaetze.csv", index=False)
    erg["vorgaben"] = {"wiederholungen": WIEDERHOLUNGEN, "folds": FOLDS, "seed_basis": SEED_BASIS, "teilmengen": TEILMENGEN,
                       "schichtung": "Groessenterzil", "richtung": RICHTUNG, "k4_k5_ohne": "PE1"}
    erg["quellen"] = json.loads((OUT / "regression.json").read_text(encoding="utf-8"))["quellen"]
    (OUT / "optimismus_ergebnis.json").write_text(json.dumps(erg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: erg[k] for k in ("kriterien", "nullprobe", "ohne_angleich", "erfuellt", "gleiche_wahl_je_fold")},
                     ensure_ascii=False, indent=1))
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80):
        print(pd.DataFrame(zp)[["platz", "e9", "anteil_wie_e9", "anteil_gleiche_fassung", "wahlen"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teil", choices=["daten", "regress", "lauf", "report", "kriterien"])
    ap.add_argument("--budget", type=float, default=150)
    a = ap.parse_args()
    t0 = time.time()
    if a.teil == "daten":
        teil_daten(t0)
    elif a.teil == "regress":
        teil_regress(t0)
    elif a.teil == "lauf":
        teil_lauf(t0, a.budget)
    elif a.teil == "kriterien":
        teil_kriterien(t0)
    else:
        teil_report()


if __name__ == "__main__":
    main()
