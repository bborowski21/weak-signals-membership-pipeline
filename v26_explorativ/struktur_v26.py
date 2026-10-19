#!/usr/bin/env python3
"""Strukturpruefung der fuenf Dimensionen, Teile B bis D (explorativ nach der Bestaetigung; Protokoll V2.6, Eintrag
vom 19.10.2026 zur Strukturpruefung, vor dem Lauf eingetragen). Aendert nichts an V2.6; nichts ist Auswahlkriterium.

Liest nur abgeleitete Ergebnisse: zugehoerigkeiten.csv der Phase 1 (output_v26_entblinden/phase1) und der Phase 2
(output_v26_bestaetigung/phase2/entblinden) mit Topicgroesse n, den Werten der 18 eingefrorenen Fassungen und den
Zugehoerigkeiten. Jeder Lesezugriff steht vorher in output_v26_explorativ/zugriffsprotokoll.jsonl (zugriff.py). Vor
jeder Rechnung werden beide Phasen geprueft (endliche Werte, n > 0, eindeutige Topics, jede Dimension mit mindestens
einer gueltigen Fassung, auch in den Varianten) und die neu gerechneten Zugehoerigkeiten mit den gespeicherten
abgeglichen; jede Verletzung bricht ab, bevor gerechnet wird.

  B Unterscheidbarkeit: Rangkorrelationen der fuenf Dimensionswerte paarweise, roh und partiell auf log n, mit
    Bootstrap-Intervallen; Eigenanteil je Dimension (1 - R^2 auf die vier anderen); effektive Zahl.
  C Kompositbedingung: je Fassung x der Dimension j und fremder Dimension k Delta = r(x, D_k) - r(D_j, D_k) * r(x, D_j)
    (Pearson, klassische z-Werte), mit Bootstrap-Intervall und Vorzeichen; Grenze |r(x, D_k)| > |r(D_j, D_k)|; roh und,
    nur berichtet, mit allen Fassungen ausser PE1 auf log n residualisiert (in jeder Ziehung neu). Fuer die
    Geschwister der benannten Nahtstellen zusaetzlich Delta mit D_j ohne die benannte Fassung (moeglicher Uebertrag).
  D Folgenrelevanz: Zugehoerigkeiten unter sieben Varianten der Zuordnung, Uebereinstimmung der dominanten Konfiguration
    mit V2.6, Anteile, mittlere Aenderung der Zugehoerigkeiten.
Eine Kennzahl, deren Ziehungen zu weniger als 99 % endlich sind (eine gueltige Fassung ist in einer Ziehung konstant),
gilt als nicht bestimmbar und wird so ausgewiesen. Die Lesarten folgen den vorab festgelegten Regeln (Konstanten unten).
Ausgabe: output_v26_explorativ/struktur/struktur_ergebnis.json, nur Kennzahlen, keine Werte je Topic; eine vorhandene
Ergebnisdatei wird nicht ueberschrieben.

Aufruf: python3 struktur_v26.py rechnen   (auf Bens Mac; liest die beiden Dateien)
        python3 struktur_v26.py test      (synthetische Daten, liest nichts; schreibt nur in ein temporaeres
                                           Verzeichnis, das danach geloescht wird)
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))

SKRIPT = "v26_explorativ/struktur_v26.py"
PLAETZE = ["EO1", "EO2", "EO3", "EO4", "EO5", "PE1", "PE2", "PE3", "DS1", "DS2", "DS3", "DI1", "DI2", "DI3", "DI4",
           "IP1", "IP2", "IP3"]
DIMS = ["EO", "PE", "DS", "DI", "IP"]
KONFIG = ["m_ws", "m_trend", "m_ec", "m_latent"]
MIN_SD = 0.01
K_BASIS, L_BASIS = 1.0, 0.5
BOOT, BOOT_SEED = 2000, 7
ANTEIL_ENDLICH = 0.99            # Mindestanteil endlicher Ziehungen je Kennzahl, sonst nicht bestimmbar
TOL_ABGLEICH = 1e-9

# Vorab festgelegte Schwellen (eigene Konventionen, Protokolleintrag vom 19.10.2026)
PAAR_ZUSAMMEN, PAAR_ENG = 0.85, 0.70
EIGEN_KAUM, EIGEN_EINGESCHRAENKT = 0.25, 0.50
DELTA_ABWEICHUNG = 0.20
FOLGEN_ARM, FOLGEN_STRUKTUR = 0.90, 0.80
BENANNT = {("IP3", "EO"), ("DS1", "IP"), ("IP1", "DS")}           # vorab benannte Nahtstellen
# Geschwister je benannter Nahtstelle: dieselbe fremde Dimension, Fassungen derselben eigenen Dimension
UEBERTRAG = {("IP3", "EO"): ("IP1", "IP2"), ("DS1", "IP"): ("DS2", "DS3"), ("IP1", "DS"): ("IP2", "IP3")}
GESCHWISTER = {(x, k): q for (q, k), xs in UEBERTRAG.items() for x in xs}

V26 = {"EO": ["EO1", "EO2", "EO3", "EO4", "EO5"], "PE": ["PE1", "PE2", "PE3"], "DS": ["DS1", "DS2", "DS3"],
       "DI": ["DI1", "DI2", "DI3", "DI4"], "IP": ["IP1", "IP2", "IP3"]}
KERN = ["EO", "PE", "DS", "DI"]
EC = ["DS1", "IP1", "IP2", "IP3"]
SPALTEN = ["topic", "n"] + [f"wert_{p}" for p in PLAETZE] + KONFIG + ["margin", "dominant"]


def struktur(dims, kern=KERN, wp="IP", ec=EC):
    return {"dims": dims, "kern": kern, "wp": wp, "ec": ec}


def ersetze(**neu):
    d = {k: list(v) for k, v in V26.items()}
    d.update({k: list(v) for k, v in neu.items()})
    return d


# Varianten der Folgenrelevanz (D1 bis D5 nach dem Rat vom 19.10., D6 und D7 nach der Inhaltszuordnung)
VARIANTEN = [
    ("D1", "IP3 in der Offenheit", struktur(ersetze(EO=V26["EO"] + ["IP3"], IP=["IP1", "IP2"]), ec=["DS1", "IP1", "IP2"])),
    ("D2", "Wirkung ohne IP3", struktur(ersetze(IP=["IP1", "IP2"]), ec=["DS1", "IP1", "IP2"])),
    ("D3", "Stadium und Diffusion als eine Achse",
     struktur({"EO": V26["EO"], "PE": V26["PE"], "DSDI": V26["DS"] + V26["DI"], "IP": V26["IP"]}, kern=["EO", "PE", "DSDI"])),
    ("D4", "Offenheit nur aus EO1, EO3, EO5", struktur(ersetze(EO=["EO1", "EO3", "EO5"]))),
    ("D5", "Stadium ohne DS1", struktur(ersetze(DS=["DS2", "DS3"]))),
    ("D6", "EO2 in der Diffusion", struktur(ersetze(EO=["EO1", "EO3", "EO4", "EO5"], DI=V26["DI"] + ["EO2"]))),
    ("D7", "Zuordnung nach der Inhaltsmehrheit des Hauptlaufs",
     struktur({"EO": ["EO3", "EO4", "EO5"], "PE": ["PE1", "PE2", "PE3"],
               "DS": ["DS1", "DS2", "DS3", "EO1", "DI1", "DI4", "IP1", "IP3"], "DI": ["DI2", "DI3", "EO2"], "IP": ["IP2"]})),
]


class Abbruch(SystemExit):
    pass


def abbruch(text):
    raise Abbruch("Abbruch: " + text)


for _, _, _s in VARIANTEN:
    _alle = sum(_s["dims"].values(), [])
    if len(_alle) != len(set(_alle)) or not set(_alle) <= set(PLAETZE):
        abbruch("Variante falsch definiert")


def r(x, k=6):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, k)


def intervall(b):
    """Perzentilintervall 2,5 und 97,5 % je Zelle ueber die endlichen Ziehungen und Anteil der endlichen Ziehungen."""
    b = np.asarray(b, float)
    fin = np.isfinite(b)
    b = np.where(fin, b, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanquantile(b, 0.025, axis=0)
        hi = np.nanquantile(b, 0.975, axis=0)
    return lo, hi, fin.mean(axis=0)


# ---------------------------------------------------------------- Schicht wie entblinden.zugehoerigkeit
def klassisch_z(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / (sd if sd > 0 else 1.0)


def robust_z(s, eps=1e-9):
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    return (s - med) / max(iqr / 2.0, eps)


def sigmoid(x, k):
    return 1.0 / (1.0 + np.exp(-k * x))


def gueltige(X):
    return [p for p in X.columns if X[p].std() > MIN_SD]


def dimensionswerte(X, gueltig, dims):
    Z = X.apply(klassisch_z, axis=0)
    out = pd.DataFrame(index=X.index)
    for d, ps in dims.items():
        ps = [p for p in ps if p in gueltig]
        out[d] = Z[ps].mean(axis=1) if ps else 0.0
    return out


def zugehoerigkeit(X, gueltig, st, k=K_BASIS, lam=L_BASIS):
    zd = dimensionswerte(X, gueltig, st["dims"]).apply(robust_z, axis=0)
    zec = X[st["ec"]].apply(robust_z, axis=0)
    kern = zd[st["kern"]].mean(axis=1)
    m = pd.DataFrame({"m_ws": sigmoid(kern, k),
                      "m_trend": sigmoid(-kern + lam * zd[st["wp"]], k),
                      "m_ec": sigmoid(zec.mean(axis=1), k),
                      "m_latent": sigmoid(-zd[st["kern"] + [st["wp"]]].mean(axis=1), k)}, index=X.index)
    s = np.sort(m[KONFIG].values, axis=1)
    m["margin"] = s[:, -1] - s[:, -2]
    m["dominant"] = m[KONFIG].idxmax(axis=1)
    return m


# ---------------------------------------------------------------- Pruefungen vor jeder Rechnung
def logn_aus(n):
    n = pd.Series(n, dtype=float)
    if not np.isfinite(n.to_numpy()).all() or (n <= 0).any():
        abbruch("Topicgroesse n nicht endlich oder nicht positiv")
    return np.log(n.to_numpy())


def pruefe_eingabe(ph, X, logn):
    if list(X.columns) != PLAETZE:
        abbruch(f"Phase {ph}: Spalten der Fassungen nicht wie erwartet")
    if not X.index.is_unique:
        abbruch(f"Phase {ph}: Topics nicht eindeutig")
    if len(X) < 30:
        abbruch(f"Phase {ph}: zu wenige Topics ({len(X)})")
    if not np.isfinite(X.to_numpy(float)).all():
        abbruch(f"Phase {ph}: nicht endliche Fassungswerte")
    ln = np.asarray(logn, float)
    if ln.shape != (len(X),) or not np.isfinite(ln).all():
        abbruch(f"Phase {ph}: log n nicht endlich oder falsche Laenge")
    gueltig = gueltige(X)
    for name, st in [("V2.6", struktur(V26))] + [(kz, st) for kz, _, st in VARIANTEN]:
        leer = [d for d, ps in st["dims"].items() if not any(p in gueltig for p in ps)]
        if leer:
            abbruch(f"Phase {ph}: {name} ohne gueltige Fassung in {leer}")
    return gueltig


def abgleich(ph, basis, gesp):
    if not gesp.index.equals(basis.index):
        abbruch(f"Phase {ph}: Topics der gespeicherten Zugehoerigkeiten weichen ab")
    g = gesp[KONFIG + ["margin"]].to_numpy(float)
    if not np.isfinite(g).all():
        abbruch(f"Phase {ph}: gespeicherte Zugehoerigkeiten nicht endlich")
    diff = float(np.abs(basis[KONFIG + ["margin"]].to_numpy(float) - g).max())
    dom = bool((basis["dominant"].astype(str) == gesp["dominant"].astype(str)).all())
    out = {"max_abs_zugehoerigkeit_und_margin": float(f"{diff:.3e}"), "dominant_gleich": dom}
    if not (diff < TOL_ABGLEICH and dom):
        abbruch(f"Phase {ph}: neu gerechnete Grundfassung weicht von der gespeicherten ab {out}")
    return out


# ---------------------------------------------------------------- B Unterscheidbarkeit
def spalten(gueltig):
    """Spaltenindizes der gueltigen Fassungen je Dimension (Reihenfolge PLAETZE)."""
    return {d: [PLAETZE.index(p) for p in V26[d] if p in gueltig] for d in DIMS}


def z_und_dim(Xa, gidx, konst=None):
    """Klassische z-Werte (Standardabweichung mit n) und Dimensionswerte als Mittel der gueltigen Fassungen, wie
    entblinden.zugehoerigkeit (dort vor robust_z, das die Korrelationen nicht aendert). Eine in der Stichprobe konstante
    Fassung erhaelt z = 0 (exakt); konst gibt die Konstanz vor, wenn Xa schon residualisiert ist (dann an den Rohwerten
    der Ziehung bestimmt), sonst gilt eine Spalte mit Maximum gleich Minimum als konstant."""
    mu = Xa.mean(axis=0)
    sd = Xa.std(axis=0)
    if konst is None:
        konst = Xa.max(axis=0) == Xa.min(axis=0)
    gut = (sd > 0) & ~np.asarray(konst, bool)
    Z = np.where(gut, (Xa - mu) / np.where(gut, sd, 1.0), 0.0)
    dm = np.column_stack([Z[:, gidx[d]].mean(axis=1) if gidx[d] else np.zeros(len(Xa)) for d in DIMS])
    return Z, dm


def raenge(dm, ln, partiell):
    """Raenge der Dimensionswerte, partiell als Residuen auf den Rang von log n; eine konstante Spalte bleibt exakt
    konstant (ihre Korrelationen werden nicht endlich)."""
    R = rankdata(dm, axis=0)
    if partiell:
        konst = R.max(axis=0) == R.min(axis=0)
        A = np.column_stack([np.ones(len(R)), rankdata(ln)])
        R = R - A @ np.linalg.lstsq(A, R, rcond=None)[0]
        R[:, konst] = 0.0
    return R


def b_kennzahlen(R):
    """Rangkorrelationen, Eigenanteil 1 - R^2 je Spalte auf die uebrigen (= 1 / Diagonale der inversen
    Korrelationsmatrix) und effektive Zahl p^2 / (p + 2 * Summe der quadrierten Korrelationen). Bei einer konstanten
    Spalte (nur in einer Ziehung moeglich) werden die betroffenen Werte nicht endlich."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(R, rowvar=False)
    p = c.shape[0]
    iu = np.triu_indices(p, 1)
    try:
        eigen = 1.0 / np.diag(np.linalg.inv(c)) if np.isfinite(c).all() else np.full(p, np.nan)
    except np.linalg.LinAlgError:
        eigen = np.full(p, np.nan)
    eff = (p * p) / (p + 2.0 * float((c[iu] ** 2).sum()))
    return c, list(eigen), eff


def teil_b(X, gueltig, logn, boot, rng_seed=BOOT_SEED):
    Xa = X[PLAETZE].to_numpy(float)
    gidx = spalten(gueltig)
    _, dm = z_und_dim(Xa, gidx)
    ln = np.asarray(logn, float)
    erg = {}
    for partiell in (False, True):
        c, eigen, eff = b_kennzahlen(raenge(dm, ln, partiell))
        if not (np.isfinite(c).all() and np.isfinite(eigen).all() and np.isfinite(eff)):
            abbruch("Teil B: Punktschaetzung nicht endlich")
        rng = np.random.default_rng(rng_seed)
        bc, be, bf = [], [], []
        for _ in range(boot):
            i = rng.integers(0, len(Xa), len(Xa))
            _, dm_b = z_und_dim(Xa[i], gidx)
            c_b, e_b, f_b = b_kennzahlen(raenge(dm_b, ln[i], partiell))
            bc.append(c_b)
            be.append(e_b)
            bf.append(f_b)
        clo, chi, cfin = intervall(bc)
        elo, ehi, efin = intervall(be)
        flo, fhi, ffin = intervall(bf)
        paare = {}
        for (a, da), (b, db) in itertools.combinations(list(enumerate(DIMS)), 2):
            w = c[a, b]
            paare[f"{da}-{db}"] = {"rho": r(w), "ci95": [r(clo[a, b]), r(chi[a, b])], "anteil_endlich": r(cfin[a, b], 4),
                                   "intervall_bestimmbar": bool(cfin[a, b] >= ANTEIL_ENDLICH),
                                   "klasse": "zusammenfallend" if abs(w) >= PAAR_ZUSAMMEN else "eng" if abs(w) >= PAAR_ENG else "unterscheidbar"}
        eig = {}
        for j, d in enumerate(DIMS):
            w = eigen[j]
            eig[d] = {"eigenanteil": r(w), "ci95": [r(elo[j]), r(ehi[j])], "anteil_endlich": r(efin[j], 4),
                      "intervall_bestimmbar": bool(efin[j] >= ANTEIL_ENDLICH),
                      "klasse": "kaum eigenständig" if w < EIGEN_KAUM else "eingeschränkt" if w < EIGEN_EINGESCHRAENKT else "eigenständig"}
        erg["partiell_logn" if partiell else "roh"] = {
            "paare": paare, "eigenanteil": eig, "effektive_zahl": r(eff), "effektive_zahl_ci95": [r(flo), r(fhi)],
            "effektive_zahl_anteil_endlich": r(ffin, 4), "max_abs_paar": max(paare, key=lambda p: abs(paare[p]["rho"]))}
    return erg


def lesart_b(je_phase):
    """Je Phase die Menge der zusammenfallenden Paare (roh oder partiell). nicht unterscheidbar: ein Paar faellt in
    beiden Phasen zusammen; unterscheidbar: in keiner Phase faellt ein Paar zusammen und jede Dimension ist in beiden
    Phasen, roh und partiell, mindestens eingeschraenkt eigenstaendig; sonst teilweise unterscheidbar."""
    def zus(ph):
        return {p for art in ("roh", "partiell_logn") for p, v in je_phase[ph][art]["paare"].items()
                if v["klasse"] == "zusammenfallend"}
    if zus(1) & zus(2):
        return "nicht unterscheidbar"
    kaum = any(v["klasse"] == "kaum eigenständig" for ph in (1, 2) for art in ("roh", "partiell_logn")
               for v in je_phase[ph][art]["eigenanteil"].values())
    return "unterscheidbar" if not (zus(1) or zus(2) or kaum) else "teilweise unterscheidbar"


# ---------------------------------------------------------------- C Kompositbedingung
def eigene_dim(p, dims=V26):
    return next(d for d, ps in dims.items() if p in ps)


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def c_deltas(Xa, gidx, gcols, own, konst=None):
    """Je gueltiger Fassung x (Zeile) und Dimension k (Spalte): r(x, D_k), r(D_j, D_k) * r(x, D_j), Delta und
    r(D_j, D_k); Pearson auf klassischen z-Werten, j die eigene Dimension von x, D_j einschliesslich x."""
    Z, dm = z_und_dim(Xa, gidx, konst)
    g = len(gcols)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(np.hstack([Z[:, gcols], dm]), rowvar=False)
    rxd, rdd = C[:g, g:], C[g:, g:]
    rxj = rxd[np.arange(g), own]
    rerw = rdd[own, :] * rxj[:, None]
    return rxd, rerw, rxd - rerw, rdd[own, :]


def delta_ohne(Xa, gueltig, x, k, q, konst=None):
    """Delta fuer die Fassung x und die fremde Dimension k mit D_j ohne die benannte Fassung q (D_k unveraendert).
    Bleibt von D_j ohne q nur x selbst, ist der Wert entartet (Delta = 0) und wird nicht bestimmt (None)."""
    Z, dm = z_und_dim(Xa, spalten(gueltig), konst)
    j = eigene_dim(x)
    rest = [PLAETZE.index(p) for p in V26[j] if p in gueltig and p != q]
    if len(rest) < 2:
        return None
    dj = Z[:, rest].mean(axis=1)
    zx, dk = Z[:, PLAETZE.index(x)], dm[:, DIMS.index(k)]
    return pearson(zx, dk) - pearson(dj, dk) * pearson(zx, dj)


def resid_np(Xa, ln):
    """Alle Fassungen ausser PE1 durch die Residuen ihrer OLS-Regression auf log n (mit Achsenabschnitt), wie
    entblinden.residualisiert."""
    A = np.column_stack([np.ones(len(Xa)), ln])
    cols = [i for i, p in enumerate(PLAETZE) if p != "PE1"]
    R = Xa.copy()
    R[:, cols] = Xa[:, cols] - A @ np.linalg.lstsq(A, Xa[:, cols], rcond=None)[0]
    return R


def teil_c(X, gueltig, logn, boot, rng_seed=BOOT_SEED):
    gidx = spalten(gueltig)
    gcols = [PLAETZE.index(p) for p in gueltig]
    own = np.array([DIMS.index(eigene_dim(p)) for p in gueltig])
    Xa0 = X[PLAETZE].to_numpy(float)
    ln = np.asarray(logn, float)
    erg = {}
    konst0 = Xa0.max(axis=0) == Xa0.min(axis=0)
    for art in ("roh", "residualisiert_logn"):
        res = art == "residualisiert_logn"
        Xa = resid_np(Xa0, ln) if res else Xa0
        robs, rerw, dlt, rjk = c_deltas(Xa, gidx, gcols, own, konst0)
        rng = np.random.default_rng(rng_seed)
        bd = np.empty((boot,) + dlt.shape)
        for t in range(boot):
            i = rng.integers(0, len(Xa0), len(Xa0))
            Xr = Xa0[i]
            konst = Xr.max(axis=0) == Xr.min(axis=0)          # Konstanz an den Rohwerten der Ziehung
            bd[t] = c_deltas(resid_np(Xr, ln[i]) if res else Xr, gidx, gcols, own, konst)[2]
        lo, hi, fin = intervall(bd)
        zeilen = []
        for ip, p in enumerate(gueltig):
            for k, dk in enumerate(DIMS):
                if k == own[ip]:
                    continue
                d = dlt[ip, k]
                bestimmbar = bool(np.isfinite(d) and fin[ip, k] >= ANTEIL_ENDLICH)
                abw = bestimmbar and abs(d) >= DELTA_ABWEICHUNG and (lo[ip, k] > 0 or hi[ip, k] < 0)
                z = {"fassung": p, "eigene": DIMS[own[ip]], "fremde": dk, "r_dimensionen": r(rjk[ip, k]),
                     "r_beobachtet": r(robs[ip, k]), "r_erwartet": r(rerw[ip, k]), "delta": r(d),
                     "vorzeichen": None if not np.isfinite(d) else ("positiv" if d > 0 else "negativ" if d < 0 else "null"),
                     "ci95": [r(lo[ip, k]), r(hi[ip, k])], "anteil_endlich": r(fin[ip, k], 4), "bestimmbar": bestimmbar,
                     "grenze_ueberschritten": bool(abs(robs[ip, k]) > abs(rjk[ip, k])), "abweichung": bool(abw),
                     "vorab_benannt": (p, dk) in BENANNT}
                if (p, dk) in GESCHWISTER:
                    q = GESCHWISTER[(p, dk)]
                    d_o = delta_ohne(Xa, gueltig, p, dk, q, konst0)
                    z["geschwister_von"] = f"{q} mit {dk}"
                    z["delta_ohne_benannte"] = r(d_o)
                    z["ohne_benannte_unter_schwelle"] = (None if d_o is None or not np.isfinite(d_o)
                                                         else bool(abs(d_o) < DELTA_ABWEICHUNG))
                zeilen.append(z)
        erg[art] = {"zeilen": zeilen,
                    "abweichungen": [f"{z['fassung']} mit {z['fremde']}" for z in zeilen if z["abweichung"]],
                    "grenze_ueberschritten": [f"{z['fassung']} mit {z['fremde']}" for z in zeilen if z["grenze_ueberschritten"]],
                    "nicht_bestimmbar": [f"{z['fassung']} mit {z['fremde']}" for z in zeilen if not z["bestimmbar"]]}
    return erg


def lesart_c(je_phase):
    """Eine Abweichung zaehlt, wenn dasselbe Paar (Fassung, fremde Dimension) in beiden Phasen roh abweicht, mit demselben
    Vorzeichen von Delta; in beiden Phasen mit entgegengesetztem Vorzeichen: gegenlaeufig, nur in einer Phase: einseitig
    (beide nur berichtet). Eine zaehlende Abweichung eines Geschwisters einer benannten Nahtstelle gilt als moeglicher
    Uebertrag, wenn die zugehoerige benannte Nahtstelle zaehlt und Delta mit D_j ohne die benannte Fassung in beiden
    Phasen unter 0,20 liegt (Betrag, ungerundet), und zaehlt dann nicht als weitere Abweichung. gewahrt: nichts zaehlt;
    gewahrt bis auf die benannten Nahtstellen: nur benannte Nahtstellen und moegliche Uebertraege zaehlen; lokalisiert
    verletzt: weitere zaehlende Abweichungen betreffen hoechstens zwei Fassungen; sonst breit verletzt."""
    zl = {ph: {(z["fassung"], z["fremde"]): z for z in je_phase[ph]["roh"]["zeilen"]} for ph in (1, 2)}
    abw = {ph: {k for k, z in zl[ph].items() if z["abweichung"]} for ph in (1, 2)}
    beide = abw[1] & abw[2]
    gleich = {k for k in beide if np.sign(zl[1][k]["delta"]) == np.sign(zl[2][k]["delta"])}

    def uebertrag(k):
        return (k in GESCHWISTER and (GESCHWISTER[k], k[1]) in gleich
                and all(zl[ph][k].get("ohne_benannte_unter_schwelle") is True for ph in (1, 2)))
    ue = {k for k in gleich - BENANNT if uebertrag(k)}
    weitere_paare = gleich - BENANNT - ue
    weitere = sorted({p for p, _ in weitere_paare})
    fmt = lambda ks: sorted(f"{p} mit {d}" for p, d in ks)  # noqa: E731
    info = {"zaehlend": fmt(gleich), "gegenlaeufig": fmt(beide - gleich),
            "einseitig": sorted(f"{p} mit {d} (Phase {ph})" for ph in (1, 2) for p, d in abw[ph] - beide),
            "benannte_bestaetigt": fmt(gleich & BENANNT), "benannte_nicht_bestaetigt": fmt(BENANNT - gleich),
            "moeglicher_uebertrag": fmt(ue), "weitere_paare": fmt(weitere_paare), "weitere_fassungen": weitere,
            "nicht_bestimmbar": sorted(f"{p} mit {d} (Phase {ph})" for ph in (1, 2) for (p, d), z in zl[ph].items()
                                       if not z["bestimmbar"])}
    if not weitere:
        return ("gewahrt bis auf die benannten Nahtstellen" if gleich & BENANNT else "gewahrt"), info
    if len(weitere) <= 2:
        return "lokalisiert verletzt", info
    return "breit verletzt", info


# ---------------------------------------------------------------- D Folgenrelevanz
def teil_d(X, gueltig, basis):
    erg = {}
    for kz, name, st in VARIANTEN:
        m = zugehoerigkeit(X, gueltig, st)
        ue = float((m["dominant"] == basis["dominant"]).mean())
        erg[kz] = {"variante": name, "uebereinstimmung": r(ue),
                   "anteile": {c: r((m["dominant"] == c).mean()) for c in KONFIG},
                   "mittlere_aenderung": {c: r((m[c] - basis[c]).abs().mean()) for c in KONFIG},
                   "wechsel": {f"{a}->{b}": int(((basis["dominant"] == a) & (m["dominant"] == b)).sum())
                               for a in KONFIG for b in KONFIG if a != b and ((basis["dominant"] == a) & (m["dominant"] == b)).any()},
                   "klasse": "folgenarm" if ue >= FOLGEN_ARM else "spürbar" if ue >= FOLGEN_STRUKTUR else "Strukturfrage"}
    return erg


def lesart_d(je_phase):
    """Je Variante die schlechtere der beiden Phasen."""
    rang = {"folgenarm": 0, "spürbar": 1, "Strukturfrage": 2}
    out = {}
    for kz, _, _ in VARIANTEN:
        k1, k2 = je_phase[1][kz]["klasse"], je_phase[2][kz]["klasse"]
        out[kz] = k1 if rang[k1] >= rang[k2] else k2
    return out


# ---------------------------------------------------------------- Kontrollwerte (wie die Bilanz, Spearman)
def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def kontrollen(X, gueltig):
    """Werte, die vor dem Lauf aus denselben Dateien bekannt sind (Bilanz, Abschnitt H und K6 im Satz; Protokoll 1b):
    Offenheit mit Wirkung, mit IP3 allein und mit der Wirkung ohne IP3; DS1 mit IP1. Spearman mit mittleren Raengen."""
    Z, dm = z_und_dim(X[PLAETZE].to_numpy(float), spalten(gueltig))
    eo, ip = dm[:, DIMS.index("EO")], dm[:, DIMS.index("IP")]
    z = lambda p: Z[:, PLAETZE.index(p)]  # noqa: E731
    return {"EO_mit_IP": r(spearman(eo, ip)), "EO_mit_IP3": r(spearman(eo, z("IP3"))),
            "EO_mit_IP_ohne_IP3": r(spearman(eo, (z("IP1") + z("IP2")) / 2.0)),
            "DS1_mit_IP1": r(spearman(X["DS1"].to_numpy(float), X["IP1"].to_numpy(float)))}


# ---------------------------------------------------------------- Ablauf
def auswerten(daten, boot=BOOT):
    """daten: {phase: (X, logn, gespeicherte Zugehoerigkeiten oder None)}. Erst beide Phasen pruefen und abgleichen,
    dann rechnen."""
    if sorted(daten) != [1, 2]:
        abbruch("es werden genau die Phasen 1 und 2 erwartet")
    vorab = {}
    for ph, (X, logn, gesp) in daten.items():
        gueltig = pruefe_eingabe(ph, X, logn)
        basis = zugehoerigkeit(X, gueltig, struktur(V26))
        vorab[ph] = (gueltig, basis, None if gesp is None else abgleich(ph, basis, gesp))
    erg = {"skript": SKRIPT,
           "parameter": {"bootstrap": boot, "seed": BOOT_SEED, "k": K_BASIS, "lambda": L_BASIS,
                         "anteil_endlich_mindestens": ANTEIL_ENDLICH,
                         "schwellen": {"paar_zusammenfallend": PAAR_ZUSAMMEN, "paar_eng": PAAR_ENG,
                                       "eigenanteil_kaum": EIGEN_KAUM, "eigenanteil_eingeschraenkt": EIGEN_EINGESCHRAENKT,
                                       "delta_abweichung": DELTA_ABWEICHUNG, "folgen_arm": FOLGEN_ARM,
                                       "folgen_strukturfrage": FOLGEN_STRUKTUR},
                         "benannt": sorted(f"{a} mit {b}" for a, b in BENANNT),
                         "geschwister": {f"{q} mit {k}": list(xs) for (q, k), xs in UEBERTRAG.items()}},
           "phasen": {}}
    B, C, Dd = {}, {}, {}
    for ph, (X, logn, _) in daten.items():
        gueltig, basis, abgl = vorab[ph]
        B[ph] = teil_b(X, gueltig, logn, boot)
        C[ph] = teil_c(X, gueltig, logn, boot)
        Dd[ph] = teil_d(X, gueltig, basis)
        erg["phasen"][ph] = {"topics": int(len(X)), "gueltige_fassungen": gueltig, "abgleich_grundfassung": abgl,
                             "anteile_grundfassung": {c: r((basis["dominant"] == c).mean()) for c in KONFIG},
                             "kontrolle": kontrollen(X, gueltig), "B": B[ph], "C": C[ph], "D": Dd[ph]}
    lc, info = lesart_c(C)
    erg["lesarten"] = {"B_unterscheidbarkeit": lesart_b(B), "C_kompositbedingung": lc, "C_einzelheiten": info,
                       "D_je_variante": lesart_d(Dd)}
    return erg


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def umgebung():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "skript_sha256": sha256(Path(__file__))}


def schreibe(ziel, erg):
    """Schreibt atomar (temporaere Datei, dann Umbenennen) und ueberschreibt kein vorhandenes Ergebnis."""
    ziel = Path(ziel)
    if ziel.exists():
        abbruch(f"{ziel.name} liegt schon vor und wird nicht ueberschrieben")
    ziel.parent.mkdir(parents=True, exist_ok=True)
    tmp = ziel.with_name(ziel.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(erg, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ziel)


def zugriffsmodul():
    if not (HIER / "zugriff.py").exists():
        abbruch("zugriff.py liegt nicht neben dem Skript (Ort: v26_explorativ)")
    import zugriff as zg
    if Path(zg.__file__).resolve().parent != HIER or HIER != zg.V26 / "v26_explorativ":
        abbruch("Skript liegt nicht in v26_explorativ neben zugriff.py")
    return zg


def lade_echt(zg):
    ordner = {1: zg.V26 / "output_v26_entblinden" / "phase1", 2: zg.V26 / "output_v26_bestaetigung" / "phase2" / "entblinden"}
    daten, quellen = {}, {}
    for ph, o in ordner.items():
        p, s = zg.lies(o / "zugehoerigkeiten.csv",
                       "Strukturprüfung B bis D: Fassungswerte, Topicgröße und Zugehörigkeiten V2.6 je Topic, nur Kennzahlen", SKRIPT)
        try:
            z = pd.read_csv(p, index_col="topic", usecols=SPALTEN, float_precision="round_trip")
        except ValueError as ex:
            abbruch(f"Phase {ph}: Spalten fehlen ({ex})")
        X = z[[f"wert_{k}" for k in PLAETZE]].astype(float)
        X.columns = PLAETZE
        daten[ph] = (X, logn_aus(z["n"]), z[KONFIG + ["margin", "dominant"]])
        quellen[ph] = s
    return daten, quellen


def rechnen():
    zg = zugriffsmodul()
    ziel = zg.AUS / "struktur" / "struktur_ergebnis.json"
    if ziel.exists():
        abbruch(f"{ziel.name} liegt schon vor und wird nicht ueberschrieben")
    daten, quellen = lade_echt(zg)
    erg = auswerten(daten)
    erg["quellen"] = {f"zugehoerigkeiten_phase{ph}": s for ph, s in quellen.items()}
    erg["umgebung"] = umgebung()
    schreibe(ziel, erg)
    print(json.dumps({"abgleich": {ph: erg["phasen"][ph]["abgleich_grundfassung"] for ph in (1, 2)},
                      "kontrolle": {ph: erg["phasen"][ph]["kontrolle"] for ph in (1, 2)},
                      "lesarten": erg["lesarten"]}, ensure_ascii=False))
    return erg


# ---------------------------------------------------------------- Tests mit synthetischen Daten
def synth(n=200, seed=1, verletze=None, gleich=None, phi=0.3):
    """Fassungen nach einem Kompositmodell: Bloecke korrelieren nur ueber ihre Komposite. verletze=(Fassung, Dimension)
    gibt der Fassung zusaetzlich einen Anteil der fremden Dimension; gleich=(d1, d2) macht zwei Bloecke nahezu gleich."""
    rng = np.random.default_rng(seed)
    C = np.full((5, 5), phi)
    np.fill_diagonal(C, 1.0)
    F = rng.multivariate_normal(np.zeros(5), C, size=n)
    rausch = {d: 0.8 for d in DIMS}
    if gleich:
        a, b = DIMS.index(gleich[0]), DIMS.index(gleich[1])
        F[:, b] = F[:, a]
        rausch[gleich[0]] = rausch[gleich[1]] = 0.3
    X = pd.DataFrame(index=[f"t{i}" for i in range(n)])
    for j, d in enumerate(DIMS):
        for p in V26[d]:
            X[p] = F[:, j] + rng.normal(0, rausch[d], n)
    if verletze:
        p, d = verletze
        X[p] = X[p] + 1.5 * F[:, DIMS.index(d)]
    logn = rng.normal(4, 1, n)
    return X, logn


def querladung(X, p, d, staerke):
    """Echte Querladung der Fassung p auf das Komposit der Dimension d (wie die Pruefinstanz, P3b)."""
    Z = (X - X.mean()) / X.std(ddof=0)
    D = Z[V26[d]].mean(axis=1)
    X = X.copy()
    X[p] = X[p] + staerke * (D - D.mean()) / D.std() * X[p].std()
    return X


def membership_bilanz(X):
    """Kopie der Schicht aus bilanz_v26.membership (dort gegen die gespeicherten Zugehoerigkeiten abgeglichen)."""
    Z = X.apply(klassisch_z, axis=0)
    gueltig = [p for p in X.columns if X[p].std() > MIN_SD]
    dm = pd.DataFrame({d: Z[[p for p in gueltig if p[:2] == d]].mean(axis=1) for d in DIMS})
    zd = dm.apply(robust_z, axis=0)
    zec = X[["DS1", "IP1", "IP2", "IP3"]].apply(robust_z, axis=0)
    kern = zd[KERN].mean(axis=1)
    sig = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
    return pd.DataFrame({"m_ws": sig(kern), "m_trend": sig(-kern + 0.5 * zd["IP"]), "m_ec": sig(zec.mean(axis=1)),
                         "m_latent": sig(-zd[KERN + ["IP"]].mean(axis=1))}, index=X.index)


def resid_pandas(X, logn):
    """Referenz wie entblinden.residualisiert (Schleife je Fassung)."""
    A = np.column_stack([np.ones(len(X)), np.asarray(logn, float)])
    R = X.copy()
    for p in X.columns:
        if p != "PE1":
            y = X[p].to_numpy(float)
            R[p] = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return R


def bricht_ab(f, *a, **kw):
    try:
        f(*a, **kw)
    except Abbruch:
        return True
    return False


def zeile(f, d, delta, abw=True, ohne=None):
    z = {"fassung": f, "fremde": d, "delta": delta, "abweichung": abw, "bestimmbar": True, "vorab_benannt": (f, d) in BENANNT}
    if ohne is not None:
        z["delta_ohne_benannte"] = ohne
        z["ohne_benannte_unter_schwelle"] = abs(ohne) < DELTA_ABWEICHUNG
    return z


def test(boot=200):
    import tempfile
    import time
    t0 = time.time()
    ok = []
    # 1 Grundschicht gleich der Schicht der Bilanz; numpy-Teile gleich den pandas-Referenzen
    X, logn = synth(seed=3)
    m = zugehoerigkeit(X, gueltige(X), struktur(V26))
    ok.append(("Grundschicht wie bilanz_v26.membership", float((m[KONFIG] - membership_bilanz(X)[KONFIG]).abs().max().max()) < 1e-12))
    _, dmn = z_und_dim(X[PLAETZE].to_numpy(float), spalten(gueltige(X)))
    ok.append(("Dimensionswerte numpy wie pandas", float(np.abs(dmn - dimensionswerte(X, gueltige(X), V26)[DIMS].to_numpy()).max()) < 1e-12))
    ok.append(("Residualisierung numpy wie entblinden.residualisiert",
               float(np.abs(resid_np(X[PLAETZE].to_numpy(float), logn) - resid_pandas(X, logn)[PLAETZE].to_numpy()).max()) < 1e-10))
    Rt = raenge(dmn, logn, False)
    _, eig_t, _ = b_kennzahlen(Rt)
    ols = []
    for j in range(5):
        A_ = np.column_stack([np.ones(len(Rt)), np.delete(Rt, j, axis=1)])
        res_ = Rt[:, j] - A_ @ np.linalg.lstsq(A_, Rt[:, j], rcond=None)[0]
        ols.append(float((res_ ** 2).sum() / ((Rt[:, j] - Rt[:, j].mean()) ** 2).sum()))
    ok.append(("Eigenanteil gleich 1 - R^2 der OLS", max(abs(a_ - b_) for a_, b_ in zip(eig_t, ols)) < 1e-10))
    # 2 Kompositmodell ohne Verletzung: keine Abweichung
    c = teil_c(X, gueltige(X), logn, boot)
    ok.append(("Kompositmodell: keine Abweichung", not c["roh"]["abweichungen"]))
    ok.append(("Kompositmodell: Grenze hoechstens knapp ueberschritten (Zufall)",
               all(abs(z_["r_beobachtet"]) - abs(z_["r_dimensionen"]) < 0.08 for z_ in c["roh"]["zeilen"] if z_["grenze_ueberschritten"])))
    ok.append(("Kompositmodell: alles bestimmbar", not c["roh"]["nicht_bestimmbar"] and not c["residualisiert_logn"]["nicht_bestimmbar"]))
    # 3 Verletzung bei EO2 mit DI wird gefunden
    Xv, lv = synth(seed=4, verletze=("EO2", "DI"))
    cv = teil_c(Xv, gueltige(Xv), lv, boot)
    ok.append(("Verletzung EO2 mit DI gefunden", "EO2 mit DI" in cv["roh"]["abweichungen"]))
    ok.append(("nur EO2 betroffen", all(a.startswith("EO2") for a in cv["roh"]["abweichungen"])))
    # 4 Unterscheidbarkeit: zwei fast gleiche Dimensionen fallen zusammen
    Xg, lg = synth(seed=5, gleich=("DS", "DI"))
    bg = teil_b(Xg, gueltige(Xg), lg, boot)
    ok.append(("DS-DI zusammenfallend", bg["roh"]["paare"]["DS-DI"]["klasse"] == "zusammenfallend"))
    b0 = teil_b(X, gueltige(X), logn, boot)
    ok.append(("ohne Zwilling: alle Paare unterscheidbar", all(v["klasse"] == "unterscheidbar" for v in b0["roh"]["paare"].values())))
    ok.append(("Lesart B zusammenfallend in beiden Phasen", lesart_b({1: bg, 2: bg}) == "nicht unterscheidbar"))
    ok.append(("Lesart B unterscheidbar", lesart_b({1: b0, 2: b0}) == "unterscheidbar"))
    # 4b Lesart B: roh in Phase 1, partiell in Phase 2 zusammenfallend ergibt nicht unterscheidbar
    def b_kunst(art_zus):
        e = json.loads(json.dumps(b0))
        e[art_zus]["paare"]["DS-DI"]["klasse"] = "zusammenfallend"
        return e
    ok.append(("Lesart B: Vereinigung roh und partiell je Phase", lesart_b({1: b_kunst("roh"), 2: b_kunst("partiell_logn")}) == "nicht unterscheidbar"))
    ok.append(("Lesart B: nur eine Phase zusammenfallend ergibt teilweise", lesart_b({1: b_kunst("roh"), 2: b0}) == "teilweise unterscheidbar"))
    # 5 Effektive Zahl: unkorrelierte Dimensionen ergeben fast 5
    rng = np.random.default_rng(9)
    _, _, eff = b_kennzahlen(rng.normal(size=(5000, 5)))
    ok.append(("effektive Zahl unkorreliert nahe 5", abs(eff - 5) < 0.05))
    # 6 Folgenrelevanz: Identitaetsvariante ergibt Uebereinstimmung 1; Varianten laufen
    basis = zugehoerigkeit(X, gueltige(X), struktur(V26))
    ok.append(("Identitaet gibt 1", float((zugehoerigkeit(X, gueltige(X), struktur(ersetze()))["dominant"] == basis["dominant"]).mean()) == 1.0))
    d = teil_d(X, gueltige(X), basis)
    ok.append(("sieben Varianten", list(d) == ["D1", "D2", "D3", "D4", "D5", "D6", "D7"] and all(0 <= v["uebereinstimmung"] <= 1 for v in d.values())))
    ok.append(("Lesart D je Variante", set(lesart_d({1: d, 2: d}).values()) <= {"folgenarm", "spürbar", "Strukturfrage"}))
    # 7 Lesart C mit Kunstzeilen
    z = {"roh": {"zeilen": [zeile("IP3", "EO", 0.3)]}}
    leer = {"roh": {"zeilen": [zeile("EO1", "DS", 0.01, abw=False)]}}
    ok.append(("Lesart C mit benannter Nahtstelle", lesart_c({1: z, 2: z})[0] == "gewahrt bis auf die benannten Nahtstellen"))
    ok.append(("Lesart C ohne Abweichung: gewahrt", lesart_c({1: leer, 2: leer})[0] == "gewahrt"))
    z2 = {"roh": {"zeilen": [zeile(f, "DS", 0.3) for f in ("EO1", "DI1", "DI4")]}}
    ok.append(("Lesart C breit, wenn in beiden Phasen", lesart_c({1: z2, 2: z2})[0] == "breit verletzt"))
    ok.append(("Lesart C: nur eine Phase zaehlt nicht", lesart_c({1: z2, 2: z})[0] == "gewahrt"
               and len(lesart_c({1: z2, 2: z})[1]["einseitig"]) == 4))
    z3 = {"roh": {"zeilen": [zeile(f, "DS", -0.3) for f in ("EO1", "DI1", "DI4")]}}
    l3, i3 = lesart_c({1: z2, 2: z3})
    ok.append(("Lesart C: entgegengesetzte Vorzeichen zaehlen nicht", l3 == "gewahrt" and len(i3["gegenlaeufig"]) == 3))
    zu = {"roh": {"zeilen": [zeile("IP3", "EO", 0.3), zeile("IP2", "EO", -0.25, ohne=-0.02)]}}
    zu2 = {"roh": {"zeilen": [zeile("IP3", "EO", 0.3), zeile("IP2", "EO", -0.25, ohne=-0.22)]}}
    lu, iu = lesart_c({1: zu, 2: zu})
    ok.append(("Lesart C: Geschwister als moeglicher Uebertrag", lu == "gewahrt bis auf die benannten Nahtstellen"
               and iu["moeglicher_uebertrag"] == ["IP2 mit EO"]))
    ok.append(("Lesart C: Uebertrag nur, wenn in beiden Phasen unter 0,20", lesart_c({1: zu, 2: zu2})[0] == "lokalisiert verletzt"))
    zo = {"roh": {"zeilen": [zeile("IP3", "EO", 0.3, abw=False), zeile("IP2", "EO", -0.25, ohne=-0.02)]}}
    lo_, io_ = lesart_c({1: zo, 2: zo})
    ok.append(("Lesart C: ohne zaehlende Nahtstelle kein Uebertrag", lo_ == "lokalisiert verletzt" and io_["weitere_paare"] == ["IP2 mit EO"]))
    zg1 = {"roh": {"zeilen": [zeile("IP3", "EO", 0.3), zeile("IP2", "EO", -0.25, ohne=-0.02)]}}
    zg2 = {"roh": {"zeilen": [zeile("IP3", "EO", -0.3), zeile("IP2", "EO", -0.25, ohne=-0.02)]}}
    ok.append(("Lesart C: gegenlaeufige Nahtstelle traegt keinen Uebertrag", lesart_c({1: zg1, 2: zg2})[0] == "lokalisiert verletzt"))
    zn = {"roh": {"zeilen": [zeile("IP3", "EO", 0.3), dict(zeile("IP2", "EO", -0.25), delta_ohne_benannte=None,
                                                          ohne_benannte_unter_schwelle=None)]}}
    ok.append(("Lesart C: nicht bestimmbarer Uebertrag zaehlt als weitere", lesart_c({1: zn, 2: zn})[0] == "lokalisiert verletzt"))
    ok.append(("delta_ohne entartet, wenn nur x bleibt", delta_ohne(X[PLAETZE].to_numpy(float), [p for p in PLAETZE if p != "DS3"],
                                                             "DS2", "IP", "DS1") is None))
    # 8 Uebertrag in Daten: echte Querladung IP3 mit EO in beiden Phasen (stark genug, dass ein Geschwister abweicht);
    #   Gegenprobe: laedt zusaetzlich IP2 auf EO, bleibt die Abweichung von IP1 ohne IP3 bestehen und zaehlt
    je, jg = {}, {}
    for ph, seed in ((1, 41), (2, 42)):
        Xq, lq = synth(n=300, seed=seed, phi=0.5)
        Xq3 = querladung(Xq, "IP3", "EO", 2.0)
        je[ph] = teil_c(Xq3, gueltige(Xq), lq, boot)
        jg[ph] = teil_c(querladung(Xq3, "IP2", "EO", 0.9), gueltige(Xq), lq, boot)
    lq_, iq = lesart_c(je)
    ok.append(("Querladung: benannte Nahtstelle bestaetigt", "IP3 mit EO" in iq["benannte_bestaetigt"]))
    ok.append(("Querladung: Geschwister weicht in beiden Phasen ab und gilt als Uebertrag",
               "IP2 mit EO" in iq["zaehlend"] and "IP2 mit EO" in iq["moeglicher_uebertrag"]
               and lq_ == "gewahrt bis auf die benannten Nahtstellen"))
    lg_, ig = lesart_c(jg)
    ok.append(("Gegenprobe: bleibende Abweichung ohne die benannte Fassung zaehlt", "IP1 mit EO" in ig["weitere_paare"]
               and lg_ == "lokalisiert verletzt"))
    # 9 Duenne Fassung: in Ziehungen konstant, dann nicht bestimmbar statt NaN-Intervall, roh und residualisiert,
    #   auch wenn der haeufige Wert nicht 0 ist (Konstanz an den Rohwerten der Ziehung)
    for modal in (0.0, 0.37):
        Xd, ld = synth(seed=6)
        Xd["DS1"] = modal
        Xd.iloc[:2, Xd.columns.get_loc("DS1")] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            try:
                cd = teil_c(Xd, gueltige(Xd), ld, boot)
                bd = teil_b(Xd, gueltige(Xd), ld, boot)
                ohne_warnung = True
            except RuntimeWarning:
                ohne_warnung = False
        ok.append((f"duenne Fassung (Wert {modal}): ohne Laufzeitwarnung", ohne_warnung))
        if not ohne_warnung:
            continue
        for art in ("roh", "residualisiert_logn"):
            nb = cd[art]["nicht_bestimmbar"]
            anteil = [z_["anteil_endlich"] for z_ in cd[art]["zeilen"] if z_["fassung"] == "DS1"]
            ok.append((f"duenne Fassung (Wert {modal}, {art}): DS1 nicht bestimmbar mit gleichem Anteil, sonst alles bestimmbar",
                       len(nb) == 4 and all(a.startswith("DS1") for a in nb) and len(set(anteil)) == 1 and anteil[0] < ANTEIL_ENDLICH))
            ok.append((f"duenne Fassung (Wert {modal}, {art}): Intervalle der uebrigen endlich",
                       all(None not in z_["ci95"] for z_ in cd[art]["zeilen"] if z_["fassung"] != "DS1")))
        ok.append((f"duenne Fassung (Wert {modal}): roh und residualisiert gleich oft endlich",
                   [z_["anteil_endlich"] for z_ in cd["roh"]["zeilen"]] == [z_["anteil_endlich"] for z_ in cd["residualisiert_logn"]["zeilen"]]))
        ok.append((f"duenne Fassung (Wert {modal}): Teil B voll bestimmbar",
                   all(v["intervall_bestimmbar"] for art in ("roh", "partiell_logn") for v in bd[art]["paare"].values())))
    # 9b Konstante Dimension in einer Ziehung: partiell bleibt die Spalte konstant (nicht endliche Korrelation)
    dk_ = np.column_stack([np.arange(50.0), np.full(50, 2.0), np.random.default_rng(3).normal(size=(50, 3))])
    with np.errstate(invalid="ignore", divide="ignore"):
        ck_ = np.corrcoef(raenge(dk_, np.random.default_rng(4).normal(size=50), True), rowvar=False)
    ok.append(("konstante Dimension partiell: Korrelationen nicht endlich", not np.isfinite(ck_[1, [0, 2, 3, 4]]).any()
               and np.isfinite(ck_[0, 2])))
    # 9c Kontrollwerte wie Spearman mit mittleren Raengen (pandas, wie die Bilanz)
    kt = kontrollen(X, gueltige(X))
    dmp = dimensionswerte(X, gueltige(X), V26)
    zp = X.apply(klassisch_z, axis=0)
    soll = {"EO_mit_IP": dmp["EO"].corr(dmp["IP"], method="spearman"), "EO_mit_IP3": dmp["EO"].corr(zp["IP3"], method="spearman"),
            "EO_mit_IP_ohne_IP3": dmp["EO"].corr(zp[["IP1", "IP2"]].mean(axis=1), method="spearman"),
            "DS1_mit_IP1": X["DS1"].corr(X["IP1"], method="spearman")}
    ok.append(("Kontrollwerte wie pandas Spearman", all(abs(kt[k] - soll[k]) < 1e-6 for k in soll)))
    # 10 Pruefungen vor der Rechnung
    Xn = X.copy()
    Xn.iloc[0, 0] = np.nan
    ok.append(("Abbruch bei fehlendem Wert", bricht_ab(pruefe_eingabe, 1, Xn, logn)))
    Xu = X.copy()
    Xu.index = ["t0"] + list(X.index[1:-1]) + ["t0"]
    ok.append(("Abbruch bei doppeltem Topic", bricht_ab(pruefe_eingabe, 1, Xu, logn)))
    Xk = X.copy()
    Xk["IP2"] = 1.0
    ok.append(("Abbruch, wenn eine Variantendimension keine gueltige Fassung hat (D7: IP nur IP2)", bricht_ab(pruefe_eingabe, 1, Xk, logn)))
    ok.append(("Abbruch bei n = 0", bricht_ab(logn_aus, [3, 0, 5])))
    gesp = basis[KONFIG + ["margin", "dominant"]]
    falsch = gesp.copy()
    falsch.iloc[5, 0] += 1e-6
    ok.append(("Abbruch, wenn der Abgleich scheitert (vor jeder Rechnung)", bricht_ab(auswerten, {1: (X, logn, gesp), 2: (Xv, lv, falsch)}, boot)))
    # 11 Gesamtlauf auf synthetischen Daten mit Abgleich; Schreiben atomar und ohne Ueberschreiben
    e = auswerten({1: (X, logn, gesp), 2: (Xv, lv, None)}, boot)
    ok.append(("Gesamtlauf mit Abgleich", e["phasen"][1]["abgleich_grundfassung"]["dominant_gleich"] and "lesarten" in e
               and e["parameter"]["bootstrap"] == boot))
    with tempfile.TemporaryDirectory() as tdir:
        ziel = Path(tdir) / "struktur" / "struktur_ergebnis.json"
        schreibe(ziel, e)
        zurueck = json.loads(ziel.read_text(encoding="utf-8"))
        ok.append(("Ergebnis geschrieben und lesbar, keine Zwischendatei", zurueck["lesarten"] == json.loads(json.dumps(e["lesarten"]))
                   and not any(p.suffix == ".tmp" for p in ziel.parent.iterdir())))
        ok.append(("vorhandenes Ergebnis wird nicht ueberschrieben", bricht_ab(schreibe, ziel, e)))
    for name, gut in ok:
        print(("ok    " if gut else "FEHLT ") + name)
    print(f"{sum(g for _, g in ok)} von {len(ok)} Tests bestanden ({time.time() - t0:.1f} s)")
    return all(g for _, g in ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teil", choices=["rechnen", "test"])
    a = ap.parse_args()
    if a.teil == "test":
        sys.exit(0 if test() else 1)
    rechnen()


if __name__ == "__main__":
    main()
