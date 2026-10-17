#!/usr/bin/env python3
"""V2.6, Entwicklungsschritt 1b (nur Phase 1): Kandidaten aus Abschnitt 0.4 des Protokolls rechnen und pruefen.

Gleiche Pruefbatterie und gleiche Pseudokorpora wie backtest_phase1.py (Teilungen mit Seed 1000 + r,
Rarefaktion 2000 + r, Label-Permutation 3000 + r), damit Kandidaten und bestehende Indikatoren gepaart
verglichen werden. Das Skript gibt nur Guetemasse und abgeleitete Werte je Topic aus, keine Zugehoerigkeiten
und keine Konfigurationen (Verblindung bis zum Einfrieren). Phase 2 wird nicht gelesen. Datensaetze verlassen
den Rechner nicht; der Zwischenspeicher _cache enthaelt dokumentweise abgeleitete Strukturen und bleibt lokal
(output_*/ und *.npz stehen in .gitignore).

Noch nicht enthalten: EO4 A1 (Zertifiziertheitsmodell, wartet auf die Modelldateien), EO5 (Umsetzung offen),
Diagnose A2 (IDF-Abgleich braucht einen allgemeinen Vergleichskorpus, den der Export nicht enthaelt).

Aufruf (im Ordner v26_entwicklung):
  python3 backtest_kandidaten.py cache              Zwischenspeicher bauen (einmalig)
  python3 backtest_kandidaten.py repro              Referenzwerte, Abdeckung, Gegenproben
  python3 backtest_kandidaten.py lauf --budget 150  fehlende Wiederholungen rechnen, bis das Zeitbudget erreicht ist
  python3 backtest_kandidaten.py report             Kennzahlen je Kandidat und Platz, Regeln R1 bis R4
  python3 backtest_kandidaten.py k7                 K7: Mischungen von Topicpaaren rechnen (17.10.)
  python3 backtest_kandidaten.py k7report           K7: Erwartungen pruefen

Die Kennzahlen werden mit sechs Nachkommastellen abgelegt, damit gerundete Angaben nicht doppelt gerundet werden.
"""
import argparse
import hashlib
import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import beta as beta_vt
from scipy.stats import kendalltau, norm, spearmanr
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.neighbors import NearestNeighbors

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import backtest_phase1 as bt  # noqa: E402  (setzt config und step02_indicators auf Phase 1)

s2 = bt.s2
OUT = bt.V26 / "output_v26_entwicklung" / "kandidaten_phase1"
CACHE = OUT / "_cache"
JAHR_MIN, JAHR_MAX = bt.JAHR_MIN, bt.JAHR_MAX

# Vor dem ersten Rechenlauf festgelegt (Protokoll, Abschnitt 1b); nach Ergebnissen nicht mehr veraendern.
M_RARE = 25            # Protokoll 0.4: Rarefaktion auf m = 25 Dokumente
R_PERM = 50            # Permutationen je Topic fuer die SES-Kandidaten
R_ZIEH = 20            # Ziehungen je Topic fuer die Zahl der Autorengruppen
MIN_DF, MAX_DF = 2, 0.25   # Phrasenvokabular am vollen Phase-1-Korpus: in mindestens 2, hoechstens 25 % der Dokumente
TAU, K_NACHBARN = 0.35, 10  # JRC-Kompaktheitsfilter (Eulaerts et al. 2022)
PSEUDO = 0.5           # IP2 A1: Pseudozaehlung in Zitationen je Jahr
AKTIV_JAHRE = 3        # DS1 A2: juengste drei Jahre des Fensters (Eulaerts et al. 2025: 2021 bis 2023)
DOV_ABSCHLAG = 0.05    # PE1 A1: Roh und Choi 2020
DOV_BODEN = 0.10       # PE1 A1: Rauschboden am 10. Perzentil (BERTrend)
MIN_ABSTRACT = 20      # EO4 A2: Mindestzahl an Woertern im Abstract
LAENGE_BINS = 10       # EO3 A1: Dezile der Textlaenge
NEU_FENSTER = 2        # DS2 A3 (17.10.): Phrase gilt als neu, wenn ihr Erstjahr im Korpus mindestens Erscheinungsjahr minus 2 ist
NEU_AB = 2003          # DS2 A3: Dokumente ab 2003 (drei Jahre Vorlauf im Fenster)
KOHORTE_AB = 2011      # IP2 A5 (17.10.): juengste fuenf Jahre des Fensters
SEMD_MIN = 20          # EO5 A1 (17.10.): Wort gueltig ab 20 Dokumenten ausserhalb des Topics
SEMD_MIN_WOERTER = 5   # EO5 A1: Dokument gueltig ab 5 gueltigen Woertern
SEMD_BODEN = 1e-3      # EO5 A1: Untergrenze der mittleren Kosinusaehnlichkeit vor dem Logarithmus
K7_SEEDS = (7000, 7001, 7002)  # K7 (17.10.): drei Aufteilungen in Topicpaare

KAND = [  # Code, Platz, Kurzbeschreibung, Feld fuer K1
    ("EO1_A1", "EO1", "Hill q = 2 der Phrasen aus Titel und Abstract (log)", "phrasen"),
    ("EO1_A2", "EO1", "SES der Keyword-Volatilitaet gegen Permutation der Jahre", "keywords"),
    ("EO2_A1", "EO2", "Hill q = 2 der WoS-Kategorien (log, an ln K normiert)", "kategorien"),
    ("EO2_A2", "EO2", "Chao-Shen-Entropie der WoS-Kategorien (an ln K normiert)", "kategorien"),
    ("EO3_A1", "EO3", "Inkohaerenz minus laengenbedingte Erwartung", "alle"),
    ("EO4_A2", "EO4", "Hedge-Dichte je 1.000 Woerter (Hyland)", "abstract"),
    ("EO5_A1", "EO5", "Semantische Diversitaet des Vokabulars ohne eigenes Topic (Ergaenzung 17.10.)", "eo5"),
    ("PE1_A1", "PE1", "DoV mit Rauschboden, Rang invertiert", "alle"),
    ("PE2_A1", "PE2", "Rauschanteil unter k = 25 Nachbarn des Zentroids", "alle"),
    ("PE2_A2", "PE2", "Rauschanteil unter k = 50 Nachbarn des Zentroids", "alle"),
    ("PE3_A1", "PE3", "inverses Kanalprestige in der WoS-Kategorie", "journal_kat"),
    ("PE3_A2", "PE3", "SES gegen hypergeometrische Erwartung", "journal"),
    ("DS1_A1", "DS1", "Paper-Vitalitaet", "alle"),
    ("DS1_A2", "DS1", "Aktivitaet: Anteil der juengsten drei Jahre", "alle"),
    ("DS2_A1", "DS2", "SES der Folgejahr-Distanz, Author Keywords", "keywords"),
    ("DS2_A2", "DS2", "SES der Folgejahr-Distanz, Phrasen", "phrasen"),
    ("DS2_A3", "DS2", "Begriffsneuheit je Jahrgang (Ergaenzung nach E3, 17.10.)", "ab2003"),
    ("DS3_A1", "DS3", "Posterior-Wahrscheinlichkeit unter dem Korpusanteil", "typ"),
    ("DI1_A1", "DI1", "Konzentration auf Autorengruppen (Komponenten unter m Dokumenten)", "autoren"),
    ("DI2_A1", "DI2", "Simpson-Konzentration der Institutionen", "institutionen"),
    ("DI3_A1", "DI3", "Simpson-Konzentration der Laender", "laender"),
    ("DI3_A2", "DI3", "1 minus Anteil international koautorierter Dokumente", "laender"),
    ("DI4_A3", "DI4", "Gini der jahrgangsnormierten Zitationen, n/(n-1) (Ergaenzung 17.10.)", "alle"),
    ("DI4_A4", "DI4", "Lognormal-Gini der jahrgangsnormierten Zitationen (Ergaenzung 17.10.)", "alle"),
    ("IP1_A1", "IP1", "Poisson-Steigung des Anteils mit Offset", "alle"),
    ("IP2_A1", "IP2", "Log-Verhaeltnis spaet zu frueh mit Pseudozaehlung", "alle"),
    ("IP2_A2", "IP2", "Poisson-Ratenverhaeltnis spaet zu frueh", "alle"),
    ("IP2_A3", "IP2", "Zitationsdynamik je Jahrgang (Ergaenzung nach E3, 17.10.)", "alle"),
    ("IP2_A4", "IP2", "Kohortentrend der Wirkung, Kendall (Ergaenzung 17.10.)", "alle"),
    ("IP2_A5", "IP2", "Wirkung der juengsten Kohorte 2011 bis 2015 (Ergaenzung 17.10.)", "ab2011"),
    ("IP3_A1", "IP3", "Hill q = 2 der Keywords Plus (log)", "kwplus"),
    ("IP3_A2", "IP3", "Anteil der Dokumente unter den Top 10 % je Jahr", "alle"),
    ("DIAG_A1", "DIAG", "Kompaktheitsfilter: Anteil mit mindestens 10 Nachbarn ausserhalb", "alle"),
]
KCODES = [k[0] for k in KAND]
PLATZ = {k[0]: k[1] for k in KAND}
FELD = {k[0]: k[3] for k in KAND}
BESCHR = {k[0]: k[2] for k in KAND}
BEVORZUGT = {"EO1": "EO1_A1", "EO2": "EO2_A1", "EO3": "EO3_A1", "EO4": "beide", "EO5": "EO5_A1", "PE1": "bestehend",
             "PE2": "PE2_A1", "PE3": "PE3_A1", "DS1": "bestehend", "DS2": "DS2_A2", "DS3": "DS3_A1",
             "DI1": "DI1_A1", "DI2": "DI2_A1", "DI3": "DI3_A1", "DI4": "keine", "IP1": "IP1_A1",
             "IP2": "IP2_A1", "IP3": "IP3_A1", "DIAG": "DIAG_A1"}

# Hedges und Booster nach Hyland (2005, Anhang); Liste vor einer Verwendung im Manuskript am Original pruefen.
HEDGE_1 = set("""about almost apparent apparently appear appeared appears approximately argue argued argues around
assume assumed broadly claim claimed claims could couldn't doubt doubtful essentially estimate estimated fairly feel
feels felt frequently generally guess indicate indicated indicates largely likely mainly may maybe might mostly often
ought perhaps plausible plausibly possible possibly postulate postulated postulates presumable presumably probable
probably quite relatively roughly seem seemed seems should sometimes somewhat suggest suggested suggests suppose
supposed supposes suspect suspects typical typically uncertain uncertainly unclear unclearly unlikely usually would
wouldn't""".split())
HEDGE_N = ["certain amount", "certain extent", "certain level", "from my perspective", "from our perspective",
           "from this perspective", "in general", "in most cases", "in most instances", "in my opinion", "in my view",
           "in this view", "in our opinion", "in our view", "on the whole", "tend to", "tended to", "tends to",
           "to my knowledge"]
BOOST_1 = set("""actually always believe believed believes certain certainly clear clearly conclusively decidedly
definite definitely demonstrate demonstrated demonstrates doubtless establish established evident evidently find finds
found incontestable incontestably incontrovertible incontrovertibly indeed indisputable indisputably know known must
never obvious obviously prove proved proves realize realized realizes really show showed shown shows sure surely think
thinks thought true truly undeniable undeniably undisputedly undoubtedly""".split())
BOOST_N = ["beyond doubt", "in fact", "no doubt", "of course", "without doubt"]
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
MEHRWORT = {}
for _p, _art in sorted([(p, "h") for p in HEDGE_N] + [(p, "b") for p in BOOST_N], key=lambda x: -len(x[0].split())):
    _t = tuple(_p.split())
    MEHRWORT.setdefault(_t[0], []).append((_t, _art))

STOP = frozenset(ENGLISH_STOP_WORDS)
SEGMENT = re.compile(r"[.,;:!?()\[\]{}\"/]+")
WORT = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


def uhr(t0, text):
    print(f"[{time.time() - t0:7.1f} s] {text}", flush=True)


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- Aufbereitung je Dokument (volles Korpus)

def phrasen(text):
    """Uni- und Bigramme innerhalb von Laeufen aus Inhaltswoertern (kein Stoppwort, keine Satzzeichen, keine Zahl)."""
    out = set()
    for seg in SEGMENT.split(text.lower()):
        lauf = []
        for roh in seg.split():
            w = roh.strip("-'")
            if len(w) >= 2 and w not in STOP and WORT.fullmatch(w):
                lauf.append(w)
                continue
            if lauf:
                out.update(lauf)
                out.update(a + " " + b for a, b in zip(lauf, lauf[1:]))
                lauf = []
        if lauf:
            out.update(lauf)
            out.update(a + " " + b for a, b in zip(lauf, lauf[1:]))
    return out


def hedges(text):
    """Hedges, Booster und Woerter eines Abstracts; Mehrwortausdruecke zuerst, ihre Woerter zaehlen nicht doppelt."""
    w = TOKEN.findall(text.lower())
    h = b = 0
    i, n = 0, len(w)
    while i < n:
        treffer = None
        for t, art in MEHRWORT.get(w[i], ()):
            if tuple(w[i:i + len(t)]) == t:
                treffer = (t, art)
                break
        if treffer:
            h += treffer[1] == "h"
            b += treffer[1] == "b"
            i += len(treffer[0])
            continue
        if w[i] in HEDGE_1:
            h += 1
        elif w[i] in BOOST_1:
            b += 1
        i += 1
    return h, b, n


def norm_name(x):
    return re.sub(r"\s+", " ", x.strip().lower())


def autoren_schluessel(a_str, o_str):
    """ORCID ueber den Namen zugeordnet (Eintraege 'Name/ID'), sonst der normierte volle Name."""
    namen = [norm_name(x) for x in str(a_str).split(";") if x.strip()]
    karte = {}
    for e in str(o_str).split(";"):
        e = e.strip()
        if "/" in e:
            nm, oid = e.rsplit("/", 1)
            if oid.strip():
                karte[norm_name(nm)] = oid.strip()
    return [("o:" + karte[n]) if n in karte else ("n:" + n) for n in namen]


def matrix(listen, zaehlen=False, vokabular=None):
    if vokabular is None:
        vokabular = {}
        for lst in listen:
            for e in lst:
                if e not in vokabular:
                    vokabular[e] = len(vokabular)
    rows, cols, vals = [], [], []
    for i, lst in enumerate(listen):
        c = Counter(e for e in lst if e in vokabular) if zaehlen else dict.fromkeys({e for e in lst if e in vokabular}, 1)
        for e, v in c.items():
            rows.append(i)
            cols.append(vokabular[e])
            vals.append(v)
    m = sparse.csr_matrix((np.asarray(vals, dtype=np.float64), (np.asarray(rows, dtype=np.int64),
                          np.asarray(cols, dtype=np.int64))), shape=(len(listen), len(vokabular)))
    return m, vokabular


def spalte(df, name):
    return df[name].fillna("").astype(str) if name in df.columns else pd.Series([""] * len(df))


def baue_cache(t0):
    df, labels, emb, red = bt.lade()
    uhr(t0, f"Daten geladen: {len(df)} Dokumente")
    N = len(df)
    # Phrasen aus Titel und Abstract, Vokabular am vollen Korpus
    ph = [phrasen(t) for t in df["text"].fillna("")]
    dfz = Counter(p for s in ph for p in s)
    vok = {p: i for i, p in enumerate(sorted(p for p, c in dfz.items() if MIN_DF <= c <= MAX_DF * N))}
    P, _ = matrix(ph, vokabular=vok)
    uhr(t0, f"Phrasen: Vokabular {len(vok)} (von {len(dfz)}), {P.nnz} Eintraege")
    KW, _ = matrix([[k.strip().lower() for k in s.split(";") if k.strip()] for s in spalte(df, "Author Keywords")],
                   zaehlen=True)
    KAT, kat_vok = matrix([[c.strip() for c in s.split(";") if c.strip()] for s in spalte(df, "WoS Categories")],
                          zaehlen=True)
    KP, _ = matrix([[k.strip().lower() for k in s.split(";") if k.strip()] for s in spalte(df, "Keywords Plus")])
    AFF, _ = matrix([[i.strip() for i in s.split(";") if i.strip()] for s in spalte(df, "Affiliations")])
    LAND, _ = matrix([[p.strip().upper() for p in s.split(";") if p.strip()] for s in spalte(df, "Country List")])
    AUT, _ = matrix([autoren_schluessel(a, o) for a, o in zip(spalte(df, "Authors"), spalte(df, "ORCIDs"))])
    uhr(t0, "Feldmatrizen gebaut")
    hb = np.array([hedges(t) for t in spalte(df, "Abstract")], dtype=np.int64)
    jahr = df["Year"].astype(int).values
    tc = df["Times Cited"].fillna(0).astype(float).values
    alter = np.clip(config_y_cutoff() - jahr + 1, 1.0, None)
    journal = spalte(df, "Source title").str.strip()
    j_codes, j_uniq = pd.factorize(journal.where(journal != "", None))
    erste = spalte(df, "WoS Categories").map(lambda s: next((c.strip() for c in s.split(";") if c.strip()), ""))
    k_codes, _ = pd.factorize(erste.where(erste != "", None))
    typ = df["Document Type"] if "Document Type" in df.columns else pd.Series([np.nan] * N)
    review = typ.map(s2._is_review).astype(bool).values
    hat_typ = typ.notna().values
    laenge = df["text_clean"].str.split().str.len().fillna(0).astype(int).values
    np.savez(CACHE / "felder.npz", jahr=jahr, tc=tc, alter=alter, journal=j_codes.astype(np.int64),
             erste_kat=k_codes.astype(np.int64), review=review, hat_typ=hat_typ, hedge=hb[:, 0], booster=hb[:, 1],
             woerter_abstract=hb[:, 2], laenge=laenge)
    for name, m in (("phrasen", P), ("keywords", KW), ("kategorien", KAT), ("kwplus", KP), ("institutionen", AFF),
                    ("laender", LAND), ("autoren", AUT)):
        sparse.save_npz(CACHE / f"{name}.npz", m.tocsr())
    meta = {"daten_sha16": sha16(bt.DATA), "labels_sha16": sha16(bt.REF / "model_results.pkl"), "n_dok": int(N),
            "phrasen_vokabular": len(vok), "phrasen_gesamt": len(dfz), "kategorien_korpus": len(kat_vok),
            "journale": int(len(j_uniq)), "min_df": MIN_DF, "max_df": MAX_DF}
    (CACHE / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    uhr(t0, f"Zwischenspeicher geschrieben: {json.dumps(meta)}")


def config_y_cutoff():
    import config
    return config.Y_CUTOFF


# ---------------------------------------------------------------- Korpus mit Zwischenspeicher

class Korpus:
    def __init__(self, t0):
        meta = json.loads((CACHE / "meta.json").read_text(encoding="utf-8"))
        if meta["daten_sha16"] != sha16(bt.DATA) or meta["labels_sha16"] != sha16(bt.REF / "model_results.pkl"):
            raise SystemExit("Zwischenspeicher passt nicht zu den Daten; 'cache' neu ausfuehren")
        self.meta = meta
        self.df, self.labels, emb, self.red = bt.lade()
        self.E = np.ascontiguousarray(emb, dtype=np.float32)
        f = np.load(CACHE / "felder.npz")
        for k in f.files:
            setattr(self, k, f[k])
        self.rate = self.tc / self.alter
        self.M = {name: sparse.load_npz(CACHE / f"{name}.npz").tocsr()
                  for name in ("phrasen", "keywords", "kategorien", "kwplus", "institutionen", "laender", "autoren")}
        self.L = {name: np.asarray(m.sum(axis=1)).ravel() for name, m in self.M.items()}
        self.ln_k = float(np.log(meta["kategorien_korpus"]))
        n = len(self.labels)
        # EO3 A1: mittlere Kosinusaehnlichkeit zwischen Laengenklassen im vollen Korpus (exakt ueber Summenvektoren)
        kanten = np.unique(np.quantile(self.laenge, np.linspace(0, 1, LAENGE_BINS + 1)[1:-1]))
        self.lbin = np.searchsorted(kanten, self.laenge, side="right")
        B = self.lbin.max() + 1
        S = np.zeros((B, self.E.shape[1]), dtype=np.float64)
        np.add.at(S, self.lbin, self.E.astype(np.float64))
        nb = np.bincount(self.lbin, minlength=B).astype(float)
        G = S @ S.T / np.outer(nb, nb)
        np.fill_diagonal(G, (np.einsum("ij,ij->i", S, S) - nb) / (nb * (nb - 1)))
        self.G = G
        self.abdeckung = {
            "phrasen": self.L["phrasen"] > 0, "keywords": self.L["keywords"] > 0,
            "kategorien": self.L["kategorien"] > 0, "kwplus": self.L["kwplus"] > 0,
            "institutionen": self.L["institutionen"] > 0, "laender": self.L["laender"] > 0,
            "autoren": self.L["autoren"] > 0, "abstract": self.woerter_abstract >= MIN_ABSTRACT,
            "journal": self.journal >= 0, "journal_kat": (self.journal >= 0) & (self.erste_kat >= 0),
            "typ": self.hat_typ, "alle": np.ones(n, dtype=bool),
            "ab2003": (self.jahr >= NEU_AB) & (self.L["phrasen"] > 0),
            "ab2011": self.jahr >= KOHORTE_AB}
        # EO5 A1 (17.10.): Unigramme des Phrasenvokabulars; Vokabular wie in baue_cache neu aufgebaut (gleiche Reihenfolge)
        ph = [phrasen(x) for x in self.df["text"].fillna("")]
        dfz = Counter(q for s in ph for q in s)
        vok = sorted(q for q, c in dfz.items() if MIN_DF <= c <= MAX_DF * len(ph))
        if len(vok) != meta["phrasen_vokabular"]:
            raise SystemExit("Phrasenvokabular passt nicht zum Zwischenspeicher")
        self.U = self.M["phrasen"][:, np.where(np.array([" " not in v for v in vok]))[0]].tocsr()
        df_uni = np.asarray(self.U.sum(axis=0)).ravel()
        self.abdeckung["eo5"] = (np.asarray(self.U[:, df_uni > SEMD_MIN].sum(axis=1)).ravel() >= SEMD_MIN_WOERTER)
        uhr(t0, f"Korpus bereit: {n} Dokumente, {len(set(self.labels[self.labels >= 0]))} Topics, {self.U.shape[1]} Unigramme")


# ---------------------------------------------------------------- Bausteine

def lam2(M):
    """Simpson-Konzentration ueber Paare von Eintraegen aus verschiedenen Dokumenten.

    Zaehler: gleiche Eintraege aus verschiedenen Dokumenten, Summe n_k^2 minus Summe x_dk^2; Nenner: alle Paare
    aus verschiedenen Dokumenten, N^2 minus Summe L_d^2. Bei einem Eintrag je Dokument identisch mit
    Summe n_k (n_k - 1) / (N (N - 1)). Verhaeltnis zweier U-Statistiken, deshalb ohne den Kleinstichproben-Bias des
    Plug-in-Schaetzers; die Rarefaktion auf m Dokumente ist bei q = 2 im Erwartungswert dieselbe Groesse.
    """
    L = np.asarray(M.sum(axis=1)).ravel()
    if (L > 0).sum() < 2:
        return np.nan
    nk = np.asarray(M.sum(axis=0)).ravel()
    N = L.sum()
    zaehler = float((nk ** 2).sum() - M.multiply(M).sum())
    nenner = float(N ** 2 - (L ** 2).sum())
    return zaehler / nenner if nenner > 0 else np.nan


def renyi2(M):
    lam = lam2(M)
    return float(-np.log(lam)) if lam == lam and lam > 0 else np.nan


def chao_shen(nk):
    nk = nk[nk > 0]
    N = nk.sum()
    if N < 2:
        return np.nan
    f1 = (nk == 1).sum()
    if f1 == N:
        f1 = N - 1
    C = 1 - f1 / N
    pa = C * nk / N
    la = 1 - (1 - pa) ** N
    return float(-np.sum(pa * np.log(pa) / la))


def wj_aus_min(m):
    """Gewichtete Jaccard-Distanz zweier normierter Haeufigkeitsvektoren aus der Summe der Minima."""
    return 1.0 - m / (2.0 - m)


def zeitdistanzen(X, jahre, modus, rng, r_perm):
    """Beobachteter Wert und Permutationswerte der Vokabulardistanz eines Topics.

    modus 'haelften' wie EO1: frueh = Jahr bis zum Median-Jahr, spaet = danach; 'folgejahre' wie DS2: Mittel ueber
    aufeinanderfolgende vorhandene Jahre, nur Paare, in denen beide Jahre Eintraege haben. Die Permutation
    vertauscht die Jahre unter den Dokumenten des Topics und erhaelt so die Abdeckung je Dokument.
    """
    n = len(jahre)
    perms = [jahre] + [rng.permutation(jahre) for _ in range(r_perm)]
    P = len(perms)
    if modus == "haelften":
        med = np.median(jahre)
        T = 2
        g = np.stack([(p > med).astype(int) for p in perms], axis=1)
    else:
        yrs = np.unique(jahre)
        T = len(yrs)
        g = np.stack([np.searchsorted(yrs, p) for p in perms], axis=1)
    if T < 2:
        return np.nan, np.full(P - 1, np.nan)
    spalten = g + T * np.arange(P)[None, :]          # n x P: Block b enthaelt die Jahresgruppen der Permutation b
    Y = sparse.csr_matrix((np.ones(n * P), (np.repeat(np.arange(n), P), spalten.ravel())), shape=(n, T * P))
    C = (Y.T @ X).tocsr()
    s = np.asarray(C.sum(axis=1)).ravel()
    inv = np.zeros_like(s)
    inv[s > 0] = 1.0 / s[s > 0]
    Pn = (sparse.diags(inv) @ C).tocsr()
    idx_a = np.concatenate([b * T + np.arange(T - 1) for b in range(P)])
    m = np.asarray(Pn[idx_a].minimum(Pn[idx_a + 1]).sum(axis=1)).ravel()
    gueltig = (s[idx_a] > 0) & (s[idx_a + 1] > 0)
    d = np.where(gueltig, wj_aus_min(m), np.nan).reshape(P, T - 1)
    werte = np.full(P, np.nan)
    hat = (~np.isnan(d)).any(axis=1)
    werte[hat] = np.nanmean(d[hat], axis=1)
    return werte[0], werte[1:]


def ses(beob, null):
    null = null[~np.isnan(null)]
    if np.isnan(beob) or len(null) < 10:
        return np.nan
    sd = null.std(ddof=1)
    return float((beob - null.mean()) / sd) if sd > 0 else np.nan


def poisson_steigung(y, A, t):
    ok = A > 0
    y, A, t = y[ok].astype(float), A[ok].astype(float), t[ok].astype(float)
    if y.sum() == 0 or len(y) < 3:
        return np.nan
    t = t - t.mean()
    a, b = np.log(y.sum() / A.sum()), 0.0
    for _ in range(200):
        mu = A * np.exp(a + b * t)
        g = np.array([np.sum(y - mu), np.sum((y - mu) * t)])
        H = np.array([[mu.sum(), (mu * t).sum()], [(mu * t).sum(), (mu * t * t).sum()]])
        try:
            schritt = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.nan
        faktor = 1.0
        ll0 = np.sum(y * (a + b * t) - mu)
        while faktor > 1e-4:
            a1, b1 = a + faktor * schritt[0], b + faktor * schritt[1]
            ll1 = np.sum(y * (a1 + b1 * t) - A * np.exp(a1 + b1 * t))
            if ll1 >= ll0 - 1e-12:
                break
            faktor /= 2
        a, b = a1, b1
        if np.abs(faktor * schritt).max() < 1e-10:
            return float(b) if abs(b) < 5 else np.nan
    return np.nan


# ---------------------------------------------------------------- Kandidaten fuer ein (Pseudo-)Korpus

def kandidaten(K, keep, lab, lauf_seed, hilf=False):
    """keep: Indizes ins volle Korpus (aufsteigend); lab: Labels dieser Dokumente (Rauschen = -1)."""
    keep = np.asarray(keep)
    tids = sorted(set(lab[lab >= 0].tolist()))
    pos = {t: np.where(lab == t)[0] for t in tids}
    voll = {t: keep[pos[t]] for t in tids}
    jahr_k = K.jahr[keep]
    zug = lab >= 0
    R = {c: {} for c in KCODES}
    H = {}

    # Korpusgroessen dieses Pseudokorpus
    jahre_fenster = np.arange(JAHR_MIN, JAHR_MAX + 1)
    A_t = np.array([np.sum(zug & (jahr_k == y)) for y in jahre_fenster])
    vorh = A_t > 0
    # PE3: Journalgroessen, Prestige und Kategorie aus dem Pseudokorpus
    jk = K.journal[keep]
    mit_j = jk >= 0
    nj = np.bincount(jk[mit_j], minlength=K.journal.max() + 1)
    median_j = float(np.median(nj[nj > 0]))
    nische_dok = np.zeros(len(keep), dtype=bool)
    nische_dok[mit_j] = nj[jk[mit_j]] <= median_j
    K_nische, N_j = int(nische_dok.sum()), int(mit_j.sum())
    rate_k = K.rate[keep]
    pres = np.bincount(jk[mit_j], weights=rate_k[mit_j], minlength=len(nj)) / np.maximum(nj, 1)
    kat_k = K.erste_kat[keep]
    jk_df = pd.DataFrame({"j": jk[mit_j], "k": kat_k[mit_j]})
    jk_df = jk_df[jk_df["k"] >= 0]
    haupt = (jk_df.groupby(["j", "k"]).size().reset_index(name="n")
             .sort_values(["j", "n", "k"], ascending=[True, False, True]).drop_duplicates("j"))
    haupt["p"] = pres[haupt["j"].values]
    haupt["pct"] = haupt.groupby("k")["p"].rank(method="average")
    haupt["pct"] = (haupt["pct"] - 0.5) / haupt.groupby("k")["p"].transform("size")
    pct_j = np.full(len(nj), np.nan)
    pct_j[haupt["j"].values] = haupt["pct"].values
    # DS3: Prior am Review-Anteil der zugeordneten Dokumente
    rev_k, typ_k = K.review[keep], K.hat_typ[keep]
    p0 = float(rev_k[zug & typ_k].sum() / max((zug & typ_k).sum(), 1))
    staerke = 2 * float(bt.config.REVIEW_ABSENCE_ALPHA)
    # IP3 A2: Top 10 % je Jahr im Pseudokorpus
    tc_k = K.tc[keep]
    top10 = np.zeros(len(keep), dtype=bool)
    pct_jahr = np.zeros(len(keep))
    tc_norm = np.zeros(len(keep))          # DI4 A3 und A4 (17.10.): Zitationen geteilt durch das Jahrgangsmittel
    for y in np.unique(jahr_k):
        w = np.where(jahr_k == y)[0]
        rk = pd.Series(tc_k[w]).rank(method="average").values / len(w)
        top10[w] = rk > 0.9
        pct_jahr[w] = rk
        mu = tc_k[w].mean()
        tc_norm[w] = tc_k[w] / mu if mu > 0 else 0.0
    # DS2 A3: Erstjahr jeder Phrase im Pseudokorpus, Anteil neuer Phrasen je Dokument, normiert je Jahrgang
    Pk = K.M["phrasen"][keep]
    Pc = Pk.tocsc()
    erst = np.full(Pc.shape[1], 10 ** 6, dtype=np.int64)
    belegt = np.diff(Pc.indptr) > 0
    if belegt.any():
        erst[belegt] = np.minimum.reduceat(jahr_k[Pc.indices], Pc.indptr[:-1][belegt])
    zeile = np.repeat(np.arange(len(keep)), np.diff(Pk.indptr))
    neu = erst[Pk.indices] >= (jahr_k[zeile] - NEU_FENSTER)
    laenge_p = np.diff(Pk.indptr).astype(float)
    mit_p = laenge_p > 0
    anteil_neu = np.full(len(keep), np.nan)
    anteil_neu[mit_p] = np.bincount(zeile, weights=neu, minlength=len(keep))[mit_p] / laenge_p[mit_p]
    neu_norm = np.full(len(keep), np.nan)
    for y in np.unique(jahr_k[mit_p]):
        w = np.where(mit_p & (jahr_k == y))[0]
        neu_norm[w] = anteil_neu[w] - anteil_neu[w].mean()
    # DI1 A1: m = min(25, kleinste Topicgroesse mit Autoren)
    n_aut = {t: int((K.L["autoren"][voll[t]] > 0).sum()) for t in tids}
    m_grp = int(min(M_RARE, min(n_aut.values())))
    # PE2: Nachbarn der Zentroide im reduzierten Raum dieses Pseudokorpus
    red_k = K.red[keep]
    zentr = np.vstack([red_k[pos[t]].mean(axis=0) for t in tids])
    nn = NearestNeighbors(n_neighbors=50).fit(red_k)
    _, nb_idx = nn.kneighbors(zentr)
    rausch_nb = lab[nb_idx] == -1
    # DIAG: Nachbarn mit Kosinus ab TAU ausserhalb des eigenen Topics
    E_k = K.E[keep]
    # EO5 A1: Summenvektoren und Dokumentzahlen je Wort im Pseudokorpus
    U_k = K.U[keep]
    E64 = E_k.astype(np.float64)
    n_w = np.asarray(U_k.sum(axis=0)).ravel()
    S_w = np.asarray(U_k.T @ E64)
    z_idx = np.where(zug)[0]
    aussen = np.zeros(len(keep), dtype=np.int64)
    for a in range(0, len(z_idx), 2048):
        blk = z_idx[a:a + 2048]
        S = E_k[blk] @ E_k.T
        S[np.arange(len(blk)), blk] = -1.0
        aussen[blk] = ((S >= TAU) & (lab[None, :] != lab[blk, None])).sum(axis=1)

    for t in tids:
        fi = voll[t]
        p = pos[t]
        n = len(fi)
        jt = K.jahr[fi]
        # EO1 A1, EO2 A1/A2, IP3 A1, DI2 A1, DI3 A1/A2
        R["EO1_A1"][t] = renyi2(K.M["phrasen"][fi])
        Mk = K.M["kategorien"][fi]
        r2 = renyi2(Mk)
        R["EO2_A1"][t] = r2 / K.ln_k if r2 == r2 else np.nan
        cs = chao_shen(np.asarray(Mk.sum(axis=0)).ravel())
        R["EO2_A2"][t] = cs / K.ln_k if cs == cs else np.nan
        R["IP3_A1"][t] = renyi2(K.M["kwplus"][fi])
        R["DI2_A1"][t] = lam2(K.M["institutionen"][fi])
        Ml = K.M["laender"][fi]
        R["DI3_A1"][t] = lam2(Ml)
        Ll = K.L["laender"][fi]
        R["DI3_A2"][t] = float(1 - np.mean(Ll[Ll > 0] >= 2)) if (Ll > 0).sum() >= 2 else np.nan
        # EO3 A1
        v = K.E[fi].astype(np.float64)
        sv = v.sum(axis=0)
        s_beob = (sv @ sv - n) / (n * (n - 1))
        c = np.bincount(K.lbin[fi], minlength=K.G.shape[0]).astype(float)
        s_erw = (c @ K.G @ c - c @ np.diag(K.G)) / (n * (n - 1))
        R["EO3_A1"][t] = float(s_erw - s_beob)
        if hilf:
            H.setdefault("EO3_beob_inkoh", {})[t] = float(1 - s_beob)
        # EO4 A2
        ok = K.woerter_abstract[fi] >= MIN_ABSTRACT
        R["EO4_A2"][t] = float(np.mean(1000 * K.hedge[fi][ok] / K.woerter_abstract[fi][ok])) if ok.sum() else np.nan
        if hilf:
            H.setdefault("EO4_booster", {})[t] = float(np.mean(1000 * K.booster[fi][ok] / K.woerter_abstract[fi][ok]))
        # PE2 A1/A2
        i_t = tids.index(t)
        R["PE2_A1"][t] = float(rausch_nb[i_t, :25].mean())
        R["PE2_A2"][t] = float(rausch_nb[i_t, :50].mean())
        # PE3 A1/A2
        jt_j = K.journal[fi]
        okj = jt_j >= 0
        vals = 1 - pct_j[jt_j[okj]]
        vals = vals[~np.isnan(vals)]
        R["PE3_A1"][t] = float(vals.mean()) if len(vals) >= 2 else np.nan
        nn_t = int(okj.sum())
        x = int(nische_dok[p][okj].sum())
        q = K_nische / N_j
        var = nn_t * q * (1 - q) * (N_j - nn_t) / (N_j - 1)
        R["PE3_A2"][t] = float((x - nn_t * q) / np.sqrt(var)) if var > 0 else np.nan
        # DS1 A1/A2
        R["DS1_A1"][t] = float(np.mean(1.0 / (JAHR_MAX - jt + 1)))
        R["DS1_A2"][t] = float(np.mean(jt > JAHR_MAX - AKTIV_JAHRE))
        # DS2 A1/A2 und EO1 A2 (SES gegen Permutation der Jahre im Topic)
        rng = np.random.default_rng([lauf_seed, int(t), 1])
        if len(np.unique(jt)) >= 3:
            b, nl = zeitdistanzen(K.M["keywords"][fi], jt, "folgejahre", rng, R_PERM)
            R["DS2_A1"][t] = ses(b, nl)
            if hilf:
                H.setdefault("DS2_beob", {})[t] = b
            rng = np.random.default_rng([lauf_seed, int(t), 2])
            b, nl = zeitdistanzen(K.M["phrasen"][fi], jt, "folgejahre", rng, R_PERM)
            R["DS2_A2"][t] = ses(b, nl)
        else:
            R["DS2_A1"][t] = R["DS2_A2"][t] = np.nan
        rng = np.random.default_rng([lauf_seed, int(t), 3])
        if n >= 4:
            b, nl = zeitdistanzen(K.M["keywords"][fi], jt, "haelften", rng, R_PERM)
            R["EO1_A2"][t] = ses(b, nl)
            if hilf:
                H.setdefault("EO1_beob", {})[t] = b
        else:
            R["EO1_A2"][t] = np.nan
        # DS3 A1
        r_t, n_t = int((K.review[fi] & K.hat_typ[fi]).sum()), int(K.hat_typ[fi].sum())
        R["DS3_A1"][t] = float(beta_vt.cdf(p0, staerke * p0 + r_t, staerke * (1 - p0) + n_t - r_t))
        # DI1 A1
        Ma = K.M["autoren"][fi]
        Ma = Ma[np.asarray(Ma.sum(axis=1)).ravel() > 0]
        na = Ma.shape[0]
        rng = np.random.default_rng([lauf_seed, int(t), 4])
        if na >= m_grp >= 2:
            ks = []
            for _ in range(1 if na == m_grp else R_ZIEH):
                sel = np.arange(na) if na == m_grp else rng.choice(na, m_grp, replace=False)
                S = Ma[sel]
                ks.append(connected_components(S @ S.T, directed=False)[0])
            R["DI1_A1"][t] = float(1 - (np.mean(ks) - 1) / (m_grp - 1))
        else:
            R["DI1_A1"][t] = np.nan
        # IP1 A1
        y_t = np.array([np.sum(jt == y) for y in jahre_fenster])
        R["IP1_A1"][t] = poisson_steigung(y_t[vorh], A_t[vorh], jahre_fenster[vorh])
        # IP2 A1/A2
        if n >= 4:
            med = np.median(jt)
            fr, sp = jt <= med, jt > med
            if fr.sum() and sp.sum():
                rt = K.rate[fi]
                R["IP2_A1"][t] = float(np.log((rt[sp].mean() + PSEUDO) / (rt[fr].mean() + PSEUDO)))
                tcs, als = K.tc[fi], K.alter[fi]
                if tcs[sp].sum() > 0 and tcs[fr].sum() > 0:
                    R["IP2_A2"][t] = float(np.log((tcs[sp].sum() / als[sp].sum()) / (tcs[fr].sum() / als[fr].sum())))
                else:
                    R["IP2_A2"][t] = np.nan
            else:
                R["IP2_A1"][t] = R["IP2_A2"][t] = np.nan
        else:
            R["IP2_A1"][t] = R["IP2_A2"][t] = np.nan
        # IP3 A2
        R["IP3_A2"][t] = float(top10[p].mean())
        # DS2 A3 (Ergaenzung 17.10.)
        okn = (jt >= NEU_AB) & ~np.isnan(neu_norm[p])
        R["DS2_A3"][t] = float(neu_norm[p][okn].mean()) if okn.any() else np.nan
        # IP2 A3 (Ergaenzung 17.10.)
        if n >= 4:
            med = np.median(jt)
            fr, sp = jt <= med, jt > med
            R["IP2_A3"][t] = float(pct_jahr[p][sp].mean() - pct_jahr[p][fr].mean()) if fr.sum() and sp.sum() else np.nan
        else:
            R["IP2_A3"][t] = np.nan
        # IP2 A4 und A5 (Ergaenzung 17.10.)
        R["IP2_A4"][t] = (float(kendalltau(jt, pct_jahr[p])[0]) if n >= 4 and len(np.unique(jt)) >= 2 else np.nan)
        okk = jt >= KOHORTE_AB
        R["IP2_A5"][t] = float(pct_jahr[p][okk].mean()) if okk.any() else np.nan
        # DI4 A3 und A4 (Ergaenzung 17.10.)
        x = tc_norm[p]
        if n >= 3 and x.sum() > 0:
            xs = np.sort(x)
            g = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * xs) / (n * xs.sum()))
            R["DI4_A3"][t] = g * n / (n - 1)
            R["DI4_A4"][t] = float(2 * norm.cdf(np.std(np.log1p(x), ddof=1) / np.sqrt(2)) - 1)
        else:
            R["DI4_A3"][t] = R["DI4_A4"][t] = np.nan
        # EO5 A1 (Ergaenzung 17.10.): semantische Diversitaet der Woerter ohne die Dokumente des eigenen Topics
        Ut = U_k[p]
        cols = np.unique(Ut.indices)
        if len(cols):
            Utc = Ut[:, cols].tocsr()
            m_w = n_w[cols] - np.asarray(Utc.sum(axis=0)).ravel()
            Sd = S_w[cols] - np.asarray(Utc.T @ E64[p])
            gueltig = m_w >= SEMD_MIN
            semd = np.full(len(cols), np.nan)
            mm = m_w[gueltig].astype(float)
            cbar = (np.einsum("ij,ij->i", Sd[gueltig], Sd[gueltig]) - mm) / (mm * (mm - 1))
            semd[gueltig] = -np.log(np.maximum(cbar, SEMD_BODEN))
            W = Utc[:, np.where(gueltig)[0]]
            anz = np.asarray(W.sum(axis=1)).ravel()
            summe = np.asarray(W @ semd[gueltig]).ravel()
            ok5 = anz >= SEMD_MIN_WOERTER
            R["EO5_A1"][t] = float((summe[ok5] / anz[ok5]).mean()) if ok5.any() else np.nan
            if hilf:
                H.setdefault("EO5_gueltige_dok", {})[t] = float(ok5.mean())
        else:
            R["EO5_A1"][t] = np.nan
        # DIAG A1
        R["DIAG_A1"][t] = float(np.mean(aussen[p] >= K_NACHBARN))

    # PE1 A1 (braucht alle Topics)
    dov = {}
    for t in tids:
        jt = K.jahr[voll[t]]
        y_t = np.array([np.sum(jt == y) for y in jahre_fenster])
        anteil = np.where(vorh, y_t / np.maximum(A_t, 1), 0.0)
        dov[t] = float(np.sum(anteil[vorh] * (1 - DOV_ABSCHLAG * (JAHR_MAX - jahre_fenster[vorh]))))
    dv = pd.Series(dov)
    dv = dv.clip(lower=dv.quantile(DOV_BODEN))
    rk = dv.rank(method="average", ascending=True)
    for t in tids:
        R["PE1_A1"][t] = float(1.0 - rk[t] / len(tids))

    out = pd.DataFrame({c: [R[c].get(t, np.nan) for t in tids] for c in KCODES}, index=pd.Index(tids, name="topic"))
    if hilf:
        return out, pd.DataFrame(H).reindex(out.index), {"m_gruppen": m_grp, "p0_review": p0}
    return out, None, {"m_gruppen": m_grp, "p0_review": p0}


# ---------------------------------------------------------------- Teile

def pseudokorpora():
    """Dieselben Pseudokorpora wie backtest_phase1.py, in derselben Reihenfolge der Zufallszahlen."""
    labels = lade_labels()
    rausch = np.where(labels < 0)[0]
    tids = sorted(set(labels[labels >= 0]))
    zug = np.where(labels >= 0)[0]
    for r in range(50):
        rng = np.random.default_rng(1000 + r)
        a, b = [], []
        for t in tids:
            idx = rng.permutation(np.where(labels == t)[0])
            h = len(idx) // 2
            a.append(idx[:h])
            b.append(idx[h:])
        for name, teile, halb in (("A", a, 0), ("B", b, 1)):
            keep = np.sort(np.concatenate([rausch] + teile))
            yield f"split_r{r:02d}_{name}", keep, labels[keep], (1000 + r) * 10 + halb
    for r in range(30):
        rng = np.random.default_rng(2000 + r)
        teile = [rng.choice(np.where(labels == t)[0], size=min(M_RARE, int((labels == t).sum())), replace=False)
                 for t in tids]
        keep = np.sort(np.concatenate([rausch] + teile))
        yield f"rare_m{M_RARE}_r{r:02d}", keep, labels[keep], 2000 + r
    for r in range(50):
        rng = np.random.default_rng(3000 + r)
        lab = labels.copy()
        lab[zug] = rng.permutation(labels[zug])
        yield f"null_r{r:02d}", np.arange(len(labels)), lab, 3000 + r


def teil_repro(t0):
    K = Korpus(t0)
    keep = np.arange(len(K.labels))
    ind, hilf, info = kandidaten(K, keep, K.labels, 0, hilf=True)
    uhr(t0, "Referenzwerte der Kandidaten gerechnet")
    OUT.mkdir(parents=True, exist_ok=True)
    ind.to_csv(OUT / "referenz.csv")
    hilf.to_csv(OUT / "referenz_hilfswerte.csv")
    # Gegenproben gegen die bestehenden Werte aus Schritt 1a
    alt = bt.lies("referenz")
    proben = {}
    for code, spalte_alt in (("EO1_beob", "keyword_volatility"), ("DS2_beob", "terminological_instability")):
        d = (hilf[code] - alt[spalte_alt]).abs()
        proben[f"{code} gegen {bt.CODE[spalte_alt]}"] = {"max_abweichung": float(d.max()),
                                                          "verglichen": int(d.notna().sum())}
    proben["EO3 Inkohaerenz exakt gegen Stichprobe 200 (rho)"] = round(rho(hilf["EO3_beob_inkoh"], alt["semantic_incoherence"]), 4)
    # DI1 von v2.5 mit ORCID ueber den Namen statt ueber die Position
    korr = {}
    for t in ind.index:
        fi = np.where(K.labels == t)[0]
        Ma = K.M["autoren"][fi]
        cnt = np.asarray(Ma.sum(axis=0)).ravel()
        cnt = cnt[cnt > 0]
        korr[t] = 1.0 if cnt.sum() < 2 else max(0, min(1, s2.gini(cnt.astype(float))))
    korr = pd.Series(korr)
    proben["DI1 v2.5 mit ORCID ueber den Namen"] = {
        "rho_mit_v25": round(rho(korr, alt["author_concentration"]), 4),
        "max_abweichung": round(float((korr - alt["author_concentration"]).abs().max()), 4),
        "mittlere_abweichung": round(float((korr - alt["author_concentration"]).abs().mean()), 4)}
    hilf["DI1_orcid_name"] = korr
    hilf.to_csv(OUT / "referenz_hilfswerte.csv")
    proben["EO4 Hedge mit Booster (rho)"] = round(rho(ind["EO4_A2"], hilf["EO4_booster"]), 4)
    # Abdeckung je Kandidat (K1)
    n_top = pd.Series(K.labels[K.labels >= 0]).value_counts().sort_index()
    abd = {}
    for code in KCODES:
        f = K.abdeckung[FELD[code]]
        je = pd.Series(f[K.labels >= 0].astype(int)).groupby(K.labels[K.labels >= 0]).sum().reindex(n_top.index)
        abd[code] = {"feld": FELD[code], "topics_unter_10_dok": int((je < 10).sum()),
                     "anteil_topics_ab_10": round(float((je >= 10).mean()), 4),
                     "anteil_dok_mit_feld": round(float(f[K.labels >= 0].mean()), 4),
                     "nicht_bestimmbar": int(ind[code].isna().sum())}
    laengen = {"mittlere_kosinus_innerhalb_der_laengenklassen": np.round(np.diag(K.G), 4).tolist()}
    res = {"meta": K.meta, "info": info, "gegenproben": proben, "abdeckung": abd, "eo3_laengenklassen": laengen,
           "diag_verteilung": np.quantile(ind["DIAG_A1"], [0, .1, .5, .9, 1]).round(4).tolist()}
    (OUT / "repro_kandidaten.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "abdeckung"}, ensure_ascii=False))
    print(pd.DataFrame(abd).T.to_string())


def teil_lauf(t0, budget):
    K = Korpus(t0)
    OUT.mkdir(parents=True, exist_ok=True)
    fertig = offen = 0
    for name, keep, lab, seed in pseudokorpora():
        if (OUT / f"{name}.csv").exists() and set(KCODES) <= set(pd.read_csv(OUT / f"{name}.csv", nrows=0).columns):
            fertig += 1
            continue
        if time.time() - t0 > budget:
            offen += 1
            continue
        ind, _, _ = kandidaten(K, keep, lab, seed)
        ind.to_csv(OUT / f"{name}.csv")
        fertig += 1
        uhr(t0, f"{name} fertig")
    print(f"STAND: {fertig} fertig, {offen} offen", flush=True)


# ---------------------------------------------------------------- Report

def rho(a, b):
    b = b.reindex(a.index)
    ok = a.notna() & b.notna()
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 10 else float("nan")


def lade_labels():
    with open(bt.REF / "model_results.pkl", "rb") as f:
        return np.asarray(pickle.load(f)["labels"])


def lies_k(name):
    return pd.read_csv(OUT / f"{name}.csv", index_col=0)


def kennzahlen(ref, lies_fn, splits, rares, nulls, logn, spalte):
    z = {}
    rs = [rho(lies_fn(f"split_{s}_A")[spalte], lies_fn(f"split_{s}_B")[spalte]) for s in splits]
    rs = [x for x in rs if not np.isnan(x)]
    if rs:
        r_h = float(np.mean(rs))
        z["K2_spearman_brown"] = round(2 * r_h / (1 + r_h), 6) if r_h > -1 else float("nan")
        z["K2_spanne_r"] = [round(min(rs), 3), round(max(rs), 3)]
        z["K2_teilungen"] = len(rs)
    z["K4_rho_logn"] = round(rho(ref[spalte], logn), 6)
    if rares:
        rm = pd.concat([lies_fn(p)[spalte] for p in rares], axis=1).mean(axis=1)
        z["K4_rare_rho_mit_voll"] = round(rho(rm, ref[spalte]), 6)
        z["K4_rare_rho_logn"] = round(rho(rm, logn), 6)
    if nulls:
        nm = pd.concat([lies_fn(p)[spalte] for p in nulls], axis=1)
        e0, s0 = nm.mean(axis=1), nm.std(axis=1)
        ok = ref[spalte].notna() & e0.notna()
        r2 = (float(np.corrcoef(ref[spalte][ok], e0[ok])[0, 1] ** 2)
              if ok.sum() > 10 and e0[ok].std() > 0 and ref[spalte][ok].std() > 0 else float("nan"))
        z["K5_null_r2"] = round(r2, 6)
        sesv = (ref[spalte] - e0) / s0.replace(0, np.nan)
        z["K5_anteil_ses_ueber_2"] = round(float((sesv.abs() > 2).mean()), 6)
        z["K5_permutationen"] = len(nulls)
    return z


def urteil(z, k1_ok, ist_pe1=False):
    """Status je Kriterium; K2 unter 0,5 verfehlt, 0,5 bis 0,75 bedingt."""
    s = {"K1": "erfuellt" if k1_ok else "verfehlt"}
    sb = z.get("K2_spearman_brown", np.nan)
    s["K2"] = "verfehlt" if not sb == sb or sb < 0.5 else ("bedingt" if sb < 0.75 else "gut")
    if ist_pe1:
        s["K4"] = s["K5"] = "entfaellt"
    else:
        s["K4"] = "erfuellt" if abs(z.get("K4_rho_logn", np.nan)) < 0.2 else "verfehlt"
        s["K5"] = "erfuellt" if z.get("K5_null_r2", np.nan) <= 0.5 else "verfehlt"
    s["K6"] = "erfuellt" if z.get("K6_max_abs_rho", 0) <= 0.9 else "verfehlt"
    s["alle"] = all(v in ("erfuellt", "gut", "bedingt", "entfaellt") for v in s.values())
    return s


def teil_report():
    ref = lies_k("referenz")
    alt_ref = bt.lies("referenz")
    labels = lade_labels()
    n = pd.Series(labels[labels >= 0]).value_counts().reindex(ref.index)
    logn = np.log(n.astype(float))
    splits = sorted({p.stem.split("_")[1] for p in OUT.glob("split_r*_B.csv")})
    rares = sorted(p.stem for p in OUT.glob(f"rare_m{M_RARE}_r*.csv"))
    nulls = sorted(p.stem for p in OUT.glob("null_r*.csv"))
    abd = json.loads((OUT / "repro_kandidaten.json").read_text(encoding="utf-8"))["abdeckung"]
    bt_tab = pd.read_csv(bt.OUT / "backtest_phase1_kennzahlen.csv").set_index("Indikator")
    alt_codes = {v: k for k, v in bt.CODE.items()}
    alt_rang = alt_ref.rank()
    kand_rang = ref.rank()
    zeilen = []
    for code in KCODES:
        z = {"Code": code, "Platz": PLATZ[code], "Beschreibung": BESCHR[code]}
        z.update(kennzahlen(ref, lies_k, splits, rares, nulls, logn, code))
        a = abd[code]
        z["K1_topics_unter_10"] = a["topics_unter_10_dok"]
        z["K1_ok"] = a["anteil_topics_ab_10"] >= 0.9
        z["nicht_bestimmbar"] = a["nicht_bestimmbar"]
        platz = PLATZ[code]
        andere_alt = [c for c in alt_ref.columns if bt.CODE[c] != platz]
        andere_k = [c for c in KCODES if PLATZ[c] != platz and PLATZ[c] != "DIAG"]
        c_alt = alt_rang[andere_alt].corrwith(kand_rang[code]).abs()
        c_k = kand_rang[andere_k].corrwith(kand_rang[code]).abs()
        z["K6_max_abs_rho"] = round(float(max(c_alt.max(), c_k.max())), 6)
        z["K6_partner"] = (bt.CODE[c_alt.idxmax()] if c_alt.max() >= c_k.max() else c_k.idxmax())
        if platz in alt_codes:
            z["rho_mit_bestehend"] = round(rho(ref[code], alt_ref[alt_codes[platz]]), 6)
        z.update({f"S_{k}": v for k, v in urteil(z, z["K1_ok"], platz == "PE1").items()})
        zeilen.append(z)
    tab = pd.DataFrame(zeilen)
    tab.to_csv(OUT / "kandidaten_kennzahlen.csv", index=False)
    (OUT / "kandidaten_kennzahlen.json").write_text(tab.to_json(orient="records", force_ascii=False, indent=1),
                                                    encoding="utf-8")
    # Bestehende Indikatoren mit denselben Statuslogiken (Werte aus Schritt 1a)
    best = {}
    for code_alt, zz in bt_tab.iterrows():
        z = {"K2_spearman_brown": zz["K2_spearman_brown"], "K4_rho_logn": zz["K4_rho_logn"],
             "K5_null_r2": zz["K5_null_r2"], "K6_max_abs_rho": zz["K6_max_abs_rho"]}
        k1_ok = code_alt not in ("EO1", "DS2")
        best[code_alt] = urteil(z, k1_ok, code_alt == "PE1")
    # Regeln R1 bis R4 je Platz
    entscheid = {}
    for platz in ["EO1", "EO2", "EO3", "EO4", "EO5", "PE1", "PE2", "PE3", "DS1", "DS2", "DS3", "DI1", "DI2", "DI3", "DI4",
                  "IP1", "IP2", "IP3"]:
        alts = tab[tab["Platz"] == platz]
        ok_alts = alts[alts["S_alle"]]["Code"].tolist()
        b = best.get(platz)
        if platz == "EO5":
            entscheid[platz] = ("aufnehmen (neue Facette, erfuellt alle Kriterien)" if ok_alts
                                else "R4 (neue Facette verfehlt Kriterien)")
            continue
        if platz == "EO4":
            entscheid[platz] = ("A2 erfuellt die Kriterien; A1 (Zertifiziertheitsmodell) steht aus, danach R3 (Doppelmessung)"
                                if ok_alts else "A2 verfehlt Kriterien; A1 steht aus")
            continue
        if b and b["alle"]:
            entscheid[platz] = "R1 behalten" + (" (K2 bedingt)" if b["K2"] == "bedingt" else "")
            continue
        bev = BEVORZUGT[platz]
        if ok_alts:
            if bev in ok_alts:
                entscheid[platz] = f"R2 ersetzen durch {bev} (bevorzugt)"
            elif len(ok_alts) == 1:
                entscheid[platz] = f"R2 ersetzen durch {ok_alts[0]} (einzige Alternative, die alle Kriterien erfuellt)"
            else:
                entscheid[platz] = f"R2 ersetzen, Wahl offen zwischen {', '.join(ok_alts)} (keine Vorabpraeferenz)"
            continue
        entscheid[platz] = "R4 eingeschraenkt fuehren oder streichen (keine Alternative erfuellt alle Kriterien)"
    # Redundanz im Satz nach den Regeln (bestehende Werte fuer R1, Kandidaten fuer R2; offene Wahl: erste Nennung)
    # Bens Entscheidungen vom 17.10.: E2 (R4 bei struktureller Groesse mit den bevorzugten Varianten), DS2 durch A3
    # eingeschraenkt (nach 1c), IP2 durch A4 mit A5 als Rueckfall bei K3 (E7, nach 1d); None heisst bestehend
    R4_WAHL = {"EO3": "EO3_A1", "PE2": "PE2_A1", "DS1": None, "DI1": "DI1_A1", "DI2": "DI2_A1", "DS2": "DS2_A3", "IP2": None}
    R2_WAHL = {"IP2": "IP2_A4"}
    satz = {"EO4 A2": kand_rang["EO4_A2"]}
    if entscheid.get("EO5", "").startswith("aufnehmen"):
        satz["EO5 A1"] = kand_rang["EO5_A1"]
    for platz, txt in entscheid.items():
        if txt.startswith("R1") and platz in alt_codes:
            satz[platz] = alt_rang[alt_codes[platz]]
        elif txt.startswith("R2"):
            wahl = R2_WAHL.get(platz) or re.findall(r"[A-Z]{2}\d_A\d", txt)[0]
            satz[wahl.replace("_", " ")] = kand_rang[wahl]
        elif txt.startswith("R4"):
            w = R4_WAHL.get(platz)
            if w:
                satz[w.replace("_", " ")] = kand_rang[w]
            elif platz in alt_codes:
                satz[platz + " (bestehend)"] = alt_rang[alt_codes[platz]]
    satz_df = pd.DataFrame(satz)
    kor = satz_df.corr().abs()
    np.fill_diagonal(kor.values, np.nan)
    paar = kor.stack().idxmax()
    satz_info = {"plaetze": list(satz_df.columns), "max_abs_rho": round(float(kor.max().max()), 6),
                 "paar": list(paar)}
    (OUT / "entscheidungsvorlage_regeln.json").write_text(json.dumps(
        {"bestehend": best, "regeln": entscheid, "satz_nach_regeln": satz_info}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print("Satz nach den Regeln:", json.dumps(satz_info, ensure_ascii=False))
    spalten = ["Code", "K1_topics_unter_10", "K2_spearman_brown", "K4_rho_logn", "K4_rare_rho_mit_voll",
               "K4_rare_rho_logn", "K5_null_r2", "K5_anteil_ses_ueber_2", "K6_max_abs_rho", "K6_partner",
               "rho_mit_bestehend", "S_alle"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab[[c for c in spalten if c in tab.columns]].to_string(index=False))
    print(json.dumps(entscheid, ensure_ascii=False, indent=1))
    print(f"Teilungen {len(splits)}, Rarefaktion {len(rares)}, Permutationen {len(nulls)}")


# ---------------------------------------------------------------- K7: eingepflanzte Struktur (17.10.)

K7_MENGE = {"EO1_A1": 1, "EO2_A1": 1, "EO3_A1": 1, "IP3_A1": 1, "DI4_A4": 1, "DI1_A1": -1, "DI2_A1": -1, "DI3": -1}
K7_MITTEL = ["EO4_A2", "EO5_A1", "PE3", "DS1", "DS2_A3", "DS3"]
K7_BESTEHEND = {"DI3": "geographic_concentration", "PE3": "journal_specificity", "DS1": "temporal_novelty",
                "DS3": "review_absence"}


def teil_k7(t0):
    """Drei Aufteilungen der Topics in Paare; jedes Paar wird im sonst unveraenderten Korpus zu einem Topic vereinigt."""
    K = Korpus(t0)
    labels = K.labels
    tids = np.array(sorted(set(labels[labels >= 0].tolist())))
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in K7_SEEDS:
        name = f"k7_s{seed}"
        if all((OUT / f"{name}_{x}.csv").exists() for x in ("kand", "best", "paare")):
            continue
        paare = np.random.default_rng(seed).permutation(tids).reshape(-1, 2)
        lab = labels.copy()
        for a, b in paare:
            lab[labels == b] = a
        kand, _, _ = kandidaten(K, np.arange(len(labels)), lab, seed)
        best, fehlt = bt.indikatoren(K.df, lab, K.E, K.red)
        for col in bt.NAN_MOEGLICH:
            best.loc[fehlt[col], col] = np.nan
        kand.to_csv(OUT / f"{name}_kand.csv")
        best.to_csv(OUT / f"{name}_best.csv")
        pd.DataFrame(paare, columns=["a", "b"]).to_csv(OUT / f"{name}_paare.csv", index=False)
        uhr(t0, f"{name} fertig: {len(paare)} Mischungen")


def teil_k7report():
    ref_k = lies_k("referenz")
    ref_b = bt.lies("referenz")
    labels = lade_labels()
    n = pd.Series(labels[labels >= 0]).value_counts()

    def wert(tab_k, tab_b, code, t):
        return tab_b.loc[t, K7_BESTEHEND[code]] if code in K7_BESTEHEND else tab_k.loc[t, code]

    sd = {c: float((ref_b[K7_BESTEHEND[c]] if c in K7_BESTEHEND else ref_k[c]).std())
          for c in list(K7_MENGE) + K7_MITTEL}
    zeilen = []
    for seed in K7_SEEDS:
        name = f"k7_s{seed}"
        mk, mb = lies_k(f"{name}_kand"), pd.read_csv(OUT / f"{name}_best.csv", index_col=0)
        for a, b in pd.read_csv(OUT / f"{name}_paare.csv").itertuples(index=False):
            na, nb = int(n[a]), int(n[b])
            for code in list(K7_MENGE) + K7_MITTEL:
                va, vb, vm = wert(ref_k, ref_b, code, a), wert(ref_k, ref_b, code, b), wert(mk, mb, code, a)
                if np.isnan(va) or np.isnan(vb) or np.isnan(vm):
                    continue
                gm = (na * va + nb * vb) / (na + nb)
                zeilen.append({"seed": seed, "a": a, "b": b, "code": code, "mischung": vm, "gewichtetes_mittel": gm,
                               "abweichung_sd": (vm - gm) / sd[code]})
    d = pd.DataFrame(zeilen)
    d.to_csv(OUT / "k7_mischungen.csv", index=False)
    erg = {}
    for code, grp in d.groupby("code"):
        if code in K7_MENGE:
            anteil = float((K7_MENGE[code] * grp["abweichung_sd"] > 0).mean())
            art = "Mengenmass, erwartet " + ("hoeher" if K7_MENGE[code] > 0 else "niedriger")
        else:
            anteil = float((grp["abweichung_sd"].abs() <= 0.25).mean())
            art = "Mittelwertmass, hoechstens 0,25 SD"
        erg[code] = {"art": art, "mischungen": int(len(grp)), "anteil_erfuellt": round(anteil, 6),
                     "erfuellt": anteil >= 0.8, "median_abweichung_sd": round(float(grp["abweichung_sd"].median()), 6)}
    (OUT / "k7_ergebnis.json").write_text(json.dumps(erg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(erg, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teil", choices=["cache", "repro", "lauf", "report", "k7", "k7report"])
    ap.add_argument("--budget", type=float, default=150.0)
    a = ap.parse_args()
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.teil == "cache":
        baue_cache(t0)
    elif a.teil == "repro":
        teil_repro(t0)
    elif a.teil == "lauf":
        teil_lauf(t0, a.budget)
    elif a.teil == "k7":
        teil_k7(t0)
    elif a.teil == "k7report":
        teil_k7report()
    else:
        teil_report()


if __name__ == "__main__":
    main()
