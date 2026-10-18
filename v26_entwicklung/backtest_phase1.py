#!/usr/bin/env python3
"""V2.6, Entwicklungsschritt 1: Backtest der 16 Indikatoren von v2.5 (Phase 1; Phase 2 nur in der Bestaetigung, 3.4).

Das Skript gibt nur Guetemasse und abgeleitete Indikatorwerte je Topic aus, keine Zugehoerigkeiten und
keine Konfigurationen (Protokoll V2.6, Verblindung bis zum Einfrieren). Datensaetze verlassen den Rechner nicht.
Die Phase waehlt phase.py (V26_PHASE, V26_AUSGABE); fuer Phase 1 ohne V26_AUSGABE aendert der Schalter nichts.

Teile, einzeln aufrufbar; jede Wiederholung wird als CSV gespeichert, ein Abbruch verliert nichts:
  repro   Referenz nachrechnen und gegen indicators_16.csv des Laufs v2.5 pruefen; Feldabdeckung je Topic
  split   Split-half-Zuverlaessigkeit: alle Topics zugleich zufaellig halbiert, Werte beider Haelften
  rare    Rarefaktion: alle Topics zugleich auf m Dokumente gezogen
  null    Label-Permutation unter den zugeordneten Dokumenten bei gleichen Topicgroessen
  report  Kennzahlen je Indikator aus den gespeicherten Wiederholungen

Aufruf (im Ordner v26_entwicklung):  python3 backtest_phase1.py repro
                                     python3 backtest_phase1.py split --r 0 10
                                     python3 backtest_phase1.py split --r 0 50 --budget 140   (setzt fort)
"""
import argparse
import contextlib
import io
import json
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HIER = Path(__file__).resolve().parent
V26 = HIER.parent
sys.path.insert(0, str(V26))
sys.path.insert(0, str(HIER))
import phase as phs  # noqa: E402  (Phasenschalter, Protokoll 3; Phase 1 wie bisher)
LAUF = phs.LAUF
REF = phs.REF
DATA = phs.DATA
OUT = phs.ordner("output_v26_entwicklung/backtest_phase1", "backtest")
JAHR_MIN, JAHR_MAX = phs.JAHR_MIN, phs.JAHR_MAX
KENNZAHLEN = f"backtest_{phs.P}_kennzahlen"

import config  # noqa: E402
config.DATA_PATH, config.OUTPUT_DIR, config.PHASE_YEAR_MIN, config.PHASE_YEAR_MAX = DATA, REF, JAHR_MIN, JAHR_MAX
import step02_indicators as s2  # noqa: E402
s2.DATA_PATH, s2.OUTPUT_DIR, s2.PHASE_YEAR_MIN, s2.PHASE_YEAR_MAX = DATA, REF, JAHR_MIN, JAHR_MAX

IND = ["keyword_volatility", "disciplinary_entropy", "semantic_incoherence",
       "relative_proportion_inv", "noise_ratio", "journal_specificity",
       "temporal_novelty", "terminological_instability", "review_absence",
       "author_concentration", "institutional_concentration", "geographic_concentration",
       "citation_concentration", "growth_rate", "citation_momentum", "field_breadth"]
CODE = dict(zip(IND, ["EO1", "EO2", "EO3", "PE1", "PE2", "PE3", "DS1", "DS2", "DS3",
                      "DI1", "DI2", "DI3", "DI4", "IP1", "IP2", "IP3"]))
FELD = {"EO1": "Author Keywords", "EO2": "WoS Categories", "EO3": "Abstract", "PE1": "Year", "PE2": "Abstract",
        "PE3": "Source title", "DS1": "Year", "DS2": "Author Keywords", "DS3": "Document Type", "DI1": "Authors",
        "DI2": "Affiliations", "DI3": "Addresses", "DI4": "Times Cited", "IP1": "Year", "IP2": "Times Cited",
        "IP3": "Keywords Plus"}
NAN_MOEGLICH = ("keyword_volatility", "terminological_instability", "citation_momentum")


def uhr(t0, text):
    print(f"[{time.time() - t0:7.1f} s] {text}", flush=True)


def lade():
    """Wie step02_indicators.run(): gleiche Spaltennamen, Filter und Zeilenfolge wie die Labels des Laufs."""
    df = pd.read_csv(phs.lies(DATA, "WoS-Export der Phase laden"))
    df = df.rename(columns={s: d for s, d in s2.CANON_COLUMNS.items() if s in df.columns})
    df["text"] = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")
    df["text_clean"] = df["text"].apply(
        lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", t.lower())).strip() if isinstance(t, str) else "")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[(df["Year"] >= JAHR_MIN) & (df["Year"] <= JAHR_MAX)].copy()
    df = df[df["text_clean"].str.split().str.len() >= 10].reset_index(drop=True)
    with open(phs.lies(REF / "model_results.pkl", "Labels und Einbettungen des berichteten Laufs"), "rb") as f:
        md = pickle.load(f)
    labels = np.asarray(md["labels"])
    emb, red = np.asarray(md["embeddings_sbert"]), np.asarray(md["embeddings_reduced"])
    assert len(labels) == len(df) == len(emb) == len(red)
    return df, labels, emb, red


def tem(df, labels):
    """Wie step01_topic_modeling.compute_tem_metrics (ohne dessen schwere Importe)."""
    d = df[["Year"]].copy()
    d["topic"] = labels
    d = d[d["topic"] >= 0]
    counts = d.groupby(["Year", "topic"]).size().unstack(fill_value=0)
    prop = counts.div(counts.sum(axis=1), axis=0)
    rows = []
    for tid in prop.columns:
        s = prop[tid].sort_index()
        nz = s[s > 0]
        if len(nz) >= 2:
            p_s, p_e = nz.iloc[0], nz.iloc[-1]
            n_y = nz.index[-1] - nz.index[0]
            g = (p_e / p_s) ** (1 / n_y) - 1 if n_y > 0 and p_s > 0 else 0.0
        else:
            g = 0.0
        rows.append({"topic": tid, "avg_proportion": s.mean(), "growth_rate": g})
    return pd.DataFrame(rows), prop


def indikatoren(df, labels, emb, red):
    """Wie compute_all_indicators; gibt zusaetzlich die Topics ohne bestimmbaren Wert zurueck."""
    tem_df, prop = tem(df, labels)
    prop.columns = prop.columns.astype(int)
    tids = sorted(set(labels[labels >= 0]))
    with contextlib.redirect_stdout(io.StringIO()):
        werte = {
            "keyword_volatility": s2.compute_keyword_volatility(df, tids, labels),
            "disciplinary_entropy": s2.compute_disciplinary_entropy(df, tids, labels),
            "semantic_incoherence": s2.compute_semantic_incoherence(df, tids, labels, emb),
            "relative_proportion_inv": s2.compute_relative_proportion(prop, tids),
            "noise_ratio": s2.compute_noise_ratio(labels, red, tids),
            "journal_specificity": s2.compute_journal_specificity(df, tids, labels),
            "temporal_novelty": s2.compute_temporal_novelty(df, tids, labels),
            "terminological_instability": s2.compute_terminological_instability(df, tids, labels),
            "review_absence": s2.compute_review_absence(df, tids, labels),
            "author_concentration": s2.compute_author_concentration(df, tids, labels),
            "institutional_concentration": s2.compute_institutional_concentration(df, tids, labels),
            "geographic_concentration": s2.compute_geographic_concentration(df, tids, labels),
            "citation_concentration": s2.compute_citation_concentration(df, tids, labels),
            "growth_rate": s2.compute_growth_rate(tem_df, tids),
            "citation_momentum": s2.compute_citation_momentum(df, tids, labels),
            "field_breadth": s2.compute_field_breadth(df, tids, labels),
        }
    ind = pd.DataFrame({k: [v.get(t, 0.5) for t in tids] for k, v in werte.items()}, index=pd.Index(tids, name="topic"))
    fehlt = ind[list(NAN_MOEGLICH)].isna()
    for col in NAN_MOEGLICH:
        if fehlt[col].any():
            med = ind[col].median()
            ind.loc[fehlt[col], col] = 0.5 if pd.isna(med) else med
    return ind[IND], fehlt


def teilkorpus(df, labels, emb, red, keep):
    keep = np.sort(np.asarray(keep))
    return df.iloc[keep].reset_index(drop=True), labels[keep], emb[keep], red[keep]


def speichere(ind, fehlt, name):
    OUT.mkdir(parents=True, exist_ok=True)
    x = ind.copy()
    for col in NAN_MOEGLICH:
        x[col + "__fehlt"] = fehlt[col].astype(int)
    phs.csv_atomar(x, OUT / f"{name}.csv")


def teil_repro(daten, t0):
    df, labels, emb, red = daten
    ind, fehlt = indikatoren(df, labels, emb, red)
    uhr(t0, "Indikatoren nachgerechnet")
    ref = pd.read_csv(phs.lies(REF / "indicators_16.csv", "Gegenprobe gegen die Indikatoren des berichteten Laufs"),
                      index_col=0)[IND]
    abw = (ind - ref.reindex(ind.index)).abs().max()
    speichere(ind, fehlt, "referenz")
    n = pd.Series(labels[labels >= 0]).value_counts().sort_index()
    abd = {}
    k1_je = {}
    tid_of = labels
    for code, feld in FELD.items():
        if feld not in df.columns:
            abd[code] = None
            continue
        voll = df[feld].notna() & (df[feld].astype(str).str.strip() != "")
        je = pd.Series(voll.values[tid_of >= 0]).groupby(tid_of[tid_of >= 0]).sum()
        k1_je[code] = je
        abd[code] = {"median_dok_mit_feld": float(je.median()), "topics_unter_10_dok": int((je < 10).sum()),
                     "anteil_dok_mit_feld": float(voll.values[tid_of >= 0].mean())}
    res = {"max_abweichung_je_indikator": {CODE[k]: float(v) for k, v in abw.items()},
           "nicht_bestimmbar": {CODE[k]: int(fehlt[k].sum()) for k in NAN_MOEGLICH},
           "n_topics": int(len(ind)), "n_dok": int(len(df)), "n_rauschen": int((labels < 0).sum()),
           "topicgroesse_min_median_max": [int(n.min()), float(n.median()), int(n.max())], "abdeckung": abd}
    (OUT / "repro.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    # K1 je Topic (Dokumente mit Feld), fuer die Zaehlung auf Phase 2 in optimismus_phase1.py (Protokoll 3, Punkt b)
    pd.DataFrame(k1_je).rename_axis("topic").to_csv(OUT / "k1_je_topic.csv")
    print(json.dumps(res, ensure_ascii=False))


def teil_split(daten, t0, r_von, r_bis, budget=None):
    df, labels, emb, red = daten
    rausch = np.where(labels < 0)[0]
    tids = sorted(set(labels[labels >= 0]))
    letzte = 0.0  # Dauer der letzten Wiederholung: keine neue beginnen, die das Budget sprengen wuerde
    for r in range(r_von, r_bis):
        if (OUT / f"split_r{r:02d}_B.csv").exists():
            continue
        if budget is not None and time.time() - t0 + letzte > budget:
            break
        t_r = time.time()
        rng = np.random.default_rng(1000 + r)
        a, b = [], []
        for t in tids:
            idx = rng.permutation(np.where(labels == t)[0])
            h = len(idx) // 2
            a.append(idx[:h]); b.append(idx[h:])
        for name, teile in (("A", a), ("B", b)):
            ind, fehlt = indikatoren(*teilkorpus(df, labels, emb, red, np.concatenate([rausch] + teile)))
            speichere(ind, fehlt, f"split_r{r:02d}_{name}")
        letzte = time.time() - t_r
        uhr(t0, f"Teilung {r} fertig")


def teil_rare(daten, t0, r_von, r_bis, m, budget=None):
    df, labels, emb, red = daten
    rausch = np.where(labels < 0)[0]
    tids = sorted(set(labels[labels >= 0]))
    letzte = 0.0
    for r in range(r_von, r_bis):
        name = f"rare_m{m}_r{r:02d}"
        if (OUT / f"{name}.csv").exists():
            continue
        if budget is not None and time.time() - t0 + letzte > budget:
            break
        t_r = time.time()
        rng = np.random.default_rng(2000 + r)
        teile = [rng.choice(np.where(labels == t)[0], size=min(m, int((labels == t).sum())), replace=False) for t in tids]
        ind, fehlt = indikatoren(*teilkorpus(df, labels, emb, red, np.concatenate([rausch] + teile)))
        speichere(ind, fehlt, name)
        letzte = time.time() - t_r
        uhr(t0, f"Rarefaktion m = {m}, Ziehung {r} fertig")


def teil_null(daten, t0, r_von, r_bis, budget=None):
    df, labels, emb, red = daten
    zug = np.where(labels >= 0)[0]
    letzte = 0.0
    for r in range(r_von, r_bis):
        name = f"null_r{r:02d}"
        if (OUT / f"{name}.csv").exists():
            continue
        if budget is not None and time.time() - t0 + letzte > budget:
            break
        t_r = time.time()
        rng = np.random.default_rng(3000 + r)
        lab = labels.copy()
        lab[zug] = rng.permutation(labels[zug])
        ind, fehlt = indikatoren(df, lab, emb, red)
        speichere(ind, fehlt, name)
        letzte = time.time() - t_r
        uhr(t0, f"Nullmodell, Permutation {r} fertig")


def lies(name):
    x = pd.read_csv(OUT / f"{name}.csv", index_col=0)
    werte = x[IND].copy()
    for col in NAN_MOEGLICH:
        werte.loc[x[col + "__fehlt"] == 1, col] = np.nan  # fuer Guetemasse nur bestimmbare Werte
    return werte


def rho(a, b):
    ok = a.notna() & b.notna()
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 10 else float("nan")


def teil_report(m):
    ref = lies("referenz")
    ref_voll = pd.read_csv(OUT / "referenz.csv", index_col=0)[IND]
    labels = pickle.load(open(phs.lies(REF / "model_results.pkl", "Labels fuer die Topicgroessen"), "rb"))["labels"]
    n = pd.Series(np.asarray(labels)[np.asarray(labels) >= 0]).value_counts().reindex(ref.index)
    logn = np.log(n.astype(float))
    splits = sorted({p.stem.split("_")[1] for p in OUT.glob("split_r*_B.csv")})
    rares = sorted(OUT.glob(f"rare_m{m}_r*.csv"))
    nulls = sorted(OUT.glob("null_r*.csv"))
    zeilen = []
    korr = ref_voll.rank().corr()
    for k in IND:
        z = {"Indikator": CODE[k], "Spalte": k}
        # K2 Split-half mit Spearman-Brown
        rs = [rho(lies(f"split_{s}_A")[k], lies(f"split_{s}_B")[k]) for s in splits]
        rs = [x for x in rs if not np.isnan(x)]
        if rs:
            r_h = float(np.mean(rs))
            z["K2_splithalf_r"] = round(r_h, 3)
            z["K2_spearman_brown"] = round(2 * r_h / (1 + r_h), 3) if r_h > -1 else float("nan")
            z["K2_spanne_r"] = [round(min(rs), 3), round(max(rs), 3)]
            z["K2_teilungen"] = len(rs)
        # K4 Groesse
        z["K4_rho_logn"] = round(rho(ref[k], logn), 3)
        if rares:
            rm = pd.concat([lies(p.stem)[k] for p in rares], axis=1).mean(axis=1)
            z[f"K4_rare_m{m}_rho_mit_voll"] = round(rho(rm, ref[k]), 3)
            z[f"K4_rare_m{m}_rho_logn"] = round(rho(rm, logn), 3)
            sd = ref[k].std()
            z[f"K4_rare_m{m}_verschiebung_sd"] = round(float((rm - ref[k]).mean() / sd), 3) if sd > 0 else float("nan")
        # K5 Nullmodell (Label-Permutation, gleiche Topicgroessen)
        if nulls:
            nm = pd.concat([lies(p.stem)[k] for p in nulls], axis=1)
            e0, s0 = nm.mean(axis=1), nm.std(axis=1)
            ok = ref[k].notna() & e0.notna()
            r2 = float(np.corrcoef(ref[k][ok], e0[ok])[0, 1] ** 2) if ok.sum() > 10 and e0[ok].std() > 0 else float("nan")
            z["K5_null_r2"] = round(r2, 3)
            sd = ref[k].std()
            z["K5_abstand_sd"] = round(float((ref[k] - e0).mean() / sd), 3) if sd > 0 else float("nan")
            ses = (ref[k] - e0) / s0.replace(0, np.nan)
            z["K5_anteil_ses_ueber_2"] = round(float((ses.abs() > 2).mean()), 3)
            z["K5_permutationen"] = len(nulls)
        # K6 Redundanz
        c = korr[k].drop(k).abs()
        z["K6_max_abs_rho"] = round(float(c.max()), 3)
        z["K6_partner"] = CODE[c.idxmax()]
        zeilen.append(z)
    tab = pd.DataFrame(zeilen)
    tab.to_csv(OUT / f"{KENNZAHLEN}.csv", index=False)
    (OUT / f"{KENNZAHLEN}.json").write_text(tab.to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab.drop(columns=["Spalte"]).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teil", choices=["repro", "split", "rare", "null", "report"])
    ap.add_argument("--r", nargs=2, type=int, default=[0, 10], metavar=("VON", "BIS"))
    ap.add_argument("--m", type=int, default=25)
    ap.add_argument("--budget", type=float, default=None, help="Zeitbudget in Sekunden (split, rare, null; setzt fort)")
    a = ap.parse_args()
    t0 = time.time()
    if a.teil == "report":
        teil_report(a.m)
        return
    daten = lade()
    uhr(t0, f"Daten geladen: {len(daten[0])} Dokumente, {len(set(daten[1][daten[1] >= 0]))} Topics")
    if a.teil == "repro":
        teil_repro(daten, t0)
    elif a.teil == "split":
        teil_split(daten, t0, *a.r, budget=a.budget)
    elif a.teil == "rare":
        teil_rare(daten, t0, *a.r, a.m, budget=a.budget)
    elif a.teil == "null":
        teil_null(daten, t0, *a.r, budget=a.budget)


if __name__ == "__main__":
    main()
