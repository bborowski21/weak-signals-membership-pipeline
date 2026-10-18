#!/usr/bin/env python3
"""V2.6: EO4 A1 aus den Zertifiziertheitswerten, Abgleich mit Bens Pruefblatt und K3 (Stabilitaet ueber Seeds).

Entwickelt auf Phase 1; auf Phase 2 nur in der Bestaetigung (3.4) mit dem Phasenschalter phase.py, dort ohne Pruefblatt,
Rarefaktion und K7. Fuer Phase 1 ohne V26_AUSGABE aendert der Schalter nichts.

Vorgaben laut Protokoll, Eintrag vom 17.10.2026 vor den Daten (Abschnitt Aenderungen, Punkte a bis c). Gelesen werden
nur abgeleitete Groessen: Modellwerte je Satz und Satzmerkmale aus output_v26_zertifiziertheit/phase1
(zertifiziertheit_mac.py), Labels und reduzierte Einbettungen weiterer Seeds aus output_v26_seeds/phase1
(seeds_phase1_mac.py) und die Zwischenstaende von backtest_kandidaten.py, dessen Funktionen unveraendert verwendet
werden. Vom Pruefblatt zaehlen nur die Spalten nr und bewertung_1_bis_6; Satztexte werden weder ausgegeben noch
gespeichert. Ausgegeben werden nur Guetemasse und Werte je Topic, keine Zugehoerigkeiten.

Aufruf (im Ordner v26_entwicklung):
  python3 auswertung_eo4_k3.py eo4          EO4 A1 und A1R je Dokument und Topic: Referenz, 180 Pseudokorpora, K7
  python3 auswertung_eo4_k3.py eo4report    K1, K2, K4 bis K7 fuer EO4 A1, A1R und R3
  python3 auswertung_eo4_k3.py vermerke     Verlagsvermerke markieren, A2 ohne Vermerke zaehlen (Eintrag vom 17.10., Bereinigung)
  python3 auswertung_eo4_k3.py blatt2       Pruefblatt 2: 100 Befundsaetze ziehen (Eintrag vom 17.10., Abweichung)
  python3 auswertung_eo4_k3.py pruefblatt   Modellwerte gegen Bens Bewertung von Blatt 2 (Hauptpruefung)
  python3 auswertung_eo4_k3.py pruefblatt --index pruefblatt_index.npy --blatt pruefblatt_saetze_claude.csv \\
          --aus pruefblatt_vergleich_claude.json                Vergleichslinie: Claudes Bewertung von Blatt 1
  python3 auswertung_eo4_k3.py k3           Indikatoren je Seed neu rechnen (mindestens ein Seed je Aufruf, setzt fort)
  python3 auswertung_eo4_k3.py k3report     Anbindung der Topics, Rangkorrelation je Indikator, Folgen fuer die Auswahl
Die Umgebungsvariablen V26_ZERT, V26_SEEDS und V26_AUSWERTUNG lenken Ein- und Ausgaben nur fuer Tests um.
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import backtest_kandidaten as bk  # noqa: E402  (Funktionen, Pseudokorpora und Zwischenstaende unveraendert)

bt = bk.bt
phs = bt.phs
ZERT = Path(os.environ.get("V26_ZERT", str(phs.ordner("output_v26_zertifiziertheit/phase1", "zertifiziertheit"))))
SEEDS = Path(os.environ.get("V26_SEEDS", str(phs.ordner("output_v26_seeds/phase1", "seeds"))))
AUS = Path(os.environ.get("V26_AUSWERTUNG", str(bk.OUT)))
OUT_EO4 = AUS / "eo4"
OUT_K3 = AUS / "k3"

# Vorab festgelegt (Protokoll, Eintrag vom 17.10.2026 vor den Daten); nach Ergebnissen nicht mehr veraendern.
EO4_CODES = ["EO4_A1", "EO4_A1R"]
R3 = "EO4_R3"
PRUEF_SCHWELLE = 0.40
BOOT, BOOT_SEED = 2000, 7
K3_SEEDS = (0, 1, 7, 2026)
K3_JACCARD = 0.75
K3_SCHWELLE = 0.75
K3_LAUFSEED = 0
K7_SD, K7_ANTEIL = 0.25, 0.8
HF_SHA16 = "c8b62d7f768ca554"  # pytorch_model.bin laut Hugging Face (pedropei/sentence-level-certainty)
BLATT2_N, BLATT2_SEED = 100, 4041  # Pruefblatt 2 (Eintrag vom 17.10., Abweichung): Befundsaetze ohne Blatt 1
# Verlagsvermerke (Eintrag vom 17.10., Bereinigung): kein Text der Autoren; fuer EO4 vor der Berechnung entfernt
VERMERK_ANFANG = re.compile(r"^\s*(\(c\)|\u00a9|copyright\b)", re.I)
VERMERK_TEXT = re.compile(r"\ball rights reserved\b|\bpublished by\b.*\b(?:elsevier|wiley|springer|press|society|institute|"
                          r"publishing)\b|\belsevier (?:b\.?\s?v\.?|ltd|inc|science|masson|gmbh)\b|\bopen access article\b|"
                          r"\bcreative commons\b", re.I)
VERMERK_MAX_WOERTER = 20
EO4_B = ["EO4_A1B", "EO4_A1RB", "EO4_A2B"]
R3B = "EO4_R3B"


def ist_vermerk(s):
    return bool(VERMERK_ANFANG.search(s)) or (len(s.split()) <= VERMERK_MAX_WOERTER and bool(VERMERK_TEXT.search(s)))
# Bens Entscheidungen vom 17.10. (E2, E7) wie in backtest_kandidaten.teil_report
E7_RUECKFALL = {"IP2_A4": "IP2_A5"}


def uhr(t0, text):
    bk.uhr(t0, text)


def runde(x, k=6):
    return None if x is None or x != x else round(float(x), k)


# ---------------------------------------------------------------- EO4 A1 je Dokument und Topic

def eo4_dokumente(n_dok):
    """EO4 A1 und A1R je Dokument aus werte.npy und saetze_meta.npz (Reihenfolge wie backtest_phase1.lade)."""
    m = np.load(ZERT / "saetze_meta.npz")
    w = np.load(ZERT / "werte.npy").astype(np.float64)
    doc, nr, anz = m["doc"].astype(np.int64), m["nr"].astype(np.int64), m["anz"].astype(np.int64)
    bef = m["befund"].astype(bool)
    if len(w) != len(doc):
        raise SystemExit(f"werte.npy ({len(w)}) passt nicht zu saetze_meta.npz ({len(doc)})")
    if doc.min() < 0 or doc.max() >= n_dok:
        raise SystemExit("Dokumentindex ausserhalb des Korpus")
    c_all = np.bincount(doc, minlength=n_dok).astype(float)
    if not np.array_equal(anz, c_all[doc].astype(np.int64)):
        raise SystemExit("Satzzahl je Abstract passt nicht zu den Saetzen")
    letzt = nr == anz - 1
    if not np.array_equal(np.bincount(doc[letzt], minlength=n_dok), (c_all > 0).astype(np.int64)):
        raise SystemExit("letzter Satz je Abstract nicht eindeutig")
    s_all = np.bincount(doc, weights=w, minlength=n_dok)
    c_b = np.bincount(doc, weights=bef.astype(float), minlength=n_dok)
    s_b = np.bincount(doc, weights=w * bef, minlength=n_dok)
    s_l = np.bincount(doc[letzt], weights=w[letzt], minlength=n_dok)
    hat = c_all > 0
    a1 = np.full(n_dok, np.nan)
    a1r = np.full(n_dok, np.nan)
    a1[hat] = -np.where(c_b[hat] > 0, s_b[hat] / np.maximum(c_b[hat], 1.0), s_l[hat])
    a1r[hat] = -s_all[hat] / c_all[hat]
    info = {"saetze": int(len(w)), "dokumente_mit_satz": int(hat.sum()), "dokumente": int(n_dok),
            "anteil_befundsaetze": runde(bef.mean()), "anteil_dok_mit_befundsatz": runde((c_b[hat] > 0).mean()),
            "werte_anteil_1_bis_6": runde(((w >= 1) & (w <= 6)).mean()),
            "werte_quantile_1_10_50_90_99": [runde(q, 4) for q in np.quantile(w, [.01, .1, .5, .9, .99])],
            "rho_a1_a1r_dokumente": runde(spearmanr(a1[hat], a1r[hat])[0])}
    x = {"EO4_A1": a1, "EO4_A1R": a1r}
    f_v = ZERT / "saetze_vermerk.npy"
    if f_v.exists():  # bereinigte Fassungen (Eintrag vom 17.10., Bereinigung)
        k = ~np.load(f_v).astype(bool)
        if len(k) != len(w):
            raise SystemExit("saetze_vermerk.npy passt nicht zu werte.npy")
        dk, nk, wk, bk_ = doc[k], nr[k], w[k], bef[k]
        c_k = np.bincount(dk, minlength=n_dok).astype(float)
        s_k = np.bincount(dk, weights=wk, minlength=n_dok)
        cb_k = np.bincount(dk, weights=bk_.astype(float), minlength=n_dok)
        sb_k = np.bincount(dk, weights=wk * bk_, minlength=n_dok)
        o = np.lexsort((nk, dk))
        do, wo = dk[o], wk[o]
        ende = np.r_[np.where(np.diff(do) != 0)[0], len(do) - 1]
        s_lk = np.zeros(n_dok)
        s_lk[do[ende]] = wo[ende]
        hk = c_k > 0
        a1b, a1rb = np.full(n_dok, np.nan), np.full(n_dok, np.nan)
        a1b[hk] = -np.where(cb_k[hk] > 0, sb_k[hk] / np.maximum(cb_k[hk], 1.0), s_lk[hk])
        a1rb[hk] = -s_k[hk] / c_k[hk]
        x.update({"EO4_A1B": a1b, "EO4_A1RB": a1rb})
        info["vermerke"] = {"saetze": int((~k).sum()), "dokumente_ohne_satz_nach_bereinigung": int((~hk).sum()),
                            "rueckfall_vorher_vermerk": int(((c_b == 0) & hat & np.isin(np.arange(n_dok), doc[letzt & ~k])).sum()),
                            "anteil_dok_mit_befundsatz_bereinigt": runde((cb_k[hk] > 0).mean())}
    f_a2 = OUT_EO4 / "a2_bereinigt.npz"
    if f_a2.exists():
        a2 = np.load(f_a2)
        wa = a2["woerter"].astype(float)
        x["EO4_A2B"] = np.where(wa >= bk.MIN_ABSTRACT, 1000 * a2["hedge"] / np.maximum(wa, 1.0), np.nan)
    return x, info


def topicwerte(x, keep, lab):
    tids = sorted(set(lab[lab >= 0].tolist()))
    out = {}
    for code, xd in x.items():
        v = xd[keep]
        ok = (lab >= 0) & ~np.isnan(v)
        out[code] = pd.Series(v[ok]).groupby(lab[ok]).mean().reindex(tids).values
    return pd.DataFrame(out, index=pd.Index(tids, name="topic"))


def z(s):
    s = s.astype(float)
    return (s - s.mean()) / s.std(ddof=0)


def mit_r3(d):
    d = d.copy()
    d[R3] = (z(d["EO4_A1"]) + z(d["EO4_A2"])) / 2
    if "EO4_A1B" in d.columns and "EO4_A2B" in d.columns:
        d[R3B] = (z(d["EO4_A1B"]) + z(d["EO4_A2B"])) / 2
    return d


_GELESEN = {}


def lies_mit_eo4(name):
    """Kandidatenwerte eines (Pseudo-)Korpus aus backtest_kandidaten, ergaenzt um EO4 A1, A1R und R3."""
    if name not in _GELESEN:
        _GELESEN[name] = mit_r3(bk.lies_k(name).join(pd.read_csv(OUT_EO4 / f"{name}.csv", index_col=0)))
    return _GELESEN[name]


def teil_eo4(t0):
    labels = bk.lade_labels()
    x, info = eo4_dokumente(len(labels))
    lauf = ZERT / f"zertifiziertheit_{phs.P}.json"
    if lauf.exists():
        z_l = json.loads(lauf.read_text(encoding="utf-8"))
        sha = z_l.get("modell_sha16", {}).get("pytorch_model.bin")
        info["modelllauf"] = {"saetze": z_l.get("saetze"), "abstracts": z_l.get("abstracts"),
                              "sekunden": z_l.get("sekunden"), "pytorch_model_sha16": sha,
                              "sha_wie_hugging_face": sha == HF_SHA16, "versionen": z_l.get("versionen")}
        if z_l.get("saetze") != info["saetze"]:
            raise SystemExit("Satzzahl im Modelllauf weicht von werte.npy ab")
    zug = labels >= 0
    info["k1"] = {}
    for c, xd in x.items():
        je = pd.Series((~np.isnan(xd[zug])).astype(int)).groupby(labels[zug]).sum()
        info["k1"][c] = {"topics_unter_10": int((je < 10).sum()), "anteil_topics_ab_10": runde((je >= 10).mean())}
    info["k1_topics_unter_10"] = info["k1"]["EO4_A1"]["topics_unter_10"]
    info["k1_anteil_topics_ab_10"] = info["k1"]["EO4_A1"]["anteil_topics_ab_10"]
    # Gegenprobe: A2 aus den Dokumentwerten von 1b ergibt die Topicwerte von backtest_kandidaten
    f = np.load(bk.CACHE / "felder.npz")
    wa = f["woerter_abstract"].astype(float)
    a2o = np.where(wa >= bk.MIN_ABSTRACT, 1000 * f["hedge"] / np.maximum(wa, 1.0), np.nan)
    a2_ref = topicwerte({"a2": a2o}, np.arange(len(labels)), labels)["a2"]
    info["gegenprobe_a2_max_abweichung"] = float((a2_ref - bk.lies_k("referenz")["EO4_A2"]).abs().max())
    OUT_EO4.mkdir(parents=True, exist_ok=True)
    topicwerte(x, np.arange(len(labels)), labels).to_csv(OUT_EO4 / "referenz.csv")
    n = 0
    for name, keep, lab, _ in bk.pseudokorpora():
        phs.csv_atomar(topicwerte(x, keep, lab), OUT_EO4 / f"{name}.csv")
        n += 1
    for seed in (bk.K7_SEEDS if phs.K7 else ()):  # dieselben Mischungen wie backtest_kandidaten.teil_k7
        lab = labels.copy()
        for a, b in pd.read_csv(bk.OUT / f"k7_s{seed}_paare.csv").itertuples(index=False):
            lab[labels == b] = a
        topicwerte(x, np.arange(len(labels)), lab).to_csv(OUT_EO4 / f"k7_s{seed}.csv")
    info["pseudokorpora"] = n
    (OUT_EO4 / "eo4_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    uhr(t0, f"EO4 A1 und A1R: Referenz, {n} Pseudokorpora und {len(bk.K7_SEEDS) if phs.K7 else 0} K7-Aufteilungen geschrieben")
    print(json.dumps(info, ensure_ascii=False))


def ci_logn(x, logn):
    rng = np.random.default_rng(BOOT_SEED)
    ok = x.notna() & logn.notna()
    xv, lv = x[ok].values, logn[ok].values
    bs = [spearmanr(xv[i], lv[i])[0] for i in (rng.integers(0, len(xv), len(xv)) for _ in range(BOOT))]
    return [runde(q) for q in np.nanquantile(bs, [0.025, 0.975])]


def teil_eo4report():
    info = json.loads((OUT_EO4 / "eo4_info.json").read_text(encoding="utf-8"))
    ref = lies_mit_eo4("referenz")
    labels = bk.lade_labels()
    n = pd.Series(labels[labels >= 0]).value_counts().reindex(ref.index)
    logn = np.log(n.astype(float))
    splits = sorted({p.stem.split("_")[1] for p in OUT_EO4.glob("split_r*_B.csv")})
    rares = sorted(p.stem for p in OUT_EO4.glob(f"rare_m{bk.M_RARE}_r*.csv"))
    nulls = sorted(p.stem for p in OUT_EO4.glob("null_r*.csv"))
    if (len(splits), len(rares), len(nulls)) != (50, phs.RARE_N, 50):
        raise SystemExit(f"Pseudokorpora unvollstaendig: {len(splits)} Teilungen, {len(rares)} Rarefaktion, {len(nulls)} Null")
    kand = pd.read_csv(bk.OUT / "kandidaten_kennzahlen.csv").set_index("Code")
    alt_rang, rang = bt.lies("referenz").rank(), ref.rank()
    andere_k = [c for c in bk.KCODES if bk.PLATZ[c] not in ("EO4", "DIAG")]
    codes = EO4_CODES + [R3] + [c for c in EO4_B + [R3B] if c in ref.columns]
    k1 = info["k1"]

    def k1_ok(c):
        return k1[c]["anteil_topics_ab_10"] >= 0.9

    zeilen = []
    for code in codes:
        zz = {"Code": code, "Platz": "EO4"}
        zz.update(bk.kennzahlen(ref, lies_mit_eo4, splits, rares, nulls, logn, code))
        zz["K4_ci95"] = ci_logn(ref[code], logn)
        if code == R3:
            zz["K1_topics_unter_10"] = k1["EO4_A1"]["topics_unter_10"]
            zz["K1_ok"] = bool(k1_ok("EO4_A1") and kand.loc["EO4_A2", "K1_ok"])
        elif code == R3B:
            zz["K1_topics_unter_10"] = max(k1["EO4_A1B"]["topics_unter_10"], k1["EO4_A2B"]["topics_unter_10"])
            zz["K1_ok"] = bool(k1_ok("EO4_A1B") and k1_ok("EO4_A2B"))
        else:
            zz["K1_topics_unter_10"] = k1[code]["topics_unter_10"]
            zz["K1_ok"] = bool(k1_ok(code))
        c_alt = alt_rang.corrwith(rang[code]).abs()
        c_k = rang[andere_k].corrwith(rang[code]).abs()
        zz["K6_max_abs_rho"] = runde(max(c_alt.max(), c_k.max()))
        zz["K6_partner"] = bt.CODE[c_alt.idxmax()] if c_alt.max() >= c_k.max() else c_k.idxmax()
        zz.update({f"S_{k}": v for k, v in bk.urteil(zz, zz["K1_ok"]).items()})
        zeilen.append(zz)
    tab = pd.DataFrame(zeilen)
    # K7: Mittelwertmass; R3 mit der Skalierung der Referenz, damit Mischung und Teile vergleichbar bleiben
    ref_roh = bk.lies_k("referenz").join(pd.read_csv(OUT_EO4 / "referenz.csv", index_col=0))
    teile = [c for c in ("EO4_A1", "EO4_A2", "EO4_A1B", "EO4_A2B") if c in ref_roh.columns]
    mu = {c: float(ref_roh[c].mean()) for c in teile}
    sg = {c: float(ref_roh[c].std(ddof=0)) for c in teile}
    sd = {c: float(ref[c].std()) for c in codes}
    k7 = []
    for seed in (bk.K7_SEEDS if phs.K7 else ()):
        mix = pd.read_csv(OUT_EO4 / f"k7_s{seed}.csv", index_col=0)
        a2 = bk.lies_k(f"k7_s{seed}_kand")["EO4_A2"].reindex(mix.index)
        mix[R3] = ((mix["EO4_A1"] - mu["EO4_A1"]) / sg["EO4_A1"] + (a2 - mu["EO4_A2"]) / sg["EO4_A2"]) / 2
        if R3B in codes:
            mix[R3B] = ((mix["EO4_A1B"] - mu["EO4_A1B"]) / sg["EO4_A1B"] + (mix["EO4_A2B"] - mu["EO4_A2B"]) / sg["EO4_A2B"]) / 2
        for a, b in pd.read_csv(bk.OUT / f"k7_s{seed}_paare.csv").itertuples(index=False):
            na, nb = int(n[a]), int(n[b])
            for code in codes:
                va, vb, vm = ref.loc[a, code], ref.loc[b, code], mix.loc[a, code]
                if np.isnan(va) or np.isnan(vb) or np.isnan(vm):
                    continue
                gm = (na * va + nb * vb) / (na + nb)
                k7.append({"seed": seed, "code": code, "abweichung_sd": (vm - gm) / sd[code]})
    k7 = pd.DataFrame(k7, columns=["seed", "code", "abweichung_sd"])
    k7_erg = {}
    for code, g in k7.groupby("code"):
        anteil = float((g["abweichung_sd"].abs() <= K7_SD).mean())
        k7_erg[code] = {"mischungen": int(len(g)), "anteil_erfuellt": runde(anteil), "erfuellt": anteil >= K7_ANTEIL,
                        "median_abweichung_sd": runde(g["abweichung_sd"].median())}
    if phs.K7:
        tab["K7_anteil"] = tab["Code"].map(lambda c: k7_erg[c]["anteil_erfuellt"])
        tab["S_K7"] = tab["Code"].map(lambda c: "erfuellt" if k7_erg[c]["erfuellt"] else "verfehlt")
    innen = {"A1_A2": runde(bk.rho(ref["EO4_A1"], ref["EO4_A2"])), "A1_A1R": runde(bk.rho(ref["EO4_A1"], ref["EO4_A1R"])),
             "R3_A1": runde(bk.rho(ref[R3], ref["EO4_A1"])), "R3_A2": runde(bk.rho(ref[R3], ref["EO4_A2"])),
             "A1_EO5_A1": runde(bk.rho(ref["EO4_A1"], ref["EO5_A1"]))}
    if R3B in codes:
        innen.update({"A1B_A2B": runde(bk.rho(ref["EO4_A1B"], ref["EO4_A2B"])),
                      "A1_A1B": runde(bk.rho(ref["EO4_A1"], ref["EO4_A1B"])),
                      "A2_A2B": runde(bk.rho(ref["EO4_A2"], ref["EO4_A2B"])),
                      "R3_R3B": runde(bk.rho(ref[R3], ref[R3B]))})
    AUS.mkdir(parents=True, exist_ok=True)
    tab.to_csv(AUS / "eo4_kennzahlen.csv", index=False)
    erg = {"kennzahlen": json.loads(tab.to_json(orient="records", force_ascii=False)), "k7": k7_erg,
           "korrelationen_im_platz": innen, "info": info,
           "a2_aus_1b": {k: (v.item() if hasattr(v, "item") else v) for k, v in kand.loc["EO4_A2"].items()
                         if k.startswith(("K", "S_"))}}
    (AUS / "eo4_ergebnis.json").write_text(json.dumps(erg, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    spalten = ["Code", "K1_topics_unter_10", "K2_spearman_brown", "K4_rho_logn", "K4_ci95", "K4_rare_rho_mit_voll",
               "K4_rare_rho_logn", "K5_null_r2", "K5_anteil_ses_ueber_2", "K6_max_abs_rho", "K6_partner", "K7_anteil",
               "S_alle"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(tab[[c for c in spalten if c in tab.columns]].to_string(index=False))
    print(json.dumps({"k7": k7_erg, "korrelationen_im_platz": innen}, ensure_ascii=False))


# ---------------------------------------------------------------- Pruefblatt

def teil_vermerke():
    """Markiert Verlagsvermerke je Satz (Reihenfolge wie saetze_meta.npz) und zaehlt Hedges und Woerter der Abstracts
    ohne Vermerke fuer A2b. Ausgegeben und gespeichert werden nur Zahlen."""
    import zertifiziertheit_mac as zm
    m = np.load(ZERT / "saetze_meta.npz")
    df, _, _, _ = bt.lade()
    f = np.load(bk.CACHE / "felder.npz")
    flag, h, b, n, mit, abweichend = [], [], [], [], [], 0
    for i, ab in enumerate(df["Abstract"].fillna("").astype(str)):
        alle = zm.saetze(ab)
        v_alle = [ist_vermerk(s) for s in alle]
        flag += [v for s, v in zip(alle, v_alle) if len(s.split()) >= zm.MIN_WOERTER]
        mit.append(any(v_alle))
        if mit[-1]:
            hb = bk.hedges(" ".join(s for s, v in zip(alle, v_alle) if not v))
        else:  # ohne Vermerk gelten die Zaehlungen aus 1b; Gegenprobe der wieder zusammengesetzten Saetze
            hb = (int(f["hedge"][i]), int(f["booster"][i]), int(f["woerter_abstract"][i]))
            abweichend += tuple(bk.hedges(" ".join(alle))) != hb
        h.append(hb[0])
        b.append(hb[1])
        n.append(hb[2])
    flag, mit, h, n = np.array(flag, dtype=bool), np.array(mit), np.array(h), np.array(n)
    if len(flag) != len(m["doc"]):
        raise SystemExit("Vermerk-Markierung passt nicht zu saetze_meta.npz")
    np.save(ZERT / "saetze_vermerk.npy", flag)
    OUT_EO4.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_EO4 / "a2_bereinigt.npz", hedge=h, booster=np.array(b), woerter=n)
    wo = f["woerter_abstract"]
    print(json.dumps({"saetze_ab_4_woertern_vermerk": int(flag.sum()), "anteil": runde(flag.mean()),
                      "dokumente_mit_vermerk": int(mit.sum()),
                      "entfernte_woerter": int((wo[mit] - n[mit]).sum()), "entfernte_hedges": int((f["hedge"][mit] - h[mit]).sum()),
                      "neu_unter_20_woertern": int(((wo >= bk.MIN_ABSTRACT) & (n < bk.MIN_ABSTRACT)).sum()),
                      "gegenprobe_ohne_vermerk_abweichend": int(abweichend),
                      "vermerke_in_blatt1": int(flag[np.load(ZERT / "pruefblatt_index.npy")].sum())
                      if (ZERT / "pruefblatt_index.npy").exists() else None,
                      "vermerke_in_blatt2": int(flag[np.load(ZERT / "pruefblatt2_index.npy")].sum())
                      if (ZERT / "pruefblatt2_index.npy").exists() else None}, ensure_ascii=False))


def teil_blatt2():
    """Zieht Pruefblatt 2 aus den Befundsaetzen ohne Blatt 1. Satzzerlegung wie zertifiziertheit_mac.py, gegen
    saetze_meta.npz geprueft; ausgegeben werden nur Zahlen, die Texte gehen nur in die Datei im Ordner phase1."""
    import zertifiziertheit_mac as zm
    ziel_csv, ziel_idx = ZERT / "pruefblatt2_befund.csv", ZERT / "pruefblatt2_index.npy"
    if ziel_csv.exists() or ziel_idx.exists():
        raise SystemExit("Pruefblatt 2 existiert schon; es wird nicht neu gezogen")
    m = np.load(ZERT / "saetze_meta.npz")
    df, _, _, _ = bt.lade()
    texte, spalten = [], {"doc": [], "nr": [], "woerter": [], "befund": []}
    for i, ab in enumerate(df["Abstract"].fillna("").astype(str)):
        ss = [s for s in zm.saetze(ab) if len(s.split()) >= zm.MIN_WOERTER]
        for j, s in enumerate(ss):
            texte.append(s)
            spalten["doc"].append(i)
            spalten["nr"].append(j)
            spalten["woerter"].append(len(s.split()))
            spalten["befund"].append(bool(zm.BEFUND.search(s)))
    for k, v in spalten.items():
        if not np.array_equal(np.asarray(v), m[k].astype(np.asarray(v).dtype)):
            raise SystemExit(f"Satzzerlegung weicht vom Mac-Lauf ab ({k})")
    blatt1 = set(np.load(ZERT / "pruefblatt_index.npy").tolist())
    pool = np.array([i for i in np.where(m["befund"].astype(bool))[0] if i not in blatt1])
    wahl = np.sort(np.random.default_rng(BLATT2_SEED).choice(pool, size=BLATT2_N, replace=False))
    with open(ziel_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nr", "satz", "bewertung_1_bis_6"])
        for k, i in enumerate(wahl):
            w.writerow([k + 1, texte[i], ""])
    np.save(ziel_idx, wahl)
    print(json.dumps({"saetze": len(texte), "befundsaetze_ohne_blatt1": int(len(pool)), "gezogen": int(len(wahl)),
                      "abstracts": int(len(set(m["doc"][wahl].tolist()))),
                      "mittlere_woerter": runde(m["woerter"][wahl].mean(), 1), "seed": BLATT2_SEED,
                      "ueberschneidung_mit_blatt1": int(len(blatt1 & set(wahl.tolist())))}, ensure_ascii=False))


def lies_bewertung(datei):
    """Nur nr und bewertung_1_bis_6; Trennzeichen und Zeilenenden wie aus Numbers oder einem Editor."""
    roh = (ZERT / datei).read_bytes()
    try:
        text = roh.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = roh.decode("mac_roman")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    kopf = text.split("\n", 1)[0]
    trenn = ";" if kopf.count(";") >= 2 else ("\t" if kopf.count("\t") >= 2 else ",")
    zeilen = list(csv.reader(io.StringIO(text), delimiter=trenn))
    namen = [h.strip().lower() for h in zeilen[0]]
    i_nr, i_b = namen.index("nr"), namen.index("bewertung_1_bis_6")
    nr, wert, leer, ungueltig = [], [], 0, 0
    for zl in zeilen[1:]:
        if not any(f.strip() for f in zl):
            continue
        k = int(float(zl[i_nr].strip().replace(",", ".")))
        s = zl[i_b].strip().replace(",", ".") if i_b < len(zl) else ""
        if not s:
            leer += 1
            continue
        try:
            v = float(s)
        except ValueError:
            ungueltig += 1
            continue
        if v != int(v) or not 1 <= v <= 6:
            ungueltig += 1
            continue
        nr.append(k)
        wert.append(int(v))
    return np.array(nr, dtype=np.int64), np.array(wert, dtype=float), leer, ungueltig, len(zeilen) - 1, trenn


def boot(x, y, f):
    rng = np.random.default_rng(BOOT_SEED)
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            bs.append(f(x[i], y[i]))
    return [runde(q) for q in np.quantile(bs, [0.025, 0.975])]


def teil_pruefblatt(index_datei, blatt_datei, aus_datei):
    wahl = np.load(ZERT / index_datei)
    w = np.load(ZERT / "werte.npy").astype(np.float64)
    bef_alle = np.load(ZERT / "saetze_meta.npz")["befund"].astype(bool)
    nr, wert, leer, ungueltig, n_zeilen, trenn = lies_bewertung(blatt_datei)
    if len(set(nr.tolist())) != len(nr) or nr.min(initial=1) < 1 or nr.max(initial=1) > len(wahl):
        raise SystemExit(f"Spalte nr im Pruefblatt ist nicht eindeutig oder ausserhalb von 1 bis {len(wahl)}")
    if len(wert) < 30:
        raise SystemExit(f"Nur {len(wert)} gueltige Bewertungen ({leer} leer, {ungueltig} ungueltig); Pruefblatt pruefen")
    idx = wahl[nr - 1]
    mod, bef = w[idx], bef_alle[idx]
    sp = lambda a, b: float(spearmanr(a, b)[0])  # noqa: E731
    pe = lambda a, b: float(pearsonr(a, b)[0])  # noqa: E731
    rho_s = sp(mod, wert)
    erg = {"blatt": blatt_datei, "index": index_datei, "zeilen": int(n_zeilen), "trennzeichen": trenn,
           "bewertet": int(len(wert)), "leer": int(leer), "ungueltig": int(ungueltig),
           "verteilung_bewertung": {str(k): int((wert == k).sum()) for k in range(1, 7)},
           "anteil_haeufigste_stufe": runde(max((wert == k).mean() for k in range(1, 7))),
           "spearman_hoechstens_erreichbar": runde(sp(np.sort(mod), np.sort(wert))),
           "spearman": runde(rho_s), "spearman_ci95": boot(mod, wert, sp),
           "pearson": runde(pe(mod, wert)), "pearson_ci95": boot(mod, wert, pe),
           "befundsaetze": {"n": int(bef.sum()),
                            "spearman": runde(sp(mod[bef], wert[bef])) if bef.sum() > 10 else None,
                            "spearman_ci95": boot(mod[bef], wert[bef], sp) if bef.sum() > 10 else None},
           "schwelle": PRUEF_SCHWELLE, "bestaetigt": bool(rho_s >= PRUEF_SCHWELLE)}
    AUS.mkdir(parents=True, exist_ok=True)
    (AUS / aus_datei).write_text(json.dumps(erg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(erg, ensure_ascii=False))


# ---------------------------------------------------------------- K3

def seed_dateien(s):
    if s == 42:
        return SEEDS / "labels_seed42_kontrolle.npy", SEEDS / "reduziert_seed42_kontrolle.npy"
    return SEEDS / f"labels_seed{s}.npy", SEEDS / f"reduziert_seed{s}.npy"


def ref_gleich():
    return bool(json.loads((SEEDS / f"seeds_{phs.P}.json").read_text(encoding="utf-8"))["kontrolle_seed42"]
                ["labels_gleich_referenz"])


def teil_k3(t0, budget):
    noetig = list(K3_SEEDS) + ([] if ref_gleich() else [42])
    offen = [s for s in noetig if not all((OUT_K3 / f"seed{s}_{x}.csv").exists() for x in ("kand", "best"))]
    if not offen:
        print("STAND: alle Seeds gerechnet", flush=True)
        return
    for s in offen:
        if not all(f.exists() for f in seed_dateien(s)):
            raise SystemExit(f"Dateien fuer Seed {s} fehlen in {SEEDS}")
    K = bk.Korpus(t0)
    red_ref = K.red
    OUT_K3.mkdir(parents=True, exist_ok=True)
    fertig = 0
    letzte = 0.0  # Dauer des letzten Seeds: keinen mehr beginnen, der das Budget sprengen wuerde
    for s in offen:
        if fertig and time.time() - t0 + letzte > budget:
            break
        t_s = time.time()
        f_lab, f_red = seed_dateien(s)
        lab, red = np.load(f_lab).astype(np.int64), np.load(f_red)
        if not len(lab) == len(red) == len(K.labels):
            raise SystemExit(f"Seed {s}: Laenge passt nicht zum Korpus")
        K.red = np.asarray(red, dtype=red_ref.dtype)
        try:
            kand, _, _ = bk.kandidaten(K, np.arange(len(lab)), lab, K3_LAUFSEED)
            best, fehlt = bt.indikatoren(K.df, lab, K.E, K.red)
        finally:
            K.red = red_ref
        for col in bt.NAN_MOEGLICH:
            best.loc[fehlt[col], col] = np.nan
        phs.csv_atomar(kand, OUT_K3 / f"seed{s}_kand.csv")
        phs.csv_atomar(best, OUT_K3 / f"seed{s}_best.csv")
        letzte = time.time() - t_s
        fertig += 1
        uhr(t0, f"Seed {s} fertig: {len(kand)} Topics")
    rest = [s for s in noetig if not (OUT_K3 / f"seed{s}_best.csv").exists()]
    print(f"STAND: {len(noetig) - len(rest)} von {len(noetig)} Seeds gerechnet, offen {rest}", flush=True)


def anbindung(ref_lab, lab):
    """Je Referenztopic das Seed-Topic mit groesstem Jaccard-Index der Dokumentmengen; ab 0,75 eindeutig."""
    n_i = pd.Series(ref_lab[ref_lab >= 0]).value_counts()
    n_j = pd.Series(lab[lab >= 0]).value_counts()
    beide = (ref_lab >= 0) & (lab >= 0)
    ct = pd.crosstab(ref_lab[beide], lab[beide])
    nij = ct.values.astype(float)
    J = nij / (n_i.reindex(ct.index).values[:, None] + n_j.reindex(ct.columns).values[None, :] - nij)
    j_arg = J.argmax(axis=1)
    paare = pd.DataFrame({"ref": ct.index.values, "seed": ct.columns.values[j_arg], "jaccard": J.max(axis=1)})
    paare = paare[paare["jaccard"] >= K3_JACCARD].reset_index(drop=True)
    if not paare["seed"].is_unique:
        raise SystemExit("Anbindung nicht eindeutig")
    return paare, n_i, n_j


def tabellen_fuer(lab, quelle, x_eo4):
    """Kandidaten, bestehende Indikatoren und EO4 (mit R3) fuer einen Lauf; quelle 'referenz' oder 'seed<s>'."""
    if quelle == "referenz":
        kand, best = bk.lies_k("referenz"), bt.lies("referenz")
    else:
        kand = pd.read_csv(OUT_K3 / f"{quelle}_kand.csv", index_col=0)
        best = pd.read_csv(OUT_K3 / f"{quelle}_best.csv", index_col=0)
    if x_eo4 is not None:
        kand = mit_r3(kand.join(topicwerte(x_eo4, np.arange(len(lab)), lab)))
    return kand, best


def teil_k3report():
    ref_ist_bericht = ref_gleich()
    labels = bk.lade_labels()
    ref_lab = labels if ref_ist_bericht else np.load(seed_dateien(42)[0]).astype(np.int64)
    x_eo4 = eo4_dokumente(len(labels))[0] if (ZERT / "werte.npy").exists() else None
    ref_kand, ref_best = tabellen_fuer(ref_lab, "referenz" if ref_ist_bericht else "seed42", x_eo4)
    codes_k = [c for c in ref_kand.columns if c != "DIAG_A1"]
    codes_b = list(ref_best.columns)
    OUT_K3.mkdir(parents=True, exist_ok=True)
    je_seed, rhos = {}, {c: {} for c in codes_k + codes_b}
    zug_ref = int((ref_lab >= 0).sum())
    for s in K3_SEEDS:
        lab = np.load(seed_dateien(s)[0]).astype(np.int64)
        paare, n_i, n_j = anbindung(ref_lab, lab)
        paare.to_csv(OUT_K3 / f"anbindung_seed{s}.csv", index=False)
        kand, best = tabellen_fuer(lab, f"seed{s}", x_eo4)
        je_seed[str(s)] = {"topics_seed": int(len(n_j)), "rauschen_seed": int((lab < 0).sum()),
                           "angebunden": int(len(paare)), "anteil_ref_topics": runde(len(paare) / len(n_i)),
                           "anteil_ref_dokumente": runde(n_i.reindex(paare["ref"]).sum() / zug_ref),
                           "median_jaccard": runde(paare["jaccard"].median())}
        for c in codes_k + codes_b:
            r_tab, s_tab = (ref_kand, kand) if c in codes_k else (ref_best, best)
            a = r_tab[c].reindex(paare["ref"]).values
            b = s_tab[c].reindex(paare["seed"]).values
            ok = ~np.isnan(a) & ~np.isnan(b)
            rhos[c][str(s)] = runde(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 10 else None
    k3 = {}
    for c, r in rhos.items():
        werte = [v for v in r.values() if v is not None]
        mittel = float(np.mean(werte)) if werte else None
        k3[c] = {"rho_je_seed": r, "mittel": runde(mittel), "min": runde(min(werte)) if werte else None,
                 "max": runde(max(werte)) if werte else None, "seeds": len(werte),
                 "erfuellt": None if mittel is None else bool(mittel >= K3_SCHWELLE)}
    folgen = None
    if phs.ENTWICKLUNG:  # Folgen fuer die Auswahl nur in der Entwicklung; in der Bestaetigung ist die Auswahl fest (3.1)
        folgen = folgen_k3(k3)
        folgen["max_abs_rho_satz_mit_k3"] = max_rho(folgen["satz_mit_k3"], ref_kand, ref_best)
    erg = {"referenz": "berichteter Lauf" if ref_ist_bericht else "Kontrolllauf Seed 42", "je_seed": je_seed,
           "k3": k3, "folgen": folgen, "schwellen": {"jaccard": K3_JACCARD, "rho_mittel": K3_SCHWELLE}}
    (AUS / "k3_ergebnis.json").write_text(json.dumps(erg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"referenz": erg["referenz"], "je_seed": je_seed}, ensure_ascii=False))
    zeilen = [{"Code": c, "mittel": v["mittel"], "min": v["min"], "max": v["max"], "seeds": v["seeds"],
               "K3": v["erfuellt"]} for c, v in k3.items()]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(pd.DataFrame(zeilen).to_string(index=False))
    if folgen is not None:
        print(json.dumps(folgen, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- Folgen fuer die Auswahl (Eintrag vom 17.10., Punkt c)

def code_von(eintrag):
    """'EO1 A1' -> EO1_A1, 'EO4 R3' -> EO4_R3, 'PE1' oder 'DS1 (bestehend)' -> Platz (bestehender Indikator)."""
    teile = eintrag.replace(" (eingeschraenkt)", "").split()
    if len(teile) > 1 and re.fullmatch(r"A\dR?B?|R3B?", teile[1]):
        return f"{teile[0]}_{teile[1]}"
    return teile[0]


def max_rho(satz, ref_kand, ref_best):
    platz_zu_ind = {v: k for k, v in bt.CODE.items()}
    spalten = {}
    for e in satz:
        c = code_von(e)
        if c in ref_kand.columns:
            spalten[e] = ref_kand[c].rank()
        elif c in platz_zu_ind:
            spalten[e] = ref_best[platz_zu_ind[c]].rank()
    kor = pd.DataFrame(spalten).corr().abs()
    np.fill_diagonal(kor.values, np.nan)
    paar = kor.stack().idxmax()
    return {"max_abs_rho": runde(kor.max().max()), "paar": list(paar), "eintraege": len(spalten)}


def folgen_k3(k3):
    """Wendet die vorab festgelegten Folgen von K3 auf die Auswahl aus entscheidungsvorlage_regeln.json an.

    Bestehende Indikatoren werden ueber ihren Platz angesprochen (Werte aus backtest_phase1), Kandidaten ueber ihren
    Code. Jede Aenderung ist eine Vorlage fuer Ben, keine stille Umstellung.
    """
    vorlage = json.loads((bk.OUT / "entscheidungsvorlage_regeln.json").read_text(encoding="utf-8"))
    tab = pd.read_csv(bk.OUT / "kandidaten_kennzahlen.csv").set_index("Code")
    alle = tab["S_alle"].astype(bool).to_dict()
    platz_zu_ind = {v: k for k, v in bt.CODE.items()}

    def k3_von(code):
        if code in k3:
            return k3[code]["erfuellt"]
        return k3.get(platz_zu_ind.get(code, ""), {}).get("erfuellt")

    pruef = AUS / "pruefblatt_auswertung.json"
    bestaetigt = json.loads(pruef.read_text(encoding="utf-8"))["bestaetigt"] if pruef.exists() else None
    eo4 = AUS / "eo4_kennzahlen.csv"
    eo4_alle = (pd.read_csv(eo4).set_index("Code")["S_alle"].astype(bool).to_dict() if eo4.exists() else {})
    neu, aenderungen, status = [], [], {}
    for eintrag in vorlage["satz_nach_regeln"]["plaetze"]:
        platz = eintrag.split()[0]
        code = code_von(eintrag)
        regel = vorlage["regeln"][platz]
        k = k3_von(code)
        status[eintrag] = k
        if platz == "EO4":  # bereinigte Fassungen, sobald gerechnet (Eintrag vom 17.10., Bereinigung)
            b_da = "EO4_A1B" in eo4_alle and "EO4_A2B" in eo4_alle
            c1, c2, zu_r3, zu_a2 = (("EO4_A1B", "EO4_A2B", "EO4 R3B", "EO4 A2B") if b_da
                                    else ("EO4_A1", "EO4_A2", "EO4 R3", eintrag))
            a1 = eo4_alle.get(c1)
            a1 = None if a1 is None or bestaetigt is None else (a1 and k3_von(c1) is not False and bestaetigt)
            a2 = bool(eo4_alle[c2] if b_da else alle["EO4_A2"]) and k3_von(c2) is not False
            status[eintrag] = k3_von(c2)
            if a1 and a2:
                neu.append(zu_r3)
                aenderungen.append({"platz": "EO4", "von": eintrag, "zu": zu_r3,
                                    "grund": "A1 erfuellt alle Kriterien und das Pruefblatt bestaetigt das Modell"})
            elif a2:
                neu.append(zu_a2)
                if a1 is False or zu_a2 != eintrag:
                    aenderungen.append({"platz": "EO4", "von": eintrag, "zu": zu_a2,
                                        "grund": ("A1 entfaellt" if a1 is False else "A1 steht aus")
                                        + ("; A2 ohne Verlagsvermerke" if b_da else "")})
            else:
                neu.append(zu_a2 + " (eingeschraenkt)")
                aenderungen.append({"platz": "EO4", "von": eintrag, "zu": zu_a2 + " (eingeschraenkt)",
                                    "grund": "A2 verfehlt K3 oder andere Kriterien; Vorlage fuer Ben", "a1_erfuellt": a1})
            continue
        if k is not False:
            neu.append(eintrag)
            continue
        if regel.startswith("R4") or "(eingeschraenkt)" in eintrag:
            neu.append(eintrag)
            aenderungen.append({"platz": platz, "von": eintrag, "zu": eintrag, "grund": "bereits eingeschraenkt; K3 verfehlt"})
        elif regel.startswith("R1"):
            neu.append(eintrag + " (eingeschraenkt)")
            aenderungen.append({"platz": platz, "von": eintrag, "zu": eintrag + " (eingeschraenkt)",
                                "grund": "nach R1 behalten, K3 verfehlt: R4"})
        elif platz == "EO5":
            neu.append(eintrag + " (eingeschraenkt)")
            aenderungen.append({"platz": platz, "von": eintrag, "zu": eintrag + " (eingeschraenkt)",
                                "grund": "neue Facette verfehlt K3; fuehren oder streichen entscheidet Ben"})
        elif code in E7_RUECKFALL:
            r = E7_RUECKFALL[code]
            if alle.get(r) and k3_von(r) is True:
                neu.append(r.replace("_", " "))
                aenderungen.append({"platz": platz, "von": eintrag, "zu": r.replace("_", " "), "grund": "E7: A4 verfehlt K3"})
            else:
                neu.append(eintrag + " (eingeschraenkt)")
                aenderungen.append({"platz": platz, "von": eintrag, "zu": eintrag + " (eingeschraenkt)",
                                    "grund": "E7: A4 und A5 verfehlen K3 oder A5 andere Kriterien"})
        else:  # R2-Wahl verfehlt K3
            uebrige = [c for c in tab.index if tab.loc[c, "Platz"] == platz and c != code and alle[c]
                       and k3_von(c) is True]
            bev = bk.BEVORZUGT.get(platz)
            if uebrige:
                zu = bev if bev in uebrige else (uebrige[0] if len(uebrige) == 1 else "offen: " + ", ".join(uebrige))
                neu.append(zu.replace("_", " "))
                aenderungen.append({"platz": platz, "von": eintrag, "zu": zu.replace("_", " "),
                                    "grund": "R2 unter den uebrigen Alternativen"})
            else:
                zu = bev if bev in bk.KCODES else code
                neu.append(zu.replace("_", " ") + " (eingeschraenkt)")
                aenderungen.append({"platz": platz, "von": eintrag, "zu": zu.replace("_", " ") + " (eingeschraenkt)",
                                    "grund": "keine Alternative erfuellt alle Kriterien mit K3: R4 mit bevorzugter Variante"})
    return {"satz_vorher": vorlage["satz_nach_regeln"]["plaetze"], "satz_mit_k3": neu, "k3_je_eintrag": status,
            "aenderungen": aenderungen, "pruefblatt_bestaetigt": bestaetigt,
            "eo4_a1_alle_ohne_k3": eo4_alle.get("EO4_A1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teil", choices=["vermerke", "eo4", "eo4report", "blatt2", "pruefblatt", "k3", "k3report"])
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--index", default="pruefblatt2_index.npy")
    ap.add_argument("--blatt", default="pruefblatt2_befund_ben.csv")
    ap.add_argument("--aus", default="pruefblatt_auswertung.json")
    a = ap.parse_args()
    t0 = time.time()
    if a.teil == "eo4":
        teil_eo4(t0)
    elif a.teil == "eo4report":
        teil_eo4report()
    elif a.teil == "blatt2":
        teil_blatt2()
    elif a.teil == "vermerke":
        teil_vermerke()
    elif a.teil == "pruefblatt":
        teil_pruefblatt(a.index, a.blatt, a.aus)
    elif a.teil == "k3":
        teil_k3(t0, a.budget)
    else:
        teil_k3report()


if __name__ == "__main__":
    main()
