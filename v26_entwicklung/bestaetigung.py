#!/usr/bin/env python3
"""V2.6, Schritt 2.1 (Protokoll V2.6, 3.4): einmalige Bestaetigung auf Phase 2.

Einziger Code, der Dateien der Phase 2 liest. Jeder Lesezugriff auf die Eingaben der Phase 2 und auf die
phasenuebergreifenden Paare geht ueber phase.lies(): vorher Eintrag ins Zugriffsprotokoll (Zeit UTC, Datei, SHA-256,
Zweck), nur in einem gestarteten Lauf. Einen zweiten Lauf verweigert phase.starte_lauf() ohne ausdruecklichen Grund.

Ablauf (Protokoll 3, Eintrag zur Umsetzung von 3.3 und 3.4, Punkt a), erst nach dem Tag v2.6-frozen und nach dem
Entblinden auf Phase 1 (entblinden.py):
  1. Ben, Pipeline-Umgebung auf dem Mac, im Ordner v26_entwicklung:
       V26_PHASE=2 V26_GETAGGT=v2.6-frozen python3 bestaetigung.py start
       V26_PHASE=2 V26_GETAGGT=v2.6-frozen python3 zertifiziertheit_mac.py
       V26_PHASE=2 V26_GETAGGT=v2.6-frozen python3 seeds_phase1_mac.py
  2. Linux-Umgebung der Geraeteanbindung, nach Bens Freigabe, so oft wie noetig (setzt fort, schreibt nur fertige Dateien):
       V26_PHASE=2 V26_GETAGGT=v2.6-frozen python3 bestaetigung.py weiter --budget 150
     Die Batterie (1a, 1b, EO4 und K3 mit denselben Skripten, Seeds und Schwellen wie auf Phase 1) wartet, bis die
     Ausgaben der Mac-Teile vorliegen. Danach Zugehoerigkeiten mit Multiversum, O3, B5, A2 und der Bericht; der Bericht
     schliesst den Lauf.
Weitere Befehle: plan (Schritte und Stand), status (Lauf und Zugriffsprotokoll).

Trockenlauf vor dem Tag (Protokoll 3, Punkt j): V26_PHASE=1 V26_AUSGABE=<Ordner> mit denselben Befehlen; gerechnet wird
nur die Batterie, B1 und B2 vergleichen sie mit den Ergebnissen aus 1a bis 1i; Zugehoerigkeiten werden verweigert.

Ausgegeben werden nur Zahlen, keine Texte, Titel oder Namen. Datensaetze verlassen den Rechner nicht.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import phase as phs  # noqa: E402  (nur Standardbibliothek: start und status laufen auch in der Pipeline-Umgebung)

TROCKEN = (phs.PHASE == 1 and not phs.ENTWICKLUNG)
SCHRITTE_DATEI = (phs.WURZEL / phs.P / "schritte.json") if phs.WURZEL is not None else None
LOGS = (phs.WURZEL / phs.P / "logs") if phs.WURZEL is not None else None
V26 = phs.V26
P1_OPT = V26 / "output_v26_entwicklung" / "optimismus_phase1"
P1_ENTBL = V26 / "output_v26_entblinden" / "phase1"
P1_REF = phs.LAUF / "prior" / "output_phase1"
P1_SEEDS = V26 / "output_v26_seeds" / "phase1"

# Eingefroren (Protokoll 3.4 und 3.5); nach dem Tag nicht mehr veraendern.
B1_MIN_REPLIZIERT = 15
B2_GRENZE = {"K1": 0.002, "K2": 0.059, "K3": 0.021, "K4": 0.047, "K5": 0.017, "K6": 0.036}  # obere Grenzen aus 1i
B5_MIN_DOK, B5_MIN_PAARE, B5_ALPHA = 20, 8, 0.05
B5_PERMUTATIONEN, B5_SEED = 10000, 2601
BOOT, BOOT_SEED = 2000, 7
A2_SCHWELLE = 0.6
O3_TAU, O3_NACHBARN, O3_TOPIC = 0.597, 10, 0
K3_SEEDS = (0, 1, 7, 2026)


def uhr(t0, text):
    print(f"[{time.time() - t0:7.1f} s] {text}", flush=True)


def runde(x, k=6):
    return None if x is None or x != x else round(float(x), k)


def schreibe_json(pfad, obj):
    pfad.parent.mkdir(parents=True, exist_ok=True)
    tmp = pfad.with_name(pfad.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, pfad)


def module():
    """Die Skripte der Batterie (laedt pandas, scikit-learn und die Pipeline; nur in der Linux-Umgebung noetig)."""
    import optimismus_phase1 as opt
    return opt, opt.ak, opt.bk, opt.bt


# ---------------------------------------------------------------- Schritte

def _alle(pfad, muster, n):
    return len(list(pfad.glob(muster))) >= n


def schritte():
    """Feste Folge. Jeder Schritt: Name, Befehl oder Funktion, Fertig-Pruefung, Voraussetzung, Budget, Mindestzeit (s)."""
    opt, ak, bk, bt = module()
    s = []
    if TROCKEN:
        s.append({"name": "mac_kopie", "funktion": kopiere_mac_ausgaben, "min": 5})
    s += [
        {"name": "bt_repro", "befehl": ["backtest_phase1.py", "repro"], "min": 60},
        {"name": "bt_split", "befehl": ["backtest_phase1.py", "split", "--r", "0", "50"], "budget": True, "min": 45,
         "vollstaendig": lambda: all((bt.OUT / f"split_r{r:02d}_B.csv").exists() for r in range(50))},
        {"name": "bt_null", "befehl": ["backtest_phase1.py", "null", "--r", "0", "50"], "budget": True, "min": 45,
         "vollstaendig": lambda: all((bt.OUT / f"null_r{r:02d}.csv").exists() for r in range(50))},
        {"name": "bt_report", "befehl": ["backtest_phase1.py", "report"], "min": 45},
        {"name": "bk_cache", "befehl": ["backtest_kandidaten.py", "cache"], "min": 60},
        {"name": "bk_repro", "befehl": ["backtest_kandidaten.py", "repro"], "min": 75},
        {"name": "bk_lauf", "befehl": ["backtest_kandidaten.py", "lauf"], "budget": True, "min": 45,
         "vollstaendig": lambda: _alle(bk.OUT, "split_r*_[AB].csv", 100) and _alle(bk.OUT, "null_r*.csv", 50)},
        {"name": "bk_report", "befehl": ["backtest_kandidaten.py", "report"], "min": 45},
        {"name": "ak_vermerke", "befehl": ["auswertung_eo4_k3.py", "vermerke"], "voraus": mac_zert_bereit, "min": 60},
        {"name": "ak_eo4", "befehl": ["auswertung_eo4_k3.py", "eo4"], "voraus": mac_zert_bereit, "min": 30},
        {"name": "ak_eo4report", "befehl": ["auswertung_eo4_k3.py", "eo4report"], "min": 30},
        {"name": "ak_k3", "befehl": ["auswertung_eo4_k3.py", "k3"], "voraus": mac_seeds_bereit, "budget": True,
         "min": 60, "vollstaendig": lambda: all((ak.OUT_K3 / f"seed{x}_best.csv").exists() for x in seeds_noetig(ak))},
        {"name": "ak_k3report", "befehl": ["auswertung_eo4_k3.py", "k3report"], "min": 20},
        {"name": "opt_daten", "befehl": ["optimismus_phase1.py", "daten"], "min": 45},
        {"name": "opt_kriterien", "befehl": ["optimismus_phase1.py", "kriterien"], "min": 20},
    ]
    if not TROCKEN:
        s += [
            {"name": "zugehoerigkeit", "befehl": ["bestaetigung.py", "zugehoerigkeit"], "voraus": phase1_entblindet,
             "min": 60},
            {"name": "o3", "befehl": ["bestaetigung.py", "o3"], "min": 45},
            {"name": "b5", "befehl": ["bestaetigung.py", "b5"], "voraus": phase1_entblindet, "min": 30},
        ]
    s += [{"name": "a2", "befehl": ["bestaetigung.py", "a2"], "min": 20},
          {"name": "bericht", "befehl": ["bestaetigung.py", "bericht"], "min": 20}]
    return s


def seeds_noetig(ak):
    try:
        return list(K3_SEEDS) + ([] if ak.ref_gleich() else [42])
    except FileNotFoundError:
        return list(K3_SEEDS)


def mac_zert_bereit():
    _, ak, _, _ = module()
    noetig = [ak.ZERT / "werte.npy", ak.ZERT / "saetze_meta.npz", ak.ZERT / f"zertifiziertheit_{phs.P}.json"]
    fehlt = [f.name for f in noetig if not f.exists()]
    return (not fehlt, "warte auf zertifiziertheit_mac.py (fehlt: " + ", ".join(fehlt) + ")")


def mac_seeds_bereit():
    _, ak, _, _ = module()
    f = ak.SEEDS / f"seeds_{phs.P}.json"
    if not f.exists():
        return False, f"warte auf seeds_phase1_mac.py (fehlt: {f.name})"
    j = json.loads(f.read_text(encoding="utf-8"))
    fehlt = [str(x) for x in K3_SEEDS if str(x) not in j.get("seeds", {})
             or not all(p.exists() for p in ak.seed_dateien(x))]
    if "kontrolle_seed42" not in j or not all(p.exists() for p in ak.seed_dateien(42)):
        fehlt.append("42 (Kontrolle)")
    return (not fehlt, "warte auf seeds_phase1_mac.py (fehlt: Seed " + ", ".join(fehlt) + ")")


def phase1_entblindet():
    f = P1_ENTBL / "entblinden_ergebnis.json"
    return (f.exists(), "Entblinden auf Phase 1 fehlt (3.3 vor 3.4): entblinden.py rechnen")


def kopiere_mac_ausgaben():
    """Trockenlauf: Ausgaben der Mac-Teile von Phase 1 in die Wurzel kopieren (die Mac-Teile laufen nicht noch einmal)."""
    _, ak, _, _ = module()
    n = 0
    for quelle, ziel in ((V26 / "output_v26_zertifiziertheit" / "phase1", ak.ZERT), (P1_SEEDS, ak.SEEDS)):
        ziel.mkdir(parents=True, exist_ok=True)
        for f in sorted(quelle.iterdir()):
            if f.suffix in (".npy", ".npz", ".json") and not f.name.startswith(("pruefblatt", "saetze_vermerk", "werte_")):
                shutil.copy2(f, ziel / f.name)
                n += 1
    print(f"{n} Dateien der Mac-Teile kopiert", flush=True)
    return 0


def lies_stand(lauf_id=None):
    """Fertige Schritte; mit lauf_id nur die dieses Laufs (ein Marker eines anderen Laufs zaehlt nicht)."""
    st = json.loads(SCHRITTE_DATEI.read_text(encoding="utf-8")) if SCHRITTE_DATEI.exists() else {}
    return st if lauf_id is None else {k: v for k, v in st.items() if v.get("lauf_id") == lauf_id}


def markiere(name, sekunden, lauf_id):
    st = lies_stand()
    st[name] = {"fertig_utc": phs.jetzt(), "sekunden": round(sekunden, 1), "lauf_id": lauf_id}
    schreibe_json(SCHRITTE_DATEI, st)


def teil_weiter(budget):
    t0 = time.time()
    lauf = phs.aktiver_lauf()
    LOGS.mkdir(parents=True, exist_ok=True)
    for sch in schritte():
        if sch["name"] in lies_stand(lauf["lauf_id"]):
            continue
        rest = budget - (time.time() - t0)
        if rest < sch.get("min", 60):
            print(f"WEITER: Zeitbudget fuer {sch['name']} reicht nicht ({rest:.0f} s); erneut aufrufen", flush=True)
            return
        if "voraus" in sch:
            ok, msg = sch["voraus"]()
            if not ok:
                print(f"WARTEN: {msg}", flush=True)
                return
        t1 = time.time()
        if "funktion" in sch:
            rc = sch["funktion"]()
            ausgabe = ""
        else:
            befehl = [sys.executable] + sch["befehl"]
            if sch.get("budget"):
                befehl += ["--budget", str(int(max(10, rest - 25)))]
            env = dict(os.environ)
            env.setdefault("PYTHONHASHSEED", "0")  # feste Reihenfolge von Mengen: DI3 in 1a sonst bis 1e-16 je Prozess anders
            r = subprocess.run(befehl, cwd=HIER, env=env, capture_output=True, text=True)
            rc, ausgabe = r.returncode, r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else "")
            n = len(list(LOGS.glob(f"{sch['name']}_*.log")))
            (LOGS / f"{sch['name']}_{n + 1:02d}.log").write_text(ausgabe, encoding="utf-8")
        dt = time.time() - t1
        if rc != 0:
            print(f"FEHLER in {sch['name']} (Rueckgabe {rc}, {dt:.0f} s); Protokoll in {phs.relativ(LOGS)}", flush=True)
            print(ausgabe[-3000:], flush=True)
            raise SystemExit(1)
        if "vollstaendig" in sch and not sch["vollstaendig"]():
            print(f"TEIL: {sch['name']} fortgesetzt ({dt:.0f} s), noch nicht vollstaendig; erneut aufrufen", flush=True)
            return
        markiere(sch["name"], dt, lauf["lauf_id"])
        print(f"FERTIG: {sch['name']} ({dt:.0f} s)", flush=True)
        if sch["name"] == "bericht":
            print("ALLES FERTIG", flush=True)
            return
    print("ALLES FERTIG", flush=True)


def teil_plan():
    reg = phs.lies_register()
    lauf_id = reg["laeufe"][-1]["lauf_id"] if reg and reg.get("laeufe") else None
    stand = lies_stand(lauf_id) if SCHRITTE_DATEI is not None else {}
    for sch in schritte():
        z = stand.get(sch["name"])
        print(f"{'fertig' if z else 'offen ':6s}  {sch['name']:15s}  {z['fertig_utc'] + ' (' + str(z['sekunden']) + ' s)' if z else ''}")


def teil_status():
    reg = phs.lies_register()
    print(json.dumps({"schalter": phs.beschreibung(), "register": phs.relativ(phs.REGISTER) if phs.REGISTER else None,
                      "laeufe": [{k: v for k, v in x.items() if k != "skripte_sha256"} for x in (reg or {}).get("laeufe", [])]},
                     ensure_ascii=False, indent=1))
    z = phs.zugriffe()
    je = {}
    for e in z:
        d = je.setdefault(e["datei"], {"zugriffe": 0, "sha256": set()})
        d["zugriffe"] += 1
        d["sha256"].add(e["sha256"][:16])
    print(json.dumps({"zugriffe": len(z), "je_datei": {k: {"zugriffe": v["zugriffe"], "sha256": sorted(v["sha256"])}
                                                        for k, v in sorted(je.items())}}, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- Zugehoerigkeit, O3, B5, A2 (Phase 2)

def teil_zugehoerigkeit():
    import entblinden
    phs.aktiver_lauf()
    entblinden.rechnen(time.time())


def teil_o3():
    """Relevanzdiagnose an T0 (Label 0 der Phase 2) mit der Schwelle aus E5, neben derselben Groesse fuer alle Topics."""
    import pickle
    import numpy as np
    import pandas as pd
    phs.aktiver_lauf()
    with open(phs.lies(phs.REF / "model_results.pkl", "Labels und Einbettungen fuer O3"), "rb") as f:
        md = pickle.load(f)
    labels = np.asarray(md["labels"])
    E = np.ascontiguousarray(md["embeddings_sbert"], dtype=np.float32)
    zug = np.where(labels >= 0)[0]
    aussen = np.zeros(len(labels), dtype=np.int64)
    for a in range(0, len(zug), 2048):
        blk = zug[a:a + 2048]
        S = E[blk] @ E.T
        S[np.arange(len(blk)), blk] = -1.0
        aussen[blk] = ((S >= O3_TAU) & (labels[None, :] != labels[blk, None])).sum(axis=1)
    ok = pd.Series((aussen[zug] >= O3_NACHBARN).astype(float)).groupby(labels[zug]).mean()
    t0_wert = float(ok.get(O3_TOPIC, np.nan))
    erg = {"tau": O3_TAU, "nachbarn": O3_NACHBARN, "topic": O3_TOPIC, "topics": int(len(ok)),
           "anteil_t0": runde(t0_wert), "anteil_dokumente_zugeordnet": runde((aussen[zug] >= O3_NACHBARN).mean()),
           "topics_median": runde(ok.median()), "topics_min": runde(ok.min()),
           "topics_quantile_10_25_75_90": [runde(q) for q in ok.quantile([.1, .25, .75, .9])],
           "topics_unter_50_prozent": int((ok < 0.5).sum()),
           "rang_t0_von_unten": int((ok < t0_wert).sum()) + 1 if t0_wert == t0_wert else None,
           "n_t0": int((labels == O3_TOPIC).sum())}
    schreibe_json(phs.WURZEL / phs.P / "o3.json", erg)
    print(json.dumps(erg, ensure_ascii=False))


def _partiell(rx, ry, rz):
    import numpy as np
    c = np.corrcoef(np.vstack([rx, ry, rz]))
    a, b, d = c[0, 1], c[0, 2], c[1, 2]
    return float((a - b * d) / np.sqrt((1 - b ** 2) * (1 - d ** 2)))


def teil_b5():
    """B5 (3.4) und Zusatz zur Offenheit; Paare aus dem berichteten Lauf v2.5.2, Phase-1-Zugehoerigkeiten aus 3.3."""
    import pickle
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu, rankdata, spearmanr
    phs.aktiver_lauf()
    paare = pd.read_csv(phs.lies(phs.KREUZ / "topic_matches_mutual.csv", "Mutual-Best-Paare des berichteten Laufs (B5)"))
    if "mutual_best" in paare.columns:
        paare = paare[paare["mutual_best"].astype(bool)]
    with open(P1_REF / "model_results.pkl", "rb") as f:
        l1 = np.asarray(pickle.load(f)["labels"])
    with open(phs.lies(phs.REF / "model_results.pkl", "Labels der Phase 2 (Topicgroessen fuer B5)"), "rb") as f:
        l2 = np.asarray(pickle.load(f)["labels"])
    n1 = pd.Series(l1[l1 >= 0]).value_counts()
    n2 = pd.Series(l2[l2 >= 0]).value_counts()
    N1, N2 = len(l1), len(l2)
    d = paare[["phase1_topic", "phase2_topic"]].astype(int).copy()
    d["n1"] = d["phase1_topic"].map(n1).fillna(0).astype(int)
    d["n2"] = d["phase2_topic"].map(n2).fillna(0).astype(int)
    alle_paare = len(d)
    d = d[(d["n1"] >= B5_MIN_DOK) & (d["n2"] >= B5_MIN_DOK)].copy()
    d["g"] = np.log((d["n2"] / N2) / (d["n1"] / N1))
    z1 = pd.read_csv(P1_ENTBL / "zugehoerigkeiten.csv", index_col=0)
    d["dominant1"] = d["phase1_topic"].map(z1["dominant"])
    d["m_ws1"] = d["phase1_topic"].map(z1["m_ws"])
    d["ip1_1"] = d["phase1_topic"].map(z1["wert_IP1"])
    ws = d.loc[d["dominant1"] == "m_ws", "g"].to_numpy(float)
    lat = d.loc[d["dominant1"] == "m_latent", "g"].to_numpy(float)
    erg = {"paare_datei": int(alle_paare), "paare_ab_20_dok": int(len(d)), "N1": int(N1), "N2": int(N2),
           "gruppen": {k: int((d["dominant1"] == k).sum()) for k in ["m_ws", "m_trend", "m_ec", "m_latent"]},
           "g_median_je_gruppe": {k: runde(d.loc[d["dominant1"] == k, "g"].median())
                                  for k in ["m_ws", "m_trend", "m_ec", "m_latent"]}}
    beurteilbar = len(ws) >= B5_MIN_PAARE and len(lat) >= B5_MIN_PAARE
    t1 = {"n_ws": int(len(ws)), "n_latent": int(len(lat))}
    if len(ws) >= 2 and len(lat) >= 2:
        u = mannwhitneyu(ws, lat, alternative="greater")
        gt = (ws[:, None] > lat[None, :]).sum()
        lt = (ws[:, None] < lat[None, :]).sum()
        t1.update({"U": runde(u.statistic), "p": runde(u.pvalue, 8), "cliffs_delta": runde((gt - lt) / (len(ws) * len(lat))),
                   "besteht": bool(u.pvalue < B5_ALPHA)})
    ok = d[["m_ws1", "g", "ip1_1"]].notna().all(axis=1)
    t2 = {"n": int(ok.sum()), "permutationen": B5_PERMUTATIONEN, "seed": B5_SEED}
    r_obs = float("nan")
    if ok.sum() >= 4:
        rx, ry, rz = (rankdata(d.loc[ok, c].to_numpy(float)) for c in ("m_ws1", "g", "ip1_1"))
        with np.errstate(invalid="ignore", divide="ignore"):
            r_obs = _partiell(rx, ry, rz)
            if np.isfinite(r_obs):
                rng = np.random.default_rng(B5_SEED)
                perm = np.array([_partiell(rx, rng.permutation(ry), rz) for _ in range(B5_PERMUTATIONEN)])
                p2 = (1 + int((perm >= r_obs).sum())) / (B5_PERMUTATIONEN + 1)
                t2.update({"partielle_spearman": runde(r_obs), "p": runde(p2, 8), "besteht": bool(p2 < B5_ALPHA)})
    if not np.isfinite(r_obs):  # nicht bestimmbar (zu wenige Paare oder keine Streuung): Test nicht bestanden
        t2.update({"partielle_spearman": None, "p": None, "besteht": False, "nicht_bestimmbar": True})
        beurteilbar = False
    if not beurteilbar:
        urteil = "nicht beurteilbar"
    else:
        urteil = "erfuellt" if (t1.get("besteht") and t2["besteht"]) else "verfehlt"
    erg.update({"test1": t1, "test2": t2, "b5": urteil})
    # Zusatz zur Offenheit (nur berichtet): Veraenderung der nullreferenzierten Offenheit gegen g
    s1 = pd.read_csv(P1_ENTBL / "ses_offenheit.csv", index_col=0)["EO_SES_MITTEL"]
    s2 = pd.read_csv(phs.ordner("output_v26_entblinden/phase2", "entblinden") / "ses_offenheit.csv",
                     index_col=0)["EO_SES_MITTEL"]
    d["delta_offenheit"] = d["phase2_topic"].map(s2).to_numpy(float) - d["phase1_topic"].map(s1).to_numpy(float)
    okz = d[["g", "delta_offenheit"]].notna().all(axis=1)
    x, y = d.loc[okz, "g"].to_numpy(float), d.loc[okz, "delta_offenheit"].to_numpy(float)
    rho = float(spearmanr(x, y)[0]) if okz.sum() > 3 else float("nan")
    rngb = np.random.default_rng(BOOT_SEED)
    bs = []
    for _ in range(BOOT if len(x) >= 4 else 0):
        i = rngb.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            bs.append(spearmanr(x[i], y[i])[0])
    erg["zusatz_offenheit"] = {"n": int(okz.sum()), "spearman": runde(rho),
                               "ci95": [runde(q) for q in np.quantile(bs, [0.025, 0.975])] if bs else None,
                               "bootstrap": BOOT, "seed": BOOT_SEED, "erwartet": "negativ"}
    d.drop(columns=["dominant1"]).to_csv(phs.WURZEL / phs.P / "b5_paare.csv", index=False)
    schreibe_json(phs.WURZEL / phs.P / "b5.json", erg)
    print(json.dumps(erg, ensure_ascii=False))


def hennig(ref_lab, seed_labels):
    """Clusterweise Stabilitaet nach Hennig 2007: je Referenztopic der groesste Jaccard-Index mit einem Topic des Laufs
    (ohne Schwelle, 0 ohne Ueberlappung), gemittelt ueber die Laeufe; zurueck der Median ueber die Topics."""
    import numpy as np
    import pandas as pd
    tids = np.array(sorted(set(ref_lab[ref_lab >= 0].tolist())))
    n_i = pd.Series(ref_lab[ref_lab >= 0]).value_counts().reindex(tids).to_numpy(float)
    J = np.zeros((len(tids), len(seed_labels)))
    for j, lab in enumerate(seed_labels):
        beide = (ref_lab >= 0) & (lab >= 0)
        ct = pd.crosstab(ref_lab[beide], lab[beide]).reindex(index=tids, fill_value=0)
        n_j = pd.Series(lab[lab >= 0]).value_counts().reindex(ct.columns).to_numpy(float)
        nij = ct.to_numpy(float)
        jac = nij / (n_i[:, None] + n_j[None, :] - nij) if nij.size else np.zeros((len(tids), 1))
        J[:, j] = jac.max(axis=1) if jac.size else 0.0
    m = J.mean(axis=1)
    return {"median": runde(np.median(m)), "mittel": runde(m.mean()), "anteil_unter_0_6": runde((m < 0.6).mean()),
            "anteil_ab_0_75": runde((m >= 0.75).mean()), "topics": int(len(tids)), "laeufe": len(seed_labels)}


def teil_a2():
    import pickle
    import numpy as np
    _, ak, bk, _ = module()
    phs.aktiver_lauf()
    ref_bericht = ak.ref_gleich()
    lab_ref = bk.lade_labels() if ref_bericht else np.load(ak.seed_dateien(42)[0]).astype(np.int64)
    seeds = [np.load(ak.seed_dateien(s)[0]).astype(np.int64) for s in K3_SEEDS]
    erg = {"referenz": "berichteter Lauf" if ref_bericht else "Kontrolllauf Seed 42", "phase": hennig(lab_ref, seeds),
           "schwelle": A2_SCHWELLE}
    if not TROCKEN:  # zum Vergleich Phase 1 (Referenz wie in 1h: der berichtete Lauf)
        with open(P1_REF / "model_results.pkl", "rb") as f:
            l1 = np.asarray(pickle.load(f)["labels"]).astype(np.int64)
        erg["phase1"] = hennig(l1, [np.load(P1_SEEDS / f"labels_seed{s}.npy").astype(np.int64) for s in K3_SEEDS])
    erg["a2_trifft_zu"] = bool(erg["phase"]["median"] < A2_SCHWELLE)
    schreibe_json(phs.WURZEL / phs.P / "a2.json", erg)
    print(json.dumps(erg, ensure_ascii=False))


# ---------------------------------------------------------------- B1, B2, B4, Deutung (3.4 und 3.5)

def _erfuellt(st, f, k, pe1):
    if k in ("K4", "K5") and pe1:
        return None  # entfaellt (E4, K4 fuer PE1 nicht anwendbar)
    v = st[f][k]
    if k == "K2":
        return v != "verfehlt"  # "bedingt" gilt nach E1 als erfuellt
    if k == "K3":
        return None if v is None else bool(v)
    return bool(v)


def b1_b2(opt, z1, z2, ohne_k3):
    import numpy as np
    st1, st2 = opt.status(z1), opt.status(z2)
    dim_von = {p: d for d, ps in {"EO": ["EO1", "EO2", "EO3", "EO4", "EO5"], "PE": ["PE1", "PE2", "PE3"],
                                  "DS": ["DS1", "DS2", "DS3"], "DI": ["DI1", "DI2", "DI3", "DI4"],
                                  "IP": ["IP1", "IP2", "IP3"]}.items() for p in ps}
    krit_b1 = [k for k in ("K1", "K2", "K3", "K4", "K5") if not (ohne_k3 and k == "K3")]
    zeilen = []
    for p in opt.PLAETZE:
        f, pe1 = opt.AUSWAHL_E9[p][0], p == "PE1"
        s1 = [k for k in krit_b1 if _erfuellt(st1, f, k, pe1) is True]
        fehl = [k for k in s1 if _erfuellt(st2, f, k, pe1) is not True]
        zeilen.append({"platz": p, "fassung": f, "dimension": dim_von[p], "phase1_erfuellt": s1, "phase2_verfehlt": fehl,
                       "repliziert": not fehl,
                       "werte_phase1": {k: runde(z1.loc[f, k]) for k in opt.KRITERIEN},
                       "werte_phase2": {k: runde(z2.loc[f, k]) for k in opt.KRITERIEN}})
    n_rep = sum(z["repliziert"] for z in zeilen)
    dims = {d: any(z["repliziert"] for z in zeilen if z["dimension"] == d) for d in ["EO", "PE", "DS", "DI", "IP"]}
    b1 = {"repliziert": n_rep, "von": len(zeilen), "mindestens": B1_MIN_REPLIZIERT, "dimensionen": dims,
          "ohne_k3": ohne_k3, "erfuellt": bool(n_rep >= B1_MIN_REPLIZIERT and all(dims.values())),
          "dimensionen_ohne_replikation": [d for d, v in dims.items() if not v], "fassungen": zeilen}
    codes = [opt.AUSWAHL_E9[p][0] for p in opt.PLAETZE]
    g1, g2 = opt.gute(z1, codes), opt.gute(z2, codes)
    b2 = {"ohne_k3": ohne_k3, "kriterien": {}}
    for k in opt.KRITERIEN:
        if ohne_k3 and k == "K3":
            b2["kriterien"][k] = {"entfaellt": "A2"}
            continue
        je = {}
        for p in opt.PLAETZE:
            if k in opt.OHNE_PE1 and p == "PE1":
                continue
            f = opt.AUSWAHL_E9[p][0]
            a, b = g1.loc[f, k], g2.loc[f, k]
            if a == a and b == b:
                je[p] = float(opt.RICHTUNG[k] * (a - b))
        abfall = float(np.mean(list(je.values()))) if je else float("nan")
        b2["kriterien"][k] = {"abfall": runde(abfall), "grenze": B2_GRENZE[k], "plaetze": len(je),
                              "erfuellt": bool(abfall <= B2_GRENZE[k]) if je else None,
                              "je_platz": {p: runde(v) for p, v in je.items()}}
    beurteilt = [v for v in b2["kriterien"].values() if v.get("erfuellt") is not None]
    b2["verfehlt"] = [k for k, v in b2["kriterien"].items() if v.get("erfuellt") is False]
    b2["nicht_beurteilt"] = [k for k, v in b2["kriterien"].items() if "entfaellt" not in v and v.get("erfuellt") is None]
    b2["alle_erfuellt"] = bool(beurteilt) and all(v["erfuellt"] for v in beurteilt) and not b2["nicht_beurteilt"]
    return b1, b2


def deutung(b1, b2, b5):
    if b1["erfuellt"] and b2["alle_erfuellt"] and b5 == "erfuellt":
        return "bestaetigt"
    if not b1["erfuellt"] or len(b2["verfehlt"]) >= 3:
        return "nicht bestaetigt"
    return "teilweise bestaetigt"


def teil_bericht():
    import pickle
    import numpy as np
    opt, ak, bk, bt = module()
    phs.aktiver_lauf()  # nach dem Abschluss wird nichts mehr geschrieben (3.5)
    reg = phs.lies_register()
    with open(P1_OPT / "daten.pkl", "rb") as f:
        D1 = pickle.load(f)
    D2 = opt.lade_daten()
    z1 = opt.kriterien(D1, D1["tids"])
    z2 = opt.kriterien(D2, D2["tids"])
    a2 = json.loads((phs.WURZEL / phs.P / "a2.json").read_text(encoding="utf-8"))
    codes18 = [opt.AUSWAHL_E9[p][0] for p in opt.PLAETZE]
    k3_unbestimmbar = bool(z2.loc[codes18, "K3"].isna().all())  # Einheiten so instabil, dass K3 nirgends bestimmbar ist
    a2["k3_auf_phase2_nicht_bestimmbar"] = k3_unbestimmbar
    ohne_k3 = bool(a2["a2_trifft_zu"] or k3_unbestimmbar)
    b1, b2 = b1_b2(opt, z1, z2, ohne_k3)
    krit2 = json.loads((opt.OUT / "kriterien_alle_topics.json").read_text(encoding="utf-8"))
    erg = {"phase": phs.PHASE, "trockenlauf": TROCKEN, "topics": {"phase1": len(D1["tids"]), "phase2": len(D2["tids"])},
           "b1": b1, "b2": b2, "a2": a2,
           "abgleich_kriterien_mit_berichten": {"bestanden": krit2["abgleich_bestanden"],
                                                "kennzahlen": krit2["kennzahlen_abweichungen"],
                                                "status": krit2["status_abweichungen"],
                                                "k1_ausserhalb_der_auswahl": krit2.get("k1_abweichungen", {})}}
    o1 = json.loads((P1_OPT / "optimismus_ergebnis.json").read_text(encoding="utf-8")) \
        if (P1_OPT / "optimismus_ergebnis.json").exists() else None
    if o1:
        erg["grenzen_aus_1i"] = {k: runde(v[1], 3) for k, v in o1["spanne_b2"].items()}
        erg["grenzen_wie_protokoll"] = all(erg["grenzen_aus_1i"][k] == B2_GRENZE[k] for k in B2_GRENZE)
    if TROCKEN:
        d = (z1.loc[z2.index, opt.KRITERIEN] - z2[opt.KRITERIEN]).abs()
        erg["trockenlauf_max_abweichung_kriterien"] = {k: runde(float(np.nanmax(d[k].to_numpy(float)))) for k in opt.KRITERIEN}
        erg["deutung"] = None
    else:
        b5 = json.loads((phs.WURZEL / phs.P / "b5.json").read_text(encoding="utf-8"))
        e1 = json.loads((P1_ENTBL / "entblinden_ergebnis.json").read_text(encoding="utf-8"))
        e2 = json.loads((phs.ordner("output_v26_entblinden/phase2", "entblinden") / "entblinden_ergebnis.json")
                        .read_text(encoding="utf-8"))
        erg["b4"] = {"phase1": e1["multiversum"], "phase2": e2["multiversum"], "mc_phase1": e1["mc"], "mc_phase2": e2["mc"]}
        erg["b5"] = b5
        erg["o3"] = json.loads((phs.WURZEL / phs.P / "o3.json").read_text(encoding="utf-8"))
        erg["zugehoerigkeit_phase2"] = {k: e2[k] for k in ("besetzung", "anteile", "margin", "v25", "gueltige_fassungen",
                                                           "ersetzt_durch_median")}
        erg["efa_phase2"] = {k: e2["efa"][k] for k in ("kmo", "n_kaiser", "n_parallel", "n_parallel_p95")}
        erg["deutung"] = deutung(b1, b2, b5["b5"])
        erg["a5_dimensionen_ohne_replikation"] = b1["dimensionen_ohne_replikation"]
    z = phs.zugriffe()
    je = {}
    for e in z:
        je.setdefault(e["datei"], {"zugriffe": 0, "sha256": set()})
        je[e["datei"]]["zugriffe"] += 1
        je[e["datei"]]["sha256"].add(e["sha256"])
    erg["zugriffsprotokoll"] = {"eintraege": len(z), "laeufe_im_protokoll": sorted({e["lauf_id"] for e in z}),
                                "dateien": {k: {"zugriffe": v["zugriffe"], "sha256": sorted(v["sha256"])}
                                            for k, v in sorted(je.items())}}
    erg["laeufe"] = [{k: v for k, v in x.items() if k not in ("skripte_sha256", "ergebnis")} for x in reg["laeufe"]]
    erg["zweite_laeufe"] = sum(1 for x in reg["laeufe"] if x.get("nummer", 1) > 1)
    erg["schalter"] = phs.beschreibung()
    erg["skripte_sha256"] = phs.skript_pruefsummen()
    f_erg = phs.WURZEL / phs.P / "bestaetigung_ergebnis.json"
    schreibe_json(f_erg, erg)
    kurz = {"deutung": erg["deutung"], "b1": {k: b1[k] for k in ("repliziert", "von", "erfuellt", "dimensionen")},
            "b2": {k: {kk: v.get(kk) for kk in ("abfall", "grenze", "erfuellt")} for k, v in b2["kriterien"].items()},
            "a2": a2["a2_trifft_zu"], "b5": erg.get("b5", {}).get("b5")}
    print(json.dumps(kurz, ensure_ascii=False))
    phs.schliesse_lauf({"deutung": erg["deutung"], "b1_erfuellt": b1["erfuellt"], "b2_verfehlt": b2["verfehlt"],
                        "b5": erg.get("b5", {}).get("b5"), "a2": a2["a2_trifft_zu"], "ohne_k3": ohne_k3}, f_erg)
    print("Lauf abgeschlossen", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teil", choices=["start", "status", "plan", "weiter", "zugehoerigkeit", "o3", "b5", "a2", "bericht"])
    ap.add_argument("--grund", default=None, help="Grund fuer einen zweiten Lauf (wird gezaehlt und berichtet)")
    ap.add_argument("--budget", type=float, default=150.0)
    a = ap.parse_args()
    if phs.ENTWICKLUNG:
        raise SystemExit("bestaetigung.py nur im Bestaetigungsmodus: V26_PHASE=2 (oder V26_PHASE=1 mit V26_AUSGABE)")
    if a.teil == "start":
        if phs.PHASE == 2:  # 3.4 nach 3.3: erst wenn das Entblinden auf Phase 1 vorliegt
            ok, msg = phase1_entblindet()
            if not ok:
                raise SystemExit(msg)
        lauf = phs.starte_lauf(a.grund)
        print(json.dumps({k: v for k, v in lauf.items() if k != "skripte_sha256"}, ensure_ascii=False))
    elif a.teil == "status":
        teil_status()
    elif a.teil == "plan":
        teil_plan()
    elif a.teil == "weiter":
        teil_weiter(a.budget)
    elif a.teil == "zugehoerigkeit":
        if TROCKEN:
            raise SystemExit("Trockenlauf: Zugehoerigkeiten werden vor dem Tag nicht gerechnet")
        teil_zugehoerigkeit()
    elif a.teil == "o3":
        teil_o3()
    elif a.teil == "b5":
        if TROCKEN:
            raise SystemExit("Trockenlauf: B5 braucht Zugehoerigkeiten und Phase 2")
        teil_b5()
    elif a.teil == "a2":
        teil_a2()
    else:
        teil_bericht()


if __name__ == "__main__":
    main()
