#!/usr/bin/env python3
"""V2.6, Schritt 1.6 (Protokoll V2.6, 3.3): Entblinden mit den 18 eingefrorenen Fassungen.

Einmal auf Phase 1 nach dem Tag v2.6-frozen; dieselben Funktionen rechnet bestaetigung.py auf Phase 2 (B4). Vor dem Tag
laeuft das Skript nur auf synthetischen Daten (Protokoll 3, Eintrag zur Umsetzung, Punkt j); auf den echten Daten
verweigert es den Lauf ohne V26_GETAGGT=v2.6-frozen.

Gelesen werden nur abgeleitete Groessen: die Topicwerte aller Fassungen aus optimismus_phase1/daten.pkl der Phase (aus 1a
bis 1h), die Null-Pseudokorpora aus 1a, 1b und 1g (fuer den Zusatz zur Offenheit) und signal_memberships.csv des
berichteten Laufs v2.5 (Wechsel je Topic). Gerechnet werden (3.3 mit den Punkten d bis g des Eintrags):
  - die vier Zugehoerigkeiten, die dominante Konfiguration und die Margin je Topic in der Grundfassung (k = 1,
    lambda = 0,5), die Anteile der Konfigurationen und der Wechsel je Topic gegenueber v2.5;
  - das Multiversum mit 30 Varianten und seine Stabilitaet;
  - Monte-Carlo-Baender im Stil der JRC-Audits (1.000 Laeufe, Seed 1600);
  - EFA mit Parallelanalyse und KMO, nur berichtet;
  - der standardisierte Abstand der fuenf EO-Fassungen vom Nullmodell je Topic (Zusatz zur Offenheit in 3.4).
Ausgaben: output_v26_entblinden/phase<N>/ (Bestaetigungsmodus: <Wurzel>/phase<N>/entblinden), nur Werte je Topic und
Kennzahlen, keine Texte. Datensaetze verlassen den Rechner nicht.

Aufruf (im Ordner v26_entwicklung, Phase 1 nach dem Tag):  V26_GETAGGT=v2.6-frozen python3 entblinden.py rechnen
"""
import argparse
import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
sys.path.insert(0, str(HIER))
import optimismus_phase1 as opt  # noqa: E402  (laedt auswertung_eo4_k3, backtest_kandidaten, backtest_phase1)

ak, bk, bt = opt.ak, opt.bk, opt.bt
phs = bt.phs
OUT = phs.ordner(f"output_v26_entblinden/{phs.P}", "entblinden")

# Eingefroren (Protokoll 3.1 und 3.3, Eintrag zur Umsetzung Punkte d bis g); nach dem Tag nicht mehr veraendern.
PLAETZE = list(opt.PLAETZE)
FASSUNG = {p: opt.AUSWAHL_E9[p][0] for p in PLAETZE}             # alle mit Vorzeichen +1
EINGESCHRAENKT = [p for p in PLAETZE if opt.AUSWAHL_E9[p][1] == "eingeschraenkt"]
DIMENSIONEN = {"Epistemische Offenheit": ["EO1", "EO2", "EO3", "EO4", "EO5"],
               "Wahrnehmbarkeit": ["PE1", "PE2", "PE3"],
               "Entwicklungsphase": ["DS1", "DS2", "DS3"],
               "Diffusion": ["DI1", "DI2", "DI3", "DI4"],
               "Wirkungspotenzial": ["IP1", "IP2", "IP3"]}
KERN = ["Epistemische Offenheit", "Wahrnehmbarkeit", "Entwicklungsphase", "Diffusion"]
WP = "Wirkungspotenzial"
EC = ["DS1", "IP1", "IP2", "IP3"]                                  # Emergent Concept: DS1, IP1, IP2 A4, IP3 A1
EO_NUR = ["EO1", "EO2", "EO3"]                                     # Variante: Offenheit nur aus EO1 bis EO3
K_BASIS, L_BASIS = 1.0, 0.5
K_GITTER, L_GITTER = [0.5, 1.0, 2.0], [0.3, 0.5, 0.7]
MIN_SD = 0.01                                                      # wie v2.5 (Rohwerte, Standardabweichung mit n - 1)
MC_LAEUFE, MC_SEED, MC_SPANNE = 1000, 1600, (0.75, 1.25)
EFA_PA_ITER, EFA_PA_SEED = 200, 42                                 # wie step03_efa_pca.py in v2.5
KONFIG = ["m_ws", "m_trend", "m_ec", "m_latent"]
ECHTE_LAEUFE = {"3fd98dc6f67d04eb", "38599d60e3078379"}            # model_results.pkl des berichteten Laufs, Phase 1 und 2

assert sorted(sum(DIMENSIONEN.values(), [])) == sorted(PLAETZE)
assert EINGESCHRAENKT == ["EO3", "PE2", "DS1", "DS2", "DI1", "DI2"]


def uhr(t0, text):
    print(f"[{time.time() - t0:7.1f} s] {text}", flush=True)


def runde(x, k=6):
    return None if x is None or x != x else round(float(x), k)


def pruefe_freigabe(D):
    """Auf den echten Daten erst nach dem Tag (Protokoll 3, Punkt j); synthetische Daten sind frei."""
    if D["quellen"].get("model_results.pkl") in ECHTE_LAEUFE and os.environ.get("V26_GETAGGT") != "v2.6-frozen":
        raise SystemExit("Echte Daten: Zugehoerigkeiten erst nach dem Tag v2.6-frozen (V26_GETAGGT=v2.6-frozen)")
    if not phs.ENTWICKLUNG and phs.PHASE == 1:
        raise SystemExit("Trockenlauf mit Phase 1: Zugehoerigkeiten werden vor dem Tag nicht gerechnet")


# ---------------------------------------------------------------- Zugehoerigkeitsschicht (wie v2.5)

def robust_z(s, eps=1e-9):
    """Wie step02_memberships.robust_z: (x - Median) / (IQR / 2), lineare Quantile."""
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    return (s - med) / max(iqr / 2.0, eps)


def klassisch_z(s):
    """Wie StandardScaler in step02_indicators.compute_dimension_scores: Standardabweichung mit n im Nenner."""
    sd = s.std(ddof=0)
    return (s - s.mean()) / (sd if sd > 0 else 1.0)


def sigmoid(x, k):
    return 1.0 / (1.0 + np.exp(-k * x))


def fassungswerte(D):
    """Topics x Plaetze mit den Rohwerten der eingefrorenen Fassungen; fehlende Werte durch den Median der Phase ersetzt
    (wie v2.5), die Ersetzungen werden gezaehlt."""
    roh = D["ref"].loc[D["tids"], [FASSUNG[p] for p in PLAETZE]].astype(float)
    roh.columns = PLAETZE
    fehlt = roh.isna()
    X = roh.copy()
    for p in PLAETZE:
        if fehlt[p].any():
            med = X[p].median()
            X.loc[fehlt[p], p] = 0.5 if pd.isna(med) else med
    return X, fehlt


def gueltige(X):
    """Fassungen mit Standardabweichung ueber 0,01 (Rohwerte, wie v2.5); dieselbe Menge gilt in allen Varianten."""
    return [p for p in X.columns if X[p].std() > MIN_SD]


def zugehoerigkeit(X, gueltig, k=K_BASIS, lam=L_BASIS, gewicht=None, standard="klassisch", ohne=(), eo_nur=None,
                   dimensionen=None, ec_liste=None):
    """Vier Zugehoerigkeiten, Margin und dominante Konfiguration je Topic (Schicht von v2.5).

    gewicht: Gewicht je Platz (Monte-Carlo), sonst gleiche Gewichte; standard: Standardisierung der Fassungen
    'klassisch' (z wie StandardScaler) oder 'robust' (Median, halber IQR); ohne: weggelassene Plaetze; eo_nur: Plaetze der
    Offenheit fuer die Variante 'nur EO1 bis EO3'. dimensionen und ec_liste nur fuer den Test gegen v2.5."""
    dimensionen = DIMENSIONEN if dimensionen is None else dimensionen
    w = pd.Series(1.0, index=X.columns) if gewicht is None else pd.Series(gewicht, dtype=float)
    Z = X.apply(klassisch_z if standard == "klassisch" else robust_z, axis=0)
    dims = pd.DataFrame(index=X.index)
    for d, ps in dimensionen.items():
        ps = [p for p in ps if p in gueltig and p not in ohne]
        if d == "Epistemische Offenheit" and eo_nur is not None:
            ps = [p for p in ps if p in eo_nur]
        dims[d] = (Z[ps] * w[ps]).sum(axis=1) / w[ps].sum() if ps else 0.0
    zd = dims.apply(robust_z, axis=0)
    ec = [p for p in (EC if ec_liste is None else ec_liste) if p not in ohne]
    zec = X[ec].apply(robust_z, axis=0)
    kern = zd[KERN].mean(axis=1)
    m = pd.DataFrame({"m_ws": sigmoid(kern, k),
                      "m_trend": sigmoid(-kern + lam * zd[WP], k),
                      "m_ec": sigmoid((zec * w[ec]).sum(axis=1) / w[ec].sum(), k),
                      "m_latent": sigmoid(-zd[KERN + [WP]].mean(axis=1), k)}, index=X.index)
    s = np.sort(m[KONFIG].values, axis=1)
    m["margin"] = s[:, -1] - s[:, -2]
    m["dominant"] = m[KONFIG].idxmax(axis=1)
    return m


def residualisiert(X, logn):
    """Variante: alle Fassungen ausser PE1 durch die Residuen ihrer OLS-Regression auf log n (mit Achsenabschnitt)."""
    A = np.column_stack([np.ones(len(X)), logn.loc[X.index].to_numpy(float)])
    R = X.copy()
    for p in X.columns:
        if p == "PE1":
            continue
        y = X[p].to_numpy(float)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        R[p] = y - A @ beta
    return R


def varianten():
    """Die 30 Varianten des Multiversums in fester Reihenfolge (3.3)."""
    v = [(f"k={k} lambda={lam}", {"k": k, "lam": lam}) for k, lam in itertools.product(K_GITTER, L_GITTER)]
    v += [(f"ohne {p}", {"ohne": (p,)}) for p in PLAETZE]
    v += [("ohne eingeschraenkte", {"ohne": tuple(EINGESCHRAENKT)}),
          ("Offenheit nur EO1 bis EO3", {"eo_nur": EO_NUR}),
          ("Residualisierung auf log n", {"resid": True})]
    assert len(v) == 30
    return v


def multiversum(X, gueltig, logn, basis):
    dom = pd.DataFrame(index=X.index)
    zeilen = []
    for name, par in varianten():
        par = dict(par)
        Xv = residualisiert(X, logn) if par.pop("resid", False) else X
        m = zugehoerigkeit(Xv, gueltig, **par)
        dom[name] = m["dominant"]
        zeilen.append({"variante": name, "uebereinstimmung": float((m["dominant"] == basis["dominant"]).mean()),
                       **{f"anteil_{c}": float((m["dominant"] == c).mean()) for c in KONFIG}})
    tab = pd.DataFrame(zeilen)
    stabil = dom.eq(basis["dominant"], axis=0).all(axis=1)
    zus = {"varianten": len(tab), "anteil_topics_stabil_in_allen": runde(stabil.mean()),
           "uebereinstimmung_median": runde(tab["uebereinstimmung"].median()),
           "uebereinstimmung_min": runde(tab["uebereinstimmung"].min()),
           "uebereinstimmung_max": runde(tab["uebereinstimmung"].max())}
    return dom, tab, stabil, zus


def mc_baender(X, gueltig, basis):
    """1.000 Laeufe (Seed 1600). Je Lauf in dieser Reihenfolge: 18 Gewichte U(0,75; 1,25) in der Reihenfolge der Plaetze
    (innerhalb der Dimension und in der EC-Zugehoerigkeit normiert), k und lambda gleichverteilt aus den Gittern, dann die
    Standardisierung der Fassungen (robust mit Wahrscheinlichkeit 1/2, sonst klassisch)."""
    rng = np.random.default_rng(MC_SEED)
    werte = {c: np.empty((MC_LAEUFE, len(X))) for c in KONFIG}
    gleich = np.zeros(len(X))
    protokoll = []
    for r in range(MC_LAEUFE):
        u = rng.uniform(*MC_SPANNE, size=len(PLAETZE))
        k = float(rng.choice(K_GITTER))
        lam = float(rng.choice(L_GITTER))
        standard = "robust" if rng.random() < 0.5 else "klassisch"
        m = zugehoerigkeit(X, gueltig, k=k, lam=lam, gewicht=dict(zip(PLAETZE, u)), standard=standard)
        for c in KONFIG:
            werte[c][r] = m[c].to_numpy(float)
        gleich += (m["dominant"] == basis["dominant"]).to_numpy()
        protokoll.append((k, lam, standard))
    tab = pd.DataFrame(index=X.index)
    for c in KONFIG:
        tab[f"{c}_median"] = np.median(werte[c], axis=0)
        tab[f"{c}_p5"] = np.percentile(werte[c], 5, axis=0)
        tab[f"{c}_p95"] = np.percentile(werte[c], 95, axis=0)
    tab["anteil_gleich_dominant"] = gleich / MC_LAEUFE
    zus = {"laeufe": MC_LAEUFE, "seed": MC_SEED,
           "anteil_gleich_mittel": runde(tab["anteil_gleich_dominant"].mean()),
           "anteil_gleich_median": runde(tab["anteil_gleich_dominant"].median()),
           "topics_ab_90_prozent_gleich": runde((tab["anteil_gleich_dominant"] >= 0.9).mean()),
           "topics_unter_50_prozent_gleich": runde((tab["anteil_gleich_dominant"] < 0.5).mean()),
           "anteil_robust": runde(np.mean([s == "robust" for _, _, s in protokoll])),
           "verteilung_k": {str(k): int(sum(1 for x, _, _ in protokoll if x == k)) for k in K_GITTER},
           "verteilung_lambda": {str(lam): int(sum(1 for _, x, _ in protokoll if x == lam)) for lam in L_GITTER}}
    return tab, zus


# ---------------------------------------------------------------- EFA, nur berichtet (Einstellungen wie v2.5)

def _kompat_sklearn():
    """Wie step03_efa_pca.py: factor_analyzer 0.5.1 ruft check_array(force_all_finite=...), scikit-learn ab 1.8 nicht mehr."""
    import inspect
    import factor_analyzer.factor_analyzer as fa_modul
    import sklearn.utils.validation as v
    if "force_all_finite" in inspect.signature(v.check_array).parameters:
        return False
    orig = v.check_array

    def check_array(*args, force_all_finite=None, **kwargs):
        if force_all_finite is not None and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = force_all_finite
        return orig(*args, **kwargs)

    fa_modul.check_array = check_array
    return True


def kmo(c):
    n = c.shape[0]
    try:
        inv = np.linalg.inv(c)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(c)
    d = np.diag(1.0 / np.sqrt(np.diag(inv)))
    pk = -d @ inv @ d
    np.fill_diagonal(pk, 0)
    r2 = np.sum(c ** 2) - n
    return float(r2 / (r2 + np.sum(pk ** 2)))


def bartlett(c, n_obs):
    from scipy.stats import chi2
    p = c.shape[0]
    det = np.linalg.det(c)
    if det <= 0:
        return 999.0, 0.0
    x2 = -(n_obs - 1 - (2 * p + 5) / 6) * np.log(det)
    return float(x2), float(1 - chi2.cdf(x2, p * (p - 1) / 2))


def parallelanalyse(n_obs, n_var):
    rng = np.random.RandomState(EFA_PA_SEED)
    eig = np.zeros((EFA_PA_ITER, n_var))
    for i in range(EFA_PA_ITER):
        rc = np.corrcoef(rng.normal(size=(n_obs, n_var)), rowvar=False)
        eig[i] = np.sort(np.linalg.eigvalsh(rc))[::-1]
    return eig.mean(axis=0), np.percentile(eig, 95, axis=0)


def n_konsekutiv(e, s):
    n = 0
    for a, b in zip(e, s):
        if a > b:
            n += 1
        else:
            break
    return n


def _phi(fa, k):
    phi, st = getattr(fa, "phi_", None), getattr(fa, "structure_", None)
    if phi is None:
        return np.eye(k)
    if st is None or k == 1:
        return np.asarray(phi)
    beste, rest = None, np.inf
    for perm in itertools.permutations(range(k)):
        kand = phi[np.ix_(perm, perm)]
        r = float(np.abs(st - fa.loadings_ @ kand).max())
        if r < rest:
            beste, rest = kand, r
    return beste


def efa_bericht(X):
    from factor_analyzer import FactorAnalyzer
    _kompat_sklearn()
    sd = X.std()
    aus = [p for p in X.columns if not sd[p] >= MIN_SD]
    Xv = X.drop(columns=aus)
    Z = Xv.apply(klassisch_z, axis=0)
    c = np.corrcoef(Z.to_numpy(float), rowvar=False)
    eig = np.sort(np.linalg.eigvalsh(c))[::-1]
    pa_m, pa_95 = parallelanalyse(len(Xv), Xv.shape[1])
    n_par = n_konsekutiv(eig, pa_m)
    n_ref = max(n_par, 3)
    n_cmp = 4 if n_ref == 5 else 5
    x2, p = bartlett(c, len(Xv))
    erg = {"fassungen": list(Xv.columns), "ausgeschlossen": aus, "kmo": runde(kmo(c)), "bartlett_chi2": runde(x2),
           "bartlett_p": runde(p, 8), "eigenwerte": [runde(e) for e in eig], "n_kaiser": int((eig > 1.0).sum()),
           "n_parallel": n_par, "n_parallel_p95": n_konsekutiv(eig, pa_95),
           "pa_schwellen_mittel": [runde(e) for e in pa_m], "loesungen": {}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for k in (n_ref, n_cmp):
            fa = FactorAnalyzer(n_factors=k, method="minres", rotation="oblimin")
            fa.fit(Z.to_numpy(float))
            ph = _phi(fa, k)
            komm = (fa.loadings_ * (fa.loadings_ @ ph)).sum(axis=1)
            erg["loesungen"][str(k)] = {
                "rolle": "Referenz" if k == n_ref else "Vergleich",
                "muster": {f: [runde(x, 4) for x in zeile] for f, zeile in zip(Xv.columns, fa.loadings_)},
                "phi": [[runde(x, 4) for x in zeile] for zeile in ph],
                "kommunalitaeten": {f: runde(x, 4) for f, x in zip(Xv.columns, komm)},
                "gemeinsame_varianz": runde(float(komm.sum() / len(komm)))}
    return erg


# ---------------------------------------------------------------- Zusatz zur Offenheit (3.4, Punkt g)

EO_FASSUNGEN = [FASSUNG[p] for p in ["EO1", "EO2", "EO3", "EO4", "EO5"]]


def ses_offenheit(D):
    """Je Topic und EO-Fassung: (Wert minus Mittel der 50 Permutationen) geteilt durch deren Standardabweichung (n - 1),
    und das Mittel ueber die fuenf Fassungen (nullreferenzierte Offenheit)."""
    nulls = sorted(p.stem for p in bk.OUT.glob("null_r*.csv"))
    if len(nulls) != 50:
        raise SystemExit(f"Nullmodell unvollstaendig: {len(nulls)} Permutationen")
    tabs = [opt.tabelle(name)[EO_FASSUNGEN] for name in nulls]
    st = pd.concat(tabs, keys=range(len(tabs)))
    mu = st.groupby(level=1).mean()
    sd = st.groupby(level=1).std()
    ref = D["ref"][EO_FASSUNGEN]
    ses = ((ref - mu.reindex(ref.index)) / sd.reindex(ref.index).replace(0, np.nan)).loc[D["tids"]]
    ses["EO_SES_MITTEL"] = ses[EO_FASSUNGEN].mean(axis=1)
    return ses


# ---------------------------------------------------------------- Ablauf

def rechnen(t0):
    D = opt.lade_daten()
    pruefe_freigabe(D)
    X, fehlt = fassungswerte(D)
    gueltig = gueltige(X)
    n = D["n"].loc[X.index]
    logn = np.log(n.astype(float))
    basis = zugehoerigkeit(X, gueltig)
    uhr(t0, f"Grundfassung: {len(X)} Topics, gueltige Fassungen {len(gueltig)} von {len(PLAETZE)}")
    f_v25 = bt.REF / "signal_memberships.csv"
    v25 = pd.read_csv(phs.lies(f_v25, "Zugehoerigkeiten des berichteten Laufs v2.5 (Wechsel je Topic)"), index_col=0)
    v25_dom = v25[KONFIG].idxmax(axis=1).reindex(X.index)
    dom, tab_mv, stabil, zus_mv = multiversum(X, gueltig, logn, basis)
    uhr(t0, "Multiversum mit 30 Varianten gerechnet")
    mc, zus_mc = mc_baender(X, gueltig, basis)
    uhr(t0, f"Monte-Carlo-Baender: {MC_LAEUFE} Laeufe")
    efa = efa_bericht(X)
    uhr(t0, f"EFA: KMO {efa['kmo']}, Faktoren nach Parallelanalyse {efa['n_parallel']}, nach Kaiser {efa['n_kaiser']}")
    ses = ses_offenheit(D)
    uhr(t0, "Zusatz zur Offenheit: standardisierte Abstaende vom Nullmodell")
    OUT.mkdir(parents=True, exist_ok=True)
    zt = pd.concat([n.rename("n"), X.add_prefix("wert_"), fehlt.add_prefix("ersetzt_").astype(int), basis], axis=1)
    zt["dominant_v25"] = v25_dom
    zt["gewechselt"] = (zt["dominant"] != zt["dominant_v25"]).astype(int)
    zt["stabil_im_multiversum"] = stabil.astype(int)
    zt.rename_axis("topic").to_csv(OUT / "zugehoerigkeiten.csv")
    dom.rename_axis("topic").to_csv(OUT / "multiversum_dominant.csv")
    tab_mv.to_csv(OUT / "multiversum_varianten.csv", index=False)
    mc.rename_axis("topic").to_csv(OUT / "mc_baender.csv")
    ses.rename_axis("topic").to_csv(OUT / "ses_offenheit.csv")
    kreuz = pd.crosstab(zt["dominant_v25"], zt["dominant"]).reindex(index=KONFIG, columns=KONFIG, fill_value=0)
    erg = {"phase": phs.PHASE, "topics": int(len(X)), "parameter": {"k": K_BASIS, "lambda": L_BASIS},
           "fassungen": FASSUNG, "eingeschraenkt": EINGESCHRAENKT, "gueltige_fassungen": gueltig,
           "ersetzt_durch_median": {p: int(fehlt[p].sum()) for p in PLAETZE if fehlt[p].any()},
           "anteile": {c: runde((basis["dominant"] == c).mean()) for c in KONFIG},
           "besetzung": {c: int((basis["dominant"] == c).sum()) for c in KONFIG},
           "margin": {"median": runde(basis["margin"].median()), "anteil_unter_0_10": runde((basis["margin"] < 0.10).mean())},
           "v25": {"besetzung": {c: int((v25_dom == c).sum()) for c in KONFIG},
                   "anteil_gewechselt": runde(zt["gewechselt"].mean()), "kreuztabelle_v25_zeilen": kreuz.values.tolist()},
           "multiversum": zus_mv, "mc": zus_mc, "efa": efa,
           "offenheit_ses": {"topics": int(ses["EO_SES_MITTEL"].notna().sum()),
                             "median": runde(ses["EO_SES_MITTEL"].median())},
           "quellen": {**D["quellen"], "signal_memberships.csv": phs.sha256(f_v25)[:16]}, "schalter": phs.beschreibung()}
    (OUT / "entblinden_ergebnis.json").write_text(json.dumps(erg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: erg[k] for k in ("phase", "topics", "gueltige_fassungen", "besetzung", "margin", "v25",
                                          "multiversum", "mc")}, ensure_ascii=False))
    print(json.dumps({k: efa[k] for k in ("kmo", "n_kaiser", "n_parallel", "n_parallel_p95")}, ensure_ascii=False))
    return erg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("teil", choices=["rechnen"])
    ap.parse_args()
    rechnen(time.time())


if __name__ == "__main__":
    main()
