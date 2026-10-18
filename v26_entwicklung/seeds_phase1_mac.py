#!/usr/bin/env python3
"""V2.6, Kriterium K3: Topicmodelle weiterer Seeds fuer Phase 1 (Phase 2 nur in der Bestaetigung, 3.4). Auf Bens Mac.

Rechnet UMAP und HDBSCAN mit den Parametern des berichteten Laufs v2.5 neu (config.py; Aufruf wie in
step05_sensitivity.random_seed_stability), fuer die Seeds 0, 1, 7 und 2026 und auf den SBERT-Einbettungen des
Referenzlaufs. Es wird nichts neu eingebettet und nichts aus dem Netz geladen. Vorab prueft ein Kontrolllauf mit
Seed 42, ob die Umgebung die Referenzlabels reproduziert. Gespeichert werden je Seed nur die Labels und die
reduzierten Einbettungen (.npy) sowie die Pruefdatei seeds_phase<N>.json; output_*/ und *.npy stehen in .gitignore.
Phase und Ausgabeordner waehlt phase.py (V26_PHASE); fuer Phase 1 ohne V26_AUSGABE wie bisher.

Aufruf auf dem Mac, in der Python-Umgebung, mit der v2.5 gerechnet wurde (umap-learn, hdbscan):
    cd ".../F3_Prototyp/Weak_Signal_Membership_Pipeline_V2.6/v26_entwicklung"
    python3 seeds_phase1_mac.py
Fuer Phase 2 (nur in der Bestaetigung, nach 'bestaetigung.py start'):
    V26_PHASE=2 V26_GETAGGT=v2.6-frozen python3 seeds_phase1_mac.py
Laufzeit je Seed einige Minuten. Nach einem Abbruch setzt ein erneuter Aufruf beim fehlenden Seed fort.
"""
import json
import pickle
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

HIER = Path(__file__).resolve().parent
V26 = HIER.parent
sys.path.insert(0, str(V26))
sys.path.insert(0, str(HIER))
import config  # noqa: E402
import phase as phs  # noqa: E402  (Phasenschalter, Protokoll 3; Phase 1 wie bisher)

REF = phs.REF
OUT = phs.ordner("output_v26_seeds/phase1", "seeds")
SEEDS = [0, 1, 7, 2026]
KONTROLLE = 42


def version(paket):
    try:
        return metadata.version(paket)
    except metadata.PackageNotFoundError:
        return None


def modell(emb, seed):
    import hdbscan
    import umap
    red = umap.UMAP(n_components=config.UMAP_N_COMPONENTS, n_neighbors=config.UMAP_N_NEIGHBORS,
                    min_dist=config.UMAP_MIN_DIST, metric=config.UMAP_METRIC, random_state=seed).fit_transform(emb)
    lab = hdbscan.HDBSCAN(min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE, min_samples=config.HDBSCAN_MIN_SAMPLES,
                          cluster_selection_method=config.HDBSCAN_CLUSTER_METHOD).fit_predict(red)
    return np.asarray(red, dtype=np.float32), np.asarray(lab, dtype=np.int32)


def main():
    from sklearn.metrics import adjusted_rand_score, v_measure_score

    OUT.mkdir(parents=True, exist_ok=True)
    with open(phs.lies(REF / "model_results.pkl", "Einbettungen und Labels des berichteten Laufs"), "rb") as f:
        md = pickle.load(f)
    emb = np.asarray(md["embeddings_sbert"])
    basis = np.asarray(md["labels"])
    f_v25 = REF / "sensitivity_seeds.csv"
    v25 = (pd.read_csv(phs.lies(f_v25, "Seed-Laeufe des berichteten Laufs (Vergleich)")).set_index("seed")
           if f_v25.exists() else pd.DataFrame({"seed": [], "n_topics": [], "n_noise": [], "ari": []}).set_index("seed"))
    pfad = OUT / f"seeds_{phs.P}.json"
    res = json.loads(pfad.read_text(encoding="utf-8")) if pfad.exists() else {"seeds": {}}
    res["versionen"] = {p: version(p) for p in ("umap-learn", "hdbscan", "numba", "pynndescent", "numpy",
                                                "scikit-learn", "scipy")}
    res["python"] = platform.python_version()
    res["plattform"] = platform.platform()
    res["parameter"] = {"umap_n_components": config.UMAP_N_COMPONENTS, "umap_n_neighbors": config.UMAP_N_NEIGHBORS,
                        "umap_min_dist": config.UMAP_MIN_DIST, "umap_metric": config.UMAP_METRIC,
                        "hdbscan_min_cluster_size": config.HDBSCAN_MIN_CLUSTER_SIZE,
                        "hdbscan_min_samples": config.HDBSCAN_MIN_SAMPLES,
                        "hdbscan_cluster_selection": config.HDBSCAN_CLUSTER_METHOD}
    res["referenz"] = {"n_dok": int(len(basis)), "n_topics": int(len(set(basis[basis >= 0].tolist()))),
                       "n_rauschen": int((basis < 0).sum())}

    def schreibe():  # erst vollstaendig schreiben, dann ersetzen
        tmp = pfad.with_name(pfad.name + ".tmp")
        tmp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(pfad)

    def speichere(name, arr):
        tmp = OUT / (name + ".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(OUT / (name + ".npy"))

    def pruefe(seed, lab, sekunden):
        z = {"n_topics": int(len(set(lab[lab >= 0].tolist()))), "n_rauschen": int((lab < 0).sum()),
             "ari": float(adjusted_rand_score(basis, lab)), "v_measure": float(v_measure_score(basis, lab)),
             "sekunden": sekunden}
        if seed in v25.index:
            z["v25"] = {"n_topics": int(v25.loc[seed, "n_topics"]), "n_rauschen": int(v25.loc[seed, "n_noise"]),
                        "ari": float(v25.loc[seed, "ari"])}
            z["wie_v25"] = bool(z["n_topics"] == z["v25"]["n_topics"] and z["n_rauschen"] == z["v25"]["n_rauschen"]
                                and abs(z["ari"] - z["v25"]["ari"]) < 1e-9)
        return z

    if "kontrolle_seed42" not in res:
        t0 = time.time()
        red, lab = modell(emb, KONTROLLE)
        speichere("labels_seed42_kontrolle", lab)
        speichere("reduziert_seed42_kontrolle", red)
        z = pruefe(KONTROLLE, lab, round(time.time() - t0, 1))
        z["labels_gleich_referenz"] = bool(np.array_equal(lab, basis))
        res["kontrolle_seed42"] = z
        schreibe()
        print(f"Kontrolle Seed 42: ARI {z['ari']:.6f}, Labels gleich Referenz: {z['labels_gleich_referenz']} "
              f"({z['sekunden']} s)", flush=True)
        if not z["labels_gleich_referenz"]:
            print("HINWEIS: Diese Umgebung reproduziert die Referenzlabels nicht. Die Seeds laufen trotzdem; "
                  "K3 vergleicht dann gegen den Kontrolllauf aus derselben Umgebung.", flush=True)
    for s in SEEDS:
        if str(s) in res["seeds"] and (OUT / f"labels_seed{s}.npy").exists():
            print(f"Seed {s} liegt schon vor", flush=True)
            continue
        t0 = time.time()
        red, lab = modell(emb, s)
        speichere(f"labels_seed{s}", lab)
        speichere(f"reduziert_seed{s}", red)
        res["seeds"][str(s)] = pruefe(s, lab, round(time.time() - t0, 1))
        schreibe()
        z = res["seeds"][str(s)]
        print(f"Seed {s}: {z['n_topics']} Topics, {z['n_rauschen']} Rauschen, ARI {z['ari']:.4f}, "
              f"wie v2.5: {z.get('wie_v25')} ({z['sekunden']} s)", flush=True)
    print(f"Fertig. Pruefdatei: {pfad}", flush=True)


if __name__ == "__main__":
    main()
