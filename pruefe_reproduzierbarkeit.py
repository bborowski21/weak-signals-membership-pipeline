#!/usr/bin/env python3
"""
Wo genau weicht ein neuer Lauf vom berichteten ab?

Der Stabilitaetslauf zur Phasengrenze hat als Nebenbefund ergeben, dass die
Kontrollzelle den berichteten Lauf nicht exakt reproduziert: 143 statt 146
Topics in Phase 1, 269 statt 265 in Phase 2. Dieses Skript grenzt ein, in
welchem Schritt die Abweichung entsteht. Es aendert nichts und schreibt nur
eine kleine Ergebnisdatei.

`model_results.pkl` enthaelt drei Dinge aus dem berichteten Lauf:
`embeddings_sbert` (Eingang von UMAP), `embeddings_reduced` (Ausgang von UMAP
und Eingang von HDBSCAN) und `labels` (Ausgang von HDBSCAN). Damit laesst sich
jede Stufe einzeln gegen ihre eigene gespeicherte Ausgabe halten.

  Test A  HDBSCAN auf den GESPEICHERTEN reduzierten Vektoren.
          Ergibt es die gespeicherten Labels, ist HDBSCAN reproduzierbar.
  Test B  UMAP auf den gespeicherten SBERT-Vektoren.
          Stimmt das Ergebnis mit den gespeicherten reduzierten Vektoren
          ueberein, ist UMAP reproduzierbar.
  Test C  HDBSCAN auf dem frisch gerechneten UMAP-Ergebnis.
          Das ist der Weg, den der Stabilitaetslauf geht.

Lesart:
  A gruen, B rot   -> die Abweichung entsteht in UMAP.
  A rot            -> HDBSCAN selbst reproduziert nicht (Version, Threading).
  A gruen, B gruen -> beide Stufen reproduzieren; dann liegt die Abweichung
                      weder an der Umgebung noch an diesen beiden Schritten,
                      sondern am Code, der den berichteten Lauf erzeugt hat.

Aufruf:
    python pruefe_reproduzierbarkeit.py            # Phase 1
    python pruefe_reproduzierbarkeit.py --phase 2
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def vergleiche_labels(a: np.ndarray, b: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score
    return {
        "topics_a": int(len(set(a[a >= 0]))),
        "topics_b": int(len(set(b[b >= 0]))),
        "noise_a": int((a == -1).sum()),
        "noise_b": int((b == -1).sum()),
        "identisch": bool(np.array_equal(a, b)),
        "adjusted_rand": float(adjusted_rand_score(a, b)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=1, choices=(1, 2))
    args = ap.parse_args()
    out_dir = BASE_DIR / f"output_phase{args.phase}"

    from config import (UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
                        UMAP_METRIC, HDBSCAN_MIN_CLUSTER_SIZE,
                        HDBSCAN_MIN_SAMPLES, HDBSCAN_CLUSTER_METHOD)
    import umap
    import hdbscan

    log(f"Lade {out_dir.name}/model_results.pkl ...")
    with open(out_dir / "model_results.pkl", "rb") as f:
        m = pickle.load(f)
    labels_ref = np.asarray(m["labels"])
    emb = np.asarray(m["embeddings_sbert"])
    red_ref = np.asarray(m["embeddings_reduced"])
    log(f"  Vektoren {emb.shape} {emb.dtype} · reduziert {red_ref.shape} "
        f"{red_ref.dtype} · {len(set(labels_ref[labels_ref >= 0]))} Topics")

    ergebnis = {"phase": args.phase,
                "topics_berichtet": int(len(set(labels_ref[labels_ref >= 0]))),
                "noise_berichtet": int((labels_ref == -1).sum())}

    def hdb(x):
        c = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                            min_samples=HDBSCAN_MIN_SAMPLES,
                            metric="euclidean",
                            cluster_selection_method=HDBSCAN_CLUSTER_METHOD,
                            prediction_data=True)
        return c.fit_predict(x)

    log("Test A: HDBSCAN auf den GESPEICHERTEN reduzierten Vektoren ...")
    t0 = time.time()
    lab_a = hdb(red_ref)
    ergebnis["A"] = vergleiche_labels(labels_ref, lab_a)
    ergebnis["A"]["sekunden"] = round(time.time() - t0, 1)
    log(f"  Topics {ergebnis['A']['topics_b']} gegen "
        f"{ergebnis['A']['topics_a']} · Rauschen {ergebnis['A']['noise_b']} "
        f"gegen {ergebnis['A']['noise_a']} · ARI "
        f"{ergebnis['A']['adjusted_rand']:.6f} · "
        f"{'IDENTISCH' if ergebnis['A']['identisch'] else 'ABWEICHEND'}")

    log("Test B: UMAP auf den gespeicherten SBERT-Vektoren ...")
    t0 = time.time()
    reducer = umap.UMAP(n_components=UMAP_N_COMPONENTS,
                        n_neighbors=UMAP_N_NEIGHBORS,
                        min_dist=UMAP_MIN_DIST,
                        metric=UMAP_METRIC,
                        random_state=42,
                        low_memory=False)
    red_neu = reducer.fit_transform(emb)
    d = np.abs(red_neu - red_ref)
    ergebnis["B"] = {
        "identisch": bool(np.array_equal(red_neu, red_ref)),
        "max_abweichung": float(d.max()),
        "mittlere_abweichung": float(d.mean()),
        "anteil_zellen_gleich": float((d == 0).mean()),
        "korrelation_je_achse": [float(np.corrcoef(red_neu[:, k], red_ref[:, k])[0, 1])
                                 for k in range(red_ref.shape[1])],
        "sekunden": round(time.time() - t0, 1),
    }
    log(f"  {'IDENTISCH' if ergebnis['B']['identisch'] else 'ABWEICHEND'} · "
        f"groesste Abweichung {ergebnis['B']['max_abweichung']:.6g} · "
        f"mittlere {ergebnis['B']['mittlere_abweichung']:.6g} · "
        f"gleiche Zellen {ergebnis['B']['anteil_zellen_gleich']:.1%}")

    log("Test C: HDBSCAN auf dem frisch gerechneten UMAP-Ergebnis ...")
    t0 = time.time()
    lab_c = hdb(red_neu)
    ergebnis["C"] = vergleiche_labels(labels_ref, lab_c)
    ergebnis["C"]["sekunden"] = round(time.time() - t0, 1)
    log(f"  Topics {ergebnis['C']['topics_b']} gegen "
        f"{ergebnis['C']['topics_a']} · Rauschen {ergebnis['C']['noise_b']} "
        f"gegen {ergebnis['C']['noise_a']} · ARI "
        f"{ergebnis['C']['adjusted_rand']:.6f}")

    if ergebnis["A"]["identisch"] and not ergebnis["B"]["identisch"]:
        urteil = ("UMAP ist der Ort der Abweichung. HDBSCAN reproduziert den "
                  "berichteten Lauf aus den gespeicherten reduzierten Vektoren "
                  "exakt.")
    elif not ergebnis["A"]["identisch"]:
        urteil = ("Schon HDBSCAN reproduziert nicht. Dann liegt es an der "
                  "hdbscan-Version oder an Threading, nicht am uebrigen Code.")
    elif ergebnis["B"]["identisch"]:
        urteil = ("Beide Stufen reproduzieren exakt. Die Abweichung des "
                  "Stabilitaetslaufs kann dann nicht aus Umgebung oder "
                  "Nichtdeterminismus stammen; zu pruefen ist der Code, der "
                  "den berichteten Lauf erzeugt hat.")
    else:
        urteil = "Unerwartete Kombination, Zahlen von Hand ansehen."
    ergebnis["urteil"] = urteil

    out = BASE_DIR / f"reproduzierbarkeit_phase{args.phase}.json"
    out.write_text(json.dumps(ergebnis, indent=2), encoding="utf-8")
    log("=" * 66)
    log(urteil)
    log(f"Geschrieben: {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
