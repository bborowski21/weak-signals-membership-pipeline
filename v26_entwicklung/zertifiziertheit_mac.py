#!/usr/bin/env python3
"""V2.6, EO4 A1: Satzweise Zertifiziertheit nach Pei und Jurgens 2021 fuer Phase 1. Auf Bens Mac auszufuehren.

Das Modell pedropei/sentence-level-certainty wird aus dem lokalen Ordner F3_Prototyp/Modelle/sentence-level-certainty
geladen (kein Netz). Die Inferenz folgt dem Paket certainty-estimator 1.6: tokenizer.encode mit Sondertokens, hoechstens
60 Tokens, Auffuellen mit dem Pad-Token auf 60, keine Attention-Maske, num_labels = 1, Rechnung auf der CPU in float32.

Bewertet wird jeder Satz der Abstracts mit mindestens 4 Woertern. Gespeichert werden je Satz nur abgeleitete Groessen
(Dokumentindex, Satznummer, Satzzahl des Abstracts, Wortzahl, Befund-Hinweis, Wert), keine Texte; die Auswahl der Saetze
fuer den Indikator (alle Saetze oder Befundsaetze) wird erst danach angewandt. Zusaetzlich entsteht ein Blatt mit 150
zufaelligen Saetzen fuer Bens fachliche Pruefung (Text und leere Bewertungsspalte, die Modellwerte in einer getrennten
Datei); beides bleibt auf dem Mac in einem Ordner, den .gitignore erfasst.

Aufruf auf dem Mac, in der Python-Umgebung der Pipeline (torch, transformers), im Ordner v26_entwicklung:
    python3 zertifiziertheit_mac.py
Nach einem Abbruch setzt ein erneuter Aufruf beim fehlenden Block fort.
"""
import hashlib
import json
import platform
import re
import sys
import time
from importlib import metadata
from pathlib import Path

import numpy as np

HIER = Path(__file__).resolve().parent
V26 = HIER.parent
sys.path.insert(0, str(HIER))
import backtest_phase1 as bt  # noqa: E402  (laedt Phase 1 mit denselben Filtern und derselben Zeilenfolge wie die Labels)

MODELL = V26.parent / "Modelle" / "sentence-level-certainty"
OUT = V26 / "output_v26_zertifiziertheit" / "phase1"
MAX_LEN = 60
BATCH = 128
MIN_WOERTER = 4
BLOCK = 20000          # Saetze je Zwischenstand
N_PRUEF = 150
SEED_PRUEF = 4040

ABK = ["e.g.", "i.e.", "et al.", "Fig.", "Figs.", "Eq.", "Eqs.", "Ref.", "Refs.", "vs.", "approx.", "ca.", "No.",
       "Phys.", "Rev.", "Lett.", "Sec.", "Dr.", "Prof.", "cf.", "resp.", "Vol.", "pp."]
BEFUND = re.compile(r"\b(show|shows|showed|shown|find|finds|found|demonstrate|demonstrates|demonstrated|observe|observes|"
                    r"observed|reveal|reveals|revealed|indicate|indicates|indicated|suggest|suggests|suggested|conclude|"
                    r"concludes|concluded|confirm|confirms|confirmed|prove|proves|proved|proven|results?)\b", re.I)


def saetze(text):
    t = " ".join(str(text).split())
    for i, a in enumerate(ABK):
        t = t.replace(a, a.replace(".", f"\u0001{i}\u0001"))
    teile = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", t)
    aus = []
    for s in teile:
        for i, a in enumerate(ABK):
            s = s.replace(a.replace(".", f"\u0001{i}\u0001"), a)
        s = s.strip()
        if s:
            aus.append(s)
    return aus


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def version(paket):
    try:
        return metadata.version(paket)
    except metadata.PackageNotFoundError:
        return None


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    noetig = ["config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]
    fehlend = [f for f in noetig if not (MODELL / f).exists()]
    if fehlend:
        raise SystemExit(f"Im Ordner {MODELL} fehlen: {', '.join(fehlend)}")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df, labels, _, _ = bt.lade()
    doc, nr, anz, woerter, befund, texte = [], [], [], [], [], []
    for i, ab in enumerate(df["Abstract"].fillna("").astype(str)):
        ss = [s for s in saetze(ab) if len(s.split()) >= MIN_WOERTER]
        for j, s in enumerate(ss):
            doc.append(i)
            nr.append(j)
            anz.append(len(ss))
            woerter.append(len(s.split()))
            befund.append(bool(BEFUND.search(s)))
            texte.append(s)
    n = len(texte)
    print(f"{n} Saetze aus {len(df)} Abstracts ({time.time() - t0:.0f} s)", flush=True)
    np.savez(OUT / "saetze_meta.npz", doc=np.array(doc, dtype=np.int32), nr=np.array(nr, dtype=np.int16),
             anz=np.array(anz, dtype=np.int16), woerter=np.array(woerter, dtype=np.int16),
             befund=np.array(befund, dtype=bool))
    tok = AutoTokenizer.from_pretrained(str(MODELL))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODELL), num_labels=1)
    model.eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    pad = tok.pad_token_id
    for start in range(0, n, BLOCK):
        ziel = OUT / f"werte_{start:07d}.npy"
        if ziel.exists():
            continue
        werte = []
        for b in range(start, min(start + BLOCK, n), BATCH):
            ids = [tok.encode(s, add_special_tokens=True, max_length=MAX_LEN, truncation=True)
                   for s in texte[b:min(b + BATCH, start + BLOCK, n)]]
            ids = [x + [pad] * (MAX_LEN - len(x)) for x in ids]
            with torch.no_grad():
                out = model(torch.tensor(ids, dtype=torch.long))
            werte.append(out[0].cpu().numpy().reshape(-1))
        np.save(ziel, np.concatenate(werte).astype(np.float32))
        print(f"Saetze {start} bis {min(start + BLOCK, n)} fertig ({time.time() - t0:.0f} s)", flush=True)
    werte = np.concatenate([np.load(OUT / f"werte_{s:07d}.npy") for s in range(0, n, BLOCK)])
    np.save(OUT / "werte.npy", werte)
    # Pruefblatt: 150 zufaellige Saetze, Text und leere Bewertung; Modellwerte getrennt
    rng = np.random.default_rng(SEED_PRUEF)
    wahl = np.sort(rng.choice(n, size=min(N_PRUEF, n), replace=False))
    import csv
    with open(OUT / "pruefblatt_saetze.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nr", "satz", "bewertung_1_bis_6"])
        for k, i in enumerate(wahl):
            w.writerow([k + 1, texte[i], ""])
    np.save(OUT / "pruefblatt_index.npy", wahl)
    res = {"saetze": int(n), "abstracts": int(len(df)), "sekunden": round(time.time() - t0, 1),
           "werte_mittel": float(werte.mean()), "werte_quantile_10_50_90": np.quantile(werte, [.1, .5, .9]).tolist(),
           "anteil_befundsaetze": float(np.mean(befund)),
           "modell_sha16": {f: sha16(MODELL / f) for f in noetig},
           "versionen": {p: version(p) for p in ("torch", "transformers", "tokenizers", "numpy")},
           "python": platform.python_version(), "plattform": platform.platform(),
           "einstellungen": {"max_len": MAX_LEN, "batch": BATCH, "min_woerter": MIN_WOERTER, "attention_maske": False,
                             "geraet": "cpu"}}
    (OUT / "zertifiziertheit_phase1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Fertig: {n} Saetze, Mittel {werte.mean():.3f}. Pruefblatt: {OUT / 'pruefblatt_saetze.csv'}", flush=True)


if __name__ == "__main__":
    main()
