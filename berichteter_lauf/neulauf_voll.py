#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vollständiger Neulauf ab Schritt 1 in der heutigen Umgebung, zwei DS3-Varianten (29.09.2026).

Anders als output_neulauf_2026-09-29 (Einheiten des berichteten Laufs übernommen, Rechnung ab
Schritt 2) bildet dieser Lauf auch die Einheiten neu: SBERT, UMAP, HDBSCAN, Matching,
Referenzüberlappung und Topic-Qualität. Der berichtete Lauf und der Neulauf ab Schritt 2 werden
nur gelesen. Alles Neue landet in diesem Ordner:

  wos_qc_phase*_....csv     neu aufbereitete Phasendateien (lizenzierte Rohdaten, bleiben lokal)
  einheiten/                von DS3 unabhängig, einmal für beide Varianten:
                            output_phase1/2 (Schritt 1, Referenzüberlappung 2b, Topic-Qualität 3c),
                            output_cross_phase/ (Matching 1b), reproduzierbarkeit_phase1/2.json,
                            topic_zuordnung_phase1/2.csv (berichtete Topics zu neuen Topics)
  alpha5/                   DS3 mit symmetrischem Prior, alpha = 5 (Spezifikation wie berichtet)
  prior/                    DS3 mit Prior auf dem Review-Anteil der Phase, gleiche Stärke
     je Variante: output_phase1/, output_phase2/, output_cross_phase/, output_phase_boundary/,
                  figures_paper/, figures_rp/, supplement_rp/, klassenanteile_null.csv,
                  standardisation_variants.csv, citation_topic_profile_appendix_rows.tex
  logs/, herkunft.json, pruefung.json

Ablauf
  1. Sicherung des berichteten Laufs (besteht schon seit dem ersten Neulauf, wird nicht angefasst).
  2. Daten: prepare_kati_data und clean_pipeline_data mit Ziel in diesem Ordner, Abgleich.
  3. Einheiten: run_phase.py (Schritt 1) je Phase, step01b --with-sbert, step02b je Phase,
     step03c, pruefe_reproduzierbarkeit.py je Phase.
  4. Je Variante: Übernahme aus einheiten/, Pipeline ab Schritt 2 wie run_all_phases.py
     (2, 3, 3b, 4, 5b, 5 je Phase, 5c), Citation-Topic-Profil (2c), Schritt 6, Perturbation,
     Nullmodelle, Klassenanteile, Standardisierungsvarianten, Phasengrenze, Abbildungen, Supplement.
  5. Prüfung: Einheiten gegen den berichteten Lauf (Topics, ARI, Zuordnung), Reproduzierbarkeit in
     dieser Umgebung (Referenzzellen der Sensitivität, Seed 42, Kontrollzelle der Phasengrenze),
     DS3 unabhängig nachgerechnet, Kontrolle, dass außerhalb dieses Ordners nichts geändert wurde.

Die Pipeline-Skripte bleiben unverändert. Jeder Schritt läuft in einem eigenen Python-Prozess;
der Treiber setzt dort vor dem Aufruf die Pfade des Skripts auf diesen Ordner und die Variante
in config.REVIEW_ABSENCE_PRIOR.

Aufruf im Pipeline-Ordner (Dauer rund 45 Minuten):
    caffeinate -i ../sbert_pipeline/.venv/bin/python output_neulauf_voll_2026-09-29/neulauf_voll.py

Teilschritte und Wiederaufnahme:
    --nur sicherung,daten,einheiten,saat,pipeline,profil,robustheit,grenze,abbildungen,supplement,pruefung
    --variante alpha5 | prior | beide        --ab-schritt 3.1 (innerhalb der Pipeline)
    --trocken   zeigt nur den Plan
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HIER = Path(__file__).resolve().parent

VARIANTEN = {"alpha5": "symmetric", "prior": "phase_share"}
DATEN = {1: ("wos_qc_phase1_2000_2015.csv", "wos_qc_phase1_2000_2015_clean.csv"),
         2: ("wos_qc_phase2_2016_2025.csv", "wos_qc_phase2_2016_2025_clean.csv")}
# Erwartete Änderungen im Feld Document Type durch das korrigierte clean_doctype (Test vom 29.09.)
ERWARTET_DT = {1: 0, 2: 46}
# Aus einheiten/ in jede Variante übernommen (Schritt 1, 2b, 3c und 1b; alle DS3-frei)
SAAT_PHASE =["model_results.pkl", "tem_metrics.csv", "topic_proportions_yearly.csv",
              "topic_keywords.csv", "topic_assignments.csv", "reference_overlap_p{ph}.csv",
              "topic_quality_per_topic.csv", "topic_quality_summary.json"]
SAAT_KREUZ = ["topic_matches_mutual.csv", "topic_matches_best_p1_to_p2.csv",
              "topic_matches_full.csv", "match_diagnostics.txt"]
ALTE_AUSGABEN = ["output_phase1", "output_phase2", "output_cross_phase", "output_phase_boundary",
                 "figures_rp", "figures_paper"]
PIPELINE = [("2", "run_phase_indicators", "Indikatoren und Memberships"),
            ("3", "run_phase_efa", "EFA/PCA"),
            ("3b", "run_phase_validation", "Externe Validierung"),
            ("4", "run_phase_viz", "Visualisierungen"),
            ("5b", "step05b_artifacts", "Sensitivitäts-Artefakte"),
            ("5", "run_phase_sensitivity", "Sensitivitätsanalyse")]
SCHWER = {"5b", "5"}
STUFEN = ["sicherung", "daten", "einheiten", "saat", "pipeline", "profil", "robustheit", "grenze",
          "abbildungen", "supplement", "pruefung"]
KATI_REFS = {1: ("Phase 1 2000-2015", "QC_2000-2015 References.csv"),
             2: ("Phase 2 2016-2025", "QC_2016-2025 References.csv")}
JAHRE = {1: (2000, 2015), 2: (2016, 2025)}
MEMB = ["m_ws", "m_trend", "m_ec", "m_latent"]
NAME = {"m_ws": "Weak Signal", "m_ec": "Emerging Concept", "m_trend": "Trend", "m_latent": "Latent"}

# Läuft in jedem Schritt-Prozess: config zuerst (Variante), dann das Skript importieren, Pfade
# umsetzen, Hauptfunktion aufrufen.
BOOT = r'''
import importlib, json, sys
from pathlib import Path
spec = json.loads(sys.argv[1])
sys.argv = spec["argv"]
import config
for k, v in spec.get("config", {}).items():
    setattr(config, k, v)
ns = {"Path": Path, "config": config, "importlib": importlib, "sys": sys}
ns.update({k: Path(v) for k, v in spec.get("pfade", {}).items()})
m = importlib.import_module(spec["modul"])
ns["m"] = m
exec(spec.get("vorher", ""), ns)
rc = eval(spec.get("aufruf", "m.main()"), ns)
sys.exit(rc if isinstance(rc, int) else 0)
'''


# factor_analyzer 0.5.1 (neueste Fassung) ruft check_array(force_all_finite=...). scikit-learn hat den
# Parameter in 1.6 in ensure_all_finite umbenannt und in 1.8 entfernt; die Pipeline-Umgebung hat 1.8.0.
# Der Aufsatz übersetzt nur das Schlüsselwort und greift nur, wenn der alte Name fehlt.
KOMPAT = r'''
def _kompat_sklearn():
    import inspect
    import sklearn.utils.validation as _v
    if "force_all_finite" in inspect.signature(_v.check_array).parameters:
        return False
    _orig = _v.check_array
    def check_array(*args, force_all_finite=None, **kwargs):
        if force_all_finite is not None and "ensure_all_finite" not in kwargs:
            kwargs["ensure_all_finite"] = force_all_finite
        return _orig(*args, **kwargs)
    import factor_analyzer.factor_analyzer as _fa
    import factor_analyzer.confirmatory_factor_analyzer as _cfa
    _fa.check_array = check_array
    _cfa.check_array = check_array
    return True
'''
KOMPAT_AN = ("print('Aufsatz scikit-learn/factor_analyzer:', "
             "'aktiv' if _kompat_sklearn() else 'nicht nötig', flush=True)\n")
INSTALL_FA = "../sbert_pipeline/.venv/bin/python -m pip install --no-deps factor_analyzer==0.5.1"


class Abbruch(RuntimeError):
    pass


def jetzt() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def dauer(s: float) -> str:
    return f"{s:.1f} s" if s < 60 else f"{s / 60:.1f} min"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Log:
    def __init__(self, pfad: Path):
        pfad.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(pfad, "a", encoding="utf-8")

    def __call__(self, *teile) -> None:
        s = " ".join(str(t) for t in teile)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def roh(self, zeile: str) -> None:
        sys.stdout.write(zeile)
        sys.stdout.flush()
        self.f.write(zeile)
        self.f.flush()


class Ctx:
    def __init__(self, a):
        self.neu = Path(a.ziel).expanduser().resolve() if a.ziel else HIER
        self.repo = Path(a.repo).expanduser().resolve() if a.repo else HIER.parent
        self.f3 = self.repo.parent
        self.kati = Path(a.kati).expanduser().resolve() if a.kati else self.f3.parent / "Data Kati"
        self.pub = (Path(a.publikation).expanduser().resolve() if a.publikation
                    else self.repo.parents[2] / "Publikation")
        self.abb = self.pub / "Analysen" / "Abbildungen_RP"
        self.nullm = self.pub / "Analysen" / "Nullmodell"
        self.vorrechnung = self.pub / "Analysen" / "DS3_Neurechnung"
        self.sicherung = (Path(a.sicherung).expanduser().resolve() if a.sicherung
                          else self.f3 / "Sicherung_berichteter_Lauf_2026-09-29")
        self.einheiten = self.neu / "einheiten"
        self.logs = self.neu / "logs"
        self.trocken = a.trocken
        self.ohne_schwer = a.ohne_schwer
        self.test_einheiten = a.test_einheiten_aus_bericht
        self.abb_nur = a.abbildungen_nur
        self.ab_schritt = a.ab_schritt
        self.bis_schritt = a.bis_schritt
        self.log = Log(self.logs / f"neulauf_voll_{dt.datetime.now():%Y%m%d_%H%M%S}.log")
        pfad = self.neu / "herkunft.json"
        self.herkunft = json.loads(pfad.read_text(encoding="utf-8")) if pfad.exists() else {}
        self.herkunft.setdefault("schritte", [])

    def speichern(self) -> None:
        if self.trocken:
            return
        (self.neu / "herkunft.json").write_text(
            json.dumps(self.herkunft, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    def im_ziel(self, *pfade: Path) -> None:
        for p in pfade:
            p = Path(p).resolve()
            if p != self.neu and self.neu not in p.parents:
                raise Abbruch(f"Schreibziel außerhalb des neuen Ordners: {p}")


# --------------------------------------------------------------------------- Schritte

def schritt(ctx: Ctx, sid: str, titel: str, spec: dict, ziele: list, env: dict | None = None) -> None:
    ctx.im_ziel(*ziele)
    ctx.log("")
    ctx.log("=" * 78)
    ctx.log(f"  {sid}  {titel}")
    ctx.log("=" * 78)
    ctx.log(f"  {spec['modul']} {' '.join(str(x) for x in spec['argv'][1:])}")
    if ctx.trocken:
        return
    umgebung = dict(os.environ)
    umgebung.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    umgebung.update(env or {})
    t0 = time.time()
    proc = subprocess.Popen([sys.executable, "-c", BOOT, json.dumps(spec)], cwd=spec["cwd"],
                            env=umgebung, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for zeile in proc.stdout:
        ctx.log.roh(zeile)
    rc = proc.wait()
    d = time.time() - t0
    ctx.herkunft["schritte"].append({"schritt": sid, "titel": titel, "modul": spec["modul"],
                                     "argv": spec["argv"], "ende": jetzt(), "sekunden": round(d, 1),
                                     "rueckgabe": rc})
    ctx.speichern()
    if rc != 0:
        raise Abbruch(f"Schritt {sid} ist fehlgeschlagen (Rückgabewert {rc}).")
    ctx.log(f"  OK {sid} ({dauer(d)})")


def spec(ctx: Ctx, modul: str, argv: list, v: str | None = None, vorher: str = "",
         aufruf: str = "m.main()", cwd: Path | None = None) -> dict:
    R = ctx.neu / v if v else ctx.neu
    return {"modul": modul, "argv": [modul + ".py"] + [str(x) for x in argv],
            "cwd": str(cwd or ctx.repo),
            "config": {"REVIEW_ABSENCE_PRIOR": VARIANTEN[v]} if v else {},
            "pfade": {"R": str(R), "D": str(ctx.neu), "E": str(ctx.einheiten), "REPO": str(ctx.repo)},
            "vorher": vorher, "aufruf": aufruf}


def kopiere(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "darwin":
        r = subprocess.run(["cp", "-c", "-p", str(src), str(dst)], capture_output=True)
        if r.returncode == 0:
            return
    shutil.copy2(src, dst)


def kopiere_baum(src: Path, dst: Path) -> None:
    if sys.platform == "darwin":
        r = subprocess.run(["cp", "-c", "-R", "-p", str(src), str(dst)], capture_output=True)
        if r.returncode == 0:
            return
        if dst.exists():
            shutil.rmtree(dst)
    shutil.copytree(src, dst, copy_function=shutil.copy2)


def bestand(root: Path) -> tuple:
    dateien = [f for f in root.rglob("*") if f.is_file()]
    return len(dateien), sum(f.stat().st_size for f in dateien)


# --------------------------------------------------------------------------- Schnappschuss

def schnappschuss(ctx: Ctx) -> dict:
    bereiche = [("repo", ctx.repo, True), ("f3", ctx.f3, False), ("abb", ctx.abb, True),
                ("nullmodell", ctx.nullm, True),
                ("perturbation", ctx.pub / "Analysen" / "Perturbationsexperiment", True),
                ("vorrechnung", ctx.vorrechnung, True)]
    res = {}
    for name, root, rekursiv in bereiche:
        if not root.exists():
            continue
        for f in (root.rglob("*") if rekursiv else root.glob("*")):
            if not f.is_file():
                continue
            rf = f.resolve()
            if ctx.neu in rf.parents or ctx.sicherung in rf.parents or HIER in rf.parents:
                continue
            rel = f.relative_to(root)
            if {"__pycache__", ".git"} & set(rel.parts) or f.name == ".DS_Store":
                continue
            st = f.stat()
            res[f"{name}/{rel}"] = [st.st_size, st.st_mtime_ns]
    return res


def schnappschuss_vergleich(ctx: Ctx) -> dict:
    pfad = ctx.logs / "schnappschuss_vorher.json"
    if not pfad.exists():
        return {"status": "kein Schnappschuss vorhanden"}
    vorher = json.loads(pfad.read_text(encoding="utf-8"))
    nachher = schnappschuss(ctx)
    geaendert = sorted(k for k in vorher if k in nachher and vorher[k] != nachher[k])
    entfernt = sorted(k for k in vorher if k not in nachher)
    neu = sorted(k for k in nachher if k not in vorher)
    return {"status": "unverändert" if not (geaendert or entfernt or neu) else "GEÄNDERT",
            "dateien": len(vorher), "geaendert": geaendert, "entfernt": entfernt, "neu": neu}


def waechter(ctx: Ctx, wann: str) -> None:
    if ctx.trocken:
        return
    v = schnappschuss_vergleich(ctx)
    ctx.herkunft.setdefault("waechter", []).append({"wann": wann, "zeit": jetzt(), **v})
    ctx.speichern()
    if v.get("status") == "GEÄNDERT":
        ctx.log(f"  WARNUNG: außerhalb des neuen Ordners hat sich etwas geändert ({wann}):")
        for k in ("geaendert", "entfernt", "neu"):
            for f in v[k][:20]:
                ctx.log(f"    {k}: {f}")
    else:
        ctx.log(f"  Wächter ({wann}): alte Ordner unverändert, {v.get('dateien')} Dateien geprüft.")


# --------------------------------------------------------------------------- Stufen

def stufe_umgebung(ctx: Ctx) -> None:
    from importlib import metadata
    pakete = {}
    for p in ["numpy", "pandas", "scipy", "scikit-learn", "umap-learn", "hdbscan",
              "factor_analyzer", "matplotlib", "seaborn", "sentence-transformers", "torch",
              "transformers", "gensim"]:
        try:
            pakete[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pakete[p] = None
    ctx.log(f"Python {platform.python_version()} ({sys.executable})")
    ctx.log("Pakete: " + ", ".join(f"{k} {v}" for k, v in pakete.items()))
    fehlend = [p for p in ["numpy", "pandas", "scipy", "scikit-learn", "factor_analyzer",
                           "matplotlib", "seaborn"] if not pakete[p]]
    if not ctx.ohne_schwer:
        fehlend += [p for p in ["umap-learn", "hdbscan", "sentence-transformers", "torch", "gensim"]
                    if not pakete[p]]
    if fehlend == ["factor_analyzer"]:
        raise Abbruch("factor_analyzer fehlt in der Umgebung (steht in requirements.txt, wird für die "
                      "Faktorenanalyse in Schritt 3 gebraucht). Einmal installieren, im Pipeline-Ordner:\n"
                      f"    {INSTALL_FA}\n"
                      "danach den Neulauf mit demselben Befehl wie eben neu starten.")
    if fehlend:
        raise Abbruch("Es fehlen Pakete: " + ", ".join(fehlend)
                      + ". Läuft der Treiber mit ../sbert_pipeline/.venv/bin/python?")
    test = KOMPAT + (
        "aktiv = _kompat_sklearn()\n"
        "import numpy as np\n"
        "from factor_analyzer import FactorAnalyzer\n"
        "X = np.random.default_rng(0).normal(size=(300, 6))\n"
        "X[:, 1] += X[:, 0]\nX[:, 3] += X[:, 2]\n"
        "fa = FactorAnalyzer(n_factors=2, method='minres', rotation='oblimin').fit(X)\n"
        "print('AUFSATZ', 'aktiv' if aktiv else 'nicht_noetig', fa.loadings_.shape)\n")
    r = subprocess.run([sys.executable, "-c", test], capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    if r.returncode != 0:
        raise Abbruch("Die Faktorenanalyse läuft in dieser Umgebung nicht:\n"
                      + "\n".join(r.stderr.strip().splitlines()[-6:]))
    zeile = [z for z in r.stdout.splitlines() if z.startswith('AUFSATZ')][-1].split()
    kompat = "aktiv" if zeile[1] == "aktiv" else "nicht nötig"
    ctx.log(f"Faktorenanalyse-Test bestanden (Aufsatz für scikit-learn: {kompat})")
    sbert = None
    if not ctx.ohne_schwer and not ctx.test_einheiten:
        # Das Sprachmodell muss vor dem langen Lauf ladbar sein (Zwischenspeicher oder Netz).
        test = ("import sys\nsys.path.insert(0, '.')\nimport config\n"
                "from sentence_transformers import SentenceTransformer\n"
                "m = SentenceTransformer(config.SBERT_MODEL)\n"
                "v = m.encode(['weak signals in foresight'], normalize_embeddings=True)\n"
                "print('SBERT', config.SBERT_MODEL, m.device, v.shape[1])\n")
        r = subprocess.run([sys.executable, "-c", test], capture_output=True, text=True, cwd=ctx.repo,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if r.returncode != 0:
            raise Abbruch("Das Sprachmodell für Schritt 1 lässt sich nicht laden:\n"
                          + "\n".join(r.stderr.strip().splitlines()[-6:]))
        sbert = [z for z in r.stdout.splitlines() if z.startswith("SBERT")][-1]
        ctx.log(f"Sprachmodell geladen: {sbert}")
    schrift = None
    if ctx.abb.exists():
        r = subprocess.run([sys.executable, "-c", "import rp_style; print(rp_style.active_font())"],
                           cwd=ctx.abb, capture_output=True, text=True,
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg"})
        schrift = (r.stdout.strip().splitlines() or ["?"])[-1]
        ctx.log(f"Schrift der Abbildungen: {schrift}")
        if "Arial" not in schrift:
            ctx.log("  WARNUNG: nicht Arial. Die Abbildungen gehören auf dem Mac in dieser Umgebung gebaut.")
    code = {}
    for f in sorted(ctx.repo.glob("*.py")):
        code[f"repo/{f.name}"] = sha256(f)
    for f in [ctx.abb / "paper_figures_rp.py", ctx.abb / "rp_style.py",
              ctx.nullm / "nullmodel_class_shares.py", Path(__file__).resolve()]:
        if f.exists():
            code[str(f.name if f.parent == HIER else f.relative_to(ctx.pub))] = sha256(f)
    ctx.herkunft["umgebung"] = {"zeit": jetzt(), "python": platform.python_version(),
                                "ausfuehrbar": sys.executable, "system": platform.platform(),
                                "pakete": pakete, "schrift": schrift, "code_sha256": code,
                                "aufsatz_sklearn_force_all_finite": kompat, "sprachmodell": sbert,
                                "pfade": {"neu": str(ctx.neu), "repo": str(ctx.repo),
                                          "kati": str(ctx.kati), "vorrechnung": str(ctx.vorrechnung),
                                          "sicherung": str(ctx.sicherung)}}
    ctx.speichern()


def stufe_sicherung(ctx: Ctx) -> None:
    ctx.log("")
    ctx.log(f"Sicherung nach {ctx.sicherung}")
    if ctx.trocken:
        return
    if ctx.sicherung.exists():
        ctx.log("  vorhanden, wird nicht angefasst")
        return
    ctx.sicherung.mkdir(parents=True)
    protokoll = {}
    for d in ALTE_AUSGABEN:
        src = ctx.repo / d
        if not src.exists():
            continue
        kopiere_baum(src, ctx.sicherung / d)
        a, b = bestand(src), bestand(ctx.sicherung / d)
        if a != b:
            raise Abbruch(f"Sicherung von {d} unvollständig: {a} gegen {b}")
        protokoll[d] = {"dateien": a[0], "bytes": a[1]}
        ctx.log(f"  {d}: {a[0]} Dateien, {a[1] / 1e6:.1f} MB")
    for roh, clean in DATEN.values():
        for f in (roh, clean):
            kopiere(ctx.f3 / f, ctx.sicherung / f)
            if (ctx.f3 / f).stat().st_size != (ctx.sicherung / f).stat().st_size:
                raise Abbruch(f"Sicherung von {f} unvollständig")
            protokoll[f] = {"bytes": (ctx.f3 / f).stat().st_size}
    (ctx.sicherung / "LIESMICH.txt").write_text(
        "Sicherung des berichteten Laufs vor dem Neulauf vom 29.09.2026 (DS3-Korrektur).\n"
        "Kopie von output_phase1/2, output_cross_phase, output_phase_boundary, figures_rp und\n"
        "figures_paper aus sbert_pipeline_membership_v3_efa sowie der vier aufbereiteten\n"
        "Phasendateien. Enthält lizenzierte Rohdaten: nicht teilen, nicht committen.\n",
        encoding="utf-8")
    ctx.herkunft["sicherung"] = {"zeit": jetzt(), "ziel": str(ctx.sicherung), "inhalt": protokoll}
    ctx.speichern()


def stufe_daten(ctx: Ctx) -> None:
    env = {"KATI_DATA_DIR": str(ctx.kati)}
    schritt(ctx, "D.1", "Aufbereitung der KATI-Daten (prepare_kati_data)",
            spec(ctx, "prepare_kati_data", [], vorher="m.OUT_DIR = D\n", aufruf="m.main(force=True)"),
            ziele=[ctx.neu], env=env)
    schritt(ctx, "D.2", "Bereinigung für Einheitenbildung und Phasengrenze (clean_pipeline_data)",
            spec(ctx, "clean_pipeline_data", [], vorher="m.PARENT_DIR = D\n", aufruf="m.main(force=True)"),
            ziele=[ctx.neu])
    if ctx.trocken:
        return
    import pandas as pd
    ctx.log("")
    ctx.log("Abgleich der neuen Phasendateien mit den vorliegenden")
    erg = {}
    for ph, (roh, clean) in DATEN.items():
        e = {}
        a = pd.read_csv(ctx.f3 / roh, dtype=str, keep_default_na=False)
        b = pd.read_csv(ctx.neu / roh, dtype=str, keep_default_na=False)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            raise Abbruch(f"Phase {ph}: {roh} hat eine andere Form ({a.shape} gegen {b.shape})")
        diff = (a != b).sum()
        andere = {c: int(n) for c, n in diff.items() if n and c != "Document Type"}
        m = a["Document Type"] != b["Document Type"]
        e["aufbereitet"] = {"zeilen": int(len(b)), "document_type_geaendert": int(m.sum()),
                            "uebergaenge": (a.loc[m, "Document Type"] + " -> "
                                            + b.loc[m, "Document Type"]).value_counts().to_dict(),
                            "andere_spalten_geaendert": andere}
        ctx.log(f"  Phase {ph} aufbereitet: {len(b)} Zeilen, Document Type in {int(m.sum())} geändert "
                f"(erwartet {ERWARTET_DT[ph]}), andere Spalten {andere or 'unverändert'}")
        if andere:
            raise Abbruch(f"Phase {ph}: Abweichung außerhalb von Document Type: {andere}")
        if int(m.sum()) != ERWARTET_DT[ph]:
            ctx.log(f"  WARNUNG: erwartet waren {ERWARTET_DT[ph]} Änderungen im Document Type")
        a = pd.read_csv(ctx.f3 / clean, dtype=str, keep_default_na=False)
        b = pd.read_csv(ctx.neu / clean, dtype=str, keep_default_na=False)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            raise Abbruch(f"Phase {ph}: {clean} hat eine andere Form ({a.shape} gegen {b.shape})")
        for c in ("Title", "Abstract", "Year"):
            if not a[c].equals(b[c]):
                raise Abbruch(f"Phase {ph}: {clean} weicht in {c} ab; die Labels aus Schritt 1 passen dann nicht.")
        diff = (a != b).sum()
        e["bereinigt"] = {"zeilen": int(len(b)), "geaenderte_zellen": {c: int(n) for c, n in diff.items() if n},
                          "gefuellt_alt": {c: round(float((a[c].str.strip() != "").mean()), 4)
                                           for c in ("Author Full Names", "ORCIDs")},
                          "gefuellt_neu": {c: round(float((b[c].str.strip() != "").mean()), 4)
                                           for c in ("Author Full Names", "ORCIDs")}}
        ctx.log(f"  Phase {ph} bereinigt: Titel, Abstracts, Jahre gleich; geänderte Zellen "
                f"{e['bereinigt']['geaenderte_zellen']}")
        erg[f"phase{ph}"] = e
    erg["sha256"] = {f: sha256(ctx.neu / f) for r_c in DATEN.values() for f in r_c}
    ctx.herkunft["daten"] = {"zeit": jetzt(), **erg}
    ctx.speichern()


def stufe_einheiten(ctx: Ctx) -> None:
    """Einheitenbildung und alles, was nur von ihr abhängt, einmal für beide Varianten."""
    E = ctx.einheiten
    ctx.im_ziel(E)
    if ctx.test_einheiten:
        # Nur für Tests ohne SBERT/UMAP: Schritt-1-Ausgaben des berichteten Laufs statt Neubildung.
        ctx.log("")
        ctx.log("TESTMODUS: Einheiten aus dem berichteten Lauf übernommen, nicht neu gebildet")
        if not ctx.trocken:
            for ph in (1, 2):
                for f in ["model_results.pkl", "tem_metrics.csv", "topic_proportions_yearly.csv",
                          "topic_keywords.csv", "topic_assignments.csv", "vectorizer.pkl"]:
                    kopiere(ctx.repo / f"output_phase{ph}" / f, E / f"output_phase{ph}" / f)
    else:
        for p in (1, 2):
            schritt(ctx, f"E.1.{p}", f"Einheitenbildung (SBERT, UMAP, HDBSCAN), Phase {p}",
                    spec(ctx, "run_phase", [p], vorher="m.BASE_DIR = E\nm.PARENT_DIR = D\n"), ziele=[E])
    schritt(ctx, "E.2", "Topic-Matching zwischen den Phasen (step01b, SBERT-Zentroide)",
            spec(ctx, "step01b_cross_phase_matching",
                 ["--phase1-dir", E / "output_phase1", "--phase2-dir", E / "output_phase2",
                  "--out-dir", E / "output_cross_phase", "--with-sbert"]),
            ziele=[E / "output_cross_phase"])
    for p in (1, 2):
        ordner, datei = KATI_REFS[p]
        schritt(ctx, f"E.3.{p}", f"Referenzüberlappung (step02b), Phase {p}",
                spec(ctx, "step02b_run_with_kati",
                     ["--phase", p, "--kati-refs", ctx.kati / ordner / datei,
                      "--topics", E / f"output_phase{p}" / "topic_assignments.csv",
                      "--out", E / f"output_phase{p}" / f"reference_overlap_p{p}.csv"]),
                ziele=[E / f"output_phase{p}"])
    vorher = "m.ROOT = E\n"
    for p in (1, 2):
        roh, clean = DATEN[p]
        vorher += (f"m.PHASE_CONFIG['phase{p}'].update(input_csv=D / '{clean}', "
                   f"model_pkl=E / 'output_phase{p}' / 'model_results.pkl', "
                   f"output_dir=E / 'output_phase{p}')\n")
    schritt(ctx, "E.4", "Topic-Qualität: Kohärenz und Diversität (step03c)",
            spec(ctx, "step03c_topic_quality", [], vorher=vorher), ziele=[E])
    if not ctx.ohne_schwer and not ctx.test_einheiten:
        for p in (1, 2):
            schritt(ctx, f"E.5.{p}", f"Reproduzierbarkeit der Einheiten in dieser Umgebung, Phase {p}",
                    spec(ctx, "pruefe_reproduzierbarkeit", ["--phase", p], vorher="m.BASE_DIR = E\n"),
                    ziele=[E])
    if not ctx.trocken:
        zuordnung_berichtet_neu(ctx)


def zuordnung_berichtet_neu(ctx: Ctx) -> None:
    """Je berichtetem Topic das neue Topic mit der größten Dokumentüberlappung (Jaccard).

    Liegen alle Dokumente eines berichteten Topics im neuen Lauf im Rauschen, bleibt topic_neu leer
    (in der Fassung vom 29.09. stand dort 0, weil idxmax bei lauter Nullen die erste Spalte liefert).
    im_rauschen_neu zählt je berichtetem Topic die Dokumente, die im neuen Lauf Rauschen sind.
    """
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score
    erg = {}
    for ph in (1, 2):
        with open(ctx.repo / f"output_phase{ph}" / "model_results.pkl", "rb") as f:
            alt = np.asarray(pickle.load(f)["labels"])
        with open(ctx.einheiten / f"output_phase{ph}" / "model_results.pkl", "rb") as f:
            neu = np.asarray(pickle.load(f)["labels"])
        if len(alt) != len(neu):
            raise Abbruch(f"Phase {ph}: {len(neu)} Dokumente im neuen Lauf, {len(alt)} im berichteten")
        kreuz = pd.crosstab(pd.Series(alt, name="alt"), pd.Series(neu, name="neu")).drop(index=-1, errors="ignore")
        rauschen = kreuz[-1] if -1 in kreuz.columns else pd.Series(0, index=kreuz.index)
        kreuz = kreuz.drop(columns=-1, errors="ignore")
        n_alt = pd.Series(alt).value_counts()
        n_neu = pd.Series(neu).value_counts()
        zeilen = []
        for t in kreuz.index:
            row = kreuz.loc[t]
            zeile = {"topic_berichtet": int(t), "topic_neu": pd.NA, "jaccard": 0.0, "n_berichtet": int(n_alt[t]),
                     "n_neu": pd.NA, "gemeinsam": 0, "im_rauschen_neu": int(rauschen[t])}
            if row.max() > 0:
                jac = row / (n_alt[t] + n_neu.reindex(row.index).values - row)
                b = jac.idxmax()
                zeile.update({"topic_neu": int(b), "jaccard": round(float(jac[b]), 4), "n_neu": int(n_neu[b]),
                              "gemeinsam": int(row[b])})
            zeilen.append(zeile)
        z = pd.DataFrame(zeilen).astype({"topic_neu": "Int64", "n_neu": "Int64"})
        z.to_csv(ctx.einheiten / f"topic_zuordnung_phase{ph}.csv", index=False)
        erg[f"phase{ph}"] = {
            "dokumente": int(len(neu)),
            "topics_berichtet": int(len(set(alt[alt >= 0]))), "topics_neu": int(len(set(neu[neu >= 0]))),
            "rauschen_berichtet": int((alt == -1).sum()), "rauschen_neu": int((neu == -1).sum()),
            "ari_gegen_berichtet": round(float(adjusted_rand_score(alt, neu)), 4),
            "jaccard_median": round(float(z["jaccard"].median()), 3),
            "anteil_jaccard_ab_0_5": round(float((z["jaccard"] >= 0.5).mean()), 3),
            "berichtet_ganz_im_rauschen": int(z["topic_neu"].isna().sum()),
            "topic0_berichtet_zu": z.loc[z["topic_berichtet"] == 0, ["topic_neu", "jaccard"]].to_dict("records")}
        ctx.log(f"  Einheiten Phase {ph}: {erg[f'phase{ph}']['topics_neu']} Topics "
                f"(berichtet {erg[f'phase{ph}']['topics_berichtet']}), ARI gegen berichtet "
                f"{erg[f'phase{ph}']['ari_gegen_berichtet']}, Jaccard-Median {erg[f'phase{ph}']['jaccard_median']}")
    ctx.herkunft["einheiten"] = {"zeit": jetzt(), "testmodus": ctx.test_einheiten, **erg}
    ctx.speichern()


def stufe_saat(ctx: Ctx, v: str) -> None:
    R = ctx.neu / v
    ctx.im_ziel(R)
    ctx.log("")
    ctx.log(f"[{v}] Übernahme aus einheiten/ (Schritt 1, 1b, 2b, 3c)")
    if ctx.trocken:
        return
    liste = {}
    E = ctx.einheiten
    paare = [(E / f"output_phase{ph}" / f.format(ph=ph), R / f"output_phase{ph}" / f.format(ph=ph))
             for ph in (1, 2) for f in SAAT_PHASE]
    paare += [(E / "output_cross_phase" / f, R / "output_cross_phase" / f) for f in SAAT_KREUZ]
    for src, dst in paare:
        kopiere(src, dst)
        h1, h2 = sha256(src), sha256(dst)
        if h1 != h2:
            raise Abbruch(f"Kopie weicht ab: {src}")
        liste[str(dst.relative_to(ctx.neu))] = {"quelle": str(src.relative_to(ctx.neu)), "sha256": h1}
    ctx.log(f"  {len(liste)} Dateien übernommen und per Prüfsumme bestätigt")
    ctx.herkunft.setdefault("saat", {})[v] = liste
    ctx.speichern()


def stufe_pipeline(ctx: Ctx, v: str) -> None:
    R = ctx.neu / v
    ids = [f"{s}.{p}" for s, _, _ in PIPELINE for p in (1, 2)] + ["5c"]
    start, ende = 0, len(ids) - 1
    for name, wert in (("ab", ctx.ab_schritt), ("bis", ctx.bis_schritt)):
        if wert and wert not in ids:
            raise Abbruch(f"Unbekannter Schritt {wert}; möglich: {ids}")
    if ctx.ab_schritt:
        start = ids.index(ctx.ab_schritt)
    if ctx.bis_schritt:
        ende = ids.index(ctx.bis_schritt)
    for s, modul, titel in PIPELINE:
        for p in (1, 2):
            sid = f"{s}.{p}"
            if not start <= ids.index(sid) <= ende or (ctx.ohne_schwer and s in SCHWER):
                continue
            vorher = "m.BASE_DIR = R\nm.PARENT_DIR = D\n"
            if modul == "run_phase_efa":
                vorher += KOMPAT + KOMPAT_AN
            if modul == "run_phase_sensitivity":
                vorher += ("import step05b_artifacts as s5b\n"
                           "s5b.BASE_DIR = R\ns5b.PARENT_DIR = D\n")
            schritt(ctx, f"[{v}] {sid}", f"{titel}, Phase {p}",
                    spec(ctx, modul, [p], v, vorher=vorher), ziele=[R])
            if sid == "2.2":
                kontrolle_ds3(ctx, v)
    if start <= ids.index("5c") <= ende:
        schritt(ctx, f"[{v}] 5c", "Cross-Phase-Sensitivität (Hybrid-alpha)",
                spec(ctx, "step05c_cross_phase_sensitivity",
                     ["--phase1-dir", R / "output_phase1", "--phase2-dir", R / "output_phase2",
                      "--out-dir", R / "output_cross_phase"], v),
                ziele=[R / "output_cross_phase"])


def kontrolle_ds3(ctx: Ctx, v: str) -> None:
    """Früher Abgleich nach Schritt 2: DS3 unabhängig aus Daten und Labels nachgerechnet."""
    if ctx.trocken:
        return
    import pickle
    import re
    import numpy as np
    import pandas as pd
    erg = {}
    for ph, (roh, _) in DATEN.items():
        y0, y1 = JAHRE[ph]
        df = pd.read_csv(ctx.neu / roh, usecols=["Title", "Abstract", "Year", "Document Type"])
        text = df["Title"].fillna("") + ". " + df["Abstract"].fillna("")
        tc = text.apply(lambda t: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", t.lower())).strip()
                        if isinstance(t, str) else "")
        jahr = pd.to_numeric(df["Year"], errors="coerce")
        behalten = (jahr >= y0) & (jahr <= y1)
        df, tc = df[behalten], tc[behalten]
        df = df[tc.str.split().str.len() >= 10].reset_index(drop=True)
        with open(ctx.neu / v / f"output_phase{ph}" / "model_results.pkl", "rb") as f:
            labels = np.asarray(pickle.load(f)["labels"])
        if len(labels) != len(df):
            raise Abbruch(f"[{v}] Phase {ph}: {len(labels)} Labels gegen {len(df)} Dokumente")
        typ = df["Document Type"]
        review = typ.apply(lambda s: isinstance(s, str)
                           and any(t.strip().lower() == "review" for t in re.split(r"[;|]", s)))
        t = pd.DataFrame({"topic": labels, "review": review.values, "hat_typ": typ.notna().values})
        t = t[(t["topic"] >= 0) & t["hat_typ"]]
        r, n = t.groupby("topic")["review"].sum(), t.groupby("topic").size()
        alpha = 5
        if VARIANTEN[v] == "symmetric":
            ds3 = 1 - (r + alpha) / (n + 2 * alpha)
        else:
            p0 = r.sum() / n.sum()
            ds3 = 1 - (r + 2 * alpha * p0) / (n + 2 * alpha)
        ind = pd.read_csv(ctx.neu / v / f"output_phase{ph}" / "indicators_16.csv", index_col=0)
        d = float((ind["review_absence"] - ds3.reindex(ind.index)).abs().max())
        erg[f"phase{ph}"] = {"topics": int(len(ind)), "reviews_in_topics": int(r.sum()),
                             "publikationen_in_topics": int(n.sum()), "max_abw": d}
        ctx.log(f"  Kontrolle [{v}] Phase {ph}: {len(ind)} Topics, {int(r.sum())} Reviews in Topics, "
                f"DS3 unabhängig nachgerechnet, max. Abweichung {d:.1e}")
        if not d <= 1e-12:
            raise Abbruch(f"[{v}] Phase {ph}: DS3 weicht von der unabhängigen Nachrechnung ab ({d:.2e})")
    ctx.herkunft.setdefault("kontrolle_ds3", {})[v] = erg
    ctx.speichern()


def stufe_profil(ctx: Ctx, v: str) -> None:
    R = ctx.neu / v
    schritt(ctx, f"[{v}] 2c", "Citation-Topic-Profil (run_step02c_phases)",
            spec(ctx, "run_step02c_phases", [], v, vorher="m.BASE = R\n"), ziele=[R])


def stufe_robustheit(ctx: Ctx, v: str) -> None:
    R = ctx.neu / v
    schritt(ctx, f"[{v}] 6", "Übergänge zwischen den Phasen",
            spec(ctx, "step06_cross_phase_transitions",
                 ["--phase1-dir", R / "output_phase1", "--phase2-dir", R / "output_phase2",
                  "--cross-dir", R / "output_cross_phase"], v),
            ziele=[R / "output_cross_phase"])
    schritt(ctx, f"[{v}] R.1", "Perturbationsexperiment",
            spec(ctx, "perturbation_experiment", ["--root", R, "--code", ctx.repo], v), ziele=[R])
    schritt(ctx, f"[{v}] R.2", "Nullmodelle",
            spec(ctx, "nullmodel_experiment", ["--root", R, "--code", ctx.repo], v), ziele=[R])
    schritt(ctx, f"[{v}] R.3", "Klassenanteile unter den Nullmodellen",
            spec(ctx, "nullmodel_class_shares", ["--root", R, "--out", R / "klassenanteile_null.csv"], v,
                 cwd=ctx.nullm),
            ziele=[R], env={"PYTHONPATH": str(ctx.repo)})
    schritt(ctx, f"[{v}] R.4", "Standardisierungsvarianten",
            spec(ctx, "standardisation_variants", ["--root", R, "--code", ctx.repo], v), ziele=[R])


def stufe_grenze(ctx: Ctx, v: str) -> None:
    if ctx.ohne_schwer:
        return
    R = ctx.neu / v
    schritt(ctx, f"[{v}] G", "Stabilität gegenüber der Phasengrenze",
            spec(ctx, "run_phase_boundary_stability", [], v,
                 vorher="m.BASE_DIR = R\nm.PARENT_DIR = D\n"),
            ziele=[R / "output_phase_boundary"])


def stufe_abbildungen(ctx: Ctx, v: str) -> None:
    R = ctx.neu / v
    if not ctx.ohne_schwer:
        schritt(ctx, f"[{v}] A.1", "Abbildung S16 (Phasengrenze)",
                spec(ctx, "make_figS16", ["--out", R / "figures_rp"], v, vorher="m.BASE_DIR = R\n"),
                ziele=[R / "figures_rp"])
    argv = ["--out", R / "figures_rp", "--perturbation-dir", R]
    if ctx.abb_nur:
        argv += ["--only", ctx.abb_nur]
    vorher = ("m.P1_DIR = R / 'output_phase1'\nm.P2_DIR = R / 'output_phase2'\n"
              "m.CROSS_DIR = R / 'output_cross_phase'\n"
              "m.PHASES[1]['dir'] = m.P1_DIR\nm.PHASES[2]['dir'] = m.P2_DIR\n")
    schritt(ctx, f"[{v}] A.2", "Abbildungen des Manuskripts (Analysen/Abbildungen_RP/paper_figures_rp.py)",
            spec(ctx, "paper_figures_rp", argv, v, vorher=vorher, cwd=ctx.abb),
            ziele=[R / "figures_rp"], env={"PYTHONPATH": str(ctx.repo)})
    # FigA9 im Manuskript stammt aus der Fassung im Pipeline-Ordner (140 x 110 mm, gebaut am 12.09.);
    # die Fassung in Abbildungen_RP hat noch die ältere Geometrie. Deshalb hier dieselbe Herkunft.
    if not ctx.abb_nur or "FigA9" in ctx.abb_nur:
        # A.3 schreibt manifest.csv (nur FigA9) und palette.json (ältere Klassennamen) neu; beides
        # wird danach aus dem Stand von A.2 zusammengeführt bzw. wiederhergestellt.
        man, pal = R / "figures_rp" / "manifest.csv", R / "figures_rp" / "palette.json"
        man_a2 = man.read_text(encoding="utf-8") if man.exists() else None
        pal_a2 = pal.read_text(encoding="utf-8") if pal.exists() else None
        schritt(ctx, f"[{v}] A.3", "Abbildung A9 aus der Fassung im Pipeline-Ordner (wie im Manuskript)",
                spec(ctx, "paper_figures_rp", ["--only", "FigA9", "--out", R / "figures_rp",
                                               "--perturbation-dir", R], v, vorher=vorher),
                ziele=[R / "figures_rp"])
        if not ctx.trocken and man_a2:
            import csv
            import io
            alt = list(csv.DictReader(io.StringIO(man_a2)))
            neu = {z["file"]: z for z in csv.DictReader(io.StringIO(man.read_text(encoding="utf-8")))}
            felder = list(alt[0].keys()) if alt else []
            puffer = io.StringIO()
            w = csv.DictWriter(puffer, fieldnames=felder, lineterminator="\r\n")
            w.writeheader()
            w.writerows([{k: neu.get(z["file"], z).get(k, "") for k in felder} for z in alt])
            man.write_text(puffer.getvalue(), encoding="utf-8")
            if pal_a2:
                pal.write_text(pal_a2, encoding="utf-8")
            ctx.log("  manifest.csv zusammengeführt (A9 aus A.3), palette.json aus A.2 wiederhergestellt")


def stufe_supplement(ctx: Ctx, v: str) -> None:
    R = ctx.neu / v
    schritt(ctx, f"[{v}] S", "Supplement-Tabellen (build_supplement.py)",
            spec(ctx, "build_supplement", ["--out", R / "supplement_rp"], v,
                 vorher="m.PHASES[1] = R / 'output_phase1'\nm.PHASES[2] = R / 'output_phase2'\n"),
            ziele=[R / "supplement_rp"])


# --------------------------------------------------------------------------- Prüfung

def stufe_pruefung(ctx: Ctx, varianten: list) -> dict:
    import numpy as np
    import pandas as pd

    def versuch(e: dict, schluessel: str, f):
        try:
            e[schluessel] = f()
        except Exception as ex:  # Prüfung soll auch bei Teilläufen einen Bericht liefern
            e[schluessel] = {"fehler": f"{type(ex).__name__}: {ex}"}

    def json_lesen(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))

    E = ctx.einheiten
    erg = {"zeit": jetzt(), "alte_ordner": schnappschuss_vergleich(ctx),
           "einheiten": ctx.herkunft.get("einheiten", {}), "varianten": {}}

    # ---- Einheiten: Reproduzierbarkeit, Matching, Qualität, Referenzüberlappung
    ein: dict = {}
    for ph in (1, 2):
        e: dict = {}
        versuch(e, "reproduzierbarkeit", lambda ph=ph: {
            k: w for k, w in json_lesen(E / f"reproduzierbarkeit_phase{ph}.json").items()
            if k in ("topics_berichtet", "noise_berichtet", "A", "C", "urteil")} | {
            "B_identisch": json_lesen(E / f"reproduzierbarkeit_phase{ph}.json")["B"]["identisch"]})
        versuch(e, "topic_qualitaet_neu_berichtet", lambda ph=ph: {
            "neu": json_lesen(E / f"output_phase{ph}" / "topic_quality_summary.json"),
            "berichtet": json_lesen(ctx.repo / f"output_phase{ph}" / "topic_quality_summary.json")})
        versuch(e, "ro_global_neu_berichtet", lambda ph=ph: [
            float(pd.read_csv(E / f"output_phase{ph}" / f"reference_overlap_p{ph}.csv")["RO_global"].iloc[0]),
            float(pd.read_csv(ctx.repo / f"output_phase{ph}" / f"reference_overlap_p{ph}.csv")["RO_global"].iloc[0])])
        ein[f"phase{ph}"] = e
    versuch(ein, "paare_mutual_neu_berichtet", lambda: [
        int(len(pd.read_csv(E / "output_cross_phase" / "topic_matches_mutual.csv"))),
        int(len(pd.read_csv(ctx.repo / "output_cross_phase" / "topic_matches_mutual.csv")))])
    erg["einheiten_pruefung"] = ein

    # ---- Varianten
    for v in varianten:
        R = ctx.neu / v
        ev = {}
        for ph in (1, 2):
            e: dict = {}
            d = R / f"output_phase{ph}"

            def memberships(d=d):
                mem = pd.read_csv(d / "signal_memberships.csv", index_col=0)
                arg = mem[MEMB].idxmax(axis=1)
                return {"topics": int(len(mem)),
                        "besetzung": {NAME[m]: int((arg == m).sum()) for m in ["m_ws", "m_ec", "m_trend", "m_latent"]},
                        "margin_unter_0_10_prozent": round(100 * float((mem["margin"] < 0.10).mean()), 2),
                        "margin_unter_0_05_prozent": round(100 * float((mem["margin"] < 0.05).mean()), 2)}
            versuch(e, "memberships", memberships)

            def ds3(d=d):
                ind = pd.read_csv(d / "indicators_16.csv", index_col=0)
                n = pd.read_csv(d / "topic_assignments.csv", usecols=["topic"])["topic"]
                n = n[n >= 0].value_counts().reindex(ind.index)
                x = ind["review_absence"]
                from scipy.stats import pearsonr, spearmanr
                return {"min_median_max": [round(float(x.min()), 4), round(float(x.median()), 4), round(float(x.max()), 4)],
                        "rho_log_n": round(float(spearmanr(x, np.log(n.astype(float)))[0]), 3),
                        "r_pe1": round(float(pearsonr(x, ind["relative_proportion_inv"])[0]), 3)}
            versuch(e, "ds3", ds3)

            def efa(d=d):
                s = json_lesen(d / "efa_summary.json")
                return {k: (round(s[k], 4) if isinstance(s[k], float) else s[k])
                        for k in ("kmo", "n_kaiser", "n_parallel", "n_parallel_p95")}
            versuch(e, "efa", efa)

            def sensitivitaet(d=d, ph=ph):
                out = {}
                hp = pd.read_csv(d / "sensitivity_parameter_hparam.csv")
                ref = hp[((hp.param == "hdbscan_min_cluster_size") & (hp.value == 25))
                         | ((hp.param == "umap_n_neighbors") & (hp.value == 15))
                         | ((hp.param == "hdbscan_min_samples") & (hp.value == 8))]
                out["referenzzellen_rho_min"] = [round(float(x), 4) for x in ref["rho_min"]]
                out["referenzzellen_topics"] = [int(x) for x in ref["n_topics"]]
                rest = hp.drop(ref.index)
                rest = rest[rest["param"] != "bertopic_min_topic_size"]
                out["andere_zellen_rho_min_spanne"] = [round(float(rest["rho_min"].min()), 3),
                                                       round(float(rest["rho_min"].max()), 3)]
                sd = pd.read_csv(d / "sensitivity_seeds.csv").set_index("seed")
                out["seed42_ari"] = round(float(sd.loc[42, "ari"]), 4)
                andere = sd.drop(index=42)
                out["andere_seeds_ari_spanne"] = [round(float(andere["ari"].min()), 3), round(float(andere["ari"].max()), 3)]
                out["andere_seeds_v_spanne"] = [round(float(andere["v_measure"].min()), 3),
                                                round(float(andere["v_measure"].max()), 3)]
                kl = pd.read_csv(d / "sensitivity_membership_kl.csv")
                nb = kl[~((kl["k"] == 1.0) & (kl["lambda_wp"] == 0.5))]
                out["kl_rho_margin_min"] = round(float(nb["rho_margin"].min()), 3)
                ab = pd.read_csv(d / "sensitivity_ablation.csv")
                out["ablation_rho_min_min"] = round(float(ab["rho_min"].min()), 2)
                out["ablation_empfindlichste"] = ab.nsmallest(4, "rho_min")[["indicator_removed", "rho_min"]].round(2).values.tolist()
                al = pd.read_csv(d / "sensitivity_parameter_alpha.csv")
                out["glaettung_rho_min"] = {int(round(float(z["value"]))): round(float(z["rho_min"]), 3) for _, z in al.iterrows()}
                gr = pd.read_csv(d / "sensitivity_growth_rightedge.csv")
                out["endjahr_stabil_prozent"] = float(gr["argmax_stable_pct"].iloc[0])
                return out
            versuch(e, "sensitivitaet", sensitivitaet)

            def robust(d=d):
                pt = pd.read_csv(d / "perturbation_summary.csv")
                nm = pd.read_csv(d / "nullmodel_summary.csv")
                return {"perturbation_spalten": list(pt.columns)[:12], "perturbation": pt.head(8).round(4).to_dict("records"),
                        "nullmodell": nm.round(4).to_dict("records")}
            versuch(e, "robustheit", robust)
            ev[f"phase{ph}"] = e

        def phasengrenze(R=R):
            j = json_lesen(R / "output_phase_boundary" / "phase_boundary_stability.json")
            return {"zellen": {f"{c.get('alt_split_year')}/P{c.get('phase')}":
                               {"topics": c.get("n_topics"), "rho_min": c.get("rho_min")} for c in j.get("cells", [])},
                    "gegen_kontrolle": {f"{c.get('alt_split_year')}/P{c.get('phase')}": c.get("rho_min")
                                        for c in j.get("cells_vs_control", [])}}
        versuch(ev, "phasengrenze", phasengrenze)
        versuch(ev, "uebergaenge", lambda R=R: json_lesen(R / "output_cross_phase" / "cross_phase_transitions.json"))
        versuch(ev, "klassenanteile_null", lambda R=R: pd.read_csv(R / "klassenanteile_null.csv").round(4).to_dict("records"))
        versuch(ev, "abbildungen_pdf", lambda R=R: sorted(f.name for f in (R / "figures_rp").glob("*.pdf")))
        versuch(ev, "supplement_dateien", lambda R=R: sorted(f.name for f in (R / "supplement_rp").glob("*")))
        versuch(ev, "citation_profil", lambda R=R: [f.name for f in R.glob("output_phase*/citation_topic_profile.csv")])
        erg["varianten"][v] = ev
    if not ctx.trocken:
        (ctx.neu / "pruefung.json").write_text(json.dumps(erg, indent=1, ensure_ascii=False, default=str),
                                              encoding="utf-8")
    return erg


def zusammenfassung(ctx: Ctx, erg: dict) -> None:
    ctx.log("")
    ctx.log("#" * 78)
    ctx.log("  ZUSAMMENFASSUNG")
    ctx.log("#" * 78)
    ctx.log(f"  Alte Ordner: {erg['alte_ordner'].get('status')}")
    for ph in (1, 2):
        e = erg.get("einheiten", {}).get(f"phase{ph}", {})
        rp = erg.get("einheiten_pruefung", {}).get(f"phase{ph}", {}).get("reproduzierbarkeit", {})
        ctx.log(f"  Einheiten Phase {ph}: {e.get('topics_neu')} Topics (berichtet {e.get('topics_berichtet')}), "
                f"ARI gegen berichtet {e.get('ari_gegen_berichtet')}; Wiederholung in dieser Umgebung: "
                f"UMAP identisch {rp.get('B_identisch')}, Topics {rp.get('C', {}).get('topics_b')}")
    ctx.log(f"  Mutual-Paare (neu, berichtet): {erg.get('einheiten_pruefung', {}).get('paare_mutual_neu_berichtet')}")
    for v, ev in erg["varianten"].items():
        for ph in (1, 2):
            e = ev.get(f"phase{ph}", {})
            s = e.get("sensitivitaet", {})
            ctx.log(f"  [{v}] Phase {ph}: Besetzung {e.get('memberships', {}).get('besetzung')}, "
                    f"Referenzzellen {s.get('referenzzellen_rho_min')}, Seed 42 ARI {s.get('seed42_ari')}")
        ctx.log(f"  [{v}] Abbildungen: {len(ev.get('abbildungen_pdf') or [])} PDF")
    ctx.log(f"  Einzelheiten: {ctx.neu / 'pruefung.json'}")


# --------------------------------------------------------------------------- Hauptprogramm

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nur", default=None, help="kommagetrennte Stufen: " + ",".join(STUFEN))
    ap.add_argument("--variante", default="beide", choices=["alpha5", "prior", "beide"])
    ap.add_argument("--ab-schritt", default=None, help="Wiederaufnahme innerhalb der Pipeline, z. B. 3.1")
    ap.add_argument("--bis-schritt", default=None, help="Pipeline nur bis zu diesem Schritt (einschließlich)")
    ap.add_argument("--trocken", action="store_true", help="nur den Plan zeigen")
    ap.add_argument("--weiter", action="store_true", help="in einen nicht leeren Variantenordner schreiben")
    ap.add_argument("--ohne-schwer", action="store_true",
                    help="Test ohne UMAP/HDBSCAN/SBERT: ohne Schritte 5b, 5, Phasengrenze, Abbildung S16, "
                         "Reproduzierbarkeitsprüfung")
    ap.add_argument("--test-einheiten-aus-bericht", action="store_true",
                    help="nur für Tests: Schritt-1-Ausgaben des berichteten Laufs statt neuer Einheitenbildung")
    ap.add_argument("--abbildungen-nur", default=None, help="an paper_figures_rp.py --only weitergeben")
    ap.add_argument("--ziel", default=None, help="Standard: der Ordner dieses Skripts")
    ap.add_argument("--repo", default=None, help="Standard: der übergeordnete Pipeline-Ordner")
    ap.add_argument("--kati", default=None)
    ap.add_argument("--publikation", default=None)
    ap.add_argument("--sicherung", default=None)
    a = ap.parse_args()

    ctx = Ctx(a)
    stufen = STUFEN if not a.nur else [s.strip() for s in a.nur.split(",") if s.strip()]
    unbekannt = [s for s in stufen if s not in STUFEN]
    if unbekannt:
        print("Unbekannte Stufe:", unbekannt)
        return 2
    varianten = list(VARIANTEN) if a.variante == "beide" else [a.variante]

    ctx.log("#" * 78)
    ctx.log(f"  VOLLSTÄNDIGER NEULAUF AB SCHRITT 1, DS3-KORREKTUR  {jetzt()}")
    ctx.log(f"  Ziel     {ctx.neu}")
    ctx.log(f"  Pipeline {ctx.repo}")
    ctx.log(f"  Stufen   {', '.join(stufen)}   Varianten {', '.join(varianten)}"
            + ("   (Trockenlauf)" if ctx.trocken else ""))
    ctx.log("#" * 78)
    try:
        if ctx.neu in [ctx.repo / d for d in ALTE_AUSGABEN] or ctx.neu == ctx.repo:
            raise Abbruch("Das Ziel darf kein Ordner des berichteten Laufs sein.")
        if ctx.neu.name == "output_neulauf_2026-09-29":
            raise Abbruch("Das Ziel darf nicht der Ordner des Neulaufs ab Schritt 2 sein.")
        if not a.nur and not a.weiter and not ctx.trocken:
            belegt = [v for v in varianten + ["einheiten"]
                      if (ctx.neu / v).exists() and any((ctx.neu / v).iterdir())]
            if belegt:
                raise Abbruch(f"Variantenordner nicht leer: {belegt}. Einzelne Stufen mit --nur, "
                              "oder bewusst überschreiben mit --weiter.")
        stufe_umgebung(ctx)
        pfad = ctx.logs / "schnappschuss_vorher.json"
        if not pfad.exists() and not ctx.trocken:
            pfad.write_text(json.dumps(schnappschuss(ctx)), encoding="utf-8")
            ctx.log(f"Schnappschuss der alten Ordner gespeichert ({pfad.name})")
        t0 = time.time()
        if "sicherung" in stufen:
            stufe_sicherung(ctx)
        if "daten" in stufen:
            stufe_daten(ctx)
            waechter(ctx, "nach der Datenaufbereitung")
        if "einheiten" in stufen:
            stufe_einheiten(ctx)
            waechter(ctx, "nach der Einheitenbildung")
        je_variante = [("saat", stufe_saat), ("pipeline", stufe_pipeline), ("profil", stufe_profil),
                       ("robustheit", stufe_robustheit), ("grenze", stufe_grenze),
                       ("abbildungen", stufe_abbildungen), ("supplement", stufe_supplement)]
        for v in varianten:
            if not any(s in stufen for s, _ in je_variante):
                break
            for s, f in je_variante:
                if s in stufen:
                    f(ctx, v)
            waechter(ctx, f"nach Variante {v}")
        if "pruefung" in stufen and not ctx.trocken:
            zusammenfassung(ctx, stufe_pruefung(ctx, varianten))
        ctx.log("")
        ctx.log(f"FERTIG nach {dauer(time.time() - t0)}")
        return 0
    except Abbruch as ex:
        ctx.log("")
        ctx.log(f"ABBRUCH: {ex}")
        ctx.log("Die alten Ordner sind nicht betroffen. Fortsetzen mit --nur <Stufe> [--variante ...] "
                "[--ab-schritt ...].")
        return 1
    except KeyboardInterrupt:
        ctx.log("Abgebrochen (Strg-C).")
        return 130


if __name__ == "__main__":
    sys.exit(main())
