#!/usr/bin/env python3
"""V2.6: Phasenschalter fuer die Skripte in v26_entwicklung (Protokoll V2.6, Abschnitt 3, Eintrag zur Umsetzung von
3.3 und 3.4, Punkte a bis c).

Umgebungsvariablen:
  V26_PHASE    1 (Standard) oder 2.
  V26_AUSGABE  Wurzel fuer alle Ausgaben (Trockenlauf, Test). Ohne Angabe gelten bei Phase 1 die Ordner der Entwicklung
               (1a bis 1i) und bei Phase 2 der Ordner output_v26_bestaetigung.

Entwicklungsmodus (Phase 1 ohne V26_AUSGABE): dieselben Pfade, derselbe Umfang (mit Rarefaktion, K7 und Pruefblatt) und
dasselbe Verhalten wie in 1a bis 1i. Bestaetigungsmodus (Phase 2, oder Phase 1 mit V26_AUSGABE fuer den Trockenlauf):
Ausgaben unter <Wurzel>/phase<N>/, Umfang wie in 3.4 (Referenz, 50 Teilungen, 50 Permutationen, vier Seeds; ohne
Rarefaktion, K7 und Pruefblatt).

Zugriffsprotokoll: Im Bestaetigungsmodus geht jeder Lesezugriff auf die Eingaben der Phase (WoS-Export und Ausgaben des
berichteten Laufs unter output_neulauf_v2.5/prior/output_phase<N>) und jeder Zugriff auf die phasenuebergreifenden Paare
ueber lies(). lies() verlangt einen gestarteten Lauf (bestaetigung.py start) und schreibt vor dem Lesen Zeit (UTC), Datei,
Groesse, SHA-256, Zweck, Skript und Teil in zugriffsprotokoll.jsonl. Fuer Phase 2 liegen Laufregister und Protokoll immer
in V26/output_v26_bestaetigung, gleich welche Ausgabewurzel gesetzt ist; ein zweiter Lauf braucht einen ausdruecklich
angegebenen Grund und wird gezaehlt. Die phasenuebergreifenden Paare sind nur in einem Lauf der Phase 2 lesbar.

Nur Standardbibliothek: dieselbe Datei laeuft in der Pipeline-Umgebung auf dem Mac und in der Linux-Umgebung.
"""
import hashlib
import json
import os
import platform
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

HIER = Path(__file__).resolve().parent
V26 = HIER.parent

_roh = os.environ.get("V26_PHASE", "1").strip() or "1"
if _roh not in ("1", "2"):
    raise SystemExit(f"V26_PHASE muss 1 oder 2 sein, nicht {_roh!r}")
PHASE = int(_roh)
P = f"phase{PHASE}"

LAUF = V26 / "output_neulauf_v2.5"
REF = LAUF / "prior" / f"output_{P}"
KREUZ = LAUF / "prior" / "output_cross_phase"
DATA = LAUF / ("wos_qc_phase1_2000_2015.csv" if PHASE == 1 else "wos_qc_phase2_2016_2025.csv")
JAHR_MIN, JAHR_MAX = (2000, 2015) if PHASE == 1 else (2016, 2025)
NEU_AB = JAHR_MIN + 3        # DS2 A3: Dokumente ab dem vierten Jahr des Fensters (Phase 1: 2003, Phase 2: 2019)
KOHORTE_AB = JAHR_MAX - 4    # IP2 A5: die juengsten fuenf Jahre (Phase 1: 2011 bis 2015, Phase 2: 2021 bis 2025)

_aus = os.environ.get("V26_AUSGABE", "").strip()
if _aus:
    WURZEL = Path(_aus).expanduser()
    if not WURZEL.is_absolute():
        WURZEL = V26 / WURZEL
elif PHASE == 2:
    WURZEL = V26 / "output_v26_bestaetigung"
else:
    WURZEL = None
ENTWICKLUNG = WURZEL is None
RARE_N = 30 if ENTWICKLUNG else 0   # Rarefaktion nur in der Entwicklung (3.4: entfaellt, geht nicht in B1 und B2 ein)
K7 = ENTWICKLUNG                    # K7 nur in der Entwicklung (ebenso)
PRUEFBLATT = ENTWICKLUNG            # Pruefblatt nur in der Entwicklung (3.4: auf Phase 2 nicht wiederholt)

# Laufregister: Phase 2 immer am festen Ort dieses Baums; Trockenlauf mit Phase 1 in seiner eigenen Wurzel
REGISTER = (V26 / "output_v26_bestaetigung") if PHASE == 2 else WURZEL

def _pruefe_wurzel():
    """Die Ausgabewurzel liegt im Pipelineordner, aber nicht in den Entwicklungsausgaben, nicht im berichteten Lauf und
    (fuer den Trockenlauf mit Phase 1) nicht im Laufregister der Phase 2."""
    w, v = WURZEL.resolve(), V26.resolve()
    if v not in w.parents:
        raise SystemExit("V26_AUSGABE muss ein Ordner innerhalb des Pipelineordners sein")
    verboten = [V26 / "output_v26_entwicklung", LAUF] + ([V26 / "output_v26_bestaetigung"] if PHASE == 1 else [])
    for d in verboten:
        d = d.resolve()
        if w == d or d in w.parents or w in d.parents:
            raise SystemExit(f"V26_AUSGABE darf nicht auf {d.name} zeigen oder darin liegen")


if not ENTWICKLUNG:
    _pruefe_wurzel()


def ordner(alt, neu):
    """Ausgabeordner: in der Entwicklung der bisherige Ordner (relativ zu V26), sonst <Wurzel>/phase<N>/<neu>."""
    return V26 / alt if ENTWICKLUNG else WURZEL / P / neu


def csv_atomar(df, pfad, **kw):
    """Schreibt df als CSV erst in eine temporaere Datei und ersetzt dann pfad; ein Abbruch hinterlaesst keinen halben Stand."""
    pfad = Path(pfad)
    tmp = pfad.with_name(pfad.name + ".tmp")
    df.to_csv(tmp, **kw)
    os.replace(tmp, pfad)


# ---------------------------------------------------------------- Laufregister und Zugriffsprotokoll

def jetzt():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def relativ(pfad):
    p = Path(pfad).resolve()
    try:
        return str(p.relative_to(V26.resolve()))
    except ValueError:
        return p.name


_SHA = {}


def sha256(pfad):
    p = Path(pfad)
    st = p.stat()
    schluessel = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if schluessel not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        _SHA[schluessel] = h.hexdigest()
    return _SHA[schluessel]


def _unter(p, basis):
    b = Path(basis).resolve()
    return p == b or b in p.parents


def geschuetzt(pfad):
    """'kreuz' fuer die phasenuebergreifenden Paare, 'phase' fuer Eingaben der Phase im Bestaetigungsmodus, sonst None."""
    p = Path(pfad).resolve()
    if _unter(p, KREUZ):
        return "kreuz"
    if not ENTWICKLUNG and (p == DATA.resolve() or _unter(p, REF)):
        return "phase"
    return None


def _register_datei():
    return REGISTER / "lauf.json"


def lies_register():
    f = _register_datei()
    return json.loads(f.read_text(encoding="utf-8")) if REGISTER is not None and f.exists() else None


def _schreibe_register(reg):
    REGISTER.mkdir(parents=True, exist_ok=True)
    tmp = _register_datei().with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, _register_datei())


def skript_pruefsummen():
    aus = {}
    for ordnername in ("v26_entwicklung", "v26_zweitumsetzung"):
        d = V26 / ordnername
        if d.is_dir():
            for f in sorted(d.glob("*.py")):
                aus[f"{ordnername}/{f.name}"] = sha256(f)
    return aus


_SKRIPTE = {}


def aktiver_lauf():
    """Der offene Lauf dieses Registers. Verlangt dieselbe Phase, dieselbe Ausgabewurzel und dieselben Skripte wie beim
    Start, fuer Phase 2 zusaetzlich V26_GETAGGT=v2.6-frozen; sonst Abbruch ohne Lesezugriff."""
    if ENTWICKLUNG:
        raise SystemExit("Im Entwicklungsmodus gibt es keinen Lauf")
    reg = lies_register()
    if reg is None or not reg.get("laeufe"):
        raise SystemExit("Kein Lauf gestartet: zuerst 'python3 bestaetigung.py start' (mit derselben V26_PHASE)")
    lauf = reg["laeufe"][-1]
    if reg.get("phase") != PHASE or lauf.get("phase") != PHASE:
        raise SystemExit(f"Das Register gehoert zu Phase {reg.get('phase')}, nicht zu Phase {PHASE}")
    if lauf.get("abgeschlossen_utc"):
        raise SystemExit(f"Lauf {lauf['lauf_id']} ist abgeschlossen; ein weiterer Lauf braucht "
                         "'python3 bestaetigung.py start --grund \"...\"' und eine neue Ausgabewurzel")
    if PHASE == 2 and os.environ.get("V26_GETAGGT") != "v2.6-frozen":
        raise SystemExit("Phase 2 erst nach dem Tag v2.6-frozen (Protokoll 3.7): V26_GETAGGT=v2.6-frozen setzen")
    if relativ(WURZEL) != lauf.get("wurzel"):
        raise SystemExit(f"Ausgabewurzel {relativ(WURZEL)} weicht vom Start des Laufs ab ({lauf.get('wurzel')})")
    if "jetzt" not in _SKRIPTE:
        _SKRIPTE["jetzt"] = skript_pruefsummen()
    if _SKRIPTE["jetzt"] != lauf.get("skripte_sha256"):
        abw = sorted(k for k in set(_SKRIPTE["jetzt"]) | set(lauf.get("skripte_sha256", {}))
                     if _SKRIPTE["jetzt"].get(k) != lauf.get("skripte_sha256", {}).get(k))
        raise SystemExit(f"Skripte seit dem Start des Laufs geaendert: {', '.join(abw)}")
    return lauf


def starte_lauf(grund=None):
    """Neuer Lauf. Besteht schon einer, nur mit ausdruecklichem Grund (wird gezaehlt und berichtet)."""
    if ENTWICKLUNG:
        raise SystemExit("Im Entwicklungsmodus gibt es keinen Lauf (V26_PHASE=2 oder V26_AUSGABE setzen)")
    if PHASE == 2 and os.environ.get("V26_GETAGGT") != "v2.6-frozen":
        raise SystemExit("Phase 2 erst nach dem Tag v2.6-frozen (Protokoll 3.7): V26_GETAGGT=v2.6-frozen setzen")
    reg = lies_register() or {"phase": PHASE, "laeufe": []}
    if reg.get("phase") != PHASE:
        raise SystemExit(f"Das Register gehoert zu Phase {reg.get('phase')}, nicht zu Phase {PHASE}")
    if reg["laeufe"] and not (grund and grund.strip()):
        alt = reg["laeufe"][-1]
        raise SystemExit(f"Es besteht bereits Lauf {alt['lauf_id']} (gestartet {alt['start_utc']}). Ein Wiederaufsetzen "
                         "gehoert zu diesem Lauf und braucht keinen Start. Ein zweiter Lauf braucht einen Grund: "
                         "'python3 bestaetigung.py start --grund \"...\"'")
    if relativ(WURZEL) in [x.get("wurzel") for x in reg["laeufe"]]:
        raise SystemExit("Ein zweiter Lauf braucht eine neue Ausgabewurzel (V26_AUSGABE); die alten Ausgaben bleiben")
    if reg["laeufe"] and not reg["laeufe"][-1].get("abgeschlossen_utc"):
        reg["laeufe"][-1]["abgebrochen_utc"] = jetzt()
    lauf = {"lauf_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3),
            "nummer": len(reg["laeufe"]) + 1, "start_utc": jetzt(), "grund": (grund or None), "phase": PHASE,
            "wurzel": relativ(WURZEL), "python": platform.python_version(), "plattform": platform.platform(),
            "skripte_sha256": skript_pruefsummen()}
    reg["laeufe"].append(lauf)
    _schreibe_register(reg)
    return lauf


def schliesse_lauf(zusammenfassung=None, ergebnis_datei=None):
    aktiver_lauf()
    reg = lies_register()
    lauf = reg["laeufe"][-1]
    lauf["abgeschlossen_utc"] = jetzt()
    if zusammenfassung is not None:
        lauf["ergebnis"] = zusammenfassung
    if ergebnis_datei is not None:
        lauf["ergebnis_datei"] = {"datei": relativ(ergebnis_datei), "sha256": sha256(ergebnis_datei)}
    _schreibe_register(reg)
    return lauf


def lies(pfad, zweck):
    """Gibt pfad zurueck. Bei geschuetzten Eingaben vorher Eintrag ins Zugriffsprotokoll (verlangt einen aktiven Lauf)."""
    art = geschuetzt(pfad)
    if art is None:
        return pfad
    if art == "kreuz" and PHASE != 2:
        raise SystemExit("Die phasenuebergreifenden Paare sind nur in einem Lauf der Phase 2 lesbar")
    lauf = aktiver_lauf()
    p = Path(pfad)
    eintrag = {"zeit_utc": jetzt(), "lauf_id": lauf["lauf_id"], "phase": PHASE, "wurzel": relativ(WURZEL), "art": art,
               "datei": relativ(p), "bytes": p.stat().st_size, "sha256": sha256(p), "zweck": zweck,
               "skript": Path(sys.argv[0]).name, "teil": " ".join(sys.argv[1:])[:200]}
    REGISTER.mkdir(parents=True, exist_ok=True)
    with open(REGISTER / "zugriffsprotokoll.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(eintrag, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return pfad


def zugriffe():
    f = (REGISTER / "zugriffsprotokoll.jsonl") if REGISTER is not None else None
    if f is None or not f.exists():
        return []
    return [json.loads(z) for z in f.read_text(encoding="utf-8").splitlines() if z.strip()]


def beschreibung():
    return {"phase": PHASE, "entwicklung": ENTWICKLUNG, "wurzel": None if WURZEL is None else relativ(WURZEL),
            "fenster": [JAHR_MIN, JAHR_MAX], "neu_ab": NEU_AB, "kohorte_ab": KOHORTE_AB, "rarefaktion": RARE_N,
            "k7": K7, "pruefblatt": PRUEFBLATT}


if __name__ == "__main__":
    print(json.dumps(beschreibung(), ensure_ascii=False))
