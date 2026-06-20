"""
step02c_citation_topic_profile.py — Citation-Topic-Charakterisierung pro Topic.

Diagnostischer Seitenpfad (Schritt 2c, analog zu Schritt 2b). Aggregiert die
über KATI/FKIE durchgereichte Clarivate-Citation-Topic-Klassifikation
(Macro/Meso/Micro) pro semantisch gebildetem Topic zu einem rein deskriptiven
Profil. Die Felder gehen NICHT in die 16 Indikatoren, die Dimensions-
aggregation oder das Membership-Scoring ein (vgl. Methodik, Schritt 2c); sie
dienen ausschließlich der inhaltlichen Verortung in der Ergebnisdiskussion.

Pro Topic und je Hierarchieebene (Macro/Meso/Micro):
  - dominante Kategorie (Modus der Citation-Topic-Labels)
  - Modalanteil (Anteil der Publikationen in der dominanten Kategorie)
  - Anzahl distinkter Kategorien

Eingang: Injektion durch den Orchestrator (df, labels, topic_ids — wie
step02b) ODER Standalone. Standalone liest die stabilen CSV-Artefakte
    output/topic_assignments.csv   (UID, topic)
    <DATA_PATH>                    (UID, Citation Topic Macro/Meso/Micro)
und merged reihenerhaltend über UID — unabhängig von Pickle-Artefakten.

Ausgabe:
  output/citation_topic_profile.csv
  output/citation_topic_profile_table.tex   (nur falls signal_memberships.csv
                                              vorliegt; WS-dominante Topics)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import OUTPUT_DIR, DATA_PATH
except ImportError:
    OUTPUT_DIR = Path(__file__).parent / "output"
    DATA_PATH = None


CT_LEVELS = {
    "macro": "Citation Topic Macro",
    "meso":  "Citation Topic Meso",
    "micro": "Citation Topic Micro",
}

_MISSING = {"", "nan", "none", "na", "null"}


def _find_col(df: pd.DataFrame, *candidates: str):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _clean_values(series: pd.Series) -> list:
    out = []
    for v in series:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if s and s.lower() not in _MISSING:
            out.append(s)
    return out


def _level_stats(series: pd.Series):
    """Dominante Kategorie, Modalanteil, Anzahl distinkter Kategorien."""
    vals = _clean_values(series)
    if not vals:
        return "", float("nan"), 0
    vc = pd.Series(vals).value_counts()
    return str(vc.index[0]), float(vc.iloc[0]) / float(len(vals)), int(vc.size)


def compute_citation_topic_profile(df: pd.DataFrame, labels,
                                   topic_ids: list) -> pd.DataFrame:
    labels = np.asarray(labels)
    cols = {lvl: _find_col(df, name) for lvl, name in CT_LEVELS.items()}

    if cols["macro"] is None:
        print("  [step02c] WARNUNG: Keine 'Citation Topic Macro'-Spalte "
              "gefunden — Profil bleibt leer.")
        return pd.DataFrame({"topic": list(topic_ids)}).set_index("topic")

    present = [c for c in cols.values() if c]
    print(f"  [step02c] Citation-Topic-Profil pro Topic (Ebenen: {present})")

    records = []
    for tid in topic_ids:
        mask = labels == tid
        sub = df.loc[mask]
        n_pub = int(mask.sum())

        macro_col = cols["macro"]
        n_covered = len(_clean_values(sub[macro_col]))
        coverage = (n_covered / n_pub) if n_pub else float("nan")

        rec = {"topic": tid, "n_pub": n_pub, "ct_coverage": coverage}
        for lvl in ("macro", "meso", "micro"):
            col = cols[lvl]
            if col is None:
                rec[f"dominant_{lvl}"] = ""
                rec[f"modal_share_{lvl}"] = float("nan")
                rec[f"n_distinct_{lvl}"] = 0
                continue
            dom, share, ndist = _level_stats(sub[col])
            rec[f"dominant_{lvl}"] = dom
            rec[f"modal_share_{lvl}"] = share
            rec[f"n_distinct_{lvl}"] = ndist
        records.append(rec)

    return pd.DataFrame(records).set_index("topic")


# ---------------------------------------------------------------------------
# Standalone-Datenbeschaffung (ohne Pickle): topic_assignments.csv ⋈ DATA_PATH
# ---------------------------------------------------------------------------
def _load_standalone(output_dir: Path):
    ta_path = output_dir / "topic_assignments.csv"
    if not ta_path.exists():
        raise FileNotFoundError(
            f"[step02c] {ta_path} fehlt — bitte zuerst step01 ausführen.")
    ta = pd.read_csv(ta_path)

    df = ta
    if DATA_PATH is not None and Path(DATA_PATH).exists():
        src = pd.read_csv(DATA_PATH)
        ct_cols = [c for c in CT_LEVELS.values() if c in src.columns]
        if ct_cols and "UID" in ta.columns and "UID" in src.columns:
            df = ta.merge(src[["UID"] + ct_cols], on="UID", how="left")
        else:
            print("  [step02c] HINWEIS: UID-Merge mit DATA_PATH nicht möglich "
                  "— fahre mit topic_assignments.csv fort.")
    else:
        print("  [step02c] HINWEIS: DATA_PATH nicht verfügbar — es werden nur "
              "Spalten aus topic_assignments.csv genutzt.")

    if "topic" not in df.columns:
        raise KeyError(
            "[step02c] Spalte 'topic' fehlt in topic_assignments.csv.")
    labels = df["topic"].to_numpy()
    topic_ids = sorted(int(t) for t in pd.unique(labels)
                       if pd.notna(t) and t >= 0)
    return df, labels, topic_ids


# ---------------------------------------------------------------------------
# Optionale LaTeX-Tabelle: WS-dominante Topics × dominante Citation-Topics
# ---------------------------------------------------------------------------
def _escape_latex(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
            .replace("$", r"\$"))


def _emit_latex_table(profile: pd.DataFrame, output_dir: Path) -> None:
    """Schreibt die fertigen Datenzeilen (nur die \\\\-terminierten Body-Zeilen)
    der WS-dominanten Topics. Diese werden in das Inline-longtable-Gerüst der
    Anhang-Subsection (Tabelle tab:app_citation_topic_profile) eingefügt und
    ersetzen dort die Platzhalterzeile."""
    mem_path = output_dir / "signal_memberships.csv"
    if profile.empty or not mem_path.exists():
        return
    mem = pd.read_csv(mem_path, index_col=0)
    mcols = ["m_ws", "m_trend", "m_ec", "m_latent"]
    if not all(c in mem.columns for c in mcols):
        return

    argmax = mem[mcols].idxmax(axis=1)
    ws_topics = [t for t in profile.index
                 if t in argmax.index and argmax.loc[t] == "m_ws"]
    if not ws_topics:
        return
    sub = profile.loc[ws_topics].join(mem[["m_ws"]]).sort_values(
        "m_ws", ascending=False)

    lines = [
        r"% Auto-generiert durch step02c_citation_topic_profile.py.",
        r"% Diese Zeilen ersetzen die Platzhalterzeile in der Tabelle",
        r"% tab:app_citation_topic_profile (appendix.tex, Schritt 2c).",
    ]
    for t, r in sub.iterrows():
        share = r["modal_share_macro"]
        share_str = "--" if pd.isna(share) else f"{share * 100:.0f}\\%"
        lines.append(
            f"{int(t)} & {r['m_ws']:.3f} & {_escape_latex(r['dominant_macro'])} "
            f"& {share_str} & {_escape_latex(r['dominant_meso'])} \\\\")

    out = output_dir / "citation_topic_profile_rows.tex"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  [step02c] LaTeX-Datenzeilen geschrieben: {out}")


def run(df: pd.DataFrame = None, labels=None, topic_ids: list = None,
        output_dir: Path = None):
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if df is None or labels is None or topic_ids is None:
        df, labels, topic_ids = _load_standalone(output_dir)

    profile = compute_citation_topic_profile(df, labels, topic_ids)
    out_path = output_dir / "citation_topic_profile.csv"
    profile.to_csv(out_path)
    print(f"  [step02c] Gespeichert: {out_path}  ({len(profile)} Topics)")

    try:
        _emit_latex_table(profile, output_dir)
    except Exception as exc:  # Tabelle ist optional, niemals fatal
        print(f"  [step02c] LaTeX-Tabelle übersprungen ({exc}).")

    return profile


if __name__ == "__main__":
    run()
